"""The goal loop — self-correcting maker/verifier iteration.

This is smartai's equivalent of /goal + Outcomes, built on open-source models:
a maker agent produces an artifact, an INDEPENDENT verifier grades it against a
rubric, and a "not met" verdict starts the next iteration with the verifier's
feedback folded in. The loop exits when the grader passes or max_iterations is
hit.

Rather than directly steering the model, we design a loop that lets it
self-correct in response to feedback and manage its own durable memory — so the
run compounds: lessons from a passing run, and open failures from a stuck run,
are written back to the state file for next time.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Callable

import memory as memory_mod
from agent import NovelAgent
from config import config
from verifier import Verdict, Verifier


def _run_check(cmd: str, cwd: str) -> tuple[bool, str]:
    """Run a behavioral check command; return (passed, tail of output).

    This is how the loop verifies the change actually *works*, not just that it
    reads correctly — the operator's own integration check (e.g. run the CLI, run
    a test), gated on the real exit code.
    """
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        return False, f"check `{cmd}` timed out after 120s"
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    return r.returncode == 0, out[-2000:]


@dataclass
class LoopResult:
    met: bool
    iterations: int
    output: str
    verdict: Verdict | None
    history: list[dict] = field(default_factory=list)


def goal_loop(
    task: str,
    *,
    rubric=None,  # str | rubric.Rubric | None — passed through to the verifier
    context: str = "",  # real source to ground the maker + grader on (fit, don't invent)
    require_file_write: bool = False,  # reject iterations that describe instead of writing files
    check_cmd: str = "",  # shell command that must exit 0 — behavioral gate, not just text grading
    max_iterations: int | None = None,
    maker_model: str | None = None,
    grader_model: str | None = None,
    client=None,
    use_memory: bool = True,
    on_event: Callable[[str, dict], None] | None = None,
) -> LoopResult:
    """Run a maker/verifier loop until the goal is met or iterations run out.

    `client` is injectable so the whole loop is testable without a live server.
    `on_event(kind, data)` receives "iteration", "verdict", "done" events.
    """
    max_iterations = max_iterations or config.max_iterations
    verifier = Verifier(model=grader_model, client=client)

    def emit(kind: str, **data) -> None:
        if on_event is not None:
            on_event(kind, data)

    history: list[dict] = []
    # Standing instructions that must persist on EVERY iteration, not just the first:
    # source grounding (fit the real code) and the explicit "use the write_file tool"
    # nudge (smaller models read "write the code" as "print code in markdown"). These
    # are prepended to each maker input — dropping them on retries is exactly when the
    # maker, having already failed once, most needs them.
    standing: list[str] = []
    if context:
        standing.append(
            "EXISTING SOURCE you must fit — match its modules, APIs, naming, and "
            "conventions; do not invent things that don't appear below:\n" + context
        )
    if require_file_write:
        standing.append(
            "You MUST apply changes by calling the write_file tool. Pasting code in a "
            "markdown block does NOT modify any file and will be rejected — actually "
            "call the tool."
        )
    standing_block = "\n\n---\n\n".join(standing)

    def _maker_input(body: str) -> str:
        return f"{standing_block}\n\n---\n\n{body}" if standing_block else body

    current_input = _maker_input(task)
    output = ""
    verdict: Verdict | None = None

    for i in range(1, max_iterations + 1):
        emit("iteration", n=i, max=max_iterations)
        # Fresh maker each iteration: it gets the original task + verifier feedback,
        # not a polluted transcript. Memory is consulted via the system prompt.
        maker = NovelAgent(
            model=maker_model, client=client, load_memory=use_memory, skills_query=task
        )
        output = maker.run(current_input)

        # The independent verifier only sees the artifact text. The loop, which can
        # see the maker's tool trace, enforces that the action actually happened: an
        # iteration that never called write_file when files were required is not-met,
        # so the loop iterates toward doing the work instead of describing it.
        if require_file_write and "write_file" not in getattr(maker, "tools_used", []):
            verdict = Verdict(
                met=False,
                score=0.0,
                feedback=(
                    "You did not call the write_file tool, so no file was changed. "
                    "Describing the edit or pasting code in text does not count — call "
                    "write_file with the path and full new contents."
                ),
            )
        else:
            verdict = verifier.grade(goal=task, artifact=output, rubric=rubric, context=context)

        # Behavioral gate: a verifier reads the artifact; this RUNS it. If the change
        # reads correct but doesn't work (e.g. a flag added to a signature but never
        # wired into the CLI), the real exit code catches what static grading misses.
        if check_cmd and verdict.met:
            ok, check_out = _run_check(check_cmd, cwd=config.workspace)
            if not ok:
                verdict = Verdict(
                    met=False,
                    score=min(verdict.score, 0.5),
                    feedback=(
                        f"The code reads correct but does NOT work: the check command "
                        f"`{check_cmd}` failed. Make it pass — update every call site, not "
                        f"just one (a new CLI flag needs both the arg parser and the "
                        f"function).\n--- check output ---\n{check_out}"
                    ),
                )
        history.append({"iteration": i, "output": output, "verdict": verdict})
        emit("verdict", n=i, met=verdict.met, score=verdict.score, feedback=verdict.feedback)

        if verdict.met:
            if use_memory:
                mem = memory_mod.load()
                mem.distill_lesson(f"Goal met in {i} iteration(s): {task}")
                mem.resolve_failure(task)
                mem.set_last_session(f"Goal loop passed: {task!r} (score {verdict.score:.2f}).")
            emit("done", met=True, iterations=i)
            return LoopResult(met=True, iterations=i, output=output, verdict=verdict, history=history)

        # Not met: feed the gap back to the next maker, re-attaching the standing
        # instructions (grounding + tool-use) so the retry keeps them.
        current_input = _maker_input(
            f"{task}\n\n"
            f"A verifier reviewed your previous attempt and it did NOT meet the goal.\n"
            f"Score: {verdict.score:.2f}\n"
            f"Required fixes:\n{verdict.feedback}\n\n"
            f"Your previous attempt was:\n{output}\n\n"
            f"Produce a corrected result that satisfies the goal and rubric."
        )

    if use_memory:
        mem = memory_mod.load()
        fb = verdict.feedback if verdict else "unknown"
        mem.log_failure(f"Goal unmet after {max_iterations} iterations: {task} — last gap: {fb}")
        mem.set_last_session(f"Goal loop stuck on: {task!r} after {max_iterations} iterations.")
    emit("done", met=False, iterations=max_iterations)
    return LoopResult(met=False, iterations=max_iterations, output=output, verdict=verdict, history=history)
