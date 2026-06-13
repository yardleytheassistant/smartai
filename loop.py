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

from dataclasses import dataclass, field
from typing import Callable

import memory as memory_mod
from agent import NovelAgent
from config import config
from verifier import Verdict, Verifier


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
    rubric: str | None = None,
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
    current_input = task
    output = ""
    verdict: Verdict | None = None

    for i in range(1, max_iterations + 1):
        emit("iteration", n=i, max=max_iterations)
        # Fresh maker each iteration: it gets the original task + verifier feedback,
        # not a polluted transcript. Memory is consulted via the system prompt.
        maker = NovelAgent(model=maker_model, client=client, load_memory=use_memory)
        output = maker.run(current_input)

        verdict = verifier.grade(goal=task, artifact=output, rubric=rubric)
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

        # Not met: feed the gap back to the next maker.
        current_input = (
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
