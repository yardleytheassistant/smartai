"""Parallel experiments — run competing approaches and keep the winner.

This wires three primitives together: fan-out (run N approaches in parallel),
sub-agent delegation (each approach is a fresh agent on a routed model), and the
independent verifier (each result is graded against the same rubric). The best
result wins. For code-producing work, `run_in_worktrees` gives each experiment
its own git checkout so they can't collide, and optionally merges the winner.

This is the "parallel structural experiments" pattern: the orchestrator explores
several hypotheses at once, collects graded results, and merges the best one.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config import config
from verifier import Verdict, Verifier
from workflows import fan_out_synthesize

# Makers fan out with clean context, so an "implement X in this repo" task with
# no source produces plausible-but-wrong code (invented modules/APIs). Folding the
# real source into BOTH the maker and the grader makes "fits the codebase" a thing
# the system can actually produce and grade, not just "sounds complete".
_MAKER_CONTEXT_HEADER = (
    "EXISTING SOURCE you must fit — match its modules, APIs, naming, and "
    "conventions. Do NOT invent files, functions, classes, imports, CLI flags, or "
    "frameworks that don't appear below. The source is already provided here, so do "
    "NOT plan to read files or describe what you would do — output the ACTUAL code "
    "(concrete edits or full functions that drop into these files):"
)


def build_context(paths, *, root=None) -> str:
    """Read the given files into a labelled source bundle for grounding.

    Paths are resolved relative to `root` (default: the current directory), so a
    caller can pass repo-relative paths like ['bench.py', 'main.py'].
    """
    base = Path(root) if root is not None else Path.cwd()
    chunks: list[str] = []
    for p in paths:
        fp = Path(p) if Path(p).is_absolute() else base / p
        try:
            text = fp.read_text(encoding="utf-8")
        except OSError as exc:
            chunks.append(f"# file: {p} (could not read: {exc})")
            continue
        chunks.append(f"# file: {p}\n{text}")
    return "\n\n".join(chunks)


@dataclass
class Experiment:
    variant: str
    output: str
    score: float
    met: bool
    branch: str = ""


@dataclass
class ExperimentReport:
    experiments: list[Experiment] = field(default_factory=list)

    def ranked(self) -> list[Experiment]:
        # Prefer experiments that met the bar, then by score.
        return sorted(self.experiments, key=lambda e: (e.met, e.score), reverse=True)

    @property
    def winner(self) -> Experiment | None:
        ranked = self.ranked()
        return ranked[0] if ranked else None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "exp"


def _normalize(variants: "list[str] | int") -> list[str]:
    if isinstance(variants, int):
        return [f"approach {i + 1}" for i in range(variants)]
    return list(variants)


def run_experiments(
    task: str,
    variants: "list[str] | int",
    *,
    rubric=None,
    context: str = "",
    client=None,
    make_fn: Callable[[str], str] | None = None,
    grade_fn: Callable[[str], Verdict] | None = None,
    max_workers: int | None = None,
) -> ExperimentReport:
    """Run each variant in parallel, grade it, and rank the results.

    `make_fn(variant)->output` and `grade_fn(output)->Verdict` are injectable;
    the defaults delegate to a routed sub-agent and the independent verifier.
    `context` (e.g. from `build_context`) is real source folded into both the
    maker and the grader so outputs fit the codebase instead of inventing it.
    """
    variants = _normalize(variants)

    if make_fn is None:
        import subagents

        def make_fn(variant: str) -> str:  # noqa: F811
            body = f"{task}\n\nApproach to try: {variant}"
            if context:
                body = f"{_MAKER_CONTEXT_HEADER}\n\n{context}\n\n---\n\n{body}"
            # With the source inlined, the maker should answer from the prompt, not
            # try to read files — tool access here just makes it emit an un-executed
            # read_file call that leaks into the graded output. So disable tools.
            return subagents.delegate(body, client=client, use_tools=not context)

    if grade_fn is None:
        verifier = Verifier(client=client)

        def grade_fn(output: str) -> Verdict:  # noqa: F811
            return verifier.grade(goal=task, artifact=output, rubric=rubric, context=context)

    def worker(variant: str) -> Experiment:
        output = make_fn(variant)
        verdict = grade_fn(output)
        return Experiment(variant=variant, output=output, score=verdict.score, met=verdict.met)

    return fan_out_synthesize(
        variants, worker, lambda exps: ExperimentReport(list(exps)), max_workers=max_workers
    )


@dataclass
class LandResult:
    """Outcome of landing a winning direction. `met` is the verifier's verdict;
    `files_written` is what actually changed on disk. A verifier 'met' with no
    files written is NOT a landed change — it means the maker described the work
    instead of doing it. `landed` requires both, so success isn't a claim on faith.
    """

    met: bool
    iterations: int
    output: str
    files_written: list[str] = field(default_factory=list)
    verdict: Verdict | None = None

    @property
    def landed(self) -> bool:
        return self.met and bool(self.files_written)


def _artifact_files(root) -> dict[str, str]:
    """Map artifact files in the workspace to a content hash, ignoring the system's
    own bookkeeping (durable memory, traces, sessions) so only real code shows up."""
    base = Path(root)
    if not base.is_dir():
        return {}
    ignore_names = {Path(config.memory_file).name, ".doctor-write-test"}
    ignore_dirs = {"traces", "sessions"}
    out: dict[str, str] = {}
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        if (rel.parts and rel.parts[0] in ignore_dirs) or p.name in ignore_names:
            continue
        try:
            out[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            continue
    return out


def land_winner(
    task: str,
    winner: Experiment,
    *,
    rubric=None,
    context: str = "",
    client=None,
    max_iterations: int | None = None,
    use_memory: bool = True,
    on_event=None,
) -> LandResult:
    """Hand the winning direction to a goal loop and verify it actually wrote code.

    Experiments explore *directions*; this is the opt-in second leg that turns the
    chosen direction into a verifier-checked artifact in the workspace. It snapshots
    the workspace before and after, so `LandResult.landed` reflects real disk writes,
    not just the verifier's verdict (a maker can describe a change without doing it).
    Note: this writes to the workspace sandbox, not the repo — for edits to existing
    repo files, use `run_in_worktrees`. The same `context` grounds maker and grader.
    """
    from loop import goal_loop

    land_task = (
        f"{task}\n\nUse this approach, chosen by a parallel experiment as the best "
        f"direction:\n{winner.output}\n\nActually write the file(s) with the "
        f"write_file tool — do not just describe the change."
    )
    before = _artifact_files(config.workspace)
    result = goal_loop(
        land_task,
        rubric=rubric,
        context=context,
        client=client,
        max_iterations=max_iterations,
        use_memory=use_memory,
        on_event=on_event,
    )
    after = _artifact_files(config.workspace)
    files_written = sorted(p for p in after if after[p] != before.get(p))
    return LandResult(
        met=result.met,
        iterations=result.iterations,
        output=result.output,
        files_written=files_written,
        verdict=result.verdict,
    )


def run_in_worktrees(
    task: str,
    variants: "list[str] | int",
    runner: Callable[[str, object], str],
    *,
    rubric=None,
    context: str = "",
    client=None,
    base: str = "HEAD",
    root=None,
    grade_fn: Callable[[str], Verdict] | None = None,
    merge_winner: bool = False,
    cleanup_losers: bool = True,
) -> ExperimentReport:
    """Run each variant in its own git worktree; optionally merge the winner.

    `runner(variant, worktree)->output` does the work inside worktree.path (e.g.
    writes code and commits). Each result is graded; the best can be merged back
    into the current branch and the losing worktrees/branches cleaned up.
    """
    import worktrees

    if grade_fn is None:
        verifier = Verifier(client=client)

        def grade_fn(output: str) -> Verdict:  # noqa: F811
            return verifier.grade(goal=task, artifact=output, rubric=rubric, context=context)

    made: list[tuple[Experiment, object]] = []
    for variant in _normalize(variants):
        branch = f"exp/{_slug(variant)}"
        wt = worktrees.create(branch, base=base, root=root)
        output = runner(variant, wt)
        verdict = grade_fn(output)
        made.append((Experiment(variant, output, verdict.score, verdict.met, branch=branch), wt))

    report = ExperimentReport([e for e, _ in made])
    winner = report.winner

    if merge_winner and winner is not None:
        worktrees._git("merge", "--no-ff", "--no-edit", winner.branch, cwd=root)

    if cleanup_losers:
        for exp, wt in made:
            if winner is not None and exp.branch == winner.branch and merge_winner:
                continue  # keep the merged winner's branch around
            try:
                worktrees.remove(wt, root=root)
                worktrees._git("branch", "-D", exp.branch, cwd=root)
            except worktrees.WorktreeError:
                pass

    return report
