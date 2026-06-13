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

import re
from dataclasses import dataclass, field
from typing import Callable

from verifier import Verdict, Verifier
from workflows import fan_out_synthesize


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
    client=None,
    make_fn: Callable[[str], str] | None = None,
    grade_fn: Callable[[str], Verdict] | None = None,
    max_workers: int | None = None,
) -> ExperimentReport:
    """Run each variant in parallel, grade it, and rank the results.

    `make_fn(variant)->output` and `grade_fn(output)->Verdict` are injectable;
    the defaults delegate to a routed sub-agent and the independent verifier.
    """
    variants = _normalize(variants)

    if make_fn is None:
        import subagents

        def make_fn(variant: str) -> str:  # noqa: F811
            return subagents.delegate(f"{task}\n\nApproach to try: {variant}", client=client)

    if grade_fn is None:
        verifier = Verifier(client=client)

        def grade_fn(output: str) -> Verdict:  # noqa: F811
            return verifier.grade(goal=task, artifact=output, rubric=rubric)

    def worker(variant: str) -> Experiment:
        output = make_fn(variant)
        verdict = grade_fn(output)
        return Experiment(variant=variant, output=output, score=verdict.score, met=verdict.met)

    return fan_out_synthesize(
        variants, worker, lambda exps: ExperimentReport(list(exps)), max_workers=max_workers
    )


def run_in_worktrees(
    task: str,
    variants: "list[str] | int",
    runner: Callable[[str, object], str],
    *,
    rubric=None,
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
            return verifier.grade(goal=task, artifact=output, rubric=rubric)

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
