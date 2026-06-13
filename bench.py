"""Benchmark models against a task set — validate the role→model mapping.

The fleet is assigned to roles by reputation; this measures it. Run a small set
of tasks across several models, grade each output with the independent verifier,
and produce a scorecard plus latency. Use it to confirm (or revise) which model
should be the orchestrator, the worker, the grader, and so on — empirically,
on your own hardware, rather than by vibes.

Tasks reuse the eval-case format (id / input / rubric), so an existing eval
suite doubles as a benchmark set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from evals import EvalCase, load_cases  # noqa: F401  (re-exported for convenience)
from verifier import Verifier


@dataclass
class BenchResult:
    model: str
    task_id: str
    score: float
    met: bool
    latency_s: float


@dataclass
class BenchReport:
    results: list[BenchResult] = field(default_factory=list)

    def models(self) -> list[str]:
        seen: list[str] = []
        for r in self.results:
            if r.model not in seen:
                seen.append(r.model)
        return seen

    def avg_score(self, model: str) -> float:
        rs = [r.score for r in self.results if r.model == model]
        return sum(rs) / len(rs) if rs else 0.0

    def pass_rate(self, model: str) -> float:
        rs = [r for r in self.results if r.model == model]
        return sum(1 for r in rs if r.met) / len(rs) if rs else 0.0

    def avg_latency(self, model: str) -> float:
        rs = [r.latency_s for r in self.results if r.model == model]
        return sum(rs) / len(rs) if rs else 0.0

    def ranked(self) -> list[str]:
        return sorted(self.models(), key=lambda m: (self.avg_score(m), -self.avg_latency(m)), reverse=True)

    def best(self) -> str | None:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def scorecard(self) -> str:
        lines = [f"{'model':<22} {'score':>6} {'pass':>6} {'latency':>9}"]
        for m in self.ranked():
            lines.append(
                f"{m:<22} {self.avg_score(m):>6.2f} {self.pass_rate(m):>5.0%} {self.avg_latency(m):>8.2f}s"
            )
        return "\n".join(lines)

    def suggested_roles(self) -> dict[str, str]:
        """A naive role suggestion from the results: best score -> orchestrator,
        fastest among those that pass the bar -> worker."""
        ranked = self.ranked()
        if not ranked:
            return {}
        passers = [m for m in ranked if self.pass_rate(m) >= 0.5] or ranked
        worker = min(passers, key=self.avg_latency)
        return {"orchestrator": ranked[0], "worker": worker, "grader": ranked[-1]}


def run_bench(models, tasks, *, client=None, run_fn=None, verifier: Verifier | None = None) -> BenchReport:
    """Run each model over each task, grade, and time it.

    `run_fn(model, task)->output` is injectable; the default runs a fresh
    tool-less agent so we measure raw model quality, not tool plumbing.
    """
    if run_fn is None:
        from agent import NovelAgent

        def run_fn(model: str, task: EvalCase) -> str:  # noqa: F811
            return NovelAgent(model=model, client=client, use_tools=False, compact=False).run(task.input)

    verifier = verifier or Verifier(client=client)
    report = BenchReport()
    for model in models:
        for task in tasks:
            start = time.perf_counter()
            output = run_fn(model, task)
            latency = time.perf_counter() - start
            verdict = verifier.grade(goal=task.input, artifact=output, rubric=task.rubric or None)
            report.results.append(
                BenchResult(model=model, task_id=task.id, score=verdict.score,
                            met=verdict.met, latency_s=latency)
            )
    return report
