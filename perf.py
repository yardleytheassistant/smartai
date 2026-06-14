"""Goal-loop battery — measure the whole compound stack, not just raw output.

`bench.py` runs single-shot, tool-less generations to rank models. This runs the
*goal loop* (maker -> independent verifier -> memory) over a task set and reports
per-task iterations, score, and wall-clock, plus an aggregate. It's the
version-controlled, reproducible form of a live "did the system actually hold up
end to end" run: same code path a live endpoint hits, exits non-zero if any task
fails.

Like every model-calling component here, the client is injectable, so the whole
battery runs offline against a fake client in tests. The runner is pure — it does
not mutate global config; callers that need workspace isolation (so the battery
doesn't write the real STATE.md) set `config.workspace` themselves. The CLI
(`main.py perf`) does this with a temp dir; tests use the `tmp_workspace` fixture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from evals import EvalCase, load_cases  # noqa: F401  (re-exported for convenience)


@dataclass
class LoopBenchResult:
    task_id: str
    met: bool
    iterations: int
    score: float
    wall_s: float


@dataclass
class LoopBenchReport:
    results: list[LoopBenchResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return sum(1 for r in self.results if r.met) / len(self.results) if self.results else 0.0

    @property
    def avg_iterations(self) -> float:
        return sum(r.iterations for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def avg_wall(self) -> float:
        return sum(r.wall_s for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.met for r in self.results)

    def scorecard(self) -> str:
        lines = [f"{'task':<24} {'met':>4} {'iters':>6} {'score':>6} {'wall':>9}"]
        for r in self.results:
            mark = "✓" if r.met else "✗"
            lines.append(
                f"{r.task_id:<24} {mark:>4} {r.iterations:>6} {r.score:>6.2f} {r.wall_s:>8.1f}s"
            )
        passed = sum(1 for r in self.results if r.met)
        lines.append("")
        lines.append(
            f"{'aggregate':<24} {f'{passed}/{len(self.results)}':>4} "
            f"{self.avg_iterations:>6.1f} {self.pass_rate:>6.0%} {self.avg_wall:>8.1f}s"
        )
        return "\n".join(lines)


def run_loop_battery(
    cases: list[EvalCase],
    *,
    client=None,
    loop_fn: Callable[[EvalCase], "object"] | None = None,
    max_iterations: int | None = None,
    use_memory: bool = True,
    on_event: Callable[[str, dict], None] | None = None,
) -> LoopBenchReport:
    """Run each case through a full goal loop, timing it and recording the verdict.

    `loop_fn(case) -> LoopResult` is injectable; the default runs the real
    `goal_loop`. The function is pure (no global state mutation) so callers
    control workspace isolation.
    """
    if loop_fn is None:
        from loop import goal_loop

        def loop_fn(case: EvalCase):  # noqa: F811
            return goal_loop(
                case.input,
                rubric=case.rubric or None,
                max_iterations=max_iterations,
                client=client,
                use_memory=use_memory,
                on_event=on_event,
            )

    report = LoopBenchReport()
    for case in cases:
        start = time.perf_counter()
        result = loop_fn(case)
        wall = time.perf_counter() - start
        score = result.verdict.score if result.verdict is not None else 0.0
        report.results.append(
            LoopBenchResult(
                task_id=case.id,
                met=result.met,
                iterations=result.iterations,
                score=score,
                wall_s=wall,
            )
        )
    return report
