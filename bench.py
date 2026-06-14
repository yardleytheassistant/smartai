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

import re
import time
from dataclasses import dataclass, field

from evals import EvalCase, load_cases  # noqa: F401  (re-exported for convenience)
from verifier import Verifier

# `novel` is built on qwen3.5 (see Modelfile), so for independence purposes it
# belongs to the qwen family — a qwen grader is NOT independent of a novel maker.
_FAMILY_ALIASES = {"novel": "qwen"}


def model_family(model: str) -> str:
    """A coarse family token so we can keep the grader independent of the maker.

    Strips the tag and the version so version bumps stay one family:
    qwen3.6:35b / qwen3.5:122b -> 'qwen', llama4:scout -> 'llama',
    deepseek-r1:32b -> 'deepseek-r'. Known custom models map via alias.
    """
    base = model.split(":")[0].strip().lower()
    if base in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[base]
    m = re.match(r"^([a-z][a-z\-]*?)\d", base)
    return m.group(1).rstrip("-") if m else base


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
        """Suggest role assignments from the results.

        best score -> orchestrator; fastest among the passers -> worker; and the
        grader is the best-scoring model from a DIFFERENT family than the
        orchestrator. Independence is the whole point of a verifier — a grader
        that shares the maker's family defeats the cross-check — so we never
        suggest a same-family grader unless the fleet has no other family.
        """
        ranked = self.ranked()
        if not ranked:
            return {}
        orchestrator = ranked[0]
        passers = [m for m in ranked if self.pass_rate(m) >= 0.5] or ranked
        worker = min(passers, key=self.avg_latency)

        orch_family = model_family(orchestrator)
        independent = [m for m in ranked if model_family(m) != orch_family]
        indep_passers = [m for m in independent if self.pass_rate(m) >= 0.5]
        # ranked is best-first, so [0] of any filtered list is its top scorer.
        grader = (
            indep_passers
            or independent
            or [m for m in ranked if m != orchestrator]
            or [orchestrator]
        )[0]
        return {"orchestrator": orchestrator, "worker": worker, "grader": grader}

    def grader_is_independent(self, roles: "dict[str, str] | None" = None) -> bool:
        """True if the suggested grader is from a different family than the orchestrator."""
        roles = roles or self.suggested_roles()
        if "grader" not in roles or "orchestrator" not in roles:
            return False
        return model_family(roles["grader"]) != model_family(roles["orchestrator"])


def run_bench(
    models, tasks, *, runs: int = 1, client=None, run_fn=None, verifier: Verifier | None = None
) -> BenchReport:
    """Run each model over each task, grade, and time it.

    `run_fn(model, task)->output` is injectable; the default runs a fresh
    tool-less agent so we measure raw model quality, not tool plumbing.

    `runs` repeats every (model, task) N times — since the maker generates at a
    non-zero temperature, a single run is noisy; averaging N tightens the score
    so the scorecard is reproducible. The aggregates (avg_score/avg_latency/
    pass_rate) already mean over all results, so N runs just feed them N samples.
    """
    if run_fn is None:
        from agent import NovelAgent

        def run_fn(model: str, task: EvalCase) -> str:  # noqa: F811
            return NovelAgent(model=model, client=client, use_tools=False, compact=False).run(task.input)

    verifier = verifier or Verifier(client=client)
    report = BenchReport()
    for model in models:
        for task in tasks:
            for _ in range(max(1, runs)):
                start = time.perf_counter()
                output = run_fn(model, task)
                latency = time.perf_counter() - start
                verdict = verifier.grade(goal=task.input, artifact=output, rubric=task.rubric or None)
                report.results.append(
                    BenchResult(model=model, task_id=task.id, score=verdict.score,
                                met=verdict.met, latency_s=latency)
                )
    return report
