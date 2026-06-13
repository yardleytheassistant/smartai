"""Eval loops — the loop that verifies a Skill or a behavior over time (step 13).

Run a set of cases (a JSONL file), grade each with the independent verifier, and
report pass/fail. Newly-failing cases can be fed straight back into a Skill's
known-failure-modes and into durable memory — so the eval suite is itself part
of the compounding loop, not a one-off check.

JSONL case format (one object per line):
    {"id": "case-1", "input": "...", "rubric": "what a correct answer must do"}
`goal` is accepted as an alias for `rubric`/`input` where convenient.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import memory as memory_mod
import skills as skills_mod
from verifier import Verdict, Verifier


@dataclass
class EvalCase:
    id: str
    input: str
    rubric: str = ""


@dataclass
class EvalResult:
    case: EvalCase
    output: str
    verdict: Verdict


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.verdict.met)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    def failures(self) -> list[EvalResult]:
        return [r for r in self.results if not r.verdict.met]

    def summary(self) -> str:
        return f"{self.passed}/{len(self.results)} passed ({self.pass_rate:.0%})"


def load_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        cases.append(
            EvalCase(
                id=str(obj.get("id", f"case-{i + 1}")),
                input=obj.get("input", obj.get("goal", "")),
                rubric=obj.get("rubric", obj.get("expect", obj.get("goal", ""))),
            )
        )
    return cases


def run_evals(
    cases: list[EvalCase],
    run_fn: Callable[[EvalCase], str],
    *,
    verifier: Verifier | None = None,
    client=None,
    grader_model: str | None = None,
    on_result: Callable[[EvalResult], None] | None = None,
    compound_into_skill: str | None = None,
    record_memory: bool = False,
) -> EvalReport:
    """Run each case through `run_fn`, grade with the verifier, and report.

    If `compound_into_skill` names a skill, each newly-failing case is appended
    to that skill's known failure modes. If `record_memory`, failures are logged
    to durable memory. This is what makes the eval suite compound.
    """
    verifier = verifier or Verifier(model=grader_model, client=client)
    report = EvalReport()
    skill = skills_mod.get(compound_into_skill) if compound_into_skill else None

    for case in cases:
        output = run_fn(case)
        verdict = verifier.grade(goal=case.input, artifact=output, rubric=case.rubric)
        result = EvalResult(case=case, output=output, verdict=verdict)
        report.results.append(result)
        if on_result is not None:
            on_result(result)
        if not verdict.met:
            if skill is not None:
                skill.add_failure_mode(f"{case.id}: {verdict.feedback or 'failed eval'}")
            if record_memory:
                memory_mod.load().log_failure(
                    f"eval {case.id} failing: {verdict.feedback or 'no feedback'}"
                )
    return report
