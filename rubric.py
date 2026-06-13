"""File-based rubrics — the Outcomes-style gradable criteria (vs a plain /goal).

A goal can be graded against a plain-text rubric, or against a structured Rubric:
a list of individually-gradable criteria with weights and a pass threshold. The
verifier scores each criterion separately, which gives a partial-credit score and
pinpoints exactly which criteria failed — better signal for the next iteration.

Load from a Markdown file (one criterion per `- ` bullet) or a JSON file:
    {"pass_threshold": 1.0, "criteria": [{"id": "c1", "text": "...", "weight": 2}]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Criterion:
    id: str
    text: str
    weight: float = 1.0


@dataclass
class Rubric:
    criteria: list[Criterion] = field(default_factory=list)
    pass_threshold: float = 1.0  # fraction of weighted criteria that must be met

    def as_prompt(self) -> str:
        return "\n".join(f"- [{c.id}] {c.text}" for c in self.criteria)

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria) or 1.0

    @classmethod
    def from_string(cls, text: str, *, pass_threshold: float = 1.0) -> "Rubric":
        criteria = []
        for i, raw in enumerate(text.splitlines()):
            line = raw.strip()
            if line.startswith(("- ", "* ")):
                criteria.append(Criterion(id=f"c{i + 1}", text=line[2:].strip()))
        if not criteria:  # a single free-text criterion
            stripped = text.strip()
            if stripped:
                criteria.append(Criterion(id="c1", text=stripped))
        return cls(criteria=criteria, pass_threshold=pass_threshold)

    @classmethod
    def from_file(cls, path: str | Path) -> "Rubric":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".json":
            obj = json.loads(text)
            criteria = [
                Criterion(
                    id=str(c.get("id", f"c{i + 1}")),
                    text=c["text"],
                    weight=float(c.get("weight", 1.0)),
                )
                for i, c in enumerate(obj.get("criteria", []))
            ]
            return cls(criteria=criteria, pass_threshold=float(obj.get("pass_threshold", 1.0)))
        return cls.from_string(text)
