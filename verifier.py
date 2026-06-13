"""Independent verifier — the grader sub-agent.

A model has a hard time grading its own work: it sees its own reasoning trail
and prefers conclusions consistent with what it already wrote. So the agent that
produced an artifact is NOT the agent that grades it. The verifier sees only the
goal, the rubric, and the artifact — never the maker's reasoning — and by
default runs on a separate (open-source) grader model at temperature 0 for
reproducible verdicts.

It returns a structured Verdict the goal loop can branch on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agent import make_client
from config import config

_VERIFIER_SYSTEM = """\
You are an exacting, impartial verifier. You did not create the work you are
grading; judge only the artifact against the goal and rubric. Reward correctness
and completeness; penalize unmet criteria, hand-waving, and unverified claims.

Respond with ONLY a single JSON object, no prose, no code fences:
{"met": <true|false>, "score": <number 0.0-1.0>, "feedback": "<specific, actionable gaps; empty if met>"}
"""


@dataclass
class Verdict:
    met: bool
    score: float
    feedback: str
    raw: str = ""


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response, tolerating fences/prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        raise ValueError("no JSON object found")
    return json.loads(candidate)


class Verifier:
    def __init__(self, *, model: str | None = None, client=None):
        self.model = model or config.grader_model
        self.client = client if client is not None else make_client()

    def grade(self, *, goal: str, artifact: str, rubric: str | None = None) -> Verdict:
        rubric_block = f"\nRUBRIC (gradable criteria):\n{rubric}\n" if rubric else ""
        user = (
            f"GOAL:\n{goal}\n{rubric_block}\n"
            f"ARTIFACT TO GRADE:\n{artifact}\n\n"
            "Grade it now. Return only the JSON verdict."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _VERIFIER_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=config.grader_temperature,
            max_tokens=config.max_tokens,
        )
        raw = response.choices[0].message.content or ""
        try:
            data = _extract_json(raw)
            met = bool(data.get("met", False))
            score = float(data.get("score", 1.0 if met else 0.0))
            feedback = str(data.get("feedback", "")).strip()
        except (ValueError, json.JSONDecodeError, TypeError):
            # Unparseable verdict is treated as "not met" so the loop keeps working.
            met, score, feedback = False, 0.0, f"Verifier returned unparseable output: {raw[:300]}"
        return Verdict(met=met, score=score, feedback=feedback, raw=raw)
