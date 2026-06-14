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

import sanitize
from agent import make_client
from config import config
from rubric import Rubric

_VERIFIER_SYSTEM = """\
You are an exacting, impartial verifier. You did not create the work you are
grading; judge only the artifact against the goal and rubric. Reward correctness
and completeness; penalize unmet criteria, hand-waving, and unverified claims.

Respond with ONLY a single JSON object — no prose, no code fences, no <think>
reasoning, no markdown. Your output must start with '{' and end with '}':
{"met": <true|false>, "score": <number 0.0-1.0>, "feedback": "<specific, actionable gaps; empty if met>"}
"""

_RUBRIC_SYSTEM = """\
You are an exacting, impartial verifier. You did not create the work you are
grading. Grade the artifact against EACH numbered rubric criterion independently.

Respond with ONLY a single JSON object — no prose, no code fences, no <think>
reasoning, no markdown. Your output must start with '{' and end with '}':
{"criteria": [{"id": "<criterion id>", "met": <true|false>, "feedback": "<gap if unmet>"}, ...]}
Include every criterion exactly once.
"""


@dataclass
class Verdict:
    met: bool
    score: float
    feedback: str
    raw: str = ""


def _context_block(context: str) -> str:
    """A source-grounding block: makes 'fits the existing code' gradeable.

    Without it, a verifier grading a code/design artifact has nothing to check
    against and rewards plausible-sounding output. With the real source in hand,
    it can penalize artifacts that reference things that don't exist.
    """
    if not context:
        return ""
    return (
        "\nEXISTING SOURCE the artifact must fit. Penalize any module, function, "
        "class, import, CLI flag, or framework the artifact references that does "
        "NOT appear below, and any deviation from these conventions:\n"
        f"{context}\n"
    )


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

    def grade(
        self, *, goal: str, artifact: str, rubric: "str | Rubric | None" = None, context: str = ""
    ) -> Verdict:
        if isinstance(rubric, Rubric):
            return self.grade_rubric(goal=goal, artifact=artifact, rubric=rubric, context=context)
        rubric_block = f"\nRUBRIC (gradable criteria):\n{rubric}\n" if rubric else ""
        user = (
            f"GOAL:\n{goal}\n{rubric_block}{_context_block(context)}\n"
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
            data = _extract_json(sanitize.strip_reasoning(raw))
            met = bool(data.get("met", False))
            score = float(data.get("score", 1.0 if met else 0.0))
            feedback = str(data.get("feedback", "")).strip()
        except (ValueError, json.JSONDecodeError, TypeError):
            # Unparseable verdict is treated as "not met" so the loop keeps working.
            met, score, feedback = False, 0.0, f"Verifier returned unparseable output: {raw[:300]}"
        return Verdict(met=met, score=score, feedback=feedback, raw=raw)

    def grade_rubric(self, *, goal: str, artifact: str, rubric: Rubric, context: str = "") -> Verdict:
        """Grade each criterion independently; aggregate to a weighted score."""
        user = (
            f"GOAL:\n{goal}\n\nRUBRIC CRITERIA:\n{rubric.as_prompt()}\n{_context_block(context)}\n"
            f"ARTIFACT TO GRADE:\n{artifact}\n\nGrade each criterion. Return only the JSON."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _RUBRIC_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=config.grader_temperature,
            max_tokens=config.max_tokens,
        )
        raw = response.choices[0].message.content or ""
        try:
            data = _extract_json(sanitize.strip_reasoning(raw))
            results = {str(r.get("id")): r for r in data.get("criteria", [])}
            earned = 0.0
            gaps: list[str] = []
            for c in rubric.criteria:
                r = results.get(c.id, {})
                if bool(r.get("met", False)):
                    earned += c.weight
                else:
                    gaps.append(f"[{c.id}] {r.get('feedback') or c.text}")
            score = earned / rubric.total_weight
            met = score >= rubric.pass_threshold
            feedback = "" if met else "Unmet criteria:\n" + "\n".join(gaps)
        except (ValueError, json.JSONDecodeError, TypeError):
            met, score, feedback = False, 0.0, f"Verifier returned unparseable output: {raw[:300]}"
        return Verdict(met=met, score=score, feedback=feedback, raw=raw)
