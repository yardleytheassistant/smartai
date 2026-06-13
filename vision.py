"""Vision self-check — verify a visual artifact against a goal with an
open-source vision-language model (step 13).

Hosted Mythos uses vision to check its own outputs (e.g. does the rendered UI
match the goal?). The open-source workaround: point an OSS VLM — LLaVA, Qwen-VL,
or similar served over the same OpenAI-compatible endpoint — at a screenshot and
have it grade against the goal. The maker that produced the UI is not the agent
that grades the screenshot.

This sends the image inline as a base64 data URL using the OpenAI vision message
format, so it works against any OpenAI-compatible server that hosts a VLM.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from agent import make_client
from config import config
from verifier import _VERIFIER_SYSTEM, Verdict, _extract_json


def encode_image(path: str | Path) -> str:
    """Read an image file and return an OpenAI-compatible data URL."""
    p = Path(path)
    data = p.read_bytes()
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


class VisionVerifier:
    def __init__(self, *, model: str | None = None, client=None):
        self.model = model or config.vision_model
        self.client = client if client is not None else make_client()

    def grade(self, *, goal: str, image_path: str | Path, rubric: str | None = None) -> Verdict:
        rubric_block = f"\nRUBRIC (gradable criteria):\n{rubric}\n" if rubric else ""
        text = (
            f"GOAL:\n{goal}\n{rubric_block}\n"
            "The image is the artifact to grade (e.g. a rendered UI or a chart). "
            "Judge whether it satisfies the goal. Return only the JSON verdict."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _VERIFIER_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
                    ],
                },
            ],
            temperature=config.grader_temperature,
            max_tokens=config.max_tokens,
        )
        raw = response.choices[0].message.content or ""
        try:
            import sanitize

            data = _extract_json(sanitize.strip_reasoning(raw))
            met = bool(data.get("met", False))
            score = float(data.get("score", 1.0 if met else 0.0))
            feedback = str(data.get("feedback", "")).strip()
        except Exception:  # noqa: BLE001 - unparseable verdict => not met, keep looping
            met, score, feedback = False, 0.0, f"Vision verifier returned unparseable output: {raw[:300]}"
        return Verdict(met=met, score=score, feedback=feedback, raw=raw)
