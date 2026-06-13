"""Strip model reasoning traces from outputs.

Reasoning models in the fleet (deepseek-r1 as grader/reasoner, gemma4 as a
thinking model) emit chain-of-thought wrapped in tags like <think>…</think>
before the actual answer. Left in place, that leaks into final answers and can
confuse JSON verdict parsing. This removes the common variants so downstream
parsing sees only the answer.
"""

from __future__ import annotations

import re

# <think>…</think>, <thinking>…</thinking>, <reasoning>…</reasoning>
_TAG_BLOCK = re.compile(
    r"<\s*(think|thinking|reasoning|thought)\s*>.*?<\s*/\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
# An unterminated opener (model got cut off mid-think): drop from the tag on.
_DANGLING_OPEN = re.compile(r"<\s*(think|thinking|reasoning|thought)\s*>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove reasoning-trace blocks, returning just the answer text."""
    if not text:
        return text
    cleaned = _TAG_BLOCK.sub("", text)
    cleaned = _DANGLING_OPEN.sub("", cleaned)
    return cleaned.strip()
