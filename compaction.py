"""Context compaction — let the agent manage its own context (long-run hygiene).

A days-long session accumulates a transcript that eventually overflows the
model's context window. Rather than truncating blindly, we summarize the
*completed* earlier turns with a cheap model and keep the current turn verbatim.

Compaction happens at a user-message boundary: everything before the last user
message is replaced by a single summary, while the system prompt and the current
in-progress turn (the last user message plus its assistant/tool exchanges) are
preserved intact. Cutting at a user boundary keeps assistant/tool-call pairs
together, so the message list stays valid for the API.
"""

from __future__ import annotations

_SUMMARIZER_SYSTEM = """\
You compress a conversation transcript for an AI agent. Preserve, tersely:
durable facts established, decisions made, tool results that matter, and any
open threads. Drop pleasantries and routine narration. Output a compact summary,
no preamble.
"""


def _content_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # vision/multimodal content parts
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def estimate_chars(messages: list[dict]) -> int:
    return sum(len(_content_text(m)) for m in messages)


def _last_user_index(messages: list[dict]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return None


def _render(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        text = _content_text(m)
        if text:
            lines.append(f"{m.get('role', '?')}: {text}")
        if m.get("tool_calls"):
            names = ", ".join(tc.get("function", {}).get("name", "?") for tc in m["tool_calls"])
            lines.append(f"assistant: (called tools: {names})")
    return "\n".join(lines)


def compact(messages: list[dict], *, client, model: str, summary_max_tokens: int = 700) -> list[dict]:
    """Return a compacted copy of `messages`, or the original if nothing to do.

    Summarizes messages between the system prompt and the last user message.
    """
    boundary = _last_user_index(messages)
    if boundary is None or boundary <= 1:
        return messages  # nothing completed before the current turn
    head, middle, tail = messages[0], messages[1:boundary], messages[boundary:]
    if not middle:
        return messages

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SUMMARIZER_SYSTEM},
            {"role": "user", "content": _render(middle)},
        ],
        temperature=0.0,
        max_tokens=summary_max_tokens,
    )
    summary = (response.choices[0].message.content or "").strip()
    if not summary:
        return messages
    note = {"role": "user", "content": f"[Summary of earlier conversation]\n{summary}"}
    return [head, note, *tail]
