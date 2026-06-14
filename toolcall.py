"""Parse tool calls that open models emit as text instead of the native field.

Not every open-source model populates the OpenAI `tool_calls` field. Hermes and
Qwen commonly wrap calls in <tool_call>{...}</tool_call>; others emit a bare
JSON object with a "name" and "arguments". This recovers those so the agent loop
works uniformly across the fleet, while avoiding false positives by only
accepting calls whose name is a registered tool.
"""

from __future__ import annotations

import json
import re

_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FUNC_BLOCK = re.compile(r"<function=([\w.\-:]+)>(.*?)</function>", re.DOTALL | re.IGNORECASE)
_PARAM = re.compile(r"<parameter=([\w.\-]+)>(.*?)</parameter>", re.DOTALL | re.IGNORECASE)


def _normalize(obj: dict, idx: int) -> dict | None:
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    args_str = args if isinstance(args, str) else json.dumps(args)
    return {"id": f"call_{idx}", "name": name, "arguments": args_str}


def parse_text_tool_calls(content: str, valid_names: set[str] | None = None) -> list[dict]:
    """Return tool calls found in text as [{id, name, arguments(json-string)}].

    If `valid_names` is given, only calls naming a registered tool are returned —
    so an ordinary JSON answer isn't mistaken for a tool call.
    """
    if not content:
        return []
    calls: list[dict] = []

    # <function=name>...</function>. Args may be JSON ({...}) OR <parameter=k>v</parameter>
    # tags (the "harmony"/GPT-OSS dialect that some Llama/Qwen-coder tunes emit). The
    # parameter form is why an un-parsed call could leak into an answer as raw XML.
    for i, (name, body) in enumerate(_FUNC_BLOCK.findall(content)):
        args: dict | None = None
        m = re.search(r"\{.*\}", body, re.DOTALL)
        if m:
            try:
                args = json.loads(m.group(0))
            except json.JSONDecodeError:
                args = None
        if args is None:
            params = _PARAM.findall(body)
            if params:
                args = {k.strip(): v.strip() for k, v in params}
        if args is None:
            continue  # a <function=...> with neither JSON nor <parameter> args
        calls.append({"id": f"call_{i}", "name": name, "arguments": json.dumps(args)})

    # <tool_call>{...}</tool_call> style (Hermes / Qwen).
    blocks = _TAG.findall(content)
    if not blocks and not calls:
        # Bare JSON object fallback (only if it looks like a call).
        m = re.search(r"\{.*\}", content, re.DOTALL)
        blocks = [m.group(0)] if m else []
    for i, block in enumerate(blocks):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        call = _normalize(obj, len(calls) + i)
        if call:
            calls.append(call)

    if valid_names is not None:
        calls = [c for c in calls if c["name"] in valid_names]
    # De-dupe by (name, arguments) while preserving order.
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for c in calls:
        key = (c["name"], c["arguments"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique
