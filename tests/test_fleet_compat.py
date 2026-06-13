"""Tests for fleet-compatibility hardening: reasoning-trace stripping and
text-based tool-call parsing (so the agent works across qwen/deepseek/llama).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --- Reasoning-trace stripping ----------------------------------------------

def test_strip_reasoning_removes_think_block():
    import sanitize
    out = sanitize.strip_reasoning("<think>lots of reasoning</think>\nThe answer is 42.")
    assert out == "The answer is 42."


def test_strip_reasoning_variants_and_multiple():
    import sanitize
    text = "<thinking>a</thinking>x<reasoning>b</reasoning>y"
    assert sanitize.strip_reasoning(text) == "xy"


def test_strip_reasoning_dangling_open():
    import sanitize
    # Model cut off mid-think with no closing tag.
    assert sanitize.strip_reasoning("answer first <think>then it trails off") == "answer first"


def test_strip_reasoning_passthrough():
    import sanitize
    assert sanitize.strip_reasoning("just a normal answer") == "just a normal answer"


def test_verifier_parses_verdict_behind_think(monkeypatch):
    from tests.conftest import FakeClient
    from verifier import Verifier
    raw = '<think>The artifact does X and Y, looks complete.</think>{"met": true, "score": 1.0, "feedback": ""}'
    v = Verifier(client=FakeClient(lambda **_: raw)).grade(goal="g", artifact="a")
    assert v.met and v.score == 1.0


# --- Text tool-call parsing --------------------------------------------------

def test_parse_tool_call_tag():
    import toolcall
    content = 'sure<tool_call>{"name": "calculate", "arguments": {"expression": "2+2"}}</tool_call>'
    calls = toolcall.parse_text_tool_calls(content, valid_names={"calculate"})
    assert len(calls) == 1
    assert calls[0]["name"] == "calculate"
    assert '"expression": "2+2"' in calls[0]["arguments"]


def test_parse_function_tag_style():
    import toolcall
    content = '<function=get_current_time>{}</function>'
    calls = toolcall.parse_text_tool_calls(content, valid_names={"get_current_time"})
    assert calls and calls[0]["name"] == "get_current_time"


def test_parse_rejects_non_tool_json():
    import toolcall
    # A plain JSON answer that isn't a registered tool must not be treated as a call.
    content = '{"answer": 42, "name": "not_a_tool"}'
    assert toolcall.parse_text_tool_calls(content, valid_names={"calculate"}) == []


def test_parse_dedupes():
    import toolcall
    content = (
        '<tool_call>{"name":"calculate","arguments":{"expression":"1+1"}}</tool_call>'
        '<tool_call>{"name":"calculate","arguments":{"expression":"1+1"}}</tool_call>'
    )
    assert len(toolcall.parse_text_tool_calls(content, valid_names={"calculate"})) == 1


def test_agent_executes_text_tool_call_then_answers(tmp_path, monkeypatch):
    import config as cfg
    from agent import NovelAgent
    from tests.conftest import FakeClient
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))

    step = {"n": 0}

    def responder(model, messages, **_):
        step["n"] += 1
        if step["n"] == 1:
            # First turn: emit a text tool call (no native tool_calls field).
            return '<tool_call>{"name": "calculate", "arguments": {"expression": "6*7"}}</tool_call>'
        # Second turn: the tool result is now in the transcript; answer.
        return "The result is 42."

    agent = NovelAgent(model="m", client=FakeClient(responder))
    answer = agent.run("what is 6*7?")
    assert answer == "The result is 42."
    # The tool result was fed back as a user turn.
    assert any(m["role"] == "user" and "42" in m["content"] for m in agent.messages)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
