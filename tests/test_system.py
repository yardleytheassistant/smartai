"""Tests for the self-improving system — memory, verifier, goal loop, workflows.

A FakeClient mimics the OpenAI chat-completions interface so the full
orchestration (maker -> independent verifier -> memory) runs offline with no
model server or GPU. This is the verifier-sub-agent / state-file / loop
machinery exercised end to end against scripted model responses.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --- Fake OpenAI-compatible client ------------------------------------------

class _Msg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Resp:
    def __init__(self, message):
        self.choices = [type("C", (), {"message": message})()]


class FakeClient:
    """Drives chat.completions.create from a responder(model, messages)->str."""

    def __init__(self, responder):
        self._responder = responder
        self.calls = []
        client = self

        class _Completions:
            def create(self, *, model, messages, **kw):
                # Copy: the agent mutates its message list after the call.
                client.calls.append({"model": model, "messages": list(messages)})
                return _Resp(_Msg(client._responder(model=model, messages=messages)))

        self.chat = type("Chat", (), {"completions": _Completions()})()


@pytest.fixture(autouse=True)
def _tmp_workspace(tmp_path, monkeypatch):
    """Point durable memory at a throwaway workspace for every test."""
    import config as cfg
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path), raising=False)
    yield


# --- Memory: the 5-stage progression ----------------------------------------

def test_memory_roundtrip_and_sections(tmp_path):
    import memory
    path = tmp_path / "STATE.md"
    mem = memory.Memory(path=path)
    mem.remember_fact("prc is in dollars, not cents")
    mem.add_rule("Always include timezone in time-bucketed metrics")
    mem.log_failure("checkout e2e flakes ~1/50; hypothesis: webhook race")
    mem.distill_lesson("Windows runners fail TLS 1.2 in PowerShell; use bash")
    mem.set_last_session("3 fixes drafted, 1 escalated")

    reloaded = memory.load(path)
    assert reloaded.facts == ["prc is in dollars, not cents"]
    assert reloaded.rules == ["Always include timezone in time-bucketed metrics"]
    assert any("webhook race" in f for f in reloaded.failures)
    assert any("TLS 1.2" in l for l in reloaded.lessons)
    assert "3 fixes drafted" in reloaded.last_session


def test_memory_dedupes_and_resolves(tmp_path):
    import memory
    path = tmp_path / "STATE.md"
    mem = memory.Memory(path=path)
    mem.remember_fact("same fact")
    mem.remember_fact("same fact")
    assert mem.facts == ["same fact"]
    mem.log_failure("auth bug in middleware order")
    mem.resolve_failure("auth bug")
    assert mem.failures == []


def test_memory_summary_injects_only_when_nonempty(tmp_path):
    import memory
    mem = memory.Memory(path=tmp_path / "STATE.md")
    assert mem.summary_for_prompt() == ""
    mem.add_rule("rule one")
    assert "Durable memory" in mem.summary_for_prompt()
    assert "rule one" in mem.summary_for_prompt()


# --- Verifier: independent grader -------------------------------------------

def test_verifier_parses_met_verdict():
    from verifier import Verifier
    client = FakeClient(lambda **_: '{"met": true, "score": 0.95, "feedback": ""}')
    v = Verifier(model="grader", client=client).grade(goal="g", artifact="a", rubric="r")
    assert v.met and v.score == 0.95


def test_verifier_parses_json_in_fences_and_prose():
    from verifier import Verifier
    noisy = 'Here is my verdict:\n```json\n{"met": false, "score": 0.3, "feedback": "missing tests"}\n```'
    v = Verifier(client=FakeClient(lambda **_: noisy)).grade(goal="g", artifact="a")
    assert not v.met and "missing tests" in v.feedback


def test_verifier_unparseable_is_not_met():
    from verifier import Verifier
    v = Verifier(client=FakeClient(lambda **_: "totally not json")).grade(goal="g", artifact="a")
    assert v.met is False and v.score == 0.0


def test_verifier_sees_artifact_not_maker_reasoning():
    from verifier import Verifier
    client = FakeClient(lambda **_: '{"met": true, "score": 1.0, "feedback": ""}')
    Verifier(model="grader", client=client).grade(
        goal="build X", artifact="THE-ARTIFACT", rubric="must do X"
    )
    sent = client.calls[-1]["messages"]
    blob = " ".join(m["content"] for m in sent)
    assert "THE-ARTIFACT" in blob and "must do X" in blob
    # The grader's own system prompt identifies it as an impartial verifier.
    assert "verifier" in sent[0]["content"].lower()


# --- Goal loop: maker -> verifier -> memory ---------------------------------

def _routed_responder(grader_payload):
    def responder(model, messages, **_):
        if model == "grader":
            return grader_payload
        return "maker's attempt"
    return responder


def test_goal_loop_stops_when_met():
    from loop import goal_loop
    client = FakeClient(_routed_responder('{"met": true, "score": 1.0, "feedback": ""}'))
    result = goal_loop(
        "do the thing", maker_model="maker", grader_model="grader",
        client=client, use_memory=False,
    )
    assert result.met and result.iterations == 1


def test_goal_loop_respects_max_iterations_and_logs_failure():
    from loop import goal_loop
    import memory
    client = FakeClient(_routed_responder('{"met": false, "score": 0.1, "feedback": "nope"}'))
    result = goal_loop(
        "hard goal", maker_model="maker", grader_model="grader",
        client=client, max_iterations=3, use_memory=True,
    )
    assert not result.met and result.iterations == 3
    # The stuck loop recorded an open failure to durable memory (stages 1-2).
    assert any("hard goal" in f for f in memory.load().failures)


def test_goal_loop_feeds_feedback_into_next_maker():
    from loop import goal_loop
    # Pass on the 2nd grade; capture what the 2nd maker was told.
    state = {"grades": 0}

    def responder(model, messages, **_):
        if model == "grader":
            state["grades"] += 1
            met = state["grades"] >= 2
            return f'{{"met": {str(met).lower()}, "score": 0.5, "feedback": "add error handling"}}'
        return "attempt body"

    client = FakeClient(responder)
    result = goal_loop(
        "ship it", maker_model="maker", grader_model="grader",
        client=client, max_iterations=3, use_memory=False,
    )
    assert result.met and result.iterations == 2
    maker_inputs = [
        m["messages"][-1]["content"] for m in client.calls if m["model"] == "maker"
    ]
    assert any("add error handling" in inp for inp in maker_inputs)


# --- Dynamic workflow primitives --------------------------------------------

def test_fan_out_synthesize_preserves_order():
    from workflows import fan_out_synthesize
    out = fan_out_synthesize([1, 2, 3, 4], worker=lambda x: x * x, synthesize=lambda rs: rs)
    assert out == [1, 4, 9, 16]


def test_fan_out_synthesize_empty():
    from workflows import fan_out_synthesize
    assert fan_out_synthesize([], worker=lambda x: x, synthesize=lambda rs: sum(rs)) == 0


def test_adversarial_verify_passes_on_third_attempt():
    from workflows import adversarial_verify
    res = adversarial_verify(
        make=lambda fb: "art",
        verify=lambda art: (False, "again"),
        max_iterations=3,
    )
    assert not res.met and res.attempts == 3

    counter = {"n": 0}

    def verify(_art):
        counter["n"] += 1
        return (counter["n"] >= 3, "keep going")

    res2 = adversarial_verify(make=lambda fb: "art", verify=verify, max_iterations=5)
    assert res2.met and res2.attempts == 3


def test_loop_until_done_stops_at_condition():
    from workflows import loop_until_done
    outs = loop_until_done(
        step=lambda i, prev: i,
        done=lambda out, prev: out >= 2,
        max_iterations=10,
    )
    assert outs == [0, 1, 2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
