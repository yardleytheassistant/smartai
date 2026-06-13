"""Tests for context compaction, file-based rubrics, and fleet status."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --- Context compaction ------------------------------------------------------

def test_compact_summarizes_completed_turns():
    import compaction
    from tests.conftest import FakeClient
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "first task " * 50},
        {"role": "assistant", "content": "did first " * 50},
        {"role": "user", "content": "current task"},
        {"role": "assistant", "content": "working on it"},
    ]
    client = FakeClient(lambda **_: "SUMMARY OF EARLIER")
    out = compaction.compact(messages, client=client, model="worker")
    assert out[0]["content"] == "SYS"                       # system preserved
    assert "SUMMARY OF EARLIER" in out[1]["content"]        # summary inserted
    assert out[-2]["content"] == "current task"             # current turn intact
    assert out[-1]["content"] == "working on it"
    assert len(out) < len(messages)


def test_compact_noop_without_completed_turns():
    import compaction
    from tests.conftest import FakeClient
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "only the current turn"},
    ]
    client = FakeClient(lambda **_: "should not be used")
    assert compaction.compact(messages, client=client, model="worker") == messages


def test_agent_compacts_when_over_budget(monkeypatch):
    import compaction
    import config as cfg
    from agent import NovelAgent
    from tests.conftest import FakeClient
    monkeypatch.setattr(cfg.config, "context_char_budget", 50)

    state = {"compacted": False}
    orig = compaction.compact

    def spy(messages, **kw):
        state["compacted"] = True
        return orig(messages, **kw)

    monkeypatch.setattr(compaction, "compact", spy)

    agent = NovelAgent(model="m", client=FakeClient(lambda **_: "final answer"))
    # Pre-load a long completed turn so the next run() is over budget.
    agent.messages += [
        {"role": "user", "content": "x" * 200},
        {"role": "assistant", "content": "y" * 200},
    ]
    agent.run("now do the thing")
    assert state["compacted"]


# --- File-based rubric (Outcomes) -------------------------------------------

def test_rubric_from_markdown_and_json(tmp_path):
    from rubric import Rubric
    md = tmp_path / "r.md"
    md.write_text("# Rubric\n\n- first criterion\n- second criterion\n")
    r = Rubric.from_file(md)
    assert [c.text for c in r.criteria] == ["first criterion", "second criterion"]

    js = tmp_path / "r.json"
    js.write_text('{"pass_threshold": 0.5, "criteria": [{"id":"a","text":"x","weight":2}]}')
    rj = Rubric.from_file(js)
    assert rj.pass_threshold == 0.5 and rj.criteria[0].weight == 2


def test_verifier_grades_rubric_per_criterion_weighted():
    from rubric import Criterion, Rubric
    from tests.conftest import FakeClient
    from verifier import Verifier
    rubric = Rubric(
        criteria=[Criterion("a", "must A", weight=3), Criterion("b", "must B", weight=1)],
        pass_threshold=0.6,
    )
    # 'a' met (weight 3), 'b' not (weight 1) => 3/4 = 0.75 >= 0.6 => met.
    payload = '{"criteria": [{"id":"a","met":true},{"id":"b","met":false,"feedback":"missing B"}]}'
    v = Verifier(client=FakeClient(lambda **_: payload)).grade(
        goal="g", artifact="art", rubric=rubric
    )
    assert v.met and abs(v.score - 0.75) < 1e-9


def test_verifier_rubric_fails_below_threshold():
    from rubric import Criterion, Rubric
    from tests.conftest import FakeClient
    from verifier import Verifier
    rubric = Rubric(criteria=[Criterion("a", "A"), Criterion("b", "B")], pass_threshold=1.0)
    payload = '{"criteria": [{"id":"a","met":true},{"id":"b","met":false,"feedback":"no B"}]}'
    v = Verifier(client=FakeClient(lambda **_: payload)).grade(goal="g", artifact="x", rubric=rubric)
    assert not v.met and "no B" in v.feedback


def test_goal_loop_accepts_rubric_object():
    from loop import goal_loop
    from rubric import Criterion, Rubric
    from tests.conftest import FakeClient
    rubric = Rubric(criteria=[Criterion("a", "must do A")])

    def responder(model, messages, **_):
        if "verifier" in messages[0]["content"].lower():
            return '{"criteria": [{"id":"a","met":true}]}'
        return "maker output"

    result = goal_loop("do A", rubric=rubric, maker_model="m", grader_model="g",
                       client=FakeClient(responder), use_memory=False)
    assert result.met and result.iterations == 1


# --- Fleet status ------------------------------------------------------------

def test_fleet_check_reports_availability(monkeypatch):
    import config as cfg
    import fleet
    monkeypatch.setattr(cfg.config, "model", "novel")
    monkeypatch.setattr(cfg.config, "worker_model", "qwen3.6:35b")
    available = {"qwen3.6:35b", "deepseek-r1:32b"}
    statuses = {s.role: s for s in fleet.check(available=available)}
    assert statuses["worker"].available is True
    assert statuses["orchestrator"].available is False  # novel not built/loaded


def test_fleet_matches_implicit_latest():
    import fleet
    assert fleet._matches("llava", {"llava:latest"})
    assert fleet._matches("qwen3-coder:30b", {"qwen3-coder:30b"})
    assert not fleet._matches("missing-model", {"other:7b"})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
