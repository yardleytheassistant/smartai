"""Tests for conversation persistence (save/resume sessions)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _tmp_workspace(tmp_path, monkeypatch):
    import config as cfg
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))
    return tmp_path


def test_session_save_load_roundtrip():
    import sessions
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    sessions.save(messages, "work")
    assert sessions.exists("work")
    assert sessions.load("work") == messages
    assert "work" in sessions.list_sessions()


def test_load_missing_raises():
    import sessions
    with pytest.raises(FileNotFoundError):
        sessions.load("nope")


def test_agent_resumes_from_messages():
    from agent import NovelAgent
    from tests.conftest import FakeClient
    saved = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    agent = NovelAgent(client=FakeClient(lambda **_: "new answer"), messages=saved)
    # The prior transcript is intact; a new turn appends to it.
    assert agent.messages[1]["content"] == "earlier question"
    agent.run("follow up")
    assert agent.messages[0]["content"] == "SYS"
    assert any(m["content"] == "follow up" for m in agent.messages)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
