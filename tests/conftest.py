"""Shared test fixtures: a fake OpenAI-compatible client and a tmp workspace."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _Msg:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


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
                client.calls.append({"model": model, "messages": list(messages)})
                return _Resp(_Msg(client._responder(model=model, messages=messages)))

        self.chat = type("Chat", (), {"completions": _Completions()})()


@pytest.fixture
def fake_client():
    return lambda responder: FakeClient(responder)


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    import config as cfg
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path), raising=False)
    return tmp_path
