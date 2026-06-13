"""Tests for the deeper layer: knowledge base, sub-agent delegation, tracing,
and reflection. All model calls are driven by the fake client from conftest.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --- Knowledge base ----------------------------------------------------------

def _seed_kb(tmp_path):
    (tmp_path / "routing.md").write_text(
        "# Routing\n\nThe worker model handles simple fan-out tasks.\n\n"
        "The reasoner model handles hard bounded reasoning.\n"
    )
    (tmp_path / "memory.md").write_text(
        "# Memory\n\nRead STATE.md at session start; write before stopping.\n"
    )
    return tmp_path


def test_knowledge_keyword_search_ranks_relevant(tmp_path):
    import knowledge
    _seed_kb(tmp_path)
    hits = knowledge.search("which model handles reasoning tasks?", directory=tmp_path)
    assert hits
    assert hits[0].chunk.doc == "routing.md"
    assert "reasoner" in hits[0].chunk.text.lower()


def test_knowledge_search_text_cites_sources(tmp_path):
    import knowledge
    _seed_kb(tmp_path)
    text = knowledge.search_text("when to read STATE.md", directory=tmp_path)
    assert "memory.md#" in text


def test_knowledge_empty_and_add(tmp_path):
    import knowledge
    assert knowledge.search("anything", directory=tmp_path) == []
    path = knowledge.add_document("note", "a fresh fact about widgets", directory=tmp_path)
    assert path.exists()
    hits = knowledge.search("widgets", directory=tmp_path)
    assert hits and "widgets" in hits[0].chunk.text


def test_knowledge_uses_embeddings_when_configured(tmp_path, monkeypatch):
    import config as cfg
    import knowledge
    _seed_kb(tmp_path)
    monkeypatch.setattr(cfg.config, "embed_model", "embed-x")

    # Fake embeddings: tag the query and the 'memory' chunk as identical vectors.
    class _Emb:
        def __init__(self, embedding):
            self.embedding = embedding

    class _Resp:
        def __init__(self, data):
            self.data = data

    class _EmbAPI:
        def create(self, *, model, input):
            vecs = []
            for text in input:
                # query + memory chunk -> [1,0]; routing chunk -> [0,1]
                vecs.append([1.0, 0.0] if "state.md" in text.lower() or "session start" in text.lower() else [0.0, 1.0])
            return _Resp([_Emb(v) for v in vecs])

    client = type("C", (), {"embeddings": _EmbAPI()})()
    hits = knowledge.search("session start", directory=tmp_path, client=client)
    assert hits[0].chunk.doc == "memory.md"


# --- Sub-agent delegation ----------------------------------------------------

def test_delegate_routes_by_role(monkeypatch):
    import config as cfg
    import subagents
    monkeypatch.setattr(cfg.config, "coder_model", "coder-x")
    from tests.conftest import FakeClient
    seen = {}

    def responder(model, messages, **_):
        seen["model"] = model
        return "sub-agent answer"

    out = subagents.delegate("refactor this", role="coder", client=FakeClient(responder))
    assert out == "sub-agent answer" and seen["model"] == "coder-x"


def test_delegate_depth_guard(monkeypatch):
    import config as cfg
    import subagents
    monkeypatch.setattr(cfg.config, "subagent_max_depth", 1)
    # Simulate already being one level deep.
    subagents._depth.value = 1
    try:
        out = subagents.delegate("do something", role="worker")
        assert "max depth" in out
    finally:
        subagents._depth.value = 0


# --- Tracing -----------------------------------------------------------------

def test_tracer_writes_and_reads_jsonl(tmp_path, monkeypatch):
    import trace

    import config as cfg
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))
    tracer = trace.Tracer("demo")
    tracer.event("iteration", n=1)
    tracer.event("verdict", n=1, met=True)
    events = trace.read_trace(tracer.path)
    assert [e["kind"] for e in events] == ["iteration", "verdict"]
    assert events[1]["met"] is True and events[0]["name"] == "demo"


def test_trace_combine_fans_out():
    import trace
    a, b = [], []
    cb = trace.combine(lambda k, d: a.append(k), None, lambda k, d: b.append(k))
    cb("x", {})
    assert a == ["x"] and b == ["x"]


# --- Reflection (meta-distillation) -----------------------------------------

def test_reflect_distills_rules_into_memory(tmp_path, monkeypatch):
    import config as cfg
    import memory
    import reflect
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))
    # Isolate skills so reflection can't mutate the repo's versioned skills/.
    monkeypatch.setattr(cfg.config, "skills_dir", str(tmp_path / "skills"))
    mem = memory.load()
    mem.log_failure("auth middleware order caused 401s")
    mem.distill_lesson("PowerShell TLS 1.2 fails on Windows CI")

    from tests.conftest import FakeClient
    payload = (
        '{"rules": ["Order auth middleware rate_limit -> jwt -> rbac"], '
        '"failure_modes": [{"skill": "ci-triage", "mode": "tls: use bash on Windows runners"}]}'
    )
    result = reflect.reflect(client=FakeClient(lambda **_: payload))
    assert result["rules"] and "rate_limit" in result["rules"][0]
    assert any("rate_limit" in r for r in memory.load().rules)


def test_reflect_noop_when_nothing_to_learn(tmp_path, monkeypatch):
    import config as cfg
    import reflect
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))
    result = reflect.reflect()  # no client needed; returns early
    assert result["rules"] == [] and "nothing to reflect" in result["note"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
