"""Offline tests for smartai — no model server or GPU required.

These cover the deterministic parts of the project: the tool registry, the
sandbox, the safe arithmetic evaluator, config defaults, and that the
Modelfile that defines the custom `smartai` model is well-formed.

Run with:  python -m pytest tests/ -q   (or: python tests/test_offline.py)
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _workspace(tmp_path, monkeypatch):
    """Point the sandbox at a throwaway dir and reload config/tools fresh."""
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    for mod in ("tools", "config"):
        sys.modules.pop(mod, None)
    yield


def _tools():
    import tools
    return tools


# --- Registry ---------------------------------------------------------------

def test_registry_exposes_expected_tools():
    tools = _tools()
    names = {t["function"]["name"] for t in tools.openai_schema()}
    assert {
        "get_current_time",
        "list_directory",
        "read_file",
        "write_file",
        "calculate",
        "run_shell",
    } <= names


def test_schema_is_valid_openai_function_format():
    tools = _tools()
    for entry in tools.openai_schema():
        assert entry["type"] == "function"
        fn = entry["function"]
        assert fn["name"] and isinstance(fn["description"], str)
        assert fn["parameters"]["type"] == "object"


def test_dispatch_unknown_tool_is_handled():
    tools = _tools()
    assert "unknown tool" in tools.dispatch("nope", "{}").lower()


def test_dispatch_bad_json_is_handled():
    tools = _tools()
    assert "valid json" in tools.dispatch("calculate", "{not json").lower()


# --- calculate --------------------------------------------------------------

def test_calculate_exact():
    tools = _tools()
    assert tools.dispatch("calculate", '{"expression": "2**16 / 7"}') == str(2**16 / 7)


def test_calculate_rejects_arbitrary_code():
    tools = _tools()
    # No eval(): name references / calls must be rejected, not executed.
    out = tools.dispatch("calculate", '{"expression": "__import__(\\"os\\").system(\\"echo hi\\")"}')
    assert out.lower().startswith("error")


# --- File tools + sandbox ---------------------------------------------------

def test_write_then_read_roundtrip():
    tools = _tools()
    tools.dispatch("write_file", '{"path": "notes/a.txt", "content": "hello"}')
    assert tools.dispatch("read_file", '{"path": "notes/a.txt"}') == "hello"


def test_sandbox_blocks_escape():
    tools = _tools()
    out = tools.dispatch("read_file", '{"path": "../../../../etc/passwd"}')
    assert out.lower().startswith("error")


def test_shell_disabled_by_default():
    tools = _tools()
    out = tools.dispatch("run_shell", '{"command": "echo hi"}')
    assert "disabled" in out.lower()


# --- Config -----------------------------------------------------------------

def test_default_model_is_novel(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    sys.modules.pop("config", None)
    from config import Config
    assert Config().model == "novel"


def test_roles_fall_back_to_primary_model(monkeypatch):
    for var in ("LLM_MODEL", "WORKER_MODEL", "GRADER_MODEL"):
        monkeypatch.delenv(var, raising=False)
    sys.modules.pop("config", None)
    from config import Config
    cfg = Config()
    assert cfg.worker_model == cfg.model == "novel"
    assert cfg.grader_model == "novel"


# --- Modelfile (the custom model definition) --------------------------------

def test_modelfile_defines_novel_on_hermes_base():
    text = (ROOT / "Modelfile").read_text()
    assert "FROM hermes3" in text, "must build on the open-source Hermes base"
    assert "SYSTEM" in text and "Novel" in text
    # Sampling + context params that make tool calling reliable.
    for param in ("temperature", "num_ctx", "stop"):
        assert f"PARAMETER {param}" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
