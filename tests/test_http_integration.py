"""Real-transport integration: drive the system through the actual `openai`
client against a loopback OpenAI-compatible mock server.

These tests don't use FakeClient — they exercise the genuine HTTP path (request
serialization, native tool_calls over the wire, /v1/models), which is exactly
what a real Ollama/MLX endpoint exposes. No external network, no GPU.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.mock_server import MockServer  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_workspace(tmp_path, monkeypatch):
    import config as cfg
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))
    return tmp_path


@pytest.fixture
def server(monkeypatch):
    import config as cfg
    with MockServer() as base_url:
        monkeypatch.setattr(cfg.config, "base_url", base_url)
        monkeypatch.setattr(cfg.config, "api_key", "test")
        yield base_url


def test_agent_real_http_tool_call(server, _tmp_workspace):
    from agent import NovelAgent
    agent = NovelAgent(model="novel")  # builds a real OpenAI client at base_url
    answer = agent.run("write fizzbuzz and tell me what it prints")
    assert "FizzBuzz" in answer
    # The native tool call really executed against the sandbox.
    assert (_tmp_workspace / "fizzbuzz.py").exists()
    assert "range(1, 16)" in (_tmp_workspace / "fizzbuzz.py").read_text()


def test_goal_loop_real_http(server, _tmp_workspace):
    import memory
    from loop import goal_loop
    from rubric import Rubric
    rubric = Rubric.from_string(
        "- defines a loop over 1..15\n- prints Fizz/Buzz/FizzBuzz correctly\n- states expected output"
    )
    result = goal_loop(
        "Write fizzbuzz.py and confirm its output",
        rubric=rubric, maker_model="novel", grader_model="deepseek-r1:32b",
    )
    assert result.met and result.iterations == 1
    assert (_tmp_workspace / "fizzbuzz.py").exists()
    assert any("Goal met" in lesson for lesson in memory.load().lessons)


def test_fleet_and_doctor_real_http(server):
    import doctor
    import fleet
    statuses = {s.role: s for s in fleet.check(probe=True)}
    assert statuses["orchestrator"].available  # 'novel' advertised by /v1/models
    assert statuses["orchestrator"].latency_s is not None and statuses["orchestrator"].latency_s >= 0
    checks = doctor.run()
    assert doctor.ok(checks)  # server reachable + critical roles loaded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
