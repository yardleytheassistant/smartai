"""Tests for the goal-loop battery (perf.py) and the expanded bench task set.

All offline: a FakeClient drives the maker and the independent verifier, so the
full goal_loop -> verifier -> memory path runs with no server.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.conftest import FakeClient  # noqa: E402


def _is_verifier(messages) -> bool:
    return bool(messages) and "verifier" in (messages[0].get("content") or "").lower()


def test_run_loop_battery_offline(tmp_workspace):
    import perf
    from evals import EvalCase

    def responder(model, messages, **_):
        if _is_verifier(messages):
            return '{"met": true, "score": 1.0, "feedback": ""}'
        return "Done — wrote the artifact."

    cases = [EvalCase("t1", "do a", "rubric a"), EvalCase("t2", "do b", "rubric b")]
    report = perf.run_loop_battery(cases, client=FakeClient(responder))

    assert report.all_passed
    assert report.pass_rate == 1.0
    assert report.avg_iterations == 1.0
    assert {r.task_id for r in report.results} == {"t1", "t2"}
    assert all(r.score == 1.0 for r in report.results)


def test_run_loop_battery_multi_iteration_offline(tmp_workspace):
    """The verifier pushes back once, then passes — the maker adapts on iter 2."""
    import perf
    from evals import EvalCase

    grade_calls = {"n": 0}

    def responder(model, messages, **_):
        if _is_verifier(messages):
            grade_calls["n"] += 1
            if grade_calls["n"] == 1:
                return '{"met": false, "score": 0.4, "feedback": "needs a loop"}'
            return '{"met": true, "score": 1.0, "feedback": ""}'
        return "Here is my attempt."

    report = perf.run_loop_battery(
        [EvalCase("t1", "do a", "rubric a")], client=FakeClient(responder)
    )
    assert report.all_passed
    assert report.results[0].iterations == 2


def test_run_loop_battery_writes_memory_to_configured_workspace(tmp_workspace):
    """A passing battery distills a lesson into STATE.md under the active workspace."""
    import memory
    import perf
    from evals import EvalCase

    def responder(model, messages, **_):
        if _is_verifier(messages):
            return '{"met": true, "score": 1.0, "feedback": ""}'
        return "Done."

    perf.run_loop_battery([EvalCase("t1", "do a", "rubric a")], client=FakeClient(responder))
    state = Path(tmp_workspace) / "STATE.md"
    assert state.exists()
    assert any("Goal met" in lesson for lesson in memory.load().lessons)


def test_cmd_perf_isolates_and_restores_workspace(monkeypatch, tmp_path):
    """cmd_perf runs the battery in a temp workspace and restores the original."""
    import config as cfg
    import main
    import perf

    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))
    monkeypatch.setattr(perf, "load_cases", lambda _path: [])

    seen = {}

    def fake_battery(cases, **kw):
        seen["workspace_during"] = cfg.config.workspace
        return perf.LoopBenchReport(
            results=[perf.LoopBenchResult("t1", met=True, iterations=1, score=1.0, wall_s=0.0)]
        )

    monkeypatch.setattr(perf, "run_loop_battery", fake_battery)
    main.cmd_perf([])  # all_passed -> no sys.exit

    assert seen["workspace_during"].startswith("/")
    assert "smartai-perf-" in seen["workspace_during"]
    assert seen["workspace_during"] != str(tmp_path)
    # Original workspace restored after the run.
    assert cfg.config.workspace == str(tmp_path)


def test_cmd_perf_exits_nonzero_on_failure(monkeypatch, tmp_path):
    import config as cfg
    import main
    import perf

    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))
    monkeypatch.setattr(perf, "load_cases", lambda _path: [])
    monkeypatch.setattr(
        perf,
        "run_loop_battery",
        lambda cases, **kw: perf.LoopBenchReport(
            results=[perf.LoopBenchResult("t1", met=False, iterations=4, score=0.3, wall_s=0.0)]
        ),
    )
    with pytest.raises(SystemExit) as exc:
        main.cmd_perf([])
    assert exc.value.code == 1
    # And the failing run did not leave the workspace pointed at the temp dir.
    assert cfg.config.workspace == str(tmp_path)


def test_battery_file_parses():
    from evals import load_cases

    cases = load_cases(str(ROOT / "bench" / "battery.jsonl"))
    assert len(cases) >= 4
    assert all(c.id and c.input and c.rubric for c in cases)
    assert len({c.id for c in cases}) == len(cases)


def test_bench_tasks_file_parses():
    from evals import load_cases

    cases = load_cases(str(ROOT / "bench" / "tasks.jsonl"))
    assert len(cases) >= 12
    assert all(c.id and c.input and c.rubric for c in cases)
    assert len({c.id for c in cases}) == len(cases)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
