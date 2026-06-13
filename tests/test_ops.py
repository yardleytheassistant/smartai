"""Tests for ops: benchmarking, availability-aware routing, and doctor."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --- Benchmark harness -------------------------------------------------------

def test_bench_ranks_models_and_suggests_roles():
    import bench
    from evals import EvalCase
    from verifier import Verdict

    tasks = [EvalCase(id="t1", input="x", rubric="r"), EvalCase(id="t2", input="y", rubric="r")]
    # model scores: A high, B mid, C low.
    table = {"A": 0.95, "B": 0.6, "C": 0.2}

    report = bench.run_bench(
        ["A", "B", "C"],
        tasks,
        run_fn=lambda model, task: model,  # output is just the model name
        verifier=type("V", (), {
            "grade": lambda self, *, goal, artifact, rubric=None: Verdict(
                met=table[artifact] >= 0.5, score=table[artifact], feedback=""
            )
        })(),
    )
    assert report.ranked() == ["A", "B", "C"]
    assert report.best() == "A"
    roles = report.suggested_roles()
    assert roles["orchestrator"] == "A" and roles["grader"] == "C"
    assert len(report.results) == 6  # 3 models x 2 tasks


def test_bench_report_aggregates():
    import bench
    r = bench.BenchReport(results=[
        bench.BenchResult("A", "t1", 1.0, True, 0.1),
        bench.BenchResult("A", "t2", 0.0, False, 0.3),
    ])
    assert r.avg_score("A") == 0.5
    assert r.pass_rate("A") == 0.5
    assert abs(r.avg_latency("A") - 0.2) < 1e-9


# --- Availability-aware routing ----------------------------------------------

def test_decide_downgrades_to_available_model(monkeypatch):
    import config as cfg
    import router
    monkeypatch.setattr(cfg.config, "model", "novel")
    monkeypatch.setattr(cfg.config, "worker_model", "qwen3.6:35b")
    # 'rename a variable' routes to worker; worker not loaded -> resolve to novel.
    d = router.decide("rename a variable", available={"novel"})
    assert d.model == "novel" and "not loaded" in d.note


def test_decide_keeps_model_when_available(monkeypatch):
    import config as cfg
    import router
    monkeypatch.setattr(cfg.config, "worker_model", "qwen3.6:35b")
    d = router.decide("rename a variable", available={"qwen3.6:35b"})
    assert d.model == "qwen3.6:35b" and "not loaded" not in (d.note or "")


def test_resolve_available_fallback_chain(monkeypatch):
    import config as cfg
    import router
    monkeypatch.setattr(cfg.config, "model", "novel")
    monkeypatch.setattr(cfg.config, "worker_model", "w")
    monkeypatch.setattr(cfg.config, "reasoner_model", "r")
    monkeypatch.setattr(cfg.config, "grader_model", "g")
    # nothing preferred is available except the reasoner.
    assert router.resolve_available("missing", {"r"}) == "r"


# --- Doctor ------------------------------------------------------------------

def test_doctor_reports_workspace_and_roles(tmp_path, monkeypatch):
    import config as cfg
    import doctor
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))
    monkeypatch.setattr(cfg.config, "model", "novel")
    monkeypatch.setattr(cfg.config, "grader_model", "deepseek-r1:32b")

    checks = doctor.run(available={"deepseek-r1:32b"})  # novel not loaded
    names = {c.name: c for c in checks}
    assert names["workspace writable"].ok
    # grader is loaded (critical, ok); orchestrator 'novel' is not (critical, fail).
    grader = next(c for c in checks if c.name.startswith("role 'grader'"))
    orch = next(c for c in checks if c.name.startswith("role 'orchestrator'"))
    assert grader.ok and not orch.ok
    assert doctor.ok(checks) is False  # a critical role is down


def test_doctor_ok_when_criticals_pass(tmp_path, monkeypatch):
    import config as cfg
    import doctor
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))
    monkeypatch.setattr(cfg.config, "model", "novel")
    monkeypatch.setattr(cfg.config, "grader_model", "novel")
    checks = doctor.run(available={"novel"})
    assert doctor.ok(checks) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
