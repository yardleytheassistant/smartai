"""Tests for parallel experiments and the fleet latency probe."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --- run_experiments (parallel make + grade + rank) -------------------------

def test_run_experiments_picks_highest_scorer():
    import experiments
    from verifier import Verdict

    scores = {"approach 1": 0.4, "approach 2": 0.9, "approach 3": 0.6}

    report = experiments.run_experiments(
        "build a thing",
        3,
        make_fn=lambda v: f"output for {v}",
        grade_fn=lambda out: Verdict(
            met=scores[out.replace("output for ", "")] >= 0.8,
            score=scores[out.replace("output for ", "")],
            feedback="",
        ),
        max_workers=1,
    )
    assert len(report.experiments) == 3
    assert report.winner.variant == "approach 2"
    assert report.ranked()[0].score == 0.9


def test_run_experiments_prefers_met_over_raw_score():
    import experiments
    from verifier import Verdict

    def grade(out):
        # "b" is met at 0.7; "a" unmet at 0.95 -> met should win.
        if out.endswith("b"):
            return Verdict(met=True, score=0.7, feedback="")
        return Verdict(met=False, score=0.95, feedback="close")

    report = experiments.run_experiments(
        "t", ["a", "b"], make_fn=lambda v: f"out-{v}", grade_fn=grade, max_workers=1
    )
    assert report.winner.variant == "b"


def test_run_experiments_default_uses_delegation(monkeypatch):
    import experiments
    from verifier import Verdict
    calls = []

    def fake_delegate(task, **kw):
        calls.append(task)
        return "delegated output"

    import subagents
    monkeypatch.setattr(subagents, "delegate", fake_delegate)

    report = experiments.run_experiments(
        "do X", ["fast", "thorough"],
        grade_fn=lambda out: Verdict(met=True, score=1.0, feedback=""),
        max_workers=1,
    )
    assert len(report.experiments) == 2
    assert all("do X" in c and "Approach to try" in c for c in calls)


def test_run_experiments_grounds_maker_and_grader_in_context(monkeypatch):
    """`context` (real source) reaches BOTH the maker and the grader, so an
    'implement X in this repo' run fits the code instead of inventing it."""
    import experiments
    from verifier import Verdict

    seen = {}

    def fake_delegate(task, **kw):
        seen["maker_task"] = task
        return "out"

    import subagents
    monkeypatch.setattr(subagents, "delegate", fake_delegate)

    class RecordingVerifier:
        def __init__(self, *a, **k):
            pass

        def grade(self, *, goal, artifact, rubric=None, context=""):
            seen["grader_context"] = context
            return Verdict(met=True, score=1.0, feedback="")

    monkeypatch.setattr(experiments, "Verifier", RecordingVerifier)

    report = experiments.run_experiments(
        "implement --runs in bench",
        1,
        context="# file: bench.py\nMARKER-SOURCE def run_bench(...)",
        max_workers=1,
    )
    assert report.winner is not None
    # Maker saw the source and the "you must fit" instruction.
    assert "MARKER-SOURCE" in seen["maker_task"]
    assert "must fit" in seen["maker_task"].lower()
    # Grader saw the same source to grade fit against.
    assert "MARKER-SOURCE" in seen["grader_context"]


def test_build_context_reads_and_labels_files(tmp_path):
    import experiments

    (tmp_path / "bench.py").write_text("def run_bench(models, tasks, *, runs=1): ...\n")
    ctx = experiments.build_context(["bench.py", "missing.py"], root=tmp_path)
    assert "# file: bench.py" in ctx and "run_bench(models, tasks" in ctx
    assert "could not read" in ctx  # a missing file is noted, not fatal


# --- run_in_worktrees (git-isolated experiments) ----------------------------

def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_run_in_worktrees_isolates_and_merges_winner(tmp_path):
    import experiments
    from verifier import Verdict

    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.email", "t@t.t", cwd=repo)
    _git("config", "user.name", "t", cwd=repo)
    _git("config", "commit.gpgsign", "false", cwd=repo)
    (repo / "seed.txt").write_text("seed")
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "init", cwd=repo)

    def runner(variant, wt):
        # Each experiment writes its own file in its isolated checkout and commits.
        (Path(wt.path) / "solution.txt").write_text(f"solution from {variant}")
        _git("add", ".", cwd=wt.path)
        _git("commit", "-m", f"sol {variant}", cwd=wt.path)
        return f"solution from {variant}"

    def grade(out):
        return Verdict(met="good" in out, score=1.0 if "good" in out else 0.2, feedback="")

    report = experiments.run_in_worktrees(
        "produce a solution",
        ["good-approach", "weak-approach"],
        runner,
        grade_fn=grade,
        root=repo,
        merge_winner=True,
    )
    assert report.winner.variant == "good-approach"
    # Winner's file is now merged into main.
    assert (repo / "solution.txt").read_text() == "solution from good-approach"


# --- fleet latency probe -----------------------------------------------------

def test_fleet_probe_returns_latency(monkeypatch):
    import config as cfg
    import fleet
    monkeypatch.setattr(cfg.config, "worker_model", "qwen3.6:35b")
    from tests.conftest import FakeClient

    client = FakeClient(lambda **_: "ok")
    statuses = {s.role: s for s in fleet.check(available={"qwen3.6:35b"}, client=client, probe=True)}
    assert statuses["worker"].available and statuses["worker"].latency_s is not None
    assert statuses["worker"].latency_s >= 0
    # Missing models aren't probed.
    assert statuses["orchestrator"].latency_s is None


def test_probe_latency_handles_unreachable():
    import fleet

    class Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**_):
                    raise RuntimeError("server down")

    assert fleet.probe_latency("m", Boom()) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
