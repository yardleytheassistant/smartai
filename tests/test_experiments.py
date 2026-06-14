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
        seen["use_tools"] = kw.get("use_tools", True)
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
    # With source inlined, tools are off so the maker answers from the prompt
    # instead of emitting an un-executed read_file call.
    assert seen["use_tools"] is False


def test_delegate_passes_use_tools_through(monkeypatch):
    """delegate(use_tools=False) builds a tool-less sub-agent (no file-read reflex)."""
    import agent
    import subagents

    captured = {}

    class FakeAgent:
        def __init__(self, **kw):
            captured.update(kw)

        def run(self, task, on_tool=None):
            return "ok"

    monkeypatch.setattr(agent, "NovelAgent", FakeAgent)
    out = subagents.delegate("do thing with inlined source", use_tools=False)
    assert out == "ok"
    assert captured["use_tools"] is False


def test_land_winner_feeds_direction_into_grounded_goal_loop(monkeypatch, tmp_path):
    """--land path: the winning direction + context are handed to a goal loop."""
    import config as cfg
    import experiments
    from experiments import Experiment

    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))
    captured = {}

    class FakeResult:
        met = True
        iterations = 1
        output = "code"
        verdict = None

    def fake_goal_loop(task, **kw):
        captured["task"] = task
        captured["context"] = kw.get("context", "")
        captured["rubric"] = kw.get("rubric")
        captured["maker_model"] = kw.get("maker_model")
        captured["require_file_write"] = kw.get("require_file_write")
        captured["check_cmd"] = kw.get("check_cmd")
        return FakeResult()

    import loop
    monkeypatch.setattr(loop, "goal_loop", fake_goal_loop)

    winner = Experiment(variant="approach 2", output="add the flag to cmd_bench", score=0.9, met=True)
    result = experiments.land_winner(
        "add a --json flag", winner, rubric="r", context="SRC-MARKER", check_cmd="pytest -q"
    )
    assert result.met
    assert "add the flag to cmd_bench" in captured["task"]  # winner's direction
    assert "add a --json flag" in captured["task"]          # original task
    assert captured["context"] == "SRC-MARKER"              # grounding threaded through
    assert captured["rubric"] == "r"
    assert captured["require_file_write"] is True            # land must actually write
    assert captured["check_cmd"] == "pytest -q"             # behavioral gate threaded through
    # Land maker defaults to the coder role (small orchestrators are weak at tools).
    assert captured["maker_model"] == cfg.config.coder_model

    # An explicit --maker override is honored.
    experiments.land_winner("t", winner, maker_model="deepseek-r1:70b")
    assert captured["maker_model"] == "deepseek-r1:70b"


def test_land_winner_landed_requires_real_file_write(tmp_workspace):
    """A maker that actually writes a file -> landed. The verifier passing alone
    is not enough; landed is anchored to disk state."""
    import experiments
    from experiments import Experiment
    from tests.conftest import FakeClient

    def responder(model, messages, **_):
        sysmsg = (messages[0].get("content") or "").lower()
        if "verifier" in sysmsg:
            return '{"met": true, "score": 1.0, "feedback": ""}'
        if any("Tool results" in (m.get("content") or "") for m in messages):
            return "Done — wrote out.txt."
        return '<tool_call>{"name": "write_file", "arguments": {"path": "out.txt", "content": "x"}}</tool_call>'

    winner = Experiment("approach 1", "create out.txt", 0.9, met=True)
    result = experiments.land_winner(
        "create a file", winner, client=FakeClient(responder), use_memory=False, max_iterations=1
    )
    assert result.met and result.files_written == ["out.txt"]
    assert result.landed is True
    assert (tmp_workspace / "out.txt").exists()


def test_land_winner_not_landed_when_maker_only_describes(tmp_workspace):
    """The demonstrated bug: maker describes the change instead of calling
    write_file. The loop now rejects the no-write iteration (require_file_write)
    AND the disk check confirms nothing landed — both layers agree."""
    import experiments
    from experiments import Experiment
    from tests.conftest import FakeClient

    def responder(model, messages, **_):
        sysmsg = (messages[0].get("content") or "").lower()
        if "verifier" in sysmsg:
            return '{"met": true, "score": 1.0, "feedback": ""}'
        return "Let me create the updated main.py file with the --quiet flag implementation:"

    winner = Experiment("approach 1", "add --quiet to doctor", 0.8, met=True)
    result = experiments.land_winner(
        "add --quiet", winner, client=FakeClient(responder), use_memory=False, max_iterations=1
    )
    assert result.met is False         # the loop rejected the describe-only iteration
    assert result.files_written == []  # and nothing was written
    assert result.landed is False      # so it is NOT landed — no false success


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


def _seed_repo(repo):
    repo.mkdir()
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.email", "t@t.t", cwd=repo)
    _git("config", "user.name", "t", cwd=repo)
    _git("config", "commit.gpgsign", "false", cwd=repo)
    (repo / "seed.txt").write_text("seed")
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "init", cwd=repo)


def test_land_in_worktree_commits_real_edit_and_isolates_main(tmp_path, monkeypatch):
    """The maker edits a REAL tracked file in an isolated checkout; git confirms the
    change, it's committed on a branch, and the main working tree is untouched."""
    import config as cfg
    import experiments
    from experiments import Experiment
    from tests.conftest import FakeClient

    repo = tmp_path / "repo"
    _seed_repo(repo)
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path / "ws"))

    def responder(model, messages, **_):
        if "verifier" in (messages[0].get("content") or "").lower():
            return '{"met": true, "score": 1.0, "feedback": ""}'
        if any("Tool results" in (m.get("content") or "") for m in messages):
            return "Edited seed.txt."
        return '<tool_call>{"name": "write_file", "arguments": {"path": "seed.txt", "content": "EDITED"}}</tool_call>'

    winner = Experiment("approach 1", "rewrite seed.txt", 0.9, met=True)
    result = experiments.land_in_worktree(
        "change the seed file", winner, client=FakeClient(responder), root=repo, max_iterations=1
    )

    assert result.committed and result.files_changed == ["seed.txt"]
    assert result.landed is True and result.merged is False
    # The change lives on the review branch, not on main.
    log = subprocess.run(
        ["git", "log", result.branch, "--oneline"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert "land:" in log
    assert (repo / "seed.txt").read_text() == "seed"  # main working tree untouched


def test_land_in_worktree_not_landed_when_maker_only_describes(tmp_path, monkeypatch):
    """No edit -> nothing committed -> not landed, and the empty branch is dropped."""
    import config as cfg
    import experiments
    from experiments import Experiment
    from tests.conftest import FakeClient

    repo = tmp_path / "repo"
    _seed_repo(repo)
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path / "ws"))

    def responder(model, messages, **_):
        if "verifier" in (messages[0].get("content") or "").lower():
            return '{"met": true, "score": 1.0, "feedback": ""}'
        return "Let me edit seed.txt to make the change."  # describes, never calls write_file

    winner = Experiment("approach 1", "rewrite seed.txt", 0.9, met=True)
    result = experiments.land_in_worktree(
        "change the seed file", winner, client=FakeClient(responder), root=repo, max_iterations=1
    )

    assert result.met is False              # loop rejected the describe-only iteration
    assert result.files_changed == [] and result.committed is False
    assert result.landed is False           # and git confirms nothing changed
    branches = subprocess.run(
        ["git", "branch", "--list", result.branch], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert branches == ""                   # empty branch was cleaned up


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
