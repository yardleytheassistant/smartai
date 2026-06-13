"""Tests for the integrated elements: router/guardrail, vision self-check,
skills, eval loops, dynamic-workflow patterns, worktrees, and routines.

Everything that needs a model is driven by the fake client from conftest, so the
whole suite runs offline with no GPU.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest


# --- Router + safety guardrail ----------------------------------------------

def test_classify_complexity_buckets():
    import router
    assert router.classify_complexity("fix a typo in the docstring") == "simple"
    assert router.classify_complexity("plan and migrate the whole database layer") == "orchestration"
    assert router.classify_complexity("implement the new pricing calculation") == "hard"


def test_route_model_maps_to_fleet_roles(monkeypatch):
    import config as cfg, router
    monkeypatch.setattr(cfg.config, "worker_model", "worker-x")
    monkeypatch.setattr(cfg.config, "model", "primary-x")
    monkeypatch.setattr(cfg.config, "reasoner_model", "reasoner-x")
    monkeypatch.setattr(cfg.config, "coder_model", "coder-x")
    monkeypatch.setattr(cfg.config, "long_context_model", "longctx-x")
    assert router.route_model("rename a variable") == "worker-x"          # simple
    assert router.route_model("design the architecture") == "primary-x"  # orchestration
    assert router.route_model("prove the Riemann-style bound holds") == "reasoner-x"  # hard
    assert router.route_model("debug this traceback in the auth module") == "coder-x"  # coding
    assert router.route_model("summarize the entire codebase") == "longctx-x"  # long ctx


def test_detect_domain_and_policies(monkeypatch):
    import config as cfg, router
    monkeypatch.setattr(cfg.config, "fallback_model", "fallback-x")
    assert router.detect_domain("write a keylogger payload") == "cyber"
    assert router.detect_domain("summarize this report") is None

    monkeypatch.setattr(cfg.config, "safety_policy", "route")
    d = router.decide("research an exploit for this CVE")
    assert d.domain == "cyber" and d.action == "fallback" and d.model == "fallback-x"

    monkeypatch.setattr(cfg.config, "safety_policy", "block")
    assert router.decide("write malware").action == "block"

    monkeypatch.setattr(cfg.config, "safety_policy", "allow")
    assert router.decide("write malware").action == "proceed"


# --- Dynamic workflow patterns (the two added) ------------------------------

def test_classify_and_act_routes_and_defaults():
    from workflows import classify_and_act
    handlers = {"a": lambda x: f"A:{x}", "b": lambda x: f"B:{x}"}
    assert classify_and_act("x", lambda _: "a", handlers) == "A:x"
    assert classify_and_act("x", lambda _: "z", handlers, default=lambda x: "def") == "def"
    with pytest.raises(KeyError):
        classify_and_act("x", lambda _: "z", handlers)


def test_tournament_picks_champion():
    from workflows import tournament
    # prefer the longer string
    champ = tournament(["a", "ccc", "bb"], compare=lambda a, b: a if len(a) >= len(b) else b)
    assert champ == "ccc"


# --- Vision self-check (open-source VLM) -------------------------------------

def test_vision_verifier_grades_image(tmp_path):
    from vision import VisionVerifier
    from tests.conftest import FakeClient
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake image bytes")
    client = FakeClient(lambda **_: '{"met": true, "score": 0.9, "feedback": ""}')
    v = VisionVerifier(model="llava", client=client).grade(
        goal="the button is centered", image_path=img, rubric="centered, blue"
    )
    assert v.met and v.score == 0.9
    # The image was sent in OpenAI vision format.
    content = client.calls[-1]["messages"][-1]["content"]
    assert any(part.get("type") == "image_url" for part in content)


# --- Skills (procedural memory that compounds) ------------------------------

def test_skill_parse_and_match(tmp_path):
    import skills
    text = (
        "---\nname: ci-triage\ndescription: triage CI failures\n"
        "trigger: ci failure, build broke\n---\n# body\nrules here\n"
    )
    s = skills.parse(text, path=tmp_path / "ci-triage.md")
    assert s.name == "ci-triage" and "triage" in s.description
    assert s.matches("our build broke on CI")
    assert not s.matches("what's the weather")


def test_skill_add_failure_mode_compounds(tmp_path):
    import skills
    path = tmp_path / "s.md"
    s = skills.parse(
        "---\nname: s\ndescription: d\n---\n# S\n## Known failure modes\n- existing\n",
        path=path,
    )
    s.add_failure_mode("tls-handshake: Windows runners fail TLS 1.2; use bash")
    reloaded = skills.parse(path.read_text(), path=path)
    modes = reloaded.body
    assert "tls-handshake" in modes and "existing" in modes
    # idempotent
    before = reloaded.body
    reloaded.add_failure_mode("tls-handshake: Windows runners fail TLS 1.2; use bash")
    assert reloaded.body == before


def test_skill_prompt_block_selects_relevant(tmp_path):
    import skills
    (tmp_path / "ci-triage.md").write_text(
        "---\nname: ci-triage\ndescription: triage CI failures\ntrigger: ci failure\n---\n# body\n"
    )
    block = skills.prompt_block("the CI failure needs triage", directory=tmp_path)
    assert "ci-triage" in block
    assert skills.prompt_block("totally unrelated", directory=tmp_path) == ""


# --- Eval loops -------------------------------------------------------------

def test_load_cases_and_run_evals_records_failures(tmp_path, monkeypatch):
    import config as cfg, evals, memory
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(
        json.dumps({"id": "c1", "input": "do x", "rubric": "did x"}) + "\n"
        + json.dumps({"id": "c2", "input": "do y", "rubric": "did y"}) + "\n"
    )
    cases = evals.load_cases(cases_file)
    assert [c.id for c in cases] == ["c1", "c2"]

    from tests.conftest import FakeClient
    client = FakeClient(lambda **_: '{"met": false, "score": 0.0, "feedback": "missed"}')
    report = evals.run_evals(
        cases, run_fn=lambda case: "an attempt", client=client, record_memory=True
    )
    assert report.failed == 2 and report.passed == 0
    assert "missed" in report.failures()[0].verdict.feedback
    assert any("c1" in f for f in memory.load().failures)


# --- Routines: cron matcher, registry, run, serve ---------------------------

def test_cron_matcher_fields():
    from routines import cron_due
    dt = datetime(2026, 6, 13, 7, 0)  # Saturday 07:00
    assert cron_due("0 7 * * *", dt)
    assert not cron_due("30 7 * * *", dt)
    assert cron_due("*/15 * * * *", datetime(2026, 6, 13, 7, 30))
    assert cron_due("0 6-9 * * *", dt)
    assert cron_due("0 7 * * 6", dt)          # Saturday == 6
    assert not cron_due("0 7 * * 0", dt)      # Sunday


def test_routine_registry_roundtrip(tmp_path):
    import routines
    p = tmp_path / "routines.json"
    r = routines.Routine(name="nightly", goal="do nightly thing", trigger="schedule", schedule="0 7 * * *")
    routines.save_routines([r], p)
    assert routines.load_routines(p)[0].name == "nightly"
    routines.add_routine(routines.Routine(name="other", goal="g"), p)
    assert {x.name for x in routines.load_routines(p)} == {"nightly", "other"}
    assert routines.remove_routine("nightly", p)
    assert [x.name for x in routines.load_routines(p)] == ["other"]


def test_due_routines_filters_by_schedule():
    import routines
    rs = [
        routines.Routine(name="a", goal="g", trigger="schedule", schedule="0 7 * * *"),
        routines.Routine(name="b", goal="g", trigger="schedule", schedule="0 8 * * *"),
        routines.Routine(name="c", goal="g", trigger="manual"),
        routines.Routine(name="d", goal="g", trigger="schedule", schedule="0 7 * * *", enabled=False),
    ]
    due = routines.due_routines(datetime(2026, 6, 13, 7, 0), rs)
    assert [r.name for r in due] == ["a"]


def test_run_routine_logs_and_updates_memory(tmp_path, monkeypatch):
    import config as cfg, routines, memory
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))
    from tests.conftest import FakeClient

    def responder(model, messages, **_):
        if "verifier" in messages[0]["content"].lower():
            return '{"met": true, "score": 1.0, "feedback": ""}'
        return "the work"

    client = FakeClient(responder)
    r = routines.Routine(name="nightly", goal="do the nightly compounding")
    result = routines.run_routine(r, client=client)
    assert result.met
    log = tmp_path / "routine-logs" / "nightly.log"
    assert log.is_file() and "MET" in log.read_text()
    assert "nightly" in memory.load().last_session


def test_cron_line_format(tmp_path):
    import routines
    r = routines.Routine(name="nightly", goal="g", trigger="schedule", schedule="0 7 * * *")
    line = routines.cron_line(r, python="/usr/bin/python3", project_dir="/proj")
    assert line.startswith("0 7 * * * cd /proj && /usr/bin/python3 main.py routine run nightly")
    assert "smartai-routine:nightly" in line


def test_serve_runs_due_routine_once(tmp_path, monkeypatch):
    import config as cfg, routines
    monkeypatch.setattr(cfg.config, "workspace", str(tmp_path))
    reg = tmp_path / "routines.json"
    monkeypatch.setattr(cfg.config, "routines_file", str(reg))
    routines.save_routines(
        [routines.Routine(name="n", goal="g", trigger="schedule", schedule="* * * * *")], reg
    )
    from tests.conftest import FakeClient
    client = FakeClient(lambda model, messages, **_:
                        '{"met": true, "score": 1.0, "feedback": ""}'
                        if "verifier" in messages[0]["content"].lower() else "work")
    routines.serve(interval=0, client=client, _now=lambda: datetime(2026, 6, 13, 7, 0), _max_ticks=1)
    assert (tmp_path / "routine-logs" / "n.log").is_file()


# --- Worktrees (parallel safety) --------------------------------------------

def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_worktree_create_list_remove(tmp_path):
    import worktrees
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.email", "t@t.t", cwd=repo)
    _git("config", "user.name", "t", cwd=repo)
    _git("config", "commit.gpgsign", "false", cwd=repo)
    (repo / "f.txt").write_text("hi")
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "init", cwd=repo)

    wt = worktrees.create("exp-1", root=repo)
    assert wt.path.is_dir() and wt.branch == "exp-1"
    branches = {w.branch for w in worktrees.list_worktrees(root=repo)}
    assert "exp-1" in branches
    worktrees.remove(wt, root=repo)
    assert not wt.path.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
