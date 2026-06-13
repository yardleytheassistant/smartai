"""End-to-end: a full goal loop through the real modules wired together.

Unlike the unit tests, this exercises the whole pipeline at once — the goal loop
drives a maker agent that issues a (text-form) tool call, the tool actually runs
in the sandbox, the result is fed back, the independent verifier grades the
answer against a file-based rubric, and durable memory is updated — all on a
fake client so it stays offline.
"""

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


def test_goal_loop_drives_tool_use_verify_and_memory(_tmp_workspace):
    import memory
    from loop import goal_loop
    from rubric import Criterion, Rubric
    from tests.conftest import FakeClient

    rubric = Rubric(criteria=[Criterion("a", "wrote the file and reported done")])

    def responder(model, messages, **_):
        if model == "grader":
            return '{"criteria": [{"id": "a", "met": true}]}'
        # maker: if the tool result is already in the transcript, give the answer.
        if any("Tool results" in (m.get("content") or "") for m in messages):
            return "Done — wrote out.txt with the greeting."
        # otherwise, issue a text-form tool call to write the file.
        return '<tool_call>{"name": "write_file", "arguments": {"path": "out.txt", "content": "hello"}}</tool_call>'

    result = goal_loop(
        "create out.txt containing a greeting, then report done",
        rubric=rubric,
        maker_model="maker",
        grader_model="grader",
        client=FakeClient(responder),
        use_memory=True,
    )

    # The loop met the goal on the first iteration.
    assert result.met and result.iterations == 1
    # The tool actually ran: the file exists in the sandbox.
    assert (_tmp_workspace / "out.txt").read_text() == "hello"
    # Durable memory compounded: a lesson was distilled and the resume pointer set.
    mem = memory.load()
    assert any("Goal met" in lesson for lesson in mem.lessons)
    assert "passed" in mem.last_session.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
