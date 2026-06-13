"""Run tracing — structured, append-only observability for agent runs.

Every loop iteration, verdict, and routine run can be recorded as a JSONL event
stream under workspace/traces/. This makes runs auditable after the fact and
gives reflection (reflect.py) and eval analysis something concrete to learn
from. It's deliberately a flat JSONL file: no service, no dependency.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import config


def _traces_root() -> Path:
    root = Path(config.workspace).expanduser() / "traces"
    root.mkdir(parents=True, exist_ok=True)
    return root


class Tracer:
    def __init__(self, name: str, *, run_id: str | None = None, root: str | Path | None = None):
        self.name = name
        self.run_id = run_id or (
            datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        )
        base = Path(root) if root is not None else _traces_root()
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"{self.run_id}.jsonl"

    def event(self, kind: str, **data) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run": self.run_id,
            "name": self.name,
            "kind": kind,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

    def on_event(self):
        """Return an on_event(kind, data) callback for goal_loop/routines."""
        def callback(kind: str, data: dict) -> None:
            self.event(kind, **data)
        return callback


def read_trace(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def combine(*callbacks):
    """Compose several on_event callbacks into one (skips None)."""
    cbs = [c for c in callbacks if c is not None]

    def callback(kind: str, data: dict) -> None:
        for c in cbs:
            c(kind, data)

    return callback
