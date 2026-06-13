"""Conversation persistence — save and resume REPL/agent sessions.

A long working session shouldn't die with the terminal. This saves the full
message transcript (system + turns + tool exchanges) to workspace/sessions/ as
JSON so it can be resumed later with `NovelAgent(messages=...)`. It's distinct
from durable memory (STATE.md): memory is distilled knowledge that compounds,
a session is the verbatim transcript you want to pick back up.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import config


def _sessions_root() -> Path:
    root = Path(config.workspace).expanduser() / "sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _path(name: str) -> Path:
    safe = name if name.endswith(".json") else f"{name}.json"
    return _sessions_root() / safe


def save(messages: list[dict], name: str) -> Path:
    path = _path(name)
    path.write_text(json.dumps(messages, indent=2), encoding="utf-8")
    return path


def load(name: str) -> list[dict]:
    path = _path(name)
    if not path.is_file():
        raise FileNotFoundError(f"no saved session named {name!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def exists(name: str) -> bool:
    return _path(name).is_file()


def list_sessions() -> list[str]:
    return sorted(p.stem for p in _sessions_root().glob("*.json"))
