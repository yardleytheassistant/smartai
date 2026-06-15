"""Persistent record of which target trades we've already mirrored.

Stored as a small JSON file so the bot can restart without re-copying old
trades. Keeping this durable is what stops a restart from double-trading.
"""

import json
from pathlib import Path


class SeenStore:
    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        self._keys: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._keys = set(data.get("seen", []))
            except (json.JSONDecodeError, OSError):
                self._keys = set()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"seen": sorted(self._keys)}), encoding="utf-8")
        tmp.replace(self.path)  # atomic on the same filesystem

    @property
    def is_empty(self) -> bool:
        return not self._keys

    def has(self, key: str) -> bool:
        return key in self._keys

    def add(self, key: str) -> None:
        self._keys.add(key)
        self._save()

    def add_many(self, keys: list[str]) -> None:
        self._keys.update(keys)
        self._save()
