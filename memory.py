"""Durable memory for smartai — the state file that makes the system compound.

This implements the 5-stage memory progression (fail -> investigate -> verify
-> distill -> consult) as a human-readable Markdown state file that survives
between sessions. Every session reads it at the start (consult) and writes to it
at the end (distill), so tomorrow's run resumes instead of restarting.

The file has five sections, each mapping to a stage:

  ## Verified facts    (stage 3) — things we stopped guessing about
  ## General rules     (stage 4) — distilled rules that apply beyond one case
  ## Open failures     (stages 1-2) — failures + hypotheses to investigate next
  ## Lessons learned   (stage 4) — distilled post-mortems
  ## Last session      (stage 5) — the resume pointer

It is deliberately dependency-free and tolerant: unknown headers are ignored,
and a missing file is treated as empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config import config

# Section header -> attribute name. Order here is the order they're written.
_SECTIONS: list[tuple[str, str]] = [
    ("Verified facts", "facts"),
    ("General rules", "rules"),
    ("Open failures", "failures"),
    ("Lessons learned", "lessons"),
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


@dataclass
class Memory:
    """An in-memory view of the state file. Mutating methods persist on save()."""

    path: Path
    facts: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    last_session: str = ""

    # --- Stage operations (each appends, dedupes, and persists) ---------------

    def remember_fact(self, fact: str) -> str:
        return self._add("facts", fact, "Saved verified fact")

    def add_rule(self, rule: str) -> str:
        return self._add("rules", rule, "Saved general rule")

    def log_failure(self, failure: str) -> str:
        return self._add("failures", f"{_now()}: {failure}", "Logged open failure")

    def distill_lesson(self, lesson: str) -> str:
        return self._add("lessons", lesson, "Distilled lesson")

    def set_last_session(self, note: str) -> str:
        self.last_session = f"{_now()} · {note.strip()}"
        self.save()
        return "Updated last-session pointer."

    def resolve_failure(self, substring: str) -> str:
        """Drop any open failure containing `substring` (e.g. once fixed)."""
        before = len(self.failures)
        self.failures = [f for f in self.failures if substring not in f]
        self.save()
        removed = before - len(self.failures)
        return f"Resolved {removed} open failure(s) matching {substring!r}."

    def _add(self, attr: str, item: str, label: str) -> str:
        item = item.strip()
        if not item:
            return "Nothing to add (empty)."
        bucket: list[str] = getattr(self, attr)
        if item not in bucket:
            bucket.append(item)
            self.save()
            return f"{label}."
        return f"{label} (already present)."

    # --- Prompt injection (stage 5: consult) ----------------------------------

    def summary_for_prompt(self, max_chars: int = 4000) -> str:
        """Render the memory for injection into a system prompt at session start."""
        if self.is_empty():
            return ""
        text = self.render(title=None).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0] + "\n… (memory truncated)"
        return (
            "## Durable memory (read this before acting; do not re-derive)\n"
            f"{text}\n"
        )

    def is_empty(self) -> bool:
        return not any([self.facts, self.rules, self.failures, self.lessons, self.last_session])

    # --- Serialization --------------------------------------------------------

    def render(self, title: str | None = "smartai memory") -> str:
        lines: list[str] = []
        if title:
            lines += [f"# {title}", ""]
        for header, attr in _SECTIONS:
            items: list[str] = getattr(self, attr)
            lines.append(f"## {header}")
            if items:
                lines += [f"- {it}" for it in items]
            else:
                lines.append("_(none yet)_")
            lines.append("")
        lines.append("## Last session")
        lines.append(self.last_session or "_(no prior session)_")
        lines.append("")
        return "\n".join(lines)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.render(), encoding="utf-8")


def _section_attr(header: str) -> str | None:
    for h, attr in _SECTIONS:
        if h.lower() == header.lower():
            return attr
    return None


def load(path: str | Path | None = None) -> Memory:
    """Load memory from disk (or return an empty Memory if the file is absent)."""
    if path is None:
        root = Path(config.workspace).expanduser().resolve()
        path = root / config.memory_file
    path = Path(path)
    mem = Memory(path=path)
    if not path.is_file():
        return mem

    current: str | None = None
    last_session_lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            current = line[3:].strip()
            continue
        if line.startswith("# "):
            current = None
            continue
        if current and current.lower() == "last session":
            if line and not line.startswith("_("):
                last_session_lines.append(line)
            continue
        if line.startswith("- "):
            attr = _section_attr(current or "")
            if attr:
                getattr(mem, attr).append(line[2:].strip())
    mem.last_session = "\n".join(last_session_lines).strip()
    return mem
