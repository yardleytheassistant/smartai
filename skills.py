"""Skills — procedural memory that compounds (step 12).

STATE.md is project memory ("what's true about this project"). Skills are
procedural memory ("how to do this kind of thing") that travels across projects.
The compounding pattern: after any non-trivial failure, write the lesson into
the Skill itself, so the Skill gets sharper every time the system runs.

A Skill is a Markdown file with a tiny frontmatter block:

    ---
    name: ci-triage
    description: Classify CI failures, draft fixes for easy ones, escalate the rest.
    when: a CI/build run fails, or during morning triage of red pipelines
    trigger: workflow_run.failure, morning triage
    ---
    # CI triage skill
    ... body ...
    ## Known failure modes      <- appended to as the loop learns
    - webhook-race: ...

`description` says what the skill does; `when` says the situation it applies to
(surfaced to the model as "Apply when:" so it can decide whether to use it);
`trigger` is keyword fuel for retrieval. No YAML dependency: the frontmatter is
simple `key: value` lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config import config

_FAILURE_MODES_HEADER = "## Known failure modes"


@dataclass
class Skill:
    name: str
    description: str
    trigger: str
    body: str
    when: str = ""
    path: Path | None = None
    meta: dict = field(default_factory=dict)

    def matches(self, query: str) -> bool:
        """True if the query plausibly invokes this skill (keyword overlap)."""
        q = query.lower()
        # `when` (applicability) is part of the haystack so situational wording
        # ("the build failed") retrieves the skill even if the name doesn't appear.
        hay = f"{self.name} {self.description} {self.trigger} {self.when}".lower()
        words = {w for w in _tokenize(hay) if len(w) > 3}
        return any(w in q for w in words) or self.name.lower() in q

    def add_failure_mode(self, lesson: str) -> str:
        """Append a learned failure mode into the Skill body and persist it."""
        lesson = lesson.strip()
        if not lesson:
            return "Nothing to add (empty)."
        body = self.body
        bullet = f"- {lesson}"
        if bullet in body:
            return "Failure mode already recorded."
        if _FAILURE_MODES_HEADER in body:
            head, _, tail = body.partition(_FAILURE_MODES_HEADER)
            # Insert the bullet right after the header line.
            rest_lines = tail.splitlines()
            first = rest_lines[0] if rest_lines else ""
            remainder = "\n".join(rest_lines[1:])
            tail = f"{first}\n{bullet}\n{remainder}".rstrip() + "\n"
            self.body = f"{head}{_FAILURE_MODES_HEADER}{tail}"
        else:
            sep = "" if body.endswith("\n") else "\n"
            self.body = f"{body}{sep}\n{_FAILURE_MODES_HEADER}\n{bullet}\n"
        self.save()
        return "Recorded failure mode into the skill."

    def render(self) -> str:
        fm = [
            "---",
            f"name: {self.name}",
            f"description: {self.description}",
        ]
        if self.when:
            fm.append(f"when: {self.when}")
        if self.trigger:
            fm.append(f"trigger: {self.trigger}")
        fm.append("---")
        return "\n".join(fm) + "\n" + self.body.lstrip("\n")

    def save(self) -> None:
        if self.path is None:
            root = skills_root()
            self.path = root / f"{self.name}.md"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.render(), encoding="utf-8")


def _tokenize(text: str) -> list[str]:
    return [t.strip(",.;:()[]") for t in text.replace("/", " ").split()]


def parse(text: str, *, path: Path | None = None) -> Skill:
    """Parse a Skill from Markdown-with-frontmatter text."""
    meta: dict = {}
    body = text
    if text.lstrip().startswith("---"):
        stripped = text.lstrip()
        _, _, rest = stripped.partition("---")
        fm_block, sep, body = rest.partition("---")
        if sep:
            for line in fm_block.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip().lower()] = val.strip()
        else:
            body = text  # no closing fence; treat whole thing as body
    name = meta.get("name") or (path.stem if path else "unnamed")
    return Skill(
        name=name,
        description=meta.get("description", ""),
        trigger=meta.get("trigger", ""),
        body=body.lstrip("\n"),
        when=meta.get("when", ""),
        path=path,
        meta=meta,
    )


def skills_root() -> Path:
    """Skills are versioned knowledge that travels — they live alongside the
    project (repo root), not in the runtime workspace. Absolute paths win."""
    p = Path(config.skills_dir).expanduser()
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent / p


def load_skills(directory: str | Path | None = None) -> list[Skill]:
    root = Path(directory) if directory is not None else skills_root()
    if not root.is_dir():
        return []
    out: list[Skill] = []
    for f in sorted(root.glob("*.md")):
        out.append(parse(f.read_text(encoding="utf-8"), path=f))
    return out


def match(query: str, directory: str | Path | None = None) -> list[Skill]:
    return [s for s in load_skills(directory) if s.matches(query)]


def get(name: str, directory: str | Path | None = None) -> Skill | None:
    for s in load_skills(directory):
        if s.name == name:
            return s
    return None


def prompt_block(query: str, *, max_skills: int = 3, directory: str | Path | None = None) -> str:
    """Render the most relevant skills for injection into a system prompt.

    Each skill leads with its `Apply when:` condition (if set), and the block
    opens with an instruction telling the model these were keyword-retrieved and
    to apply each only when its condition fits the task — so the model decides
    relevance, not just the retrieval heuristic.
    """
    relevant = match(query, directory)[:max_skills]
    if not relevant:
        return ""
    chunks = []
    for s in relevant:
        head = f"### Skill: {s.name}"
        if s.when:
            head += f"\n**Apply when:** {s.when}"
        if s.description:
            head += f"\n{s.description}"
        chunks.append(f"{head}\n\n{s.body.strip()}")
    intro = (
        "## Relevant skills (procedural memory)\n"
        "These were retrieved by keyword and may not all apply. Use a skill only "
        "when its **Apply when** condition fits the task; otherwise ignore it.\n\n"
    )
    return intro + "\n\n".join(chunks) + "\n"
