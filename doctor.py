"""Operational self-check — is the system ready to run?

Verifies the things that silently break a routine at 3am: the workspace is
writable, the model server is reachable, and each role's model is actually
loaded. Returns structured checks so the CLI can render them and exit non-zero
if anything critical fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    critical: bool = True


def _workspace_check() -> Check:
    try:
        root = Path(config.workspace).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return Check("workspace writable", True, str(root))
    except OSError as exc:
        return Check("workspace writable", False, str(exc))


def run(available: "set[str] | None" = None, client=None) -> list[Check]:
    """Run all checks. If `available` is given, skip the live server query."""
    checks: list[Check] = [_workspace_check()]

    # Knowledge/skills dirs are optional (warn, don't fail).
    import knowledge
    import skills
    checks.append(Check("skills directory", skills.skills_root().is_dir(),
                        str(skills.skills_root()), critical=False))
    checks.append(Check("knowledge directory", knowledge.knowledge_root().is_dir(),
                        str(knowledge.knowledge_root()), critical=False))

    # Model server + per-role availability.
    if available is None:
        try:
            import fleet
            available = fleet.available_models(client)
            checks.append(Check("model server reachable", True, f"{len(available)} models"))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("model server reachable", False, str(exc)))
            return checks

    import fleet
    for status in fleet.check(available=available):
        # Only the orchestrator/grader are critical to the core loop.
        critical = status.role in {"orchestrator", "grader"}
        checks.append(Check(f"role '{status.role}' ({status.model})", status.available,
                            "loaded" if status.available else "not loaded", critical=critical))
    return checks


def ok(checks: list[Check]) -> bool:
    """True if no critical check failed."""
    return all(c.ok for c in checks if c.critical)
