"""Routines — saved configs that run on a trigger (step 9), open-source.

Hosted Routines run on Anthropic-managed cloud so your laptop can be off. The
open-source workaround is two-fold and uses only local infrastructure:

  1. cron        — install_cron() writes crontab entries that invoke
                   `python main.py routine run <name>`. Survives logout; the
                   machine just needs to be on (a always-on Mac/mini/box).
  2. serve()     — a built-in scheduler daemon (`python main.py routine serve`)
                   that runs due routines on an interval. Use it under nohup,
                   tmux, systemd, or launchd when you don't want to touch cron.

Triggers:
  schedule  — a 5-field cron expression (own minimal matcher, no dependency).
  event     — fire when a sentinel/watch file appears or changes (the local
              analog of "CI failed -> fire a routine"; a CI hook/webhook just
              touches the file).
  manual    — only runs when invoked explicitly.

Each run executes a goal loop, logs the outcome, and updates durable memory, so
routines are how the system compounds while you sleep.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import memory as memory_mod
from config import config
from loop import LoopResult, goal_loop


@dataclass
class Routine:
    name: str
    goal: str
    trigger: str = "manual"           # schedule | event | manual
    schedule: str = ""                # cron expr for trigger == schedule
    watch_path: str = ""              # sentinel file for trigger == event
    model: str = ""                   # blank => primary model
    rubric: str = ""
    max_iterations: int = 0           # 0 => config default
    enabled: bool = True


# --- Minimal cron matcher (no dependency) -----------------------------------

def _match_field(field: str, value: int, lo: int, hi: int) -> bool:
    if field == "*":
        return True
    for part in field.split(","):
        part = part.strip()
        if part.startswith("*/"):
            step = int(part[2:])
            if step > 0 and (value - lo) % step == 0:
                return True
        elif "-" in part:
            a, b = part.split("-")
            if int(a) <= value <= int(b):
                return True
        elif part.isdigit() and int(part) == value:
            return True
    return False


def cron_due(expr: str, dt: datetime) -> bool:
    """True if a 5-field cron expression (m h dom mon dow) matches dt.

    dow is 0-6 with Sunday=0, standard cron convention.
    """
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"cron expression must have 5 fields, got {expr!r}")
    minute, hour, dom, mon, dow = fields
    py_dow = (dt.weekday() + 1) % 7  # Mon=0 -> Sun=0 convention
    return (
        _match_field(minute, dt.minute, 0, 59)
        and _match_field(hour, dt.hour, 0, 23)
        and _match_field(dom, dt.day, 1, 31)
        and _match_field(mon, dt.month, 1, 12)
        and _match_field(dow, py_dow, 0, 6)
    )


# --- Registry ----------------------------------------------------------------

def _registry_path() -> Path:
    p = Path(config.routines_file)
    return p if p.is_absolute() else Path.cwd() / p


def load_routines(path: str | Path | None = None) -> list[Routine]:
    p = Path(path) if path is not None else _registry_path()
    if not p.is_file():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [Routine(**obj) for obj in data]


def save_routines(routines: list[Routine], path: str | Path | None = None) -> None:
    p = Path(path) if path is not None else _registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([asdict(r) for r in routines], indent=2), encoding="utf-8")


def add_routine(routine: Routine, path: str | Path | None = None) -> None:
    routines = [r for r in load_routines(path) if r.name != routine.name]
    routines.append(routine)
    save_routines(routines, path)


def get_routine(name: str, path: str | Path | None = None) -> Routine | None:
    for r in load_routines(path):
        if r.name == name:
            return r
    return None


def remove_routine(name: str, path: str | Path | None = None) -> bool:
    routines = load_routines(path)
    kept = [r for r in routines if r.name != name]
    save_routines(kept, path)
    return len(kept) != len(routines)


# --- Scheduling logic (pure, testable) --------------------------------------

def due_routines(now: datetime, routines: list[Routine] | None = None) -> list[Routine]:
    """Schedule-triggered routines whose cron expression matches `now`."""
    routines = routines if routines is not None else load_routines()
    out: list[Routine] = []
    for r in routines:
        if not r.enabled or r.trigger != "schedule" or not r.schedule:
            continue
        if cron_due(r.schedule, now):
            out.append(r)
    return out


def fired_events(routines: list[Routine] | None = None) -> list[Routine]:
    """Event-triggered routines whose watch/sentinel file currently exists."""
    routines = routines if routines is not None else load_routines()
    return [
        r for r in routines
        if r.enabled and r.trigger == "event" and r.watch_path and Path(r.watch_path).exists()
    ]


# --- Execution ---------------------------------------------------------------

def _log_path(name: str) -> Path:
    root = Path(config.workspace).expanduser() / "routine-logs"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{name}.log"


def run_routine(routine: Routine, *, client=None, on_event=None) -> LoopResult:
    """Execute a routine's goal loop, log it, and update durable memory."""
    result = goal_loop(
        routine.goal,
        rubric=routine.rubric or None,
        max_iterations=routine.max_iterations or None,
        maker_model=routine.model or None,
        client=client,
        on_event=on_event,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = "MET" if result.met else "UNMET"
    with _log_path(routine.name).open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {status} in {result.iterations} iter — {routine.goal}\n")
    memory_mod.load().set_last_session(
        f"Routine {routine.name!r} ran: {status} in {result.iterations} iter."
    )
    # Clear an event sentinel so it doesn't re-fire until touched again.
    if routine.trigger == "event" and routine.watch_path:
        try:
            Path(routine.watch_path).unlink(missing_ok=True)
        except OSError:
            pass
    return result


def run_by_name(name: str, *, client=None, on_event=None) -> LoopResult:
    routine = get_routine(name)
    if routine is None:
        raise KeyError(f"no routine named {name!r}")
    return run_routine(routine, client=client, on_event=on_event)


# --- cron installer ----------------------------------------------------------

_CRON_MARKER = "# smartai-routine"


def cron_line(routine: Routine, *, python: str | None = None, project_dir: str | Path | None = None) -> str:
    import sys
    python = python or sys.executable
    project = Path(project_dir) if project_dir else Path(__file__).resolve().parent
    log = _log_path(routine.name)
    cmd = f"cd {project} && {python} main.py routine run {routine.name} >> {log} 2>&1"
    return f"{routine.schedule} {cmd} {_CRON_MARKER}:{routine.name}"


def install_cron(names: list[str] | None = None, *, python: str | None = None,
                 project_dir: str | Path | None = None) -> str:
    """Install/refresh crontab entries for schedule-triggered routines.

    Returns the new crontab text. Replaces any prior smartai-routine lines.
    """
    import subprocess

    routines = [r for r in load_routines() if r.trigger == "schedule" and r.schedule]
    if names:
        routines = [r for r in routines if r.name in names]

    existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    lines = [] if existing.returncode != 0 else [
        ln for ln in existing.stdout.splitlines() if _CRON_MARKER not in ln
    ]
    lines += [cron_line(r, python=python, project_dir=project_dir) for r in routines]
    new_crontab = "\n".join(lines) + "\n"
    subprocess.run(["crontab", "-"], input=new_crontab, text=True, check=True)
    return new_crontab


# --- Built-in scheduler daemon (cron-free workaround) -----------------------

def serve(*, interval: float = 30.0, client=None, on_event=None, _now=None, _max_ticks=None) -> None:
    """Run due schedule/event routines on an interval until interrupted.

    Use under nohup/tmux/systemd/launchd for laptop-off operation without cron.
    `_now` and `_max_ticks` exist for testing; production runs unbounded.
    """
    import time

    now_fn = _now or (lambda: datetime.now())
    seen_minute: dict[str, str] = {}
    ticks = 0
    while True:
        now = now_fn()
        minute_key = now.strftime("%Y-%m-%d %H:%M")
        for r in due_routines(now):
            # Don't run the same scheduled routine twice within one minute.
            if seen_minute.get(r.name) == minute_key:
                continue
            seen_minute[r.name] = minute_key
            run_routine(r, client=client, on_event=on_event)
        for r in fired_events():
            run_routine(r, client=client, on_event=on_event)
        ticks += 1
        if _max_ticks is not None and ticks >= _max_ticks:
            return
        time.sleep(interval)
