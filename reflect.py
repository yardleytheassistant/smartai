"""Reflection — meta-distillation (closing the self-improvement loop).

The goal loop logs open failures and lessons to memory. Reflection reads those
back and asks the strongest reasoning model to generalize them: propose durable
general rules and concrete known-failure-modes for Skills. The proposals are
written back to memory (and optionally into a Skill), so accumulated specific
failures become reusable general knowledge.

This is the step that turns "the system recorded what happened" into "the system
got sharper" — run it on a schedule (a Routine) to compound while you sleep.
"""

from __future__ import annotations

from agent import make_client
from config import config
from verifier import _extract_json

import memory as memory_mod
import skills as skills_mod

_REFLECT_SYSTEM = """\
You are a reflective engineer reviewing an agent's accumulated failures and
lessons. Generalize them into reusable knowledge — do not restate specifics.

Return ONLY a JSON object, no prose, no code fences:
{
  "rules": ["general rule that prevents a class of failure", ...],
  "failure_modes": [{"skill": "<skill-name>", "mode": "<symptom>: <fix>"}, ...]
}
Propose rules only when well-supported by the evidence. Empty arrays are fine.
"""


def reflect(*, client=None, model: str | None = None, skill: str | None = None) -> dict:
    """Distill durable rules / failure modes from memory; write them back.

    Returns a summary dict: {rules: [...], failure_modes: [...], note: str}.
    """
    mem = memory_mod.load()
    if not (mem.failures or mem.lessons):
        return {"rules": [], "failure_modes": [], "note": "nothing to reflect on yet"}

    model = model or config.reasoner_model
    client = client if client is not None else make_client()

    evidence = "## Open failures\n" + "\n".join(f"- {f}" for f in mem.failures)
    evidence += "\n\n## Lessons\n" + "\n".join(f"- {l}" for l in mem.lessons)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _REFLECT_SYSTEM},
            {"role": "user", "content": evidence + "\n\nReflect now. Return only the JSON."},
        ],
        temperature=config.grader_temperature,
        max_tokens=config.max_tokens,
    )
    raw = response.choices[0].message.content or ""
    try:
        data = _extract_json(raw)
    except Exception:  # noqa: BLE001 - unparseable reflection is a no-op
        return {"rules": [], "failure_modes": [], "note": f"unparseable reflection: {raw[:200]}"}

    rules = [r for r in data.get("rules", []) if isinstance(r, str) and r.strip()]
    for rule in rules:
        mem.add_rule(rule)

    applied: list[str] = []
    for fm in data.get("failure_modes", []):
        if not isinstance(fm, dict):
            continue
        target = fm.get("skill") or skill
        mode = (fm.get("mode") or "").strip()
        if not (target and mode):
            continue
        sk = skills_mod.get(target)
        if sk is not None:
            sk.add_failure_mode(mode)
            applied.append(f"{target}: {mode}")

    mem.set_last_session(f"Reflection added {len(rules)} rule(s), {len(applied)} failure mode(s).")
    return {"rules": rules, "failure_modes": applied, "note": "reflection applied"}
