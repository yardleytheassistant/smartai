"""Model routing + safety guardrail — the open-source analog of the
cost-capability matrix (step 4) and the Mythos safety boundary (step 14).

Hosted Mythos models ship classifiers that silently downgrade certain domains.
Open-source models don't block — so the boundary becomes a *policy you control*:
detect sensitive-domain tasks, then route them to a designated fallback model
and/or flag them for human review. Nothing is silent; everything is configurable.

Routing picks an open-source model by task complexity instead of always paying
for the heaviest model: simple fan-out work goes to the worker model, hard or
orchestration work to the primary model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config import config

# --- Complexity routing (cost-capability matrix) ----------------------------

_SIMPLE_HINTS = (
    "lint", "format", "rename", "typo", "docstring", "comment", "import",
    "bump", "scaffold", "boilerplate", "stub", "spelling",
)
_ORCHESTRATION_HINTS = (
    "plan", "migrate", "migration", "refactor across", "multi-step", "orchestrate",
    "architecture", "design the", "end-to-end", "across the codebase", "for days",
)
_CODING_HINTS = (
    "debug", "stack trace", "traceback", "refactor", "unit test", "compile",
    "regex", "endpoint", "code review", "pull request", ".py", "function signature",
    "implement the function", "write a function", "fix the bug",
)
_LONG_CONTEXT_HINTS = (
    "entire codebase", "whole repo", "whole repository", "all the files",
    "long document", "large document", "1m context", "10m", "huge file",
)


def classify_complexity(task: str) -> str:
    """Heuristically bucket a task as 'simple' | 'hard' | 'orchestration'.

    Cheap and deterministic — no model call. Override per-call by passing an
    explicit model anywhere routing is used.
    """
    t = task.lower()
    if any(h in t for h in _ORCHESTRATION_HINTS):
        return "orchestration"
    if any(h in t for h in _SIMPLE_HINTS) and len(t) < 200:
        return "simple"
    return "hard"


def route_model(task: str) -> str:
    """Pick the best-fit open-source model from the local fleet for a task.

    Priority: coding specialist -> long-context model -> complexity bucket
    (simple->worker, orchestration->orchestrator, hard->reasoner).
    """
    t = task.lower()
    if any(h in t for h in _CODING_HINTS):
        return config.coder_model
    if any(h in t for h in _LONG_CONTEXT_HINTS):
        return config.long_context_model
    bucket = classify_complexity(task)
    if bucket == "simple":
        return config.worker_model
    if bucket == "orchestration":
        return config.model
    return config.reasoner_model  # 'hard' but bounded -> delegate to the reasoner


# --- Safety guardrail (configurable, never silent) --------------------------

# Domain -> signal keywords. Broad on purpose, like the documented classifier.
SENSITIVE_DOMAINS: dict[str, tuple[str, ...]] = {
    "cyber": ("exploit", "malware", "ransomware", "vulnerability research", "0day",
              "zero-day", "payload", "rootkit", "keylogger", "sql injection"),
    "bio": ("pathogen", "virus synthesis", "bioweapon", "toxin", "select agent"),
    "chem": ("nerve agent", "explosive synthesis", "precursor chemical", "chemical weapon"),
    "distillation": ("distill the model", "extract model weights", "clone the model",
                     "replicate weights", "steal the model"),
}


@dataclass
class Decision:
    task: str
    model: str
    complexity: str
    domain: str | None      # sensitive domain matched, if any
    action: str             # "proceed" | "fallback" | "block" | "review"
    note: str = ""


def detect_domain(task: str) -> str | None:
    t = task.lower()
    for domain, signals in SENSITIVE_DOMAINS.items():
        if any(re.search(rf"\b{re.escape(s)}", t) for s in signals):
            return domain
    return None


def decide(task: str) -> Decision:
    """Route a task and apply the safety policy. Returns a transparent Decision.

    Policy is set by SAFETY_POLICY in config:
      route  (default) — sensitive tasks go to the fallback model, flagged
      review           — sensitive tasks are held for human review
      block            — sensitive tasks are refused
      allow            — no special handling (you accept the risk)
    """
    complexity = classify_complexity(task)
    model = route_model(task)
    domain = detect_domain(task)
    if domain is None:
        return Decision(task, model, complexity, None, "proceed")

    policy = config.safety_policy
    if policy == "allow":
        return Decision(task, model, complexity, domain, "proceed",
                        f"sensitive domain '{domain}' allowed by policy")
    if policy == "block":
        return Decision(task, config.fallback_model, complexity, domain, "block",
                        f"refused: sensitive domain '{domain}'")
    if policy == "review":
        return Decision(task, config.fallback_model, complexity, domain, "review",
                        f"held for human review: sensitive domain '{domain}'")
    # default: route to the fallback model, transparently
    return Decision(task, config.fallback_model, complexity, domain, "fallback",
                    f"routed to fallback model for sensitive domain '{domain}'")
