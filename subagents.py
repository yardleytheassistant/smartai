"""Sub-agent delegation — the orchestrator spawning specialist workers.

The orchestrator delegates a bounded subtask to a fresh agent that runs on the
best-fit model for the work (by explicit role, explicit model, or automatic
routing) with its own clean context. Each sub-agent loads durable memory and
relevant skills, so delegates inherit the compounding state too.

A thread-local depth counter caps recursion so the system can't spawn agents
without bound (config.subagent_max_depth).
"""

from __future__ import annotations

import threading

from config import config
from router import route_model

_depth = threading.local()

_ROLE_MODELS = {
    "orchestrator": "model",
    "worker": "worker_model",
    "grader": "grader_model",
    "reasoner": "reasoner_model",
    "coder": "coder_model",
    "long_context": "long_context_model",
    "heavy": "heavy_model",
}


def role_model(role: str | None) -> str | None:
    if not role:
        return None
    attr = _ROLE_MODELS.get(role)
    return getattr(config, attr) if attr else None


def current_depth() -> int:
    return getattr(_depth, "value", 0)


def delegate(
    task: str,
    *,
    role: str | None = None,
    model: str | None = None,
    client=None,
    load_memory: bool = True,
    use_tools: bool = True,
    on_tool=None,
) -> str:
    """Run a subtask on a fresh sub-agent and return its answer.

    Model selection precedence: explicit `model` > `role` mapping > automatic
    routing by task shape. Refuses to recurse past config.subagent_max_depth.
    `use_tools=False` makes the delegate answer from its prompt without the tool
    loop — used when the relevant source is already inlined into the task.
    """
    depth = current_depth()
    if depth >= config.subagent_max_depth:
        return (
            f"Error: sub-agent max depth ({config.subagent_max_depth}) reached; "
            "refusing to delegate further."
        )
    chosen = model or role_model(role) or route_model(task)

    from agent import NovelAgent  # lazy: avoids import cycle (agent -> tools -> subagents)

    _depth.value = depth + 1
    try:
        agent = NovelAgent(
            model=chosen, client=client, load_memory=load_memory,
            use_tools=use_tools, skills_query=task,
        )
        return agent.run(task, on_tool=on_tool)
    finally:
        _depth.value = depth
