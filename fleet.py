"""Fleet status — which role models are actually available on the server.

A self-improving system routes across many models; this reports, for each role,
whether its configured model is loaded on the local OpenAI-compatible server
(Ollama's /v1/models). Operational glue: catch a missing pull or an unbuilt
'novel' before a routine fails at 3am.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import make_client
from config import config


def role_models() -> dict[str, str]:
    return {
        "orchestrator": config.model,
        "worker": config.worker_model,
        "grader": config.grader_model,
        "reasoner": config.reasoner_model,
        "coder": config.coder_model,
        "long_context": config.long_context_model,
        "heavy": config.heavy_model,
        "vision": config.vision_model,
        "fallback": config.fallback_model,
    }


@dataclass
class FleetStatus:
    role: str
    model: str
    available: bool


def available_models(client=None) -> set[str]:
    """Model ids the server reports (via the OpenAI-compatible /models endpoint)."""
    client = client if client is not None else make_client()
    return {m.id for m in client.models.list().data}


def _matches(model: str, available: set[str]) -> bool:
    # Ollama reports tags like "qwen3.5:122b"; also accept an implicit ":latest".
    return model in available or f"{model}:latest" in available or any(
        a.split(":", 1)[0] == model for a in available
    )


def check(available: set[str] | None = None, client=None) -> list[FleetStatus]:
    if available is None:
        available = available_models(client)
    seen: dict[str, FleetStatus] = {}
    for role, model in role_models().items():
        seen.setdefault(role, FleetStatus(role=role, model=model, available=_matches(model, available)))
    return list(seen.values())
