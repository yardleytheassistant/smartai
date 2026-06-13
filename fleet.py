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
    latency_s: float | None = None  # first-token-ish latency from a 1-token probe


def available_models(client=None) -> set[str]:
    """Model ids the server reports (via the OpenAI-compatible /models endpoint)."""
    client = client if client is not None else make_client()
    return {m.id for m in client.models.list().data}


def probe_latency(model: str, client) -> float | None:
    """Time a minimal (1-token) completion. Returns seconds, or None on failure.

    This loads the model server-side, so it can be slow for large models — it's
    opt-in (fleet --probe) for exactly that reason.
    """
    import time

    start = time.perf_counter()
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1,
            temperature=0.0,
        )
    except Exception:  # noqa: BLE001 - unreachable / not pulled
        return None
    return time.perf_counter() - start


def _matches(model: str, available: set[str]) -> bool:
    # Ollama reports tags like "qwen3.5:122b"; also accept an implicit ":latest".
    return model in available or f"{model}:latest" in available or any(
        a.split(":", 1)[0] == model for a in available
    )


def check(available: set[str] | None = None, client=None, *, probe: bool = False) -> list[FleetStatus]:
    if available is None:
        available = available_models(client)
    if probe and client is None:
        client = make_client()
    seen: dict[str, FleetStatus] = {}
    for role, model in role_models().items():
        if role in seen:
            continue
        is_up = _matches(model, available)
        latency = probe_latency(model, client) if (probe and is_up) else None
        seen[role] = FleetStatus(role=role, model=model, available=is_up, latency_s=latency)
    return list(seen.values())
