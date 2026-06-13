"""Runtime configuration for the smartai self-improving agent system.

Everything is driven by environment variables (see .env.example) so the same
code runs against any OpenAI-compatible *local, open-source* model server —
Ollama, mlx_lm.server, llama.cpp, LM Studio — by changing only the endpoint
and the model names. No cloud APIs, no Claude/Anthropic dependency: the models
are open-source checkpoints you run on your own hardware.

Model routing (step 4 of the compound stack): different roles in the system
get different open-source models. The orchestrator does the heavy planning, a
worker handles bounded fan-out tasks, and an independent grader verifies work.
By default all three point at the custom `smartai` model so the system runs out
of the box; override any role to route to a smaller/faster open model.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SYSTEM_PROMPT = """\
You are smartai, a capable local AI agent running on the user's own machine on
open-source models. No cloud, no per-token cost.

You can call tools to read and write files, inspect the workspace, run shell
commands (when enabled), do exact arithmetic, and record what you learn to
durable memory. Prefer using a tool over guessing when a tool can give you a
precise answer, and pass arguments as valid JSON.

Work in small, verifiable steps. When you discover a durable fact, save it with
remember_fact; when you derive a general rule, save it with add_rule; when
something fails, record it with log_failure. When you have enough information to
answer, stop calling tools and answer directly: lead with the outcome, then any
supporting detail. Do not narrate routine actions.
"""


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    # Local servers don't check the key, but the OpenAI client requires a non-empty value.
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "ollama"))

    # --- Model routing (all open-source) ---
    # `model` is the primary/orchestrator model. worker/grader default to it but
    # can be routed to other open models (e.g. a smaller Hermes/Qwen for grading).
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "novel"))
    worker_model: str = field(default_factory=lambda: os.getenv("WORKER_MODEL", ""))
    grader_model: str = field(default_factory=lambda: os.getenv("GRADER_MODEL", ""))

    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096")))
    # Graders should be near-deterministic so verdicts are reproducible.
    grader_temperature: float = field(default_factory=lambda: float(os.getenv("GRADER_TEMPERATURE", "0.0")))

    # --- Agent loop ---
    max_steps: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_STEPS", "12")))
    enable_shell: bool = field(default_factory=lambda: _get_bool("AGENT_ENABLE_SHELL", False))
    workspace: str = field(default_factory=lambda: os.getenv("AGENT_WORKSPACE", "./workspace"))
    system_prompt: str = field(default_factory=lambda: os.getenv("AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT))

    # --- Self-improvement layer ---
    # Max maker->verifier iterations in a goal loop before giving up.
    max_iterations: int = field(default_factory=lambda: int(os.getenv("GOAL_MAX_ITERATIONS", "4")))
    # Durable memory / state file (relative to the workspace), the 5-stage store.
    memory_file: str = field(default_factory=lambda: os.getenv("AGENT_MEMORY_FILE", "STATE.md"))

    def __post_init__(self) -> None:
        # Roles fall back to the primary model so the system runs out of the box.
        self.worker_model = self.worker_model or self.model
        self.grader_model = self.grader_model or self.model


config = Config()
