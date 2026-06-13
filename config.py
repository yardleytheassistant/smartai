"""Runtime configuration for the Hermes agent.

Everything is driven by environment variables (see .env.example) so the same
code runs against any OpenAI-compatible local server — Ollama, mlx_lm.server,
llama.cpp, LM Studio — by changing only LLM_BASE_URL / LLM_MODEL.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SYSTEM_PROMPT = """\
You are Hermes, a capable local AI agent running on the user's own machine.

You can call tools to read and write files, inspect the workspace, run shell
commands (when enabled), and do exact arithmetic. Prefer using a tool over
guessing when a tool can give you a precise answer.

Work in small, verifiable steps. When you have enough information to answer,
stop calling tools and give a clear, direct final answer. Lead with the
outcome, then any supporting detail. Do not narrate routine actions.
"""


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    # Local servers don't check the key, but the OpenAI client requires a non-empty value.
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "ollama"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "smartai"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096")))

    max_steps: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_STEPS", "12")))
    enable_shell: bool = field(default_factory=lambda: _get_bool("AGENT_ENABLE_SHELL", False))
    workspace: str = field(default_factory=lambda: os.getenv("AGENT_WORKSPACE", "./workspace"))
    system_prompt: str = field(default_factory=lambda: os.getenv("AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT))


config = Config()
