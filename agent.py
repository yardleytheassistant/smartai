"""The Novel agent loop.

Talks to any OpenAI-compatible endpoint (Ollama, mlx_lm.server, llama.cpp,
LM Studio) serving an open-source model, and drives a standard function-calling
loop: ask the model, run any tools it requests, feed the results back, repeat
until it answers.

The agent supports model routing (pass `model=` to run a role on a specific
open-source model) and client injection (pass `client=` so the loop is testable
without a live server). At construction it can fold durable memory into the
system prompt so the session resumes instead of restarting.
"""

from __future__ import annotations

import memory as memory_mod
import tools
from config import config


def make_client():
    """Build an OpenAI-compatible client pointed at the local model server."""
    from openai import OpenAI

    return OpenAI(base_url=config.base_url, api_key=config.api_key)


class NovelAgent:
    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        client=None,
        use_tools: bool = True,
        load_memory: bool = False,
        skills_query: str | None = None,
    ):
        self.client = client if client is not None else make_client()
        self.model = model or config.model
        self.use_tools = use_tools

        prompt = system_prompt or config.system_prompt
        if load_memory:
            summary = memory_mod.load().summary_for_prompt()
            if summary:
                prompt = f"{prompt}\n\n{summary}"
        if skills_query:
            # Procedural memory: fold in the most relevant Skills for this task.
            import skills as skills_mod

            block = skills_mod.prompt_block(skills_query)
            if block:
                prompt = f"{prompt}\n\n{block}"
        self.messages: list[dict] = [{"role": "system", "content": prompt}]

    def run(self, user_input: str, *, on_tool=None) -> str:
        """Send a user message and return the agent's final text answer.

        on_tool, if given, is called as on_tool(name, arguments, result) after
        each tool execution so a UI can show progress.
        """
        self.messages.append({"role": "user", "content": user_input})
        schema = tools.openai_schema() if self.use_tools else None

        for _ in range(config.max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=schema,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            message = response.choices[0].message

            # Record the assistant turn (including any tool calls) before acting.
            self.messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls]
                    if message.tool_calls
                    else None,
                }
            )

            if not message.tool_calls:
                return message.content or ""

            for call in message.tool_calls:
                name = call.function.name
                arguments = call.function.arguments
                result = tools.dispatch(name, arguments)
                if on_tool is not None:
                    on_tool(name, arguments, result)
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )

        return "Stopped: reached the maximum number of tool-calling steps without a final answer."


# Backwards-compatible alias.
HermesAgent = NovelAgent
