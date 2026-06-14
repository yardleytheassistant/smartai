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

import compaction
import memory as memory_mod
import sanitize
import toolcall
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
        compact: bool = True,
        compact_model: str | None = None,
        messages: list[dict] | None = None,
    ):
        self.client = client if client is not None else make_client()
        self.model = model or config.model
        self.use_tools = use_tools
        self.compact = compact
        self.compact_model = compact_model or config.worker_model

        if messages is not None:
            # Resume from a saved transcript (must already include the system turn).
            self.messages = list(messages)
            return

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
        # Record tool names used this run so callers (e.g. the goal loop) can tell
        # whether the model actually acted or merely described the action.
        self.tools_used: list[str] = []

        for _ in range(config.max_steps):
            # Manage context: summarize completed earlier turns if we're over budget.
            if self.compact and compaction.estimate_chars(self.messages) > config.context_char_budget:
                self.messages = compaction.compact(
                    self.messages, client=self.client, model=self.compact_model
                )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=schema,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            message = response.choices[0].message
            content = message.content or ""

            # Prefer native tool calls; otherwise recover ones emitted as text
            # (Hermes/Qwen/Llama tunes don't always populate the native field).
            native = list(message.tool_calls or [])
            if native:
                calls = [
                    {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in native
                ]
            elif self.use_tools:
                calls = toolcall.parse_text_tool_calls(content, valid_names=set(tools._REGISTRY))
            else:
                calls = []

            self.messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [tc.model_dump() for tc in native] if native else None,
                }
            )

            if not calls:
                return sanitize.strip_reasoning(content)

            text_results: list[str] = []
            for call in calls:
                self.tools_used.append(call["name"])
                result = tools.dispatch(call["name"], call["arguments"])
                if on_tool is not None:
                    on_tool(call["name"], call["arguments"], result)
                if native:
                    self.messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": result}
                    )
                else:
                    text_results.append(f"{call['name']} -> {result}")
            if text_results:
                # No native tool_calls to attach a tool role to: feed results back
                # as a user turn, which every OpenAI-compatible server accepts.
                self.messages.append(
                    {"role": "user", "content": "Tool results:\n" + "\n".join(text_results)}
                )

        return "Stopped: reached the maximum number of tool-calling steps without a final answer."


# Backwards-compatible alias.
HermesAgent = NovelAgent
