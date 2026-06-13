"""The Hermes agent loop.

Talks to any OpenAI-compatible endpoint (Ollama, mlx_lm.server, llama.cpp,
LM Studio) and drives a standard function-calling loop: ask the model, run any
tools it requests, feed the results back, repeat until it answers.
"""

from openai import OpenAI

import tools
from config import config


class HermesAgent:
    def __init__(self, *, system_prompt: str | None = None):
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt or config.system_prompt}
        ]

    def run(self, user_input: str, *, on_tool=None) -> str:
        """Send a user message and return the agent's final text answer.

        on_tool, if given, is called as on_tool(name, arguments, result) after
        each tool execution so a UI can show progress.
        """
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(config.max_steps):
            response = self.client.chat.completions.create(
                model=config.model,
                messages=self.messages,
                tools=tools.openai_schema(),
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
