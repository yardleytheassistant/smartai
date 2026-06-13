"""A minimal OpenAI-compatible mock server for tests and live smoke runs.

It mimics the endpoints an Ollama/MLX/llama.cpp server exposes — GET /v1/models
and POST /v1/chat/completions — so the *real* `openai` client (and therefore the
real transport, routing, and tool-call handling) can be exercised end to end with
no external network and no GPU. The responder is scriptable; the default plays
out the fizzbuzz goal-loop scenario (maker issues a native tool call to write the
file, then answers; grader returns a passing verdict).

Use as a context manager:

    with MockServer() as base_url:
        client = OpenAI(base_url=base_url, api_key="x")
        ...
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_MODELS = [
    "novel", "qwen3.6:35b", "deepseek-r1:32b", "deepseek-r1:70b",
    "qwen3-coder:30b", "llama4:scout", "qwen3:235b", "kimi-vl",
]


def _is_grader(messages: list[dict]) -> bool:
    sys_msg = (messages[0].get("content") or "") if messages else ""
    return "verifier" in sys_msg.lower()


def _has_tool_result(messages: list[dict]) -> bool:
    for m in messages:
        if m.get("role") == "tool":
            return True
        if m.get("role") == "user" and "Tool results" in (m.get("content") or ""):
            return True
    return False


def default_responder(model: str, messages: list[dict]):
    """Return (content, tool_calls) for the fizzbuzz scenario."""
    if _is_grader(messages):
        # Works for both plain and rubric grading prompts.
        last = (messages[-1].get("content") or "")
        if "CRITERIA" in last or "criterion" in last.lower():
            return '{"criteria": [{"id": "c1", "met": true}, {"id": "c2", "met": true}, {"id": "c3", "met": true}]}', None
        return '{"met": true, "score": 1.0, "feedback": ""}', None

    if not _has_tool_result(messages):
        tool_calls = [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps({
                    "path": "fizzbuzz.py",
                    "content": (
                        "for i in range(1, 16):\n"
                        "    if i % 15 == 0: print('FizzBuzz')\n"
                        "    elif i % 3 == 0: print('Fizz')\n"
                        "    elif i % 5 == 0: print('Buzz')\n"
                        "    else: print(i)\n"
                    ),
                }),
            },
        }]
        return "", tool_calls
    return (
        "Done. Wrote fizzbuzz.py. It prints 1, 2, Fizz, 4, Buzz, Fizz, 7, 8, "
        "Fizz, Buzz, 11, Fizz, 13, 14, FizzBuzz.",
        None,
    )


class MockServer:
    def __init__(self, responder=default_responder, models=None):
        self.responder = responder
        self.models = models or DEFAULT_MODELS
        self.requests: list[dict] = []
        self._httpd = None
        self._thread = None

    def __enter__(self) -> str:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/v1/models"):
                    self._send(200, {"object": "list", "data": [
                        {"id": m, "object": "model"} for m in server.models
                    ]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                server.requests.append(body)
                model = body.get("model", "")
                messages = body.get("messages", [])
                content, tool_calls = server.responder(model, messages)
                message = {"role": "assistant", "content": content}
                if tool_calls:
                    message["tool_calls"] = tool_calls
                self._send(200, {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                })

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        port = self._httpd.server_address[1]
        return f"http://127.0.0.1:{port}/v1"

    def __exit__(self, *exc):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
