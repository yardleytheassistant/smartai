"""Tool definitions for the Novel agent.

Each tool is a plain Python function registered with @tool. The registry
exposes them in OpenAI function-calling format and dispatches calls by name.
File and shell tools are sandboxed to config.workspace so the agent can't
touch paths outside it.
"""

import ast
import json
import operator
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import memory
from config import config


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]


_REGISTRY: dict[str, Tool] = {}


def tool(description: str, parameters: dict):
    """Register a function as an agent tool. `parameters` is a JSON Schema object."""

    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        _REGISTRY[func.__name__] = Tool(
            name=func.__name__,
            description=description,
            parameters=parameters,
            func=func,
        )
        return func

    return decorator


# --- Sandbox helpers ---------------------------------------------------------

def _workspace_root() -> Path:
    root = Path(config.workspace).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_path(relative: str) -> Path:
    """Resolve a path and guarantee it stays inside the workspace."""
    root = _workspace_root()
    candidate = (root / relative).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"Path '{relative}' escapes the workspace sandbox.")
    return candidate


# --- Tools -------------------------------------------------------------------

@tool(
    description="Get the current local date and time.",
    parameters={"type": "object", "properties": {}, "required": []},
)
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()


@tool(
    description="List files and directories inside the workspace.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace. Use '.' for the root."}
        },
        "required": ["path"],
    },
)
def list_directory(path: str = ".") -> str:
    target = _safe_path(path)
    if not target.exists():
        return f"Error: '{path}' does not exist."
    if target.is_file():
        return f"{path} is a file ({target.stat().st_size} bytes)."
    entries = []
    for child in sorted(target.iterdir()):
        kind = "dir " if child.is_dir() else "file"
        size = "" if child.is_dir() else f" ({child.stat().st_size} b)"
        entries.append(f"[{kind}] {child.name}{size}")
    return "\n".join(entries) or "(empty)"


@tool(
    description="Read the contents of a text file in the workspace.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path relative to the workspace."}},
        "required": ["path"],
    },
)
def read_file(path: str) -> str:
    target = _safe_path(path)
    if not target.is_file():
        return f"Error: '{path}' is not a readable file."
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a UTF-8 text file."


@tool(
    description="Write text to a file in the workspace, creating parent directories as needed. Overwrites existing files.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the workspace."},
            "content": {"type": "string", "description": "Text to write."},
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str) -> str:
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}."


# Safe arithmetic evaluator (no eval()): supports + - * / ** % and unary minus.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression.")


@tool(
    description="Evaluate an arithmetic expression exactly, e.g. '2**16 / 7'. Supports + - * / ** %.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "The arithmetic expression."}},
        "required": ["expression"],
    },
)
def calculate(expression: str) -> str:
    try:
        return str(_eval_node(ast.parse(expression, mode="eval").body))
    except Exception as exc:  # noqa: BLE001 - report any parse/eval failure to the model
        return f"Error: could not evaluate '{expression}': {exc}"


_BLOCKED_SHELL = ("rm -rf /", "mkfs", ":(){", "dd if=", "> /dev/")


@tool(
    description="Run a shell command inside the workspace directory and return its output. Disabled unless AGENT_ENABLE_SHELL=1.",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The shell command to run."}},
        "required": ["command"],
    },
)
def run_shell(command: str) -> str:
    if not config.enable_shell:
        return "Error: shell tool is disabled. Set AGENT_ENABLE_SHELL=1 to enable it."
    if any(bad in command for bad in _BLOCKED_SHELL):
        return "Error: command blocked for safety."
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(_workspace_root()),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 60s."
    out = ((result.stdout or "") + (result.stderr or "")).strip()
    if not out:
        return f"(exit {result.returncode}, no output)"
    return f"(exit {result.returncode})\n{out}"


# --- Memory tools (the 5-stage durable store) --------------------------------

@tool(
    description="Save a verified fact to durable memory (stage 3: something you stopped guessing about and confirmed). Persists across sessions.",
    parameters={
        "type": "object",
        "properties": {"fact": {"type": "string", "description": "The confirmed fact, stated plainly."}},
        "required": ["fact"],
    },
)
def remember_fact(fact: str) -> str:
    return memory.load().remember_fact(fact)


@tool(
    description="Save a general rule to durable memory (stage 4: a distilled rule that applies beyond the specific case). Persists across sessions.",
    parameters={
        "type": "object",
        "properties": {"rule": {"type": "string", "description": "The general rule."}},
        "required": ["rule"],
    },
)
def add_rule(rule: str) -> str:
    return memory.load().add_rule(rule)


@tool(
    description="Log an open failure to durable memory (stages 1-2: what failed plus a hypothesis to investigate next session).",
    parameters={
        "type": "object",
        "properties": {"failure": {"type": "string", "description": "What failed and your current hypothesis."}},
        "required": ["failure"],
    },
)
def log_failure(failure: str) -> str:
    return memory.load().log_failure(failure)


@tool(
    description="Distill a lesson to durable memory (stage 4: a post-mortem takeaway worth keeping).",
    parameters={
        "type": "object",
        "properties": {"lesson": {"type": "string", "description": "The lesson learned."}},
        "required": ["lesson"],
    },
)
def distill_lesson(lesson: str) -> str:
    return memory.load().distill_lesson(lesson)


# --- Registry interface ------------------------------------------------------

def openai_schema() -> list[dict]:
    """Return all registered tools in OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in _REGISTRY.values()
    ]


def dispatch(name: str, arguments: str) -> str:
    """Execute a tool by name with JSON-string arguments; never raises."""
    tool_obj = _REGISTRY.get(name)
    if tool_obj is None:
        return f"Error: unknown tool '{name}'."
    try:
        kwargs = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return f"Error: arguments for '{name}' were not valid JSON: {arguments!r}"
    try:
        return str(tool_obj.func(**kwargs))
    except Exception as exc:  # noqa: BLE001 - surface tool errors back to the model
        return f"Error running '{name}': {exc}"
