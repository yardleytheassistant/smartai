"""Interactive CLI for Novel, smartai's self-improving local agent.

Usage:
    python main.py                         # start a chat REPL
    python main.py "your task"             # run a single task and exit
    python main.py goal "your goal"        # run a self-correcting maker/verifier loop
        [--rubric "criteria"] [--max-iterations N]
    python main.py memory                  # print the durable state file
"""

import sys

from rich.console import Console
from rich.panel import Panel

import memory as memory_mod
from agent import NovelAgent
from config import config
from loop import goal_loop

console = Console()


def _show_tool(name: str, arguments: str, result: str) -> None:
    preview = result if len(result) <= 500 else result[:500] + " …"
    console.print(
        Panel(
            f"[bold]args[/bold] {arguments}\n[bold]result[/bold] {preview}",
            title=f"🛠  {name}",
            border_style="cyan",
            expand=False,
        )
    )


def _banner() -> None:
    routing = config.model
    if config.worker_model != config.model or config.grader_model != config.model:
        routing = f"orchestrator [cyan]{config.model}[/cyan] · worker [cyan]{config.worker_model}[/cyan] · grader [cyan]{config.grader_model}[/cyan]"
    console.print(
        Panel(
            f"[bold]Novel[/bold] — smartai self-improving agent\n"
            f"model    [cyan]{routing}[/cyan]\n"
            f"endpoint [cyan]{config.base_url}[/cyan]\n"
            f"shell    [cyan]{'enabled' if config.enable_shell else 'disabled'}[/cyan]\n"
            f"workspace[cyan] {config.workspace}[/cyan]",
            border_style="green",
            expand=False,
        )
    )


def run_once(task: str) -> None:
    agent = NovelAgent(load_memory=True)
    answer = agent.run(task, on_tool=_show_tool)
    console.print(Panel(answer, title="answer", border_style="green", expand=False))


def run_goal(task: str, rubric: str | None, max_iterations: int | None) -> None:
    def on_event(kind: str, data: dict) -> None:
        if kind == "iteration":
            console.print(f"[dim]→ iteration {data['n']}/{data['max']}…[/dim]")
        elif kind == "verdict":
            mark = "[green]met[/green]" if data["met"] else "[yellow]not met[/yellow]"
            console.print(f"  verifier: {mark} (score {data['score']:.2f})")
            if not data["met"] and data["feedback"]:
                console.print(f"  [dim]feedback:[/dim] {data['feedback']}")

    result = goal_loop(task, rubric=rubric, max_iterations=max_iterations, on_event=on_event)
    title = "goal met ✓" if result.met else "goal not met ✗"
    style = "green" if result.met else "red"
    console.print(
        Panel(result.output, title=f"{title}  ({result.iterations} iter)", border_style=style, expand=False)
    )


def show_memory() -> None:
    mem = memory_mod.load()
    if mem.is_empty():
        console.print("[dim]No durable memory yet.[/dim]")
        return
    console.print(Panel(mem.render(), title="durable memory", border_style="cyan", expand=False))


def repl() -> None:
    _banner()
    console.print("Type your message. [dim]Ctrl-C or 'exit' to quit.[/dim]\n")
    agent = NovelAgent(load_memory=True)
    while True:
        try:
            user_input = console.input("[bold blue]you ›[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        if user_input.lower() in {"exit", "quit"}:
            console.print("[dim]bye[/dim]")
            return
        if not user_input:
            continue
        answer = agent.run(user_input, on_tool=_show_tool)
        console.print(Panel(answer, title="novel", border_style="green", expand=False))


def _parse_goal_args(args: list[str]) -> tuple[str, str | None, int | None]:
    task_parts: list[str] = []
    rubric: str | None = None
    max_iterations: int | None = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--rubric" and i + 1 < len(args):
            rubric = args[i + 1]
            i += 2
        elif arg == "--max-iterations" and i + 1 < len(args):
            max_iterations = int(args[i + 1])
            i += 2
        else:
            task_parts.append(arg)
            i += 1
    return " ".join(task_parts), rubric, max_iterations


def main() -> None:
    args = sys.argv[1:]
    if not args:
        repl()
        return
    if args[0] == "goal":
        task, rubric, max_iterations = _parse_goal_args(args[1:])
        if not task:
            console.print("[red]usage:[/red] python main.py goal \"your goal\" [--rubric ...] [--max-iterations N]")
            return
        run_goal(task, rubric, max_iterations)
        return
    if args[0] == "memory":
        show_memory()
        return
    run_once(" ".join(args))


if __name__ == "__main__":
    main()
