"""Interactive CLI for the Hermes agent.

Usage:
    python main.py                # start a chat REPL
    python main.py "your task"    # run a single task and exit
"""

import sys

from rich.console import Console
from rich.panel import Panel

from agent import HermesAgent
from config import config

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
    console.print(
        Panel(
            f"[bold]Hermes agent[/bold]\n"
            f"model    [cyan]{config.model}[/cyan]\n"
            f"endpoint [cyan]{config.base_url}[/cyan]\n"
            f"shell    [cyan]{'enabled' if config.enable_shell else 'disabled'}[/cyan]\n"
            f"workspace[cyan] {config.workspace}[/cyan]",
            border_style="green",
            expand=False,
        )
    )


def run_once(task: str) -> None:
    agent = HermesAgent()
    answer = agent.run(task, on_tool=_show_tool)
    console.print(Panel(answer, title="answer", border_style="green", expand=False))


def repl() -> None:
    _banner()
    console.print("Type your message. [dim]Ctrl-C or 'exit' to quit.[/dim]\n")
    agent = HermesAgent()
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
        console.print(Panel(answer, title="hermes", border_style="green", expand=False))


def main() -> None:
    if len(sys.argv) > 1:
        run_once(" ".join(sys.argv[1:]))
    else:
        repl()


if __name__ == "__main__":
    main()
