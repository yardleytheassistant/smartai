"""Interactive CLI for Novel, smartai's self-improving local agent.

Usage:
    python main.py                         # start a chat REPL
    python main.py --resume NAME           # resume a saved REPL session
    python main.py "your task"             # run a single task (routed + guardrailed)
    python main.py goal "your goal"        # run a self-correcting maker/verifier loop
        [--rubric "criteria" | --rubric-file PATH] [--max-iterations N]
    python main.py route "your task"       # show the routing/safety decision only
    python main.py fleet [--probe]         # which role models are live (+ latency)
    python main.py status                  # print the resolved configuration
    python main.py doctor                  # operational self-check (exits non-zero if unhealthy)
    python main.py bench --models a,b,c    # benchmark models to validate role assignments [--runs N]
    python main.py perf                    # run the goal-loop battery (full maker/verifier loop)
    python main.py experiment "task" [--variants N] [--context-file PATH] [--land | --land-worktree [--merge]] [--maker MODEL] [--check "CMD"]  # explore, keep best, optionally land it
    python main.py memory                  # print the durable state file
    python main.py skills [list|show NAME] # inspect procedural-memory skills
    python main.py kb [search Q|add NAME C]# query/extend the knowledge base
    python main.py reflect                 # distill rules from accumulated failures
    python main.py eval CASES.jsonl        # run an eval suite [--skill NAME to compound]
    python main.py routine [list|run NAME|serve|install-cron]   # scheduled/triggered runs
"""

import sys

from rich.console import Console
from rich.panel import Panel

import memory as memory_mod
import router
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
    decision = router.decide(task)
    if decision.action in {"block", "review"}:
        console.print(
            Panel(
                f"[bold]{decision.action.upper()}[/bold] — {decision.note}\n"
                f"Set SAFETY_POLICY=route or allow in .env to change this.",
                title="safety guardrail", border_style="red", expand=False,
            )
        )
        return
    if decision.domain:
        console.print(f"[yellow]guardrail:[/yellow] {decision.note}")
    agent = NovelAgent(model=decision.model, load_memory=True, skills_query=task)
    answer = agent.run(task, on_tool=_show_tool)
    console.print(Panel(answer, title=f"answer · {decision.model}", border_style="green", expand=False))


def show_route(task: str) -> None:
    d = router.decide(task)
    console.print(
        Panel(
            f"model      [cyan]{d.model}[/cyan]\n"
            f"complexity [cyan]{d.complexity}[/cyan]\n"
            f"domain     [cyan]{d.domain or '—'}[/cyan]\n"
            f"action     [cyan]{d.action}[/cyan]\n"
            f"{d.note}",
            title="routing decision", border_style="cyan", expand=False,
        )
    )


def run_goal(task: str, rubric, max_iterations: int | None) -> None:
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


def cmd_routine(args: list[str]) -> None:
    import routines as routines_mod

    sub = args[0] if args else "list"
    if sub == "list":
        rs = routines_mod.load_routines()
        if not rs:
            console.print("[dim]No routines defined. Add one to routines.json.[/dim]")
            return
        for r in rs:
            trig = r.schedule if r.trigger == "schedule" else (r.watch_path or r.trigger)
            state = "on" if r.enabled else "off"
            console.print(f"[bold]{r.name}[/bold] ([{state}]) · {r.trigger} {trig} → {r.goal}")
    elif sub == "run" and len(args) > 1:
        def on_event(kind, data):
            if kind == "verdict":
                console.print(f"  [dim]iter {data['n']}: {'met' if data['met'] else 'not met'}[/dim]")
        result = routines_mod.run_by_name(args[1], on_event=on_event)
        console.print(f"[green]done[/green]: {'met' if result.met else 'unmet'} in {result.iterations} iter")
    elif sub == "serve":
        console.print("[green]scheduler running[/green] (Ctrl-C to stop). Use under nohup/tmux for laptop-off runs.")
        try:
            routines_mod.serve()
        except KeyboardInterrupt:
            console.print("\n[dim]stopped[/dim]")
    elif sub == "install-cron":
        text = routines_mod.install_cron()
        console.print(Panel(text, title="installed crontab", border_style="green", expand=False))
    else:
        console.print("[red]usage:[/red] routine [list | run <name> | serve | install-cron]")


def cmd_skills(args: list[str]) -> None:
    import skills as skills_mod

    sub = args[0] if args else "list"
    if sub == "list":
        sk = skills_mod.load_skills()
        if not sk:
            console.print(f"[dim]No skills in {skills_mod.skills_root()}.[/dim]")
            return
        for s in sk:
            console.print(f"[bold]{s.name}[/bold] — {s.description}")
    elif sub == "show" and len(args) > 1:
        s = skills_mod.get(args[1])
        if s is None:
            console.print(f"[red]no skill named {args[1]!r}[/red]")
            return
        console.print(Panel(s.render(), title=f"skill · {s.name}", border_style="cyan", expand=False))
    else:
        console.print("[red]usage:[/red] skills [list | show <name>]")


def cmd_eval(args: list[str]) -> None:
    import evals as evals_mod

    if not args:
        console.print("[red]usage:[/red] eval <cases.jsonl> [--skill name]")
        return
    path = args[0]
    skill = args[args.index("--skill") + 1] if "--skill" in args else None
    cases = evals_mod.load_cases(path)

    def run_fn(case):
        return NovelAgent(load_memory=True, skills_query=case.input).run(case.input)

    def on_result(r):
        mark = "[green]✓[/green]" if r.verdict.met else "[red]✗[/red]"
        console.print(f"  {mark} {r.case.id} (score {r.verdict.score:.2f})")

    report = evals_mod.run_evals(
        cases, run_fn, on_result=on_result, compound_into_skill=skill, record_memory=True
    )
    console.print(Panel(report.summary(), title="eval report", border_style="green", expand=False))


def cmd_kb(args: list[str]) -> None:
    import knowledge as kb

    sub = args[0] if args else "search"
    if sub == "search" and len(args) > 1:
        console.print(Panel(kb.search_text(" ".join(args[1:])), title="knowledge", border_style="cyan", expand=False))
    elif sub == "add" and len(args) > 2:
        path = kb.add_document(args[1], " ".join(args[2:]))
        console.print(f"[green]added[/green] {path}")
    else:
        console.print("[red]usage:[/red] kb [search <query> | add <name> <content>]")


def cmd_fleet(args: list[str]) -> None:
    import fleet as fleet_mod

    probe = "--probe" in args
    try:
        statuses = fleet_mod.check(probe=probe)
    except Exception as exc:  # noqa: BLE001 - server may be down
        console.print(f"[red]could not reach the model server:[/red] {exc}")
        return
    lines = []
    for s in statuses:
        mark = "[green]✓ up[/green]" if s.available else "[red]✗ missing[/red]"
        lat = f"  [dim]{s.latency_s:.2f}s[/dim]" if s.latency_s is not None else ""
        lines.append(f"{mark}  [bold]{s.role:<13}[/bold] {s.model}{lat}")
    console.print(Panel("\n".join(lines), title="fleet", border_style="cyan", expand=False))


def cmd_experiment(args: list[str]) -> None:
    import experiments as exp_mod

    n = 3
    task_parts: list[str] = []
    rubric = None
    context_files: list[str] = []
    land = False
    land_worktree = False
    merge = False
    maker_model = None
    check_cmd = ""
    i = 0
    while i < len(args):
        if args[i] == "--variants" and i + 1 < len(args):
            n = int(args[i + 1])
            i += 2
        elif args[i] == "--rubric" and i + 1 < len(args):
            rubric = args[i + 1]
            i += 2
        elif args[i] == "--context-file" and i + 1 < len(args):
            context_files.append(args[i + 1])
            i += 2
        elif args[i] == "--maker" and i + 1 < len(args):
            maker_model = args[i + 1]
            i += 2
        elif args[i] == "--check" and i + 1 < len(args):
            check_cmd = args[i + 1]
            i += 2
        elif args[i] == "--land":
            land = True
            i += 1
        elif args[i] == "--land-worktree":
            land_worktree = True
            i += 1
        elif args[i] == "--merge":
            merge = True
            i += 1
        else:
            task_parts.append(args[i])
            i += 1
    task = " ".join(task_parts)
    if not task:
        console.print("[red]usage:[/red] experiment \"task\" [--variants N] [--rubric ...] [--context-file PATH ...] [--land | --land-worktree [--merge]]")
        return
    context = exp_mod.build_context(context_files) if context_files else ""
    grounding = f", grounded in {len(context_files)} source file(s)" if context_files else ""
    console.print(f"[dim]running {n} approaches in parallel{grounding}…[/dim]")
    report = exp_mod.run_experiments(task, n, rubric=rubric, context=context)
    ranked = report.ranked()
    body = "\n".join(
        f"{'★' if e is report.winner else ' '} score {e.score:.2f} "
        f"{'met' if e.met else 'unmet':<5} — {e.variant}"
        for e in ranked
    )
    console.print(Panel(body, title="experiments", border_style="green", expand=False))
    if report.winner:
        console.print(Panel(report.winner.output, title="winner", border_style="green", expand=False))
        if not (land or land_worktree):
            console.print(
                "[dim]experiment = grounded design exploration: a winning direction that fits "
                "the code. Re-run with --land (writes a workspace artifact) or "
                "--land-worktree (edits real repo files on a review branch).[/dim]"
            )

    if land and report.winner:
        if not report.winner.met:
            console.print(
                "[red]winner did not meet the rubric[/red] — refusing to land a bad seed. "
                "Tighten the approach/rubric, or use run_in_worktrees."
            )
            return
        console.print("[dim]landing the winning direction via a goal loop…[/dim]")

        def on_event(kind: str, data: dict) -> None:
            if kind == "verdict":
                mark = "[green]met[/green]" if data["met"] else "[yellow]not met[/yellow]"
                console.print(f"  [dim]iter {data['n']}:[/dim] {mark} (score {data['score']:.2f})")
                if not data["met"] and data["feedback"]:
                    console.print(f"  [dim]feedback:[/dim] {data['feedback']}")

        result = exp_mod.land_winner(
            task, report.winner, rubric=rubric, context=context,
            maker_model=maker_model, check_cmd=check_cmd, on_event=on_event,
        )
        # 'landed' requires real files on disk, not just the verifier's verdict.
        if result.landed:
            title, style = "landed ✓", "green"
            footer = "wrote: " + ", ".join(result.files_written)
        elif result.met:
            title, style = "verifier approved but NO files written ⚠ — not landed", "yellow"
            footer = "the maker described the change instead of writing it (workspace unchanged)"
        else:
            title, style = "land attempt ✗", "red"
            footer = "verifier did not pass"
        console.print(
            Panel(f"{result.output}\n\n[dim]{footer}[/dim]",
                  title=f"{title}  ({result.iterations} iter)", border_style=style, expand=False)
        )

    if land_worktree and report.winner:
        if not report.winner.met:
            console.print(
                "[red]winner did not meet the rubric[/red] — refusing to land a bad seed. "
                "Tighten the approach/rubric first."
            )
            return

        def on_event(kind: str, data: dict) -> None:
            if kind == "verdict":
                mark = "[green]met[/green]" if data["met"] else "[yellow]not met[/yellow]"
                console.print(f"  [dim]iter {data['n']}:[/dim] {mark} (score {data['score']:.2f})")
                if not data["met"] and data["feedback"]:
                    console.print(f"  [dim]feedback:[/dim] {data['feedback']}")

        console.print("[dim]landing as a real repo edit in an isolated worktree…[/dim]")
        try:
            result = exp_mod.land_in_worktree(
                task, report.winner, rubric=rubric, context=context,
                maker_model=maker_model, check_cmd=check_cmd, merge=merge, on_event=on_event,
            )
        except Exception as exc:  # noqa: BLE001 - git/worktree failures shouldn't crash the CLI
            console.print(f"[red]could not land in a worktree:[/red] {exc}")
            return
        # Git is the source of truth: 'landed' = verifier met AND a real commit.
        if result.landed:
            title, style = "landed ✓", "green"
            footer = (
                f"committed to branch [bold]{result.branch}[/bold] "
                f"({len(result.files_changed)} file(s)); "
                + ("merged into the current branch." if result.merged
                   else f"review with: git diff ...{result.branch}  ·  merge with: git merge {result.branch}")
            )
        elif result.committed:
            title, style = "changed files but verifier did NOT pass ⚠ — not landed", "yellow"
            footer = f"left on branch [bold]{result.branch}[/bold] for inspection; not merged"
        else:
            title, style = "no repo edit made ✗", "red"
            footer = "the maker did not edit any file (nothing committed)"
        detail = (result.diffstat + "\n\n" if result.diffstat else "") + f"[dim]{footer}[/dim]"
        console.print(
            Panel(detail, title=f"{title}  ({result.iterations} iter)", border_style=style, expand=False)
        )


def cmd_status(_args: list[str]) -> None:
    body = (
        f"endpoint     {config.base_url}\n"
        f"orchestrator {config.model}\n"
        f"worker       {config.worker_model}\n"
        f"grader       {config.grader_model}\n"
        f"reasoner     {config.reasoner_model}\n"
        f"coder        {config.coder_model}\n"
        f"long_context {config.long_context_model}\n"
        f"heavy        {config.heavy_model}\n"
        f"vision       {config.vision_model}\n"
        f"fallback     {config.fallback_model}\n"
        f"safety       {config.safety_policy}\n"
        f"max_iter     {config.max_iterations}   ctx_budget {config.context_char_budget}\n"
        f"workspace    {config.workspace}"
    )
    console.print(Panel(body, title="status", border_style="cyan", expand=False))


def cmd_doctor(_args: list[str]) -> None:
    import doctor as doctor_mod

    checks = doctor_mod.run()
    lines = []
    for c in checks:
        mark = "[green]✓[/green]" if c.ok else ("[red]✗[/red]" if c.critical else "[yellow]–[/yellow]")
        lines.append(f"{mark} {c.name}  [dim]{c.detail}[/dim]")
    healthy = doctor_mod.ok(checks)
    border = "green" if healthy else "red"
    console.print(Panel("\n".join(lines), title="doctor", border_style=border, expand=False))
    if not healthy:
        sys.exit(1)


def cmd_bench(args: list[str]) -> None:
    import bench as bench_mod

    models = None
    tasks_path = None
    runs = 1
    i = 0
    while i < len(args):
        if args[i] == "--models" and i + 1 < len(args):
            models = [m.strip() for m in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--tasks" and i + 1 < len(args):
            tasks_path = args[i + 1]
            i += 2
        elif args[i] == "--runs" and i + 1 < len(args):
            runs = max(1, int(args[i + 1]))
            i += 2
        else:
            i += 1
    if not models:
        console.print("[red]usage:[/red] bench --models a,b,c [--tasks cases.jsonl] [--runs N]")
        return
    tasks = bench_mod.load_cases(tasks_path) if tasks_path else bench_mod.load_cases("bench/tasks.jsonl")
    suffix = f" ×{runs} runs each" if runs > 1 else ""
    console.print(f"[dim]benchmarking {len(models)} models over {len(tasks)} tasks{suffix}…[/dim]")
    report = bench_mod.run_bench(models, tasks, runs=runs)
    console.print(Panel(report.scorecard(), title="benchmark", border_style="green", expand=False))
    roles = report.suggested_roles()
    if roles:
        console.print("suggested: " + "  ".join(f"{r}={m}" for r, m in roles.items()))
        if not report.grader_is_independent(roles):
            console.print(
                "[yellow]warning:[/yellow] suggested grader shares the orchestrator's "
                "model family — no independent grader was in this set. Add a "
                "different-family model so verification stays a real cross-check."
            )


def cmd_perf(args: list[str]) -> None:
    import tempfile

    import perf as perf_mod
    from config import config

    tasks_path = "bench/battery.jsonl"
    max_iterations = None
    use_memory = True
    i = 0
    while i < len(args):
        if args[i] == "--tasks" and i + 1 < len(args):
            tasks_path = args[i + 1]
            i += 2
        elif args[i] == "--max-iterations" and i + 1 < len(args):
            max_iterations = int(args[i + 1])
            i += 2
        elif args[i] == "--no-memory":
            use_memory = False
            i += 1
        else:
            i += 1

    cases = perf_mod.load_cases(tasks_path)
    console.print(f"[dim]goal-loop battery: {len(cases)} tasks…[/dim]")

    def on_event(kind: str, data: dict) -> None:
        if kind == "verdict":
            mark = "[green]met[/green]" if data["met"] else "[yellow]not met[/yellow]"
            console.print(f"  [dim]iter {data['n']}:[/dim] {mark} (score {data['score']:.2f})")

    # Run against an isolated workspace so the battery never writes the real STATE.md.
    original = config.workspace
    tmp = tempfile.mkdtemp(prefix="smartai-perf-")
    try:
        config.workspace = tmp
        report = perf_mod.run_loop_battery(
            cases, max_iterations=max_iterations, use_memory=use_memory, on_event=on_event
        )
    finally:
        config.workspace = original

    border = "green" if report.all_passed else "red"
    console.print(Panel(report.scorecard(), title="goal-loop battery", border_style=border, expand=False))
    console.print(
        "[dim]note: this is the goal-loop battery only. "
        "Run `python main.py bench --models a,b,c` (or `make perf`) for the model scorecard.[/dim]"
    )
    if not report.all_passed:
        sys.exit(1)


def cmd_reflect(_args: list[str]) -> None:
    import reflect as reflect_mod

    result = reflect_mod.reflect()
    body = (
        f"note: {result['note']}\n"
        f"rules added: {len(result['rules'])}\n"
        + "\n".join(f"  • {r}" for r in result["rules"])
        + (f"\nfailure modes: {len(result['failure_modes'])}" if result["failure_modes"] else "")
    )
    console.print(Panel(body, title="reflection", border_style="green", expand=False))


def repl(resume: str | None = None) -> None:
    import sessions

    _banner()
    console.print(
        "Type your message. [dim]Ctrl-C or 'exit' to quit; "
        "/save NAME and /load NAME to persist sessions.[/dim]\n"
    )
    if resume and sessions.exists(resume):
        agent = NovelAgent(messages=sessions.load(resume))
        console.print(f"[dim]resumed session '{resume}'[/dim]")
    else:
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
        if user_input.startswith("/save "):
            path = sessions.save(agent.messages, user_input[6:].strip())
            console.print(f"[dim]saved → {path}[/dim]")
            continue
        if user_input.startswith("/load "):
            name = user_input[6:].strip()
            try:
                agent = NovelAgent(messages=sessions.load(name))
                console.print(f"[dim]loaded session '{name}'[/dim]")
            except FileNotFoundError as exc:
                console.print(f"[red]{exc}[/red]")
            continue
        answer = agent.run(user_input, on_tool=_show_tool)
        console.print(Panel(answer, title="novel", border_style="green", expand=False))


def _parse_goal_args(args: list[str]):
    task_parts: list[str] = []
    rubric = None
    max_iterations: int | None = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--rubric" and i + 1 < len(args):
            rubric = args[i + 1]
            i += 2
        elif arg == "--rubric-file" and i + 1 < len(args):
            from rubric import Rubric

            rubric = Rubric.from_file(args[i + 1])
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
    if args[0] == "--resume":
        repl(resume=args[1] if len(args) > 1 else None)
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
    if args[0] == "route":
        show_route(" ".join(args[1:]))
        return
    if args[0] == "routine":
        cmd_routine(args[1:])
        return
    if args[0] == "skills":
        cmd_skills(args[1:])
        return
    if args[0] == "eval":
        cmd_eval(args[1:])
        return
    if args[0] == "kb":
        cmd_kb(args[1:])
        return
    if args[0] == "reflect":
        cmd_reflect(args[1:])
        return
    if args[0] == "fleet":
        cmd_fleet(args[1:])
        return
    if args[0] == "experiment":
        cmd_experiment(args[1:])
        return
    if args[0] == "status":
        cmd_status(args[1:])
        return
    if args[0] == "doctor":
        cmd_doctor(args[1:])
        return
    if args[0] == "bench":
        cmd_bench(args[1:])
        return
    if args[0] == "perf":
        cmd_perf(args[1:])
        return
    run_once(" ".join(args))


if __name__ == "__main__":
    main()
