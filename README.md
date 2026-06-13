# smartai — Novel, a self-improving local agent system

**smartai** is a self-improving agent *system* built around **Novel**, a custom
tool-using model that runs entirely on **open-source models, locally** (Mac
Studio / Apple Silicon). No cloud APIs, no per-token cost, **no Claude or other
hosted model** — every model in the system is an open-source checkpoint you run
on your own hardware over an OpenAI-compatible endpoint (Ollama, MLX, llama.cpp,
LM Studio).

The point isn't to prompt a model and close the tab. It's a system that
**compounds**: every run leaves the next run smarter. An independent verifier
grades each attempt, a durable state file accumulates verified facts and
distilled rules, Skills sharpen with each post-mortem, and scheduled routines
keep the system learning while you sleep. The model is stateless; the system
around it isn't.

## Novel — the agent model

`novel` is the project's own model, defined in a [`Modelfile`](./Modelfile) on
top of a strong local open-source instruct model (`qwen3.5:122b`). It bakes in
Novel's identity, sampling tuned for reliable function/tool calling, and a large
context window so the agent loop (tools + memory + skills + history) has room.

```bash
ollama pull qwen3.5:122b           # the open-source base (you already have it)
ollama create novel -f Modelfile   # or: ./build_model.sh
ollama run novel                   # try it directly
SMARTAI_BASE_MODEL=qwen3:235b ./build_model.sh   # build a max-capability variant
```

## The model fleet → roles

Different roles get the open-source model that fits their job, instead of paying
the heaviest model for everything. Defaults (override any via `.env`):

| Role | Default model | Why it fits |
| ---- | ------------- | ----------- |
| **Orchestrator** (`novel`) | `qwen3.5:122b` | strong instruct + tool calling, multilingual, ~81 GB RAM sweet spot for days-long driving |
| **Heavy / max-capability** | `qwen3:235b` | most capable verified model — reserve for the hardest planning |
| **Reasoner** (hard + fallback) | `deepseek-r1:70b` | strongest pure reasoning; the "delegate-the-hard-part" tier |
| **Grader / verifier** | `deepseek-r1:32b` | fast R1 distill; a sharp *independent* judge, different family from the maker |
| **Worker** (fan-out) | `qwen3.6:35b` | MoE 35B/3B-active, 1M ctx, fast — high-volume cheap work |
| **Coder** | `qwen3-coder:30b` | coding specialist for code-shaped tasks (→ Kimi-Dev 72B) |
| **Long-context** | `llama4:scout` | 10M-context MoE for whole-repo / long-document reads |
| **Vision** | `kimi-vl` | VLM for vision self-checks (served via llama.cpp/ollama) |

Routing (`router.py`) picks by task shape: coding hints → coder, huge-context
hints → long-context, otherwise by complexity (simple → worker, orchestration →
orchestrator, hard → reasoner). `python main.py route "<task>"` shows the
decision without running anything. `python main.py fleet [--probe]` reports
which role models are actually live on the server (and, with `--probe`, a
1-token latency per model — useful for deciding which models are interactive vs
batch-only, like the 405B).

Other models in a typical fleet map to alternates: `mistral-large` (creative /
multilingual drafts), `gemma4:26b` or `phi4:14b` (cheap graders / cheapest
workers), `llama3.3:70b-instruct` (alternate orchestrator). `llama3.1:405b` is
**offline-batch only** — at ~243 GB on 256 GB RAM its first-token latency makes
it impractical for interactive loops.

## The compound stack

Four layers, one feedback loop. Every output flows up to the self-improvement
layer, gets graded and distilled, and is written back to memory — so tomorrow's
run inherits sharpened state.

| Layer | What it is | In this repo |
| ----- | ---------- | ------------ |
| **1 · Primitives** | the model, tools, sandbox, worktrees, sub-agents | `agent.py`, `tools.py`, `worktrees.py`, `subagents.py` |
| **2 · Orchestration** | self-correcting loops, dynamic workflows, routines, tracing | `loop.py`, `workflows.py`, `routines.py`, `trace.py` |
| **3 · Memory** | state file + compounding skills + knowledge base | `memory.py`, `skills.py`, `knowledge.py` |
| **4 · Self-improvement** | verifier, vision-verify, eval loops, reflection | `verifier.py`, `vision.py`, `evals.py`, `reflect.py` |

### Goal loop — self-correcting maker/verifier iteration

A maker agent produces an artifact; an **independent verifier** (a separate
model — `deepseek-r1:32b` by default — that sees only the artifact and rubric,
never the maker's reasoning) grades it; a *not-met* verdict feeds the gap back
and starts the next iteration. A verifier sub-agent beats self-critique.

```bash
python main.py goal "write workspace/fizzbuzz.py and prove it runs" \
    --rubric "file exists; running it prints 1..15 with Fizz/Buzz/FizzBuzz" \
    --max-iterations 4
```

On success the loop distills a lesson and updates the resume pointer; if it gets
stuck it logs an open failure — both to durable memory.

For sharper signal, grade against a **file-based rubric** (Outcomes-style) where
each criterion is scored independently for partial credit and pinpointed gaps:

```bash
python main.py goal "write workspace/fizzbuzz.py and prove it runs" \
    --rubric-file rubrics/fizzbuzz.md
```

`rubric.py` loads criteria from Markdown (one `- ` bullet each) or JSON (with
weights and a `pass_threshold`); the verifier returns a weighted score and the
exact unmet criteria.

### Context compaction — long-run hygiene

`compaction.py` keeps days-long sessions inside the context window: once the
transcript exceeds `CONTEXT_CHAR_BUDGET`, completed earlier turns are summarized
by the cheap worker model while the system prompt and the current turn stay
verbatim. Compaction cuts at a user-message boundary so assistant/tool-call
pairs are never split. On by default in `NovelAgent`.

### Durable memory — the 5-stage state file

`memory.py` maintains `STATE.md` in the workspace, structured around the
progression **fail → investigate → verify → distill → consult**:

```
## Verified facts    (stage 3)   ## General rules     (stage 4)
## Open failures     (stages 1-2) ## Lessons learned   (stage 4)
## Last session      (stage 5 — resume, don't restart)
```

Every session reads it at the start (folded into the system prompt) and writes
to it as it learns, via tools `remember_fact` / `add_rule` / `log_failure` /
`distill_lesson`. `python main.py memory` prints it.

### Skills — procedural memory that compounds

`skills/` holds versioned Markdown Skills (frontmatter + body). The most relevant
ones are folded into the maker's prompt; after a confirmed failure the lesson is
written **into the Skill** (`## Known failure modes`), so it sharpens every run.
See [`skills/ci-triage.md`](./skills/ci-triage.md). `python main.py skills list`.

### Knowledge base — the retrieval layer

`knowledge/` holds versioned reference docs. `knowledge.py` chunks and searches
them (deterministic token-overlap by default; cosine over an open-source
embedding model when `EMBED_MODEL` is set). The agent queries it with the
`search_knowledge` tool to cite facts instead of re-deriving them.
`python main.py kb search "<query>"`.

### Sub-agent delegation

`subagents.py` lets the orchestrator hand a bounded subtask to a fresh agent on
the best-fit model — by explicit `role` (worker/coder/reasoner/long_context/…),
explicit model, or automatic routing — with its own clean context. Exposed as
the `delegate` tool and depth-guarded by `SUBAGENT_MAX_DEPTH`.

### Reflection — meta-distillation

`reflect.py` reads the open failures and lessons accumulated in memory and asks
the reasoner model to generalize them into durable rules and concrete skill
failure-modes, written back to memory/skills. Run it on a schedule so specific
failures become reusable knowledge while you sleep. `python main.py reflect`.

### Tracing

`trace.py` writes an append-only JSONL event stream per run under
`workspace/traces/`, so loops and routines are auditable after the fact (and
give reflection something concrete to learn from).

### Dynamic workflow primitives

`workflows.py` — `fan_out_synthesize` (parallel independent work), `adversarial_verify`
(maker + independent verifier), `loop_until_done`, `classify_and_act` (triage /
model routing), and `tournament` (taste-based pairwise ranking).

### Vision self-check

`vision.py` points an open-source VLM (`kimi-vl`) at a screenshot and grades it
against the goal — the open-source version of "use vision to check outputs
against goals". Same independent-verifier structure as the text grader.

### Eval loops

`evals.py` runs a JSONL suite (e.g. [`evals/ci-triage-cases.jsonl`](./evals/ci-triage-cases.jsonl)),
grades each case with the verifier, and can feed newly-failing cases back into a
Skill and into memory — so the eval suite itself compounds.

```bash
python main.py eval evals/ci-triage-cases.jsonl --skill ci-triage
```

### Worktrees & parallel experiments

`worktrees.py` wraps `git worktree` so parallel agents each get an isolated
checkout that can't collide. `experiments.py` builds on it: `run_experiments`
fans out N approaches (each a delegated sub-agent on a routed model), grades
each with the independent verifier, and keeps the highest scorer;
`run_in_worktrees` runs code-producing variants in isolated checkouts and can
merge the winning branch back. `python main.py experiment "task" --variants 3`.

### Routines — scheduled / triggered runs (laptop-off)

Hosted "Routines" run in the cloud; the open-source workaround uses local
infrastructure two ways:

```bash
cp routines.example.json routines.json
python main.py routine list
python main.py routine run nightly-eval-compounding   # run one now
python main.py routine install-cron                   # install crontab entries
python main.py routine serve                          # built-in scheduler daemon
```

- **cron** — `install-cron` writes crontab entries that invoke the routine.
- **serve** — a built-in scheduler daemon (use under `nohup`/`tmux`/`launchd`)
  for laptop-off operation without touching cron.

Triggers: `schedule` (a 5-field cron expression), `event` (fire when a sentinel
file appears — the local analog of "CI failed → fire a routine"), and `manual`.
Each run executes a goal loop, logs the outcome, and updates memory — so the
system compounds while you sleep.

### Safety policy — the configurable boundary

Open-source models don't force-block, so the Mythos safety boundary becomes a
*policy you control* (`router.py`). Tasks matching a sensitive domain
(cyber/bio/chem/distillation) are, per `SAFETY_POLICY`: routed to the fallback
model (`route`, default), held for human review (`review`), refused (`block`),
or run normally (`allow`). Nothing is silent — `route` annotates every decision.

## Quick start

```bash
chmod +x setup.sh
./setup.sh                 # installs Ollama, pulls the base, builds novel + .venv

source .venv/bin/activate
python main.py             # chat REPL (Novel, loads memory + skills)
python main.py "task"      # one task, routed + guardrailed
python main.py goal "..."  # self-correcting goal loop
python main.py route "..." # show the routing/safety decision

python examples/compounding_demo.py   # live end-to-end demo: watch it compound
```

## Configuration

All settings are environment variables (see [`.env.example`](./.env.example)):
endpoint (`LLM_BASE_URL`/`LLM_API_KEY`), the role models above, generation
(`LLM_TEMPERATURE`, `GRADER_TEMPERATURE=0.0`, `LLM_MAX_TOKENS`), the agent loop
(`AGENT_MAX_STEPS`, `AGENT_ENABLE_SHELL`, `AGENT_WORKSPACE`), the
self-improvement layer (`GOAL_MAX_ITERATIONS`, `AGENT_MEMORY_FILE`,
`AGENT_SKILLS_DIR`, `AGENT_ROUTINES_FILE`), and `SAFETY_POLICY`.

## Project layout

```
Modelfile            # the custom 'novel' model on the open-source qwen3.5 base
build_model.sh       # ollama create novel (with optional base override)
config.py            # env config + model-fleet routing (orchestrator/worker/grader/…)
router.py            # task→model routing + configurable safety guardrail
tools.py             # sandboxed tool registry (memory, knowledge, delegate, …)
agent.py             # NovelAgent — OpenAI-compatible tool-calling loop
subagents.py         # sub-agent delegation (role-routed, depth-guarded)
compaction.py        # context compaction for long sessions
rubric.py + rubrics/ # file-based gradable criteria (Outcomes-style)
fleet.py             # which role models are live on the server
memory.py            # durable 5-stage state file (STATE.md)
skills.py + skills/  # procedural memory that compounds
knowledge.py + knowledge/  # retrieval layer over versioned reference docs
verifier.py          # independent text grader sub-agent
vision.py            # independent vision (VLM) grader
loop.py              # goal loop — maker → verifier → memory
reflect.py           # meta-distillation: failures → durable rules
trace.py             # append-only JSONL run tracing
evals.py + evals/    # eval-suite runner that feeds failures back in
workflows.py         # dynamic workflow primitives (5 patterns)
worktrees.py         # git-worktree helpers for parallel safety
experiments.py       # parallel approaches (fan-out + delegate + verify), keep winner
routines.py          # scheduled/triggered runs (cron + built-in daemon)
main.py              # CLI (chat, goal, route, fleet, memory, skills, kb, reflect, eval, routine)
setup.sh             # one-time install + model build
examples/            # live end-to-end demo (needs a running server)
tests/               # offline tests (no model/GPU needed)
```

## Tests

The whole system — tools, memory, verifier, goal loop, workflows, router,
skills, evals, routines (cron matcher, registry, scheduler), and worktrees — is
covered by offline tests. A fake OpenAI-compatible client drives every model
call against scripted responses, so nothing needs a server:

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```
