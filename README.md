# smartai — Novel, a self-improving local agent

**smartai** is a self-improving agent *system* built around **Novel**, a custom
tool-using model that runs entirely on **open-source models, locally** (Mac
Studio / Apple Silicon). No cloud APIs, no per-token cost, **no Claude or other
hosted model** — every model in the system is an open-source checkpoint you run
on your own hardware.

The point isn't to prompt a model and close the tab. It's a system that
**compounds**: every run leaves the next run smarter. An independent verifier
grades each attempt, a durable state file accumulates verified facts and
distilled rules, and a goal loop self-corrects until the work actually meets its
criteria. The model is stateless; the system around it isn't.

## Novel — the agent model

The project ships its own model, **`novel`**, defined in a [`Modelfile`](./Modelfile)
on top of an open-source Nous Hermes checkpoint. Building on Hermes inherits a
base already tuned for reliable function/tool calling; Novel layers on its
identity, sampling defaults, and a roomy context window — rather than training
from scratch.

- **`FROM hermes3:70b`** — the open-source base, pulled by Ollama.
- **`SYSTEM`** — Novel's identity, so even bare `ollama run novel` behaves like
  the agent.
- **`PARAMETER`s** — sampling tuned for steady, reproducible tool calling, an 8K
  `num_ctx` so multi-step tool + memory transcripts fit, and ChatML `stop` tokens.

```bash
ollama pull hermes3:70b            # the open-source base
ollama create novel -f Modelfile   # or: ./build_model.sh
ollama run novel                   # try it directly
SMARTAI_BASE_MODEL=hermes3:8b ./build_model.sh   # build on a smaller open base
```

It talks to any **OpenAI-compatible** local server, so the same code runs on
Ollama, MLX, llama.cpp, or LM Studio by changing one environment variable.

## The compound stack

Four layers, one feedback loop. Every output flows up to the self-improvement
layer, gets graded and distilled, and is written back to memory — so tomorrow's
run inherits sharpened state. All of it runs on open-source models.

| Layer | What it is | In this repo |
| ----- | ---------- | ------------ |
| **1 · Primitives** | the model, tools, sandbox | `agent.py`, `tools.py` |
| **2 · Orchestration** | self-correcting loops + dynamic workflows | `loop.py`, `workflows.py` |
| **3 · Memory** | a durable state file that survives sessions | `memory.py`, `STATE.md` |
| **4 · Self-improvement** | independent verifier, rule distillation | `verifier.py` |

### Goal loop — self-correcting maker/verifier iteration

A maker agent produces an artifact; an **independent verifier** (a separate
model instance that sees only the artifact and rubric, never the maker's
reasoning) grades it; a *not-met* verdict feeds the gap back and starts the next
iteration. The loop exits when the grader passes or `max_iterations` is hit. A
verifier sub-agent beats self-critique — a model grading its own work prefers
conclusions consistent with what it already wrote.

```bash
python main.py goal "write workspace/fizzbuzz.py and prove it runs" \
    --rubric "file exists; running it prints 1..15 with Fizz/Buzz/FizzBuzz" \
    --max-iterations 4
```

On success the loop distills a lesson and updates the resume pointer; if it gets
stuck, it logs an open failure — both to durable memory.

### Durable memory — the 5-stage state file

`memory.py` maintains `STATE.md` in the workspace, structured around the
progression **fail → investigate → verify → distill → consult**:

```
## Verified facts    (stage 3) — things we stopped guessing about
## General rules     (stage 4) — distilled rules that apply beyond one case
## Open failures     (stages 1-2) — failures + hypotheses to investigate next
## Lessons learned   (stage 4) — distilled post-mortems
## Last session      (stage 5) — resume, don't restart
```

Every session reads it at the start (folded into the system prompt) and writes
to it as it learns. The agent has tools for each stage — `remember_fact`,
`add_rule`, `log_failure`, `distill_lesson` — and `python main.py memory` prints
the current state.

### Dynamic workflow primitives

`workflows.py` provides composable, model-agnostic orchestration:

- **`fan_out_synthesize`** — split work into N independent pieces, run each in
  its own clean context (in parallel), then synthesize.
- **`adversarial_verify`** — pair a maker with an independent verifier; iterate
  until it passes.
- **`loop_until_done`** — keep stepping until a stop condition is met.

### Model routing

Different roles get different open-source models (`config.py`): an
**orchestrator** for heavy planning, a **worker** for bounded fan-out, an
independent **grader** for verification. By default all three point at `novel`
so the system runs out of the box; route any role to a smaller open model (e.g.
a small Hermes/Qwen grader) via `WORKER_MODEL` / `GRADER_MODEL`.

## Quick start (Ollama)

```bash
chmod +x setup.sh
./setup.sh                 # installs Ollama, pulls Hermes, builds the novel model + .venv

source .venv/bin/activate
python main.py             # interactive chat REPL (uses Novel, loads memory)
python main.py goal "..."  # run a self-correcting goal loop
python main.py memory      # print the durable state file
```

## Model sizing on 256 GB RAM

| Model            | Quant     | Approx. RAM | Notes                              |
| ---------------- | --------- | ----------- | ---------------------------------- |
| Hermes 70B       | Q4_K_M    | ~42 GB      | Fast, great default                |
| Hermes 70B       | fp16      | ~140 GB     | Highest quality at 70B             |
| Hermes 405B      | 4-bit     | ~200–230 GB | Feasible on 256 GB; slower         |

`novel` is built from `hermes3:70b` by default. Point it at a different base
with `SMARTAI_BASE_MODEL` when building, or switch the agent's model with
`LLM_MODEL` in `.env`.

## Using MLX instead

MLX serves an OpenAI-compatible endpoint too, so only `.env` changes:

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Hermes-3-Llama-3.1-70B-4bit --port 8080
```

```
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=mlx
LLM_MODEL=mlx-community/Hermes-3-Llama-3.1-70B-4bit
```

## Built-in tools

| Tool             | What it does                                              |
| ---------------- | -------------------------------------------------------- |
| `get_current_time` | Current local date/time                                |
| `list_directory` | List files in the workspace                              |
| `read_file`      | Read a text file in the workspace                        |
| `write_file`     | Write/overwrite a text file in the workspace            |
| `calculate`      | Exact arithmetic (safe — no `eval`)                     |
| `run_shell`      | Run a shell command (disabled by default)               |
| `remember_fact`  | Save a verified fact to durable memory (stage 3)        |
| `add_rule`       | Save a general rule to durable memory (stage 4)         |
| `log_failure`    | Log an open failure to durable memory (stages 1-2)      |
| `distill_lesson` | Distill a lesson to durable memory (stage 4)            |

**Sandboxing:** file and shell tools are confined to `AGENT_WORKSPACE`
(default `./workspace`). `run_shell` is **off by default** — set
`AGENT_ENABLE_SHELL=1` in `.env` to enable it; it refuses obviously destructive
commands. Add your own tool by dropping a `@tool`-decorated function in
`tools.py`; it's automatically advertised to the model and dispatched by name.

## Configuration

All settings are environment variables (see `.env.example`):

| Variable             | Default                      | Purpose                              |
| -------------------- | ---------------------------- | ------------------------------------ |
| `LLM_BASE_URL`       | `http://localhost:11434/v1`  | OpenAI-compatible endpoint           |
| `LLM_MODEL`          | `novel`                      | Primary / orchestrator model         |
| `WORKER_MODEL`       | (= `LLM_MODEL`)              | Model for fan-out worker tasks       |
| `GRADER_MODEL`       | (= `LLM_MODEL`)              | Model for the independent verifier   |
| `LLM_TEMPERATURE`    | `0.7`                        | Sampling temperature                 |
| `GRADER_TEMPERATURE` | `0.0`                        | Verifier temperature (reproducible)  |
| `LLM_MAX_TOKENS`     | `4096`                       | Max tokens per response              |
| `AGENT_MAX_STEPS`    | `12`                         | Max tool-calling steps per turn      |
| `GOAL_MAX_ITERATIONS`| `4`                          | Max maker→verifier iterations        |
| `AGENT_MEMORY_FILE`  | `STATE.md`                   | Durable state file (in workspace)    |
| `AGENT_ENABLE_SHELL` | `0`                          | Allow the shell tool                 |
| `AGENT_WORKSPACE`    | `./workspace`                | File/shell sandbox directory         |

## Project layout

```
Modelfile        # defines the custom 'novel' model on the open-source Hermes base
build_model.sh   # ollama create novel (with optional base override)
config.py        # env-driven config + model routing (orchestrator/worker/grader)
tools.py         # tool definitions + registry (sandboxed) incl. memory tools
agent.py         # NovelAgent — the OpenAI-compatible tool-calling loop
memory.py        # durable 5-stage state file (STATE.md)
verifier.py      # independent grader sub-agent
loop.py          # goal loop — maker -> verifier -> memory
workflows.py     # dynamic workflow primitives (fan-out, adversarial, loop-until)
main.py          # CLI / REPL (chat, goal, memory)
setup.sh         # one-time install: Ollama + base pull + model build + Python env
tests/           # offline tests (no model/GPU needed)
```

## Tests

The whole system — tool registry, sandbox, memory, the verifier, the goal loop,
and the workflow primitives — is covered by offline tests that need **no model
server**. A fake OpenAI-compatible client drives the orchestration against
scripted responses:

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```
