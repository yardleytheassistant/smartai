# CLAUDE.md — guide for agents working in this repo

**smartai** is a self-improving agent system named **Novel**, running entirely on
local **open-source** models over an OpenAI-compatible endpoint (Ollama/MLX/
llama.cpp). There is **no Claude/Anthropic or other hosted-model dependency** —
do not add one. The `novel` model is built from `qwen3.5:122b` via the
`Modelfile`.

## Commands

```bash
make install-dev     # deps (or: pip install -r requirements-dev.txt)
make test            # python -m pytest  (fully offline — no model server)
make lint            # ruff check .
make fmt             # ruff format + import sort
make build-model     # ollama create the 'novel' model
make demo            # live demo (needs a running server)
```

## Architecture (the compound stack)

- **Primitives**: `agent.py` (NovelAgent tool loop), `tools.py` (sandboxed
  registry), `subagents.py` (delegation), `worktrees.py`, `compaction.py`.
- **Orchestration**: `loop.py` (goal loop), `workflows.py` (5 patterns),
  `routines.py` (cron + daemon), `trace.py`, `experiments.py`.
- **Memory**: `memory.py` (STATE.md, 5-stage), `skills.py` (`skills/`),
  `knowledge.py` (`knowledge/`).
- **Self-improvement**: `verifier.py` (independent grader), `vision.py`,
  `evals.py`, `reflect.py`, `rubric.py`.
- **Routing/ops**: `config.py` (fleet→role config), `router.py` (routing +
  safety + availability), `fleet.py`, `bench.py`, `doctor.py`, `sessions.py`.
- `main.py` is the CLI; `examples/` holds the live demo.

## Conventions

- **Model calls are injectable.** Every component that calls a model accepts a
  `client=` (and usually a `model=`). Tests pass a fake OpenAI-compatible client
  (`tests/conftest.py::FakeClient`); never hit a real server in tests.
- **Config is a single live singleton** (`config.config`). In tests, mutate it
  with `monkeypatch.setattr(cfg.config, "...", ...)`. **Do not** pop `config`
  from `sys.modules` — that creates duplicate singletons that desync across
  modules (this bit us once; see git history).
- **Reasoning models** (deepseek-r1, gemma4) emit `<think>` traces; parse model
  output through `sanitize.strip_reasoning` before JSON.
- **Tool calls** may arrive as native `tool_calls` or as text (`<tool_call>`,
  `<function=...>`); `toolcall.parse_text_tool_calls` handles the text forms.
- **Memory vs Skills vs Knowledge**: project memory (STATE.md, distilled) vs
  procedural memory (versioned `skills/`) vs reference docs (`knowledge/`).
- Keep new model-calling code testable: thread `client`/`model` through and add
  an offline test with `FakeClient`. Run `make lint && make test` before pushing.
