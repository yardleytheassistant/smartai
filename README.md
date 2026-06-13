# smartai — Hermes local agent

A custom tool-using agent that runs **open-source Nous Hermes models locally**
on a Mac Studio (Apple Silicon). No cloud APIs, no per-token cost — the model
runs on your own hardware.

The agent talks to any **OpenAI-compatible** local server, so the same code
runs on Ollama, MLX, llama.cpp, or LM Studio. You switch runtimes by changing
one environment variable.

## Why this stack

- **Runtime: Ollama (default).** First-class function/tool calling, one-line
  model pulls, a built-in OpenAI-compatible server. The path of least
  resistance for a tool-using agent.
- **Runtime: MLX (performance upgrade).** Apple's native framework — the
  fastest and most memory-efficient option on M-series silicon. Use it when you
  want maximum throughput or to run the 405B at low quant.
- **Model: Nous Hermes.** Hermes models are specifically tuned for reliable
  structured tool/function calling, which is exactly what an agent loop needs.
- **Language: Python.** The local-LLM ecosystem (MLX, the Ollama client, the
  tooling) is Python-native.

## Model sizing on 256 GB RAM

| Model            | Quant     | Approx. RAM | Notes                              |
| ---------------- | --------- | ----------- | ---------------------------------- |
| Hermes 70B       | Q4_K_M    | ~42 GB      | Fast, great default                |
| Hermes 70B       | fp16      | ~140 GB     | Highest quality at 70B             |
| Hermes 405B      | 4-bit     | ~200–230 GB | Feasible on 256 GB; slower         |

Default is `hermes3:70b`. Change `LLM_MODEL` in `.env` to go bigger or smaller.

## Quick start (Ollama)

```bash
chmod +x setup.sh
./setup.sh                 # installs Ollama, pulls Hermes, builds .venv

source .venv/bin/activate
python main.py             # interactive chat REPL
python main.py "create notes.md in the workspace summarizing what you can do"
```

## Using MLX instead

MLX serves an OpenAI-compatible endpoint too, so only `.env` changes:

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Hermes-3-Llama-3.1-70B-4bit --port 8080
```

Then in `.env`:

```
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=mlx
LLM_MODEL=mlx-community/Hermes-3-Llama-3.1-70B-4bit
```

Browse `mlx-community` on Hugging Face for other Hermes conversions (including
Hermes 4 and 405B builds).

## Built-in tools

The agent ships with a small, safe starter toolset (see `tools.py`):

| Tool             | What it does                                              |
| ---------------- | -------------------------------------------------------- |
| `get_current_time` | Current local date/time                                |
| `list_directory` | List files in the workspace                              |
| `read_file`      | Read a text file in the workspace                        |
| `write_file`     | Write/overwrite a text file in the workspace            |
| `calculate`      | Exact arithmetic (safe — no `eval`)                     |
| `run_shell`      | Run a shell command (disabled by default; see below)    |

**Sandboxing:** file and shell tools are confined to `AGENT_WORKSPACE`
(default `./workspace`). The `run_shell` tool is **off by default** — set
`AGENT_ENABLE_SHELL=1` in `.env` to enable it, and it refuses obviously
destructive commands.

### Adding your own tool

Drop a function in `tools.py` and decorate it:

```python
@tool(
    description="Fetch the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
def get_weather(city: str) -> str:
    ...
    return "sunny, 22°C"
```

It's automatically advertised to the model and dispatched by name.

## Configuration

All settings are environment variables (see `.env.example`):

| Variable             | Default                      | Purpose                          |
| -------------------- | ---------------------------- | -------------------------------- |
| `LLM_BASE_URL`       | `http://localhost:11434/v1`  | OpenAI-compatible endpoint       |
| `LLM_MODEL`          | `hermes3:70b`                | Model name/tag                   |
| `LLM_TEMPERATURE`    | `0.7`                        | Sampling temperature             |
| `LLM_MAX_TOKENS`     | `4096`                       | Max tokens per response          |
| `AGENT_MAX_STEPS`    | `12`                         | Max tool-calling steps per turn  |
| `AGENT_ENABLE_SHELL` | `0`                          | Allow the shell tool             |
| `AGENT_WORKSPACE`    | `./workspace`                | File/shell sandbox directory     |

## Project layout

```
config.py   # env-driven configuration
tools.py    # tool definitions + registry (sandboxed)
agent.py    # the OpenAI-compatible tool-calling loop
main.py     # CLI / REPL
setup.sh    # one-time install for Ollama + Python env
```
