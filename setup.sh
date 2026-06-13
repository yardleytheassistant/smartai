#!/usr/bin/env bash
# One-time setup for the Hermes agent on a Mac Studio (Apple Silicon).
# Installs Ollama, pulls a Hermes model, and creates a Python environment.
set -euo pipefail

MODEL="${LLM_MODEL:-hermes3:70b}"

echo "==> Installing Ollama (if missing)"
if ! command -v ollama >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install ollama
  else
    echo "Homebrew not found. Install from https://ollama.com/download then re-run." >&2
    exit 1
  fi
fi

echo "==> Starting the Ollama server (background)"
# 'ollama serve' is a no-op if the app is already running.
ollama serve >/dev/null 2>&1 &
sleep 2

echo "==> Pulling model: ${MODEL}"
ollama pull "${MODEL}"

echo "==> Creating Python virtual environment (.venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Writing .env from template (if missing)"
[ -f .env ] || cp .env.example .env

cat <<EOF

Setup complete.

  source .venv/bin/activate
  python main.py            # start the chat REPL
  python main.py "list the files in the workspace and tell me what's there"

Model in use: ${MODEL}  (edit LLM_MODEL in .env to change it)
EOF
