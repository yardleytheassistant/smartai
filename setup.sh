#!/usr/bin/env bash
# One-time setup for smartai on a Mac Studio (Apple Silicon).
# Installs Ollama, pulls the open-source Hermes base, builds the custom
# 'smartai' model from it, and creates a Python environment.
set -euo pipefail

cd "$(dirname "$0")"

BASE_MODEL="${SMARTAI_BASE_MODEL:-hermes3:70b}"
MODEL_NAME="${SMARTAI_MODEL_NAME:-smartai}"

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

echo "==> Pulling open-source base model: ${BASE_MODEL}"
ollama pull "${BASE_MODEL}"

echo "==> Building the custom '${MODEL_NAME}' model"
SMARTAI_MODEL_NAME="${MODEL_NAME}" SMARTAI_BASE_MODEL="${BASE_MODEL}" ./build_model.sh

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

Model in use: ${MODEL_NAME}  (built from ${BASE_MODEL}; edit LLM_MODEL in .env to change it)
EOF
