#!/usr/bin/env bash
# Build the custom `novel` agent model from the open-source base.
#
# Usage:
#   ./build_model.sh                         # builds 'novel' from Modelfile
#   SMARTAI_MODEL_NAME=novel-fast \
#   SMARTAI_BASE_MODEL=phi4:14b ./build_model.sh     # smaller/faster base, custom name
#   SMARTAI_MODEL_NAME=novel-max \
#   SMARTAI_BASE_MODEL=qwen3:235b ./build_model.sh   # max-capability base
#
# The Modelfile is the canonical definition (FROM qwen3.5:122b). If
# SMARTAI_BASE_MODEL is set to something else, we build from a temporary copy
# with the FROM line swapped, so a single Modelfile is the source of truth.
set -euo pipefail

cd "$(dirname "$0")"

MODEL_NAME="${SMARTAI_MODEL_NAME:-novel}"
BASE_MODEL="${SMARTAI_BASE_MODEL:-}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Error: 'ollama' not found. Run ./setup.sh first (or install from https://ollama.com)." >&2
  exit 1
fi

MODELFILE="Modelfile"
CLEANUP=""

if [ -n "${BASE_MODEL}" ]; then
  CURRENT_BASE="$(grep -m1 '^FROM ' Modelfile | awk '{print $2}')"
  if [ "${BASE_MODEL}" != "${CURRENT_BASE}" ]; then
    echo "==> Building on override base: ${BASE_MODEL} (Modelfile default is ${CURRENT_BASE})"
    echo "==> Pulling base model: ${BASE_MODEL}"
    ollama pull "${BASE_MODEL}"
    MODELFILE="$(mktemp)"
    CLEANUP="${MODELFILE}"
    # Swap only the FROM line; keep every other directive intact.
    sed "s|^FROM .*|FROM ${BASE_MODEL}|" Modelfile > "${MODELFILE}"
  fi
fi

echo "==> Creating model '${MODEL_NAME}' from ${MODELFILE}"
ollama create "${MODEL_NAME}" -f "${MODELFILE}"

[ -n "${CLEANUP}" ] && rm -f "${CLEANUP}"

cat <<EOF

Built model: ${MODEL_NAME}

Try it directly:
  ollama run ${MODEL_NAME}

Or point the agent at it (already the default in .env.example):
  LLM_MODEL=${MODEL_NAME} python main.py
EOF
