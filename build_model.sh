#!/usr/bin/env bash
# Build the custom `smartai` model from the open-source Hermes base.
#
# Usage:
#   ./build_model.sh                         # builds 'smartai' from Modelfile
#   SMARTAI_MODEL_NAME=smartai-fast \
#   SMARTAI_BASE_MODEL=hermes3:8b ./build_model.sh   # smaller base, custom name
#
# The Modelfile is the canonical definition (FROM hermes3:70b). If
# SMARTAI_BASE_MODEL is set to something else, we build from a temporary copy
# with the FROM line swapped, so a single Modelfile is the source of truth.
set -euo pipefail

cd "$(dirname "$0")"

MODEL_NAME="${SMARTAI_MODEL_NAME:-smartai}"
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
