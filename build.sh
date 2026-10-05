#!/usr/bin/env bash
# Build / run the voice-driven SDA harness.
#
#   ./build.sh          # set up: venv, python deps, fetch tools, pull the model
#   ./build.sh run      # launch: model server + orchestrator, then open the hub
#
# This script wires the harness to tools you fetch. It does not redistribute the
# tools; scripts/fetch_tools.sh points at their upstream projects.
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "copy .env.example to .env and edit it first"; exit 1; }
set -a; . ./.env; set +a

sysdeps() {
  echo "== system packages (toolchain, cmake, ffmpeg) =="
  # The tool builds (llama.cpp, whisper.cpp) need cmake + a C/C++ compiler, and
  # the voice loop needs ffmpeg. These are OS packages, not pip deps. Auto-install
  # on apt systems; elsewhere install the equivalents by hand (see README).
  need=""
  command -v cmake  >/dev/null 2>&1 || need="$need cmake"
  command -v ffmpeg >/dev/null 2>&1 || need="$need ffmpeg"
  command -v c++    >/dev/null 2>&1 || need="$need build-essential"
  if [ -z "$need" ]; then echo "  all present"; return 0; fi
  echo "  missing:$need"
  if command -v apt-get >/dev/null 2>&1; then
    SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"
    $SUDO apt-get update && $SUDO apt-get install -y build-essential cmake ffmpeg
  else
    echo "  no apt-get: install$need with your package manager, then re-run." >&2
    exit 1
  fi
}

setup() {
  sysdeps
  echo "== python venv + deps =="
  python3 -m venv .venv
  ./.venv/bin/pip install -U pip
  ./.venv/bin/pip install -r requirements.txt

  echo "== fetch + build open-source tools (llama.cpp, whisper.cpp, Piper, SatSim; GMAT optional) =="
  bash scripts/fetch_tools.sh

  echo "== pull the language model =="
  bash scripts/pull_model.sh

  echo
  echo "setup complete. run:  ./build.sh run"
}

run() {
  mkdir -p "${EPHEMERIS_DIR:-./data/ephemeris}" "${SCENE_DIR:-./data/scenes}"

  echo "== model server on ${LLM_BASE_URL} =="
  # Default: llama.cpp llama-server. For Ollama, start `ollama serve` instead
  # and set LLM_BASE_URL to its OpenAI-compatible endpoint in .env.
  if [ -n "${LLAMA_SERVER_BIN:-}" ] && [ -n "${MODEL_PATH:-}" ]; then
    nohup "${LLAMA_SERVER_BIN}" -m "${MODEL_PATH}" \
      --host 127.0.0.1 --port "${LLM_PORT:-8080}" -c 8192 -ngl 99 \
      > logs/llama.log 2>&1 &
    echo "  llama-server pid $!"
  else
    echo "  (LLAMA_SERVER_BIN / MODEL_PATH not set; assuming an external model at ${LLM_BASE_URL})"
  fi

  echo "== orchestrator on :${ORCH_PORT:-8703} =="
  ./.venv/bin/python -m orchestrator.server &
  echo "  orchestrator pid $!"

  sleep 2
  echo
  echo "open http://127.0.0.1:${ORCH_PORT:-8703}/  and ask a question."
}

mkdir -p logs
case "${1:-setup}" in
  setup) setup ;;
  run)   run ;;
  *)     echo "usage: ./build.sh [setup|run]"; exit 1 ;;
esac
