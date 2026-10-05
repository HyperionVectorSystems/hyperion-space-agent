#!/usr/bin/env bash
# Pull the GGUF model named in .env (MODEL_ID / MODEL_FILE) into ./models/.
# Uses the Hugging Face CLI; install it with: pip install -U "huggingface_hub[cli]"
#
# Default is Llama-3.1-8B-Instruct. Swap MODEL_ID / MODEL_FILE in .env for
# Gemma-2-9b-it or an MoE option (all listed there).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
mkdir -p models

: "${MODEL_ID:?set MODEL_ID in .env}"
: "${MODEL_FILE:?set MODEL_FILE in .env}"

echo "pulling ${MODEL_FILE} from ${MODEL_ID} ..."
# huggingface_hub >= 0.34 ships the `hf` command; the old `huggingface-cli` was
# removed in hub 2.0 ("deprecated and no longer works"). Prefer the venv's hf (build.sh
# installs it there, not on PATH), then a global hf, then the legacy CLI as a last resort.
if [ -x ./.venv/bin/hf ]; then
  HF="./.venv/bin/hf"
elif command -v hf >/dev/null 2>&1; then
  HF="hf"
elif [ -x ./.venv/bin/huggingface-cli ]; then
  HF="./.venv/bin/huggingface-cli"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF="huggingface-cli"
else
  HF=""
fi

if [ -n "${HF}" ]; then
  "${HF}" download "${MODEL_ID}" "${MODEL_FILE}" --local-dir models
else
  echo "No Hugging Face CLI found. Install it:  pip install -U huggingface_hub"
  echo "then:  hf download ${MODEL_ID} ${MODEL_FILE} --local-dir models"
  echo "or download ${MODEL_FILE} from https://huggingface.co/${MODEL_ID} into ./models/ by hand."
  exit 1
fi
echo "model at ./models/${MODEL_FILE}"
