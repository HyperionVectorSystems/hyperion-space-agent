#!/usr/bin/env bash
# Fetch + build the open-source tools into ./tools/ and create the SatSim venv.
# This builds a WORKING pipeline. Versions are pinned at the top; bump them
# deliberately. Each upstream keeps its own license (reviewed in README).
#
# CUDA is auto-detected: if `nvcc` is present the GPU builds are used, else CPU.
# Nothing here is required for orbit propagation (the harness has an analytical
# propagator); these tools add voice, real GMAT fidelity, and synthetic scenes.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tools models data/ephemeris data/scenes

# ---- pinned versions (edit deliberately) ----
LLAMA_CPP_REF="${LLAMA_CPP_REF:-b4400}"          # llama.cpp release tag
WHISPER_CPP_REF="${WHISPER_CPP_REF:-v1.7.4}"     # whisper.cpp release tag
WHISPER_GGML="${WHISPER_GGML:-base.en}"          # ggml model NAME to download (not the .env WHISPER_MODEL path)
PIPER_REF="${PIPER_REF:-2023.11.14-2}"           # piper release
PIPER_VOICE_URL="${PIPER_VOICE_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx}"
SATSIM_VER="${SATSIM_VER:-0.19.0}"               # satsim (verify on first run; not run in our stack)
TF_VER="${TF_VER:-2.15.1}"                       # TensorFlow for the satsim venv
GMAT_URL="${GMAT_URL:-https://sourceforge.net/projects/gmat/files/GMAT/GMAT-R2025a/}"  # official R2025a page

# Put the CUDA toolkit on PATH if it is installed but not exported. Common on GPU
# cloud images: the driver is present and nvcc lives under /usr/local/cuda*/bin,
# but PATH does not include it, so the nvcc probe below would wrongly fall back to
# a CPU build on a GPU box.
for _cuda in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
  if [ -x "$_cuda/nvcc" ]; then case ":$PATH:" in *":$_cuda:"*) ;; *) PATH="$_cuda:$PATH";; esac; fi
done
export PATH

if command -v nvcc >/dev/null 2>&1; then CUDA=1; echo "CUDA detected: GPU builds"; else CUDA=0; echo "no CUDA: CPU builds"; fi

echo "== llama.cpp (${LLAMA_CPP_REF}) =="
if [ ! -x tools/llama.cpp/build/bin/llama-server ]; then
  git clone --depth 1 --branch "${LLAMA_CPP_REF}" https://github.com/ggml-org/llama.cpp tools/llama.cpp || \
    git clone --depth 1 https://github.com/ggml-org/llama.cpp tools/llama.cpp
  # -DLLAMA_CURL=OFF drops llama-server's libcurl model-download feature (we pull
  # the GGUF ourselves via the HF CLI), which removes libcurl-dev as a build dep.
  if [ "$CUDA" = 1 ]; then
    cmake -S tools/llama.cpp -B tools/llama.cpp/build -DGGML_CUDA=on -DLLAMA_CURL=OFF
  else
    cmake -S tools/llama.cpp -B tools/llama.cpp/build -DLLAMA_CURL=OFF
  fi
  cmake --build tools/llama.cpp/build --config Release -j --target llama-server
fi

echo "== whisper.cpp (${WHISPER_CPP_REF}) + ggml-${WHISPER_GGML} =="
if [ ! -x tools/whisper.cpp/build/bin/whisper-cli ]; then
  git clone --depth 1 --branch "${WHISPER_CPP_REF}" https://github.com/ggml-org/whisper.cpp tools/whisper.cpp || \
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp tools/whisper.cpp
  if [ "$CUDA" = 1 ]; then
    cmake -S tools/whisper.cpp -B tools/whisper.cpp/build -DGGML_CUDA=on
  else
    cmake -S tools/whisper.cpp -B tools/whisper.cpp/build
  fi
  cmake --build tools/whisper.cpp/build --config Release -j
  bash tools/whisper.cpp/models/download-ggml-model.sh "${WHISPER_GGML}"
fi

echo "== Piper (${PIPER_REF}) + lessac voice =="
if [ ! -x tools/piper/piper ]; then
  mkdir -p tools/piper/voices
  ARCH=$([ "$(uname -m)" = aarch64 ] && echo arm64 || echo amd64)
  curl -fsSL "https://github.com/rhasspy/piper/releases/download/${PIPER_REF}/piper_linux_x86_64.tar.gz" -o /tmp/piper.tgz
  tar -xzf /tmp/piper.tgz -C tools/
  curl -fsSL "${PIPER_VOICE_URL}" -o tools/piper/voices/en_US-lessac-medium.onnx
  curl -fsSL "${PIPER_VOICE_URL}.json" -o tools/piper/voices/en_US-lessac-medium.onnx.json
fi

echo "== SatSim venv (.venv-satsim, TensorFlow ${TF_VER} + satsim ${SATSIM_VER}) =="
echo "   NOTE: SatSim was not integrated in our own stack; verify one scene after this."
# SatSim is OPTIONAL and was not exercised in our stack. Run it in a subshell so a
# failed install cannot abort the rest of setup (set -e): the core harness (text,
# voice, propagation, globe) does not depend on it.
if [ ! -x .venv-satsim/bin/python ]; then
  ( set -e
    python3 -m venv .venv-satsim
    ./.venv-satsim/bin/pip install -U pip
    if [ "$CUDA" = 1 ]; then
      ./.venv-satsim/bin/pip install "tensorflow[and-cuda]==${TF_VER}" "satsim==${SATSIM_VER}"
    else
      ./.venv-satsim/bin/pip install "tensorflow==${TF_VER}" "satsim==${SATSIM_VER}"
    fi
  ) || echo "WARN: SatSim venv setup failed (optional; scenes unavailable). Continuing."
fi

echo
echo "== GMAT R2025a (OPTIONAL, higher-fidelity propagation) =="
echo "   The analytical propagator already makes /viz work; GMAT is the upgrade."
echo "   Download GMAT-R2025a for Linux and unpack into tools/GMAT/, then set"
echo "   GMAT_BIN in .env:  ${GMAT_URL}"
echo
echo "== Cesium 1.132 + Aladin Lite v3: loaded from CDN by web/index.html; nothing to fetch."
echo
echo "fetch complete. Text chat + voice + propagation + globe work now."
echo "SatSim scenes work once you set SATSIM_CONFIG; GMAT fidelity once you set GMAT_BIN."
