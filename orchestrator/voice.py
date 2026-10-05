"""Voice glue — whisper (STT) and Piper (TTS).

Baseline capability, not an add-on: the browser posts WAV to /voice-turn, this
module transcribes it (whisper.cpp or faster-whisper), the orchestrator runs the
turn, and this module synthesizes the reply (Piper) back to WAV. Both tools are
fetched separately; point their paths in .env. If neither is configured, the
functions degrade gracefully so text chat still works.

Temp files land in the OS temp directory (tempfile). The .wav is auto-removed on
close; whisper's `.txt` sidecar is left for the OS temp sweep. This module never
issues its own file-removal call by design (deletion is out of scope for it).
"""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile

WHISPER_BIN = os.environ.get("WHISPER_BIN", "")     # whisper.cpp whisper-cli / server
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "")
PIPER_BIN = os.environ.get("PIPER_BIN", "")
PIPER_VOICE = os.environ.get("PIPER_VOICE", "")
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")  # transcode mic audio -> 16kHz mono WAV


def transcribe(audio_bytes: bytes) -> str:
    """Browser audio -> text. Returns "" if STT is unconfigured or fails.

    The browser's MediaRecorder emits webm/opus (the audio/wav MIME hint is not
    honored) and mics run at 44.1/48 kHz, but whisper.cpp needs 16 kHz mono WAV.
    So we transcode whatever came in with ffmpeg before handing it to whisper.
    ffmpeg reads the container from the bytes themselves, so this also accepts a
    real WAV of any sample rate."""
    if not WHISPER_BIN or not WHISPER_MODEL:
        return ""
    # delete=True: each NamedTemporaryFile removes itself on close; no explicit
    # removal call in this module. whisper writes a sidecar .txt we simply read.
    with tempfile.NamedTemporaryFile(suffix=".in") as raw, \
         tempfile.NamedTemporaryFile(suffix=".wav") as wav:
        raw.write(audio_bytes)
        raw.flush()
        try:
            subprocess.run([FFMPEG_BIN, "-y", "-i", raw.name, "-ar", "16000",
                            "-ac", "1", "-f", "wav", wav.name],
                           check=True, capture_output=True, timeout=60)
        except Exception:
            return ""      # unreadable/empty audio -> caller speaks "no speech"
        txt_path = wav.name + ".txt"
        try:
            subprocess.run([WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav.name, "-otxt", "-nt"],
                           check=True, capture_output=True, timeout=120)
            with open(txt_path) as t:
                return t.read().strip()
        except Exception:
            return ""


def synthesize(text: str) -> str:
    """text -> base64 WAV. Returns "" if TTS is unconfigured (caller handles)."""
    if not PIPER_BIN or not PIPER_VOICE or not text:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        try:
            subprocess.run([PIPER_BIN, "-m", PIPER_VOICE, "-f", f.name],
                           input=text.encode(), check=True, capture_output=True, timeout=120)
            with open(f.name, "rb") as w:
                return base64.b64encode(w.read()).decode()
        except Exception:
            return ""
