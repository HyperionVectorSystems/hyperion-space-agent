"""Orchestrator — FastAPI service on :8703.

The single backend for the hub. It:
  1. takes a turn (text via /chat/send, audio via /voice-turn),
  2. runs a two-pass loop: pass 1 lets the model pick a tool; the tool runs;
     pass 2 grounds the model's spoken answer in the tool result,
  3. returns text + any artifact (table / chart / image / viz update), and for
     voice, synthesized speech.

It also serves the hub (/) and the viz + image routes the hub reads.

This is a reference implementation: readable over clever. The tool loop is
deterministic Python, not model-driven control flow, so it behaves the same
regardless of how well a given small model calls functions.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from . import llm_client
from . import propagate
from .tools import build_registry
from .router import Router, CHAT_INTENT
from .macros import default_macros
from . import voice

WEB_DIR = os.environ.get("WEB_DIR", os.path.join(os.path.dirname(__file__), "..", "web"))
EPHEMERIS_DIR = os.environ.get("EPHEMERIS_DIR", "./data/ephemeris")
SCENE_DIR = os.environ.get("SCENE_DIR", "./data/scenes")

app = FastAPI(title="Voice-driven SDA toolchain (reference harness)")
registry = build_registry()
router = Router(registry, macros=default_macros())   # deterministic dispatch + macros; the model does not route

SYSTEM = (
    "You are an assistant for Space Domain Awareness (tracking satellites and "
    "orbits) — NOT any religious or other meaning of 'SDA'. Answer in one or two "
    "short spoken sentences, grounded ONLY in the information provided to you. "
    "Plain speech: no markdown, no lists."
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "tools": registry.names(), "model": llm_client.LLM_MODEL}


@app.post("/chat/send")
async def chat_send(req: Request) -> JSONResponse:
    body = await req.json()
    text = (body.get("message") or "").strip()
    if not text:
        return JSONResponse({"error": True, "message": "empty message"}, status_code=400)
    reply, artifact = _turn(text)
    return JSONResponse({"response": reply, "artifact": artifact})


@app.post("/voice-turn")
async def voice_turn(req: Request) -> JSONResponse:
    """Audio in, audio + text + artifact out. WAV bytes are posted raw."""
    wav = await req.body()
    text = voice.transcribe(wav)                 # whisper
    if not text:
        return JSONResponse({"error": True, "message": "no speech recognized"}, status_code=400)
    reply, artifact = _turn(text)
    audio_b64 = voice.synthesize(reply)          # piper -> base64 wav
    return JSONResponse({"heard": text, "response": reply, "artifact": artifact, "audio": audio_b64})


def _turn(text: str) -> tuple[str, dict[str, Any] | None]:
    """One deterministic turn. Returns (spoken_reply, artifact_or_None).

    The ROUTER decides what runs (classify → dispatch), not the model. The model
    is used only to (a) answer conversationally when no tool applies, or (b)
    ground the spoken answer in a tool result. This is the architecture-spec.json
    control flow: a closed set parsed in Python, dispatched in Python.
    """
    intent, args = router.classify(text)
    print(f"[route] input={text!r} -> intent={intent} args={args}", flush=True)
    if intent == CHAT_INTENT:
        return (_chat_answer(text), None)

    result = router.dispatch(intent, args)
    if result.error:
        return (result.error, None)                # speak the failure plainly
    artifact = result.artifact or None             # already rendered by dispatch

    # Grounding pass: the model narrates the tool result (no tools offered here).
    try:
        grounded = llm_client.chat(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text},
                {"role": "user", "content": f"Tool {intent} returned: {result.text}\nData: {json.dumps(result.data)[:800]}"},
            ],
            max_tokens=256,
        )
        reply = llm_client.first_message(grounded).get("content", "").strip()
    except Exception:
        reply = result.text                        # fall back to the tool summary
    return (reply or result.text, artifact)


def _chat_answer(text: str) -> str:
    """No tool applies: answer conversationally. No tools are offered, so the
    model cannot route from here — routing is the router's job alone."""
    try:
        resp = llm_client.chat(
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            max_tokens=512,
        )
        return llm_client.first_message(resp).get("content", "").strip() or "I could not answer that."
    except Exception as e:
        return (f"The model is unavailable ({e}).")


# --- viz + image routes the hub reads -------------------------------------

@app.get("/viz/files")
def viz_files() -> dict[str, Any]:
    """Newest-first OEM list; the globe polls this to know when to reload. Any
    change to the set (a new object, or a clear) changes the head, which triggers
    a reload of the merged multi-satellite CZML."""
    paths = sorted(propagate.list_oems(EPHEMERIS_DIR), key=os.path.getmtime, reverse=True)
    return {"files": [{"name": os.path.basename(p), "modified": os.path.getmtime(p)} for p in paths],
            "count": len(paths)}


@app.get("/viz/czml")
def viz_czml() -> JSONResponse:
    """Merge ALL current OEMs into one CZML (one entity per satellite)."""
    return JSONResponse(propagate.oems_to_czml(EPHEMERIS_DIR))


@app.post("/viz/clear")
def viz_clear() -> dict[str, Any]:
    """Reset the globe: move all OEMs into a _cleared/ subdir (not a delete)."""
    return {"cleared": propagate.clear_oems(EPHEMERIS_DIR)}


# Serve generated data (FITS scenes, OEM) to the browser so the Aladin/Cesium
# panes can fetch the actual bytes. Mounted before the web root.
for _p in (EPHEMERIS_DIR, SCENE_DIR):
    os.makedirs(_p, exist_ok=True)
app.mount("/data/ephemeris", StaticFiles(directory=EPHEMERIS_DIR), name="ephemeris")
app.mount("/data/scenes", StaticFiles(directory=SCENE_DIR), name="scenes")

# Serve the hub last so API routes win.
if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main() -> None:
    import uvicorn
    host = os.environ.get("ORCH_HOST", "127.0.0.1")
    port = int(os.environ.get("ORCH_PORT", "8703"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
