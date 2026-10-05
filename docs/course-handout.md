# Voice-driven SDA toolchain — one-page handout

A local-first reference harness: ask a question in plain language, and a local
model drives open-source space-domain-awareness tools to an answer plus a live
visualization. You speak or type; the model routes; the tools do the work; the
answer comes back as speech, text, a table or chart, and a Cesium globe that
redraws itself.

## The chain it runs

`natural language -> orchestrator -> (model routes) -> SDA tool -> grounded answer + viz`

One request drives orbit propagation (GMAT), synthetic scene generation
(SatSim), an annotation overlay (SatNet), image display (FITS viewer), and an
example detector, then speaks the result and updates the globe. The point is
that plain language drives tools nobody wants to hand-wire.

## What you get vs what you fetch

- This repo provides the hard part: the orchestrator, the tool-interface
  contract, the voice pipeline glue, and a single-page web hub. MIT licensed,
  free to use and build on.
- You fetch the tools and the model (the build script builds/points at them):
  SatSim, Cesium, Aladin Lite, whisper.cpp, Piper, llama.cpp or Ollama with a
  GGUF model, and optionally GMAT R2025a. Each keeps its own license; this repo
  does not ship them. Propagation needs nothing fetched, the harness has its own
  analytical propagator, so the globe works immediately; GMAT only raises fidelity.

## Baseline capabilities (work after the build script)

Text chat and a **voice loop** (push-to-talk in the browser, whisper to model
to Piper), orbit propagation to a self-updating globe, synthetic scene
generation, a SatNet truth overlay, and image display with the example
detector's marks. Voice is baseline, it was the hook.

## Quickstart

```
cp .env.example .env        # pick your model + set tool paths
./build.sh                  # fetch tools, make the venv, pull the model
./build.sh run              # start the model server + orchestrator
```

Open `http://localhost:8703/` and say or type "propagate the ISS for one orbit."

## Choosing a model (Sept 2026)

Default is **Llama-3.1-8B-Instruct** (widest support, the path we verified). Swaps in
`.env.example`: Gemma-2-9B (quality on a memory budget) or an MoE (gpt-oss-20B /
Qwen3-Coder-30B) for ~7B speed at higher quality. Use GGUF Q4_K_M. The harness
routes and grounds deterministically, so you are not tied to any model's
tool-calling reliability, a small instruct model is enough. Avoid thinking models
(e.g. Qwen3): they wrap the routing JSON in prose and route worse.

## The map and going deeper

`docs/capability-map.html` is the one-page architecture picture (open in a
browser): teal = this repo, amber = you fetch, plus the layer stack and the
three functional loops. Then `README.md`, `docs/architecture.md` (how a turn
flows), and `docs/interface-contract.md` (add your own tool in one class).

## Note

This release ships an **example detector only** (a simple threshold stub), so
the chain has a working detection step. It includes no proprietary detection or
estimation method. Swap in your own detector via the interface contract.
