# Voice-driven SDA toolchain — reference harness

A small, local-first harness that lets natural language drive a heterogeneous
space-domain-awareness (SDA) toolchain to an answer and a live visualization.
You speak or type a question; a local language model routes it; open-source SDA
tools (orbit propagation, synthetic scene generation, catalog/annotation
overlay, image display) do the work; the answer comes back as speech, text, a
table or chart, and a Cesium globe that redraws itself.

This repository is the **harness**: the orchestrator, the tool-interface
contract, the voice pipeline glue, and a single-page web hub. It is not a
bundle of the SDA tools themselves. The tools are established open-source
projects; you fetch them with the build script and this harness wires them
together. The value here is the glue and the contract, the part that is hard to
get right, not the tools, which are already excellent and freely available.

> This is a reference implementation released for the AMOS short course. It
> ships an **example detector only** (a simple matched-filter stub). It does not
> include any proprietary detection or estimation methods.

## The capability map

`docs/capability-map.html` is the one-page picture of how the pieces connect,
color-coded by what this repo provides versus what you fetch, with the layer
stack and the three functional loops. Open it in any browser.
`docs/architecture.md` walks the map in prose.

The short version, left to right:

- **Browser hub** (this repo): chat + push-to-talk on the left half, a tool
  artifact pane (table / chart / zoom graphic) top-right, a Cesium globe
  bottom-right, a light FITS image viewer, each pane expandable to full screen.
- **Orchestrator** (this repo): a FastAPI service on `:8703`. It takes a turn
  (`/chat/send` text or `/voice-turn` audio), routes it, calls the right tool
  adapter, grounds the model's answer in the tool result, and returns
  speech + text + any artifact + a viz update.
- **Tool adapters** (this repo): thin wrappers that conform to one interface
  contract (`orchestrator/tools/contract.py`). Baseline adapters: GMAT
  propagation, SatSim scene generation, the SatNet annotation overlay, and the
  example detector.
- **The tools and the model** (you fetch): GMAT headless, SatSim, Cesium, a
  Aladin Lite (sky/FITS viewer), whisper.cpp, Piper, and a local model served by
  llama.cpp `llama-server` or Ollama.

## Baseline capabilities

What runs after `./build.sh` (which pulls the model and runs `fetch_tools.sh`):

- **Text chat** that drives the toolchain. Works as soon as the model server is up.
- **Orbit propagation to a self-updating globe, multi-satellite.** The harness
  has its own analytical propagator (SGP4 for bundled objects like ISS, two-body
  for `altitude=<km>`); it writes an OEM and Cesium redraws. Ask for more objects
  and they append, the globe shows every propagated satellite at once (one color +
  label each), with a Clear control to reset. No external tool needed. A real
  GMAT R2025a install (set `GMAT_BIN`) is a higher-fidelity swap, not a requirement.
- **Voice loop**: push-to-talk in the browser, whisper STT, model, Piper TTS,
  playback. Live once `fetch_tools.sh` has built whisper.cpp + Piper, and with
  **ffmpeg** on PATH — the browser records webm/opus, which the orchestrator
  transcodes to the 16 kHz mono WAV whisper needs. Browsers only allow the mic on
  a secure context, so open the hub over `http://localhost` (or HTTPS), not a bare
  LAN/remote IP.
- **Image display + overlays** in Aladin Lite (sky context + RA/Dec marks),
  with the example detector's candidates.
- **Synthetic scene generation** via SatSim. This is the one piece that needs
  configuration: `fetch_tools.sh` creates the SatSim venv, and you point
  `SATSIM_CONFIG` at a scene config to generate. SatSim was not exercised in our
  own stack, so verify one scene before relying on it.

Honest state: text + propagation + globe run immediately; voice runs after the
whisper/Piper build; scenes run after you set `SATSIM_CONFIG`; GMAT fidelity is
optional. Nothing here is a stub except the detector (below).

## Prerequisites and the open-source tools to fetch

The build script fetches or points at each of these. Licenses are the upstream
projects' own; check them before redistribution.

| Tool | Purpose | Where | License (verify upstream) |
|------|---------|-------|---------------------------|
| GMAT R2025a (headless) | orbit propagation, OPTIONAL higher fidelity | NASA GMAT | NASA Open Source Agreement |
| SatSim | synthetic SDA scenes + truth (own TF venv) | open source | check upstream |
| Cesium 1.132 | 3D globe viz | CDN | Apache-2.0 |
| Aladin Lite v3 | in-browser sky/FITS viewer + RA/Dec overlay | CDN (CDS) | GPL/CeCILL |
| whisper.cpp | speech-to-text | ggml/whisper.cpp | MIT |
| Piper | text-to-speech | rhasspy/piper | MIT |
| ffmpeg | transcode browser mic audio to 16 kHz WAV for whisper | system pkg (apt/brew) | LGPL/GPL |
| llama.cpp / Ollama | local model server | ggml/llama.cpp or Ollama | MIT |
| a GGUF model | the language model | Hugging Face | model-specific |

Default model is **Llama-3.1-8B-Instruct** (widely compatible and the path we
verified). `.env.example` lists one-line swaps: Qwen2.5-7B-Instruct, Gemma-2-9B-it
(quality on a memory budget), and an MoE option (gpt-oss-20B / Qwen3-Coder-30B) for
more VRAM at ~7B speed. Routing is **deterministic Python** (`orchestrator/router.py`),
not native tool-calling: the model is used only as a bounded transducer that returns
a small `{"intent","args"}` JSON object drawn from a closed set, and Python validates
it and dispatches the tool in a two-pass loop (route, then ground). Control flow —
whether a tool runs, which one, in what order — lives in the router, so the harness
behaves the same regardless of how well a given small model calls functions. The one
model requirement is that it emit that clean JSON: most instruct models do; reasoning
/ "thinking" models (e.g. the Qwen3 line) tend to wrap it in prose or return their
reasoning, which routes worse — prefer a straight instruct model. Any OpenAI-compatible
endpoint works, so a remote API or a local Ollama can replace the model by changing
one base URL.

## Build and run

Install the system packages the tool builds need first (these are not pip deps): a
C/C++ toolchain, **cmake**, and **ffmpeg** (voice audio transcode). On Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake ffmpeg
```

`build.sh` also attempts this automatically when `apt-get` is available. Then:

```bash
cp .env.example .env          # pick your model + ports
./build.sh                    # fetch tools, make the venv, pull the model
./build.sh run                # launch the model server + orchestrator, open the hub
```

Then open `http://localhost:8703/` and ask "propagate the ISS for one orbit."
See `build.sh` for the step-by-step it runs, and `docs/architecture.md` for what
each step wires.

**Running on a remote host (cloud GPU, RunPod, etc.):** the browser only grants the
microphone on a secure context, so you cannot open the hub on a bare remote IP.
Either tunnel the orchestrator port to your machine over SSH and open `localhost`:

```bash
ssh -L 8703:127.0.0.1:8703 <user>@<host> [-p <port>]   # then open http://localhost:8703
```

or put the orchestrator behind HTTPS (for example a RunPod HTTP proxy URL), which is
a secure context and needs no tunnel. Text chat works over any of these; voice needs
the secure context.

## Troubleshooting

- "The model is unavailable": the model server is not up. Check `logs/llama.log`,
  or confirm `LLM_BASE_URL` points at a running endpoint (`ollama serve`, or your
  remote API).
- Port already in use: the orchestrator is `:8703`, the model `:8080`. Change
  `ORCH_PORT` / `LLM_PORT` in `.env`.
- No GPU: drop `-ngl 99` from the `llama-server` line in `build.sh` (CPU is
  slower but works); `fetch_tools.sh` already builds CPU-only when `nvcc` is absent.
- Voice does nothing: whisper/Piper are not built or their paths in `.env` are
  wrong; text chat still works without them.
- The globe is empty: no orbit propagated yet. Ask for one; the globe polls
  `/viz/files` and redraws.

## Contributing

The entry point is the tool interface contract: add a capability by writing one
adapter class. See **`docs/interface-contract.md`** and the "Adding your own
tool" section below.

## Adding your own tool

Implement the contract in `orchestrator/tools/contract.py` (one class, three
methods: `schema`, `run`, `render`) and register it. `docs/interface-contract.md`
is the full guide. The orchestrator discovers registered adapters and offers
them to the model automatically; no orchestrator edits needed.

## Roadmap (not in this release)

A hosted version, accounts, and a remote model API behind the orchestrator (so
users need no local GPU) are planned. The orchestrator already calls the model
over HTTP, so that is a base-URL change, not a rearchitecture. This release is
the local-first harness.

## License

Harness code: see `LICENSE`. The fetched tools and the model carry their own
licenses; this repo does not redistribute them.
