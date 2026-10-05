# Voice testing — SDA toolchain phrasings

A reference of spoken commands for live-testing the harness. Open `http://localhost:8703/`
(over `localhost` or an SSH tunnel — the mic needs a secure context), **press and hold** the talk button, speak, release. Numbers work as
words ("six") or digits ("6"). After a batch, the routing log (`logs/orch-live.log`, one
`[route]` line per turn) shows exactly how each transcribed and dispatched.

Expected Δv scales ~6 m/s per degree of separation (phasing dominates): 6°≈37, 8°≈49,
10°≈61, 12°≈74, 15°≈92 m/s.

## Full RPO scene — routes to `model_rpo` (observer + client + servicer)
| Say this | Exercises |
|---|---|
| "model a GEO refueling RPO with the servicer six degrees behind the client" | baseline scene |
| "model a refueling RPO at GEO with a twelve degree separation" | larger phase, deeper drift |
| "model an RPO with the servicer fifteen degrees behind" | largest phase |
| "model a GEO RPO with the observer at eight hundred kilometers" | observer altitude |

## Maneuver only — routes to `plan_maneuver` (client + servicer, no observer)
| Say this | Exercises |
|---|---|
| "plan a rendezvous at GEO with a ten degree separation" | tool + phase, v-bar default |
| "plan a rendezvous at GEO with an eight degree separation using a radial approach" | r-bar path |
| "plan a rendezvous at twenty thousand kilometers with a ten degree separation" | non-GEO target |

## Single orbits — routes to `propagate_orbit`
| Say this | Exercises |
|---|---|
| "propagate the ISS for two hours" | bundled TLE |
| "put a satellite in GEO" | circular by altitude |
| "propagate a satellite at twenty thousand kilometers, thirty five degrees inclination" | altitude + inclination |

## Utility / questions
| Say this | Routes to |
|---|---|
| "clear the globe" / "reset the view" | `clear_view` (resets the scene) |
| "what is a v-bar approach?" / "how far away is GEO?" | `chat` (answered, no tool) |

## Speaking tips (from live-test failure modes)
- **Approach axis:** say **"v-bar"** (or leave it out — v-bar is the default) for along-track;
  say **"radial approach"** for r-bar. **Avoid saying "r-bar" aloud** — whisper mangles it
  ("our bar", "R-R"); and avoid "along-track" — say "v-bar" instead.
- **Numbers degrade gracefully:** if the model drops a spoken number, the router recovers it
  from your words (digits or spelled-out); if it zeroes it, a sensible default is used. A
  mis-heard number is never silently wrong in a way that breaks the scene.
- **Don't name the satellites** — they default to "client"/"servicer"; just narrate normally.

## What the harness auto-corrects (so nothing reads as broken)
The small local model routinely emits junk optional args; the tool/router layer catches each:
- observer altitude copied from the target → falls back to a LEO default (550 km)
- zeroed phase / standoff / hold → sensible defaults; hold kept inside standoff
- a word grabbed as the approach ("rendezvous") → v-bar default
- invented args (e.g. `servicer_phase_deg`) and model-supplied entity names → dropped

If a command does something unexpected, check the `[route]` line — it shows the raw
`intent` + `args` the router acted on.
