"""Observer synthetic-view adapter — "show me what the observer sees".

Renders the scene from an observer's line of sight to one or more targets: each
LOS becomes an RA/Dec, the targets are drawn as point sources on a synthetic star
field, and the result is a FITS (with a valid WCS) plus an overlay carrying the
target RA/Dec. The hub opens the FITS in the sky pane and marks the targets.

General by construction — any observer, any target list, any epoch, any field of
view. It reads whatever ephemerides are on the globe (the OEMs written by the
propagate / maneuver tools). With no arguments it picks the lowest orbit as the
observer and treats every other body as a target, so after an RPO scene "show me
what the observer sees" just works; the GEO refueling case is one instance, not a
hardcoded path.

Compute lives in orchestrator/observer_render.py; this adapter only resolves which
OEMs to use, calls the renderer, and shapes the result into an image artifact.
"""

from __future__ import annotations

import os
from typing import Any

from .contract import Tool, ToolResult
from .. import observer_render
from .. import propagate

EPHEMERIS_DIR = os.environ.get("EPHEMERIS_DIR", "./data/ephemeris")
SCENE_DIR = os.environ.get("SCENE_DIR", "./data/scenes")


def _oem_index() -> dict[str, str]:
    """Map each current OEM's object name (basename without .oem) to its path."""
    out: dict[str, str] = {}
    for path in propagate.list_oems(EPHEMERIS_DIR):
        name = os.path.splitext(os.path.basename(path))[0]
        out[name] = path
    return out


def _match(index: dict[str, str], query: str) -> str | None:
    """Resolve a caller-supplied name to an OEM: exact (case-insensitive) first,
    then a unique substring match. Returns None if nothing or several match."""
    q = (query or "").strip().lower()
    if not q:
        return None
    for name in index:
        if name.lower() == q:
            return name
    hits = [name for name in index if q in name.lower() or name.lower() in q]
    return hits[0] if len(hits) == 1 else None


class ObserverView(Tool):
    name = "render_observer_view"

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Render what an observer sees looking at one or more target "
                "satellites: the line of sight becomes RA/Dec and the targets are "
                "drawn as point sources on a synthetic star field (a FITS image). "
                "Use for 'show me what the observer sees', 'render the observer "
                "view', 'what does the telescope see'."
            ),
            # No arguments are advertised. The small model fills any optional arg it
            # is shown with junk (live testing produced observer='Earth', fov_deg=10,
            # npix=1024 from a request that named none of them) — the same failure the
            # maneuver tool fixed by removing name params from its schema. run() reads
            # observer / targets / time_s / fov_deg / npix from args ONLY for
            # programmatic and macro callers; a voice request uses the robust defaults
            # (lowest orbit = observer, every other body = target, auto-framed FOV,
            # 512 px). Explicit observer/FOV by voice would need deterministic text
            # recovery (x_recover), not a bare advertised arg.
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

    def run(self, args: dict[str, Any]) -> ToolResult:
        index = _oem_index()
        if len(index) < 2:
            return ToolResult(error=(
                "Need at least two objects on the globe (an observer and a target). "
                "Model an orbit or an RPO scene first, then ask for the observer view."))

        # Observer: the caller's choice if it resolves, else the lowest orbit.
        observer = _match(index, str(args.get("observer") or "")) if args.get("observer") else None
        if observer is None:
            observer = min(index, key=lambda n: observer_render.mean_radius_km(index[n]))

        # Targets: the caller's list if given, else every other body on the globe.
        raw_targets = args.get("targets")
        target_names: list[str] = []
        if isinstance(raw_targets, (list, tuple)):
            for q in raw_targets:
                m = _match(index, str(q))
                if m and m != observer and m not in target_names:
                    target_names.append(m)
        if not target_names:
            target_names = [n for n in index if n != observer]
        if not target_names:
            return ToolResult(error="No target distinct from the observer to look at.")

        # Epoch: all TARGETS are read at one common time so their mutual separation
        # is meaningful (a rendezvous ends as an unresolved pair only when both
        # bodies are sampled at the same instant). Default to the latest epoch every
        # target covers = min of the target end-times, so the shortest-lived target
        # sits at its final state (e.g. the servicer at its hold point) and none is
        # extrapolated. The observer is read at the same time, clamped to its own
        # span (its LEO orbit is far shorter than a multi-hour maneuver; its exact
        # phase is not the teaching point, the target pair is).
        time_s = args.get("time_s")
        if isinstance(time_s, (int, float)):
            time_s = float(time_s)
        else:
            ends = [propagate._parse_oem(index[n])[2] for n in target_names]
            time_s = min(ends) if ends else None

        obs_pos = observer_render.read_state_at(index[observer], time_s)
        if obs_pos is None:
            return ToolResult(error=f"Could not read the observer state for {observer!r}.")
        targets = []
        for n in target_names:
            pos = observer_render.read_state_at(index[n], time_s)
            if pos is not None:
                targets.append({"name": n, "pos_km": pos})
        if not targets:
            return ToolResult(error="Could not read any target state.")

        fov = args.get("fov_deg")
        fov = float(fov) if isinstance(fov, (int, float)) and float(fov) > 0 else None
        npix = int(args.get("npix") or 512)
        npix = max(64, min(2048, npix))               # keep the render bounded

        out = observer_render.render_observer_view(
            obs_pos, targets, SCENE_DIR, basename="observer_view", fov_deg=fov, npix=npix)
        ov = out["overlay"]

        # A short, speakable summary the model grounds its spoken answer on.
        parts = [f"{t['name']} at RA {t['ra']:.2f}, Dec {t['dec']:.2f} degrees "
                 f"({t['range_km']:.0f} km)" for t in ov["targets"]]
        sep = ov.get("separation_arcsec", 0.0)
        sep_txt = (f" They are {sep:.0f} arcsec apart." if len(parts) > 1 and sep else "")
        text = (f"From {observer}, the observer sees " + "; ".join(parts) + "."
                + sep_txt + f" Field of view {ov['fov_deg']:.2f} degrees.")

        return ToolResult(
            text=text,
            data={"observer": observer, "targets": ov["targets"],
                  "center": ov["center"], "fov_deg": ov["fov_deg"],
                  "separation_arcsec": sep},
            artifact={"kind": "image",
                      "fits": _served(out["fits"]),
                      "overlay": _served(out["overlay_path"])},
        )

    def render(self, result: ToolResult) -> dict[str, Any]:
        return result.artifact or {}


def _served(path: str) -> str:
    """Map a scene-dir path to the browser-served /data/scenes URL."""
    return "/data/scenes/" + os.path.basename(path) if path else ""
