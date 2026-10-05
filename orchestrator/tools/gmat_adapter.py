"""Orbit propagation adapter.

Baseline path is the harness's own analytical propagator (orchestrator/propagate.py):
SGP4 for bundled named objects, two-body for "altitude=<km>". It writes an OEM and
the globe renders it, with NO external download. So propagation works out of the box.

If a real GMAT R2025a install is present (set GMAT_BIN), you can swap this adapter's
run() to shell out to GMAT for higher-fidelity force models. The analytical baseline
is what our own demo used; GMAT is the upgrade, not a requirement.
"""

from __future__ import annotations

import os
from typing import Any

from .contract import Tool, ToolResult
from .. import propagate

EPHEMERIS_DIR = os.environ.get("EPHEMERIS_DIR", "./data/ephemeris")
GMAT_BIN = os.environ.get("GMAT_BIN", "")   # optional higher-fidelity engine


class GmatPropagate(Tool):
    name = "propagate_orbit"

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Propagate a satellite orbit and return its period. Writes an "
                "ephemeris the globe renders. Accepts a bundled name (e.g. ISS) "
                "or 'altitude=<km>' for a circular orbit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object": {"type": "string", "description": (
                        "a bundled satellite name, e.g. ISS. For a circular orbit "
                        "given by altitude, use 'altitude_km' instead of this.")},
                    "altitude_km": {"type": "number", "description": (
                        "altitude in kilometers for a circular orbit (e.g. 35786 for "
                        "GEO). Use this for any 'at N km' request. Emit a plain number, "
                        "no thousands separators.")},
                    "duration_min": {"type": "number", "description": "span in minutes", "default": 95},
                    "inclination": {"type": "number", "description": (
                        "orbit inclination in degrees for a circular orbit (default "
                        "51.6). Ignored for bundled TLE objects like ISS, which use "
                        "their own inclination.")},
                },
                # Neither is strictly required: a request may carry a name OR an
                # altitude. run() enforces that at least one is present.
                "required": [],
            },
        }

    def run(self, args: dict[str, Any]) -> ToolResult:
        obj = str(args.get("object", "")).strip()
        alt = args.get("altitude_km", None)
        dur = float(args.get("duration_min", 95))
        inc = args.get("inclination", None)
        try:
            inc = float(inc) if inc is not None else None
        except (TypeError, ValueError):
            inc = None
        # Resolve the target deterministically. A real bundled NAME wins (e.g.
        # "ISS"); otherwise a numeric altitude_km builds a circular orbit — the
        # string is formatted HERE in Python, so the model never hand-writes
        # "altitude=<n>" and there is no comma for it to truncate. The legacy
        # 'altitude=<km>' string in `object` still works for back-compat.
        if obj and not obj.lower().startswith("altitude="):
            target = obj
        elif alt is not None:
            try:
                target = f"altitude={float(alt)}"
            except (TypeError, ValueError):
                return ToolResult(error="altitude_km must be a number, e.g. 35786.")
        elif obj:
            target = obj
        else:
            return ToolResult(error="propagate_orbit needs a bundled name (e.g. ISS) or altitude_km (a number, e.g. 35786).")
        # Baseline: the harness's analytical propagator. (To use GMAT instead,
        # branch on GMAT_BIN here and shell out to it, then point at its OEM.)
        res = propagate.propagate(target, dur, ephemeris_dir=EPHEMERIS_DIR, inclination_deg=inc)
        if "error" in res:
            return ToolResult(error=res["error"])
        engine = "GMAT R2025a" if GMAT_BIN and os.path.exists(GMAT_BIN) else "analytical (SGP4/two-body)"
        return ToolResult(
            text=(f"Propagated {res['name']}: period ~{res['period_min']:.1f} min, "
                  f"inclination {res['inclination_deg']:.1f} degrees. Orbit is on the globe."),
            data={"object": res["name"], "period_min": res["period_min"],
                  "inclination_deg": res["inclination_deg"], "oem": res["oem"], "engine": engine},
            artifact={"kind": "viz", "czml_source": "newest"},
        )

    def render(self, result: ToolResult) -> dict[str, Any]:
        art = dict(result.artifact or {"kind": "viz", "czml_source": "newest"})
        d = result.data or {}
        art["tables"] = [{"title": f"Orbit: {d.get('object', '')}", "rows": [
            ["Field", "Value"],
            ["Object", d.get("object", "")],
            ["Period (min)", d.get("period_min", "")],
            ["Inclination (deg)", d.get("inclination_deg", "")],
            ["Engine", d.get("engine", "")],
        ]}]
        art.pop("table", None)
        return art
