"""Rendezvous / proximity maneuver adapter.

Models a chaser approaching a target on a circular orbit: computes the Δv budget
and burn schedule (orchestrator/maneuver.py) and realizes both bodies' trajectories
(orchestrator/gmat_engine.py — GMAT if installed, two-body fallback otherwise),
writing OEMs the globe renders. General: any circular target orbit, any initial
separation, v-bar or r-bar approach. A GEO refueling approach is one instance —
nothing here is hardcoded to it.
"""

from __future__ import annotations

import os
from typing import Any

from .contract import Tool, ToolResult
from .. import maneuver
from .. import gmat_engine

EPHEMERIS_DIR = os.environ.get("EPHEMERIS_DIR", "./data/ephemeris")
RE = 6378.137     # Earth equatorial radius, km


def _num(value: Any, default: float) -> float:
    """Coerce to float; fall back to default on None/blank/non-numeric."""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pos(value: Any, default: float) -> float:
    """Like _num, but a non-positive value is treated as unset (the model zeroing
    an optional arg must not collapse the maneuver)."""
    v = _num(value, default)
    return v if v > 0 else default


def _approach(value: Any) -> str:
    """Canonicalize the approach axis to 'v-bar' or 'r-bar'. Only real tokens flip
    it; a word the model grabbed from the sentence (e.g. 'rendezvous') falls back
    to the v-bar default instead of silently mapping to r-bar."""
    a = str(value or "").strip().lower()
    if any(t in a for t in ("r-bar", "rbar", "r bar", "radial")):
        return "r-bar"
    if any(t in a for t in ("v-bar", "vbar", "v bar", "along")):
        return "v-bar"
    return "v-bar"


ORBIT_COLS = ["Satellite", "Altitude (km)", "Period (min)", "Incl (deg)", "Note"]


def orbit_details_table(d: dict[str, Any]) -> dict[str, Any]:
    """Orbit-details table for the target + chaser (the macro appends an observer
    row of the same shape). Reusable so both the tool and the macro share it."""
    alt, per, inc = d.get("target_altitude_km", ""), d.get("target_period_min", ""), d.get("inclination_deg", "")
    rows = [ORBIT_COLS,
            [d.get("target", "client"), alt, per, inc, "target"],
            [d.get("chaser", "servicer"), alt, per, inc, f"drift SMA {d.get('drift_sma_km', '')} km"]]
    return {"title": "Orbit details", "rows": rows}


def burn_plan_table(d: dict[str, Any]) -> dict[str, Any]:
    rows = [["Burn", "Time", "Δv", "Note"]]
    for b in d.get("burns", []):
        rows.append([b.get("leg", ""), f"{b.get('t_s', 0) / 3600:.1f} hr",
                     f"{b.get('dv_kms', 0) * 1000:.2f} m/s", b.get("note", "")])
    return {"title": "Burn plan", "rows": rows}


class PlanManeuver(Tool):
    name = "plan_maneuver"

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Model a rendezvous/proximity maneuver: a chaser closes an angular "
                "gap to a target on a circular orbit and approaches to a hold point. "
                "Returns the delta-v budget and draws both trajectories on the globe. "
                "Use for RPO / rendezvous / 'approach' / servicing requests."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_altitude_km": {"type": "number", "description":
                        "circular orbit altitude of the target in km (e.g. 35786 for GEO)"},
                    "phase_deg": {"type": "number", "description":
                        "initial angular separation the chaser must close, degrees (default 6)",
                        "x_recover": {"pattern": r"(\d+(?:\.\d+)?)\s*-?\s*deg", "nonpositive": True}},
                    "inclination_deg": {"type": "number", "description":
                        "orbit inclination in degrees (default 0)"},
                    "standoff_km": {"type": "number", "description":
                        "standoff distance before final approach, km (default 20)"},
                    "hold_km": {"type": "number", "description":
                        "final hold distance from the target, km (default 0.5)"},
                    "approach": {"type": "string", "description":
                        "'v-bar' (along-track) or 'r-bar' (radial) approach (default v-bar)"},
                    # target_name/chaser_name intentionally NOT exposed: the small model
                    # grabbed prompt nouns as names ("GEO", "rendezvous"). run() defaults
                    # them to client/servicer; validate_args drops any the model sends.
                },
                "required": ["target_altitude_km"],
            },
        }

    def run(self, args: dict[str, Any]) -> ToolResult:
        alt = args.get("target_altitude_km")
        if alt is None:
            return ToolResult(error="plan_maneuver needs target_altitude_km (e.g. 35786 for GEO).")
        radius = RE + float(alt)
        # The small model routinely zeroes/empties optional args, which overrides
        # a plain args.get(default). So every optional value is validated for
        # sensibility, not just presence: a rendezvous needs a non-zero gap, a
        # positive standoff, a hold inside the standoff, and non-empty names.
        phase = _pos(args.get("phase_deg"), 6.0)
        inc = _num(args.get("inclination_deg"), 0.0)
        standoff = _pos(args.get("standoff_km"), 20.0)
        hold = _pos(args.get("hold_km"), 0.5)
        if hold >= standoff:                       # hold must be closer than standoff
            hold = standoff * 0.05
        approach = _approach(args.get("approach"))
        target_name = (str(args.get("target_name") or "").strip() or "client")
        chaser_name = (str(args.get("chaser_name") or "").strip() or "servicer")

        plan = maneuver.plan_rendezvous(radius, phase, standoff, hold, approach)
        if "error" in plan:
            return ToolResult(error=f"maneuver planning failed: {plan['error']}")
        duration = plan["total_time_s"]

        # Place the target (phase 0) and the maneuvering chaser (starts phase_deg behind).
        gmat_engine.propagate_circular(radius, 0.0, inc, duration, EPHEMERIS_DIR, target_name)
        man = gmat_engine.propagate_maneuver(radius, phase, inc, plan, EPHEMERIS_DIR,
                                             name=chaser_name, engine="auto")
        if "error" in man:
            return ToolResult(error=man["error"])

        dv_ms = plan["total_dv_ms"]
        hours = duration / 3600.0
        return ToolResult(
            text=(f"Planned {approach} rendezvous: {chaser_name} closes {phase:g} degrees to "
                  f"{target_name} for {dv_ms:.1f} m/s total delta-v over {hours:.1f} hours "
                  f"({man['engine']}). Both orbits are on the globe."),
            data={"total_dv_ms": round(dv_ms, 2), "duration_hours": round(hours, 2),
                  "engine": man["engine"], "approach": approach, "phase_deg": phase,
                  "burns": plan["burns"], "target": target_name, "chaser": chaser_name,
                  "target_altitude_km": round(float(alt), 1), "inclination_deg": round(inc, 2),
                  "target_period_min": round(maneuver.period(radius) / 60.0, 1),
                  "drift_sma_km": round(plan["phasing"]["drift_sma_km"])},
            artifact={"kind": "viz", "czml_source": "newest"},
        )

    def render(self, result: ToolResult) -> dict[str, Any]:
        art = dict(result.artifact or {"kind": "viz", "czml_source": "newest"})
        d = result.data or {}
        art["title"] = (f"Rendezvous {d.get('chaser', '')} → {d.get('target', '')} · "
                        f"{d.get('approach', '')} · {d.get('phase_deg', '')}° · "
                        f"total Δv {d.get('total_dv_ms', '')} m/s over "
                        f"{d.get('duration_hours', '')} hr · {d.get('engine', '')}")
        art["tables"] = [orbit_details_table(d), burn_plan_table(d)]
        art.pop("table", None)
        return art
