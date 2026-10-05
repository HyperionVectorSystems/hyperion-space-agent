"""Router macros — deterministic multi-tool procedures.

A macro is a fixed Python sequence of tool dispatches the router runs as one
intent. This is how a multi-step workflow stays deterministic: the orchestrator
expands the intent into a known pipeline of validated tool calls, rather than
letting the model improvise the sequence.

`model_rpo` composes existing GENERAL tools (propagate_orbit + plan_maneuver) into
a full rendezvous-and-proximity scene. It is a convenience composition, not a new
capability — every tool it calls is independently callable, and its parameters
default to a GEO servicing scene but are overridable, so nothing is hardcoded.
"""

from __future__ import annotations

from typing import Any

from .tools import ToolResult


def model_rpo(router, args: dict[str, Any]) -> ToolResult:
    """Place a LEO observer, a target, and a servicer that maneuvers to rendezvous.
    Defaults describe a GEO servicing scene; all are overridable via args."""
    args = args or {}

    def num(key: str, default: float) -> float:
        try:
            return float(args.get(key, default))
        except (TypeError, ValueError):
            return default

    target_alt = num("target_altitude_km", 35786.0)
    phase = num("phase_deg", 6.0)
    if phase <= 0:                 # a rendezvous needs a gap to close; the model
        phase = 6.0                # sometimes zeroes it — restore a meaningful default
    inc = num("inclination_deg", 0.0)
    obs_alt = num("observer_altitude_km", 550.0)
    if obs_alt <= 0 or obs_alt >= target_alt:   # a LEO observer sits BELOW the target;
        obs_alt = 550.0                          # the model sometimes copies the target alt
    obs_inc = num("observer_inclination_deg", 51.6)

    # 1) LEO observer. 2) target + maneuvering servicer (plan_maneuver places both).
    obs = router.dispatch("propagate_orbit",
                          {"altitude_km": obs_alt, "inclination": obs_inc, "duration_min": 95})
    man = router.dispatch("plan_maneuver",
                          {"target_altitude_km": target_alt, "phase_deg": phase,
                           "inclination_deg": inc})
    if man.error:
        return man

    data = dict(man.data)
    data["observer_altitude_km"] = obs_alt
    # Reuse plan_maneuver's rendered artifact (dispatch already ran its render(), so
    # it carries the orbit-details + burn-plan tables) and append the observer as a
    # third row of the orbit-details table (same 5-column shape).
    art = dict(man.artifact or {"kind": "viz", "czml_source": "newest"})
    tables = [dict(t) for t in art.get("tables", [])]
    if tables:
        od = obs.data or {}
        obs_row = ["observer", round(obs_alt), od.get("period_min", ""),
                   od.get("inclination_deg", ""), "LEO observer"]
        tables[0] = {**tables[0], "rows": tables[0]["rows"] + [obs_row]}
        art["tables"] = tables
    return ToolResult(
        text=(f"Modeled an RPO scene: a LEO observer at {obs_alt:g} km altitude, plus "
              + (man.text or "the target and servicer are on the globe.")),
        data=data,
        artifact=art,
    )


# Advertised to the router's slot-filler so the model can route to it and fill args.
model_rpo.description = (
    "Model a full rendezvous & proximity operations (RPO) scene: place a LEO "
    "observer, a target satellite, and a servicer that maneuvers to rendezvous "
    "with it. Use for 'model an RPO', 'refueling scenario', 'rendezvous demo'."
)
model_rpo.parameters = {
    "type": "object",
    "properties": {
        "target_altitude_km": {"type": "number", "description":
            "target circular orbit altitude in km (default 35786 = GEO)"},
        "phase_deg": {"type": "number", "description":
            "initial angular separation to close, degrees (default 6)",
            "x_recover": {"pattern": r"(\d+(?:\.\d+)?)\s*-?\s*deg", "nonpositive": True}},
        "inclination_deg": {"type": "number", "description": "orbit inclination in degrees (default 0)"},
        "observer_altitude_km": {"type": "number", "description": "LEO observer altitude in km (default 550)"},
    },
}


def default_macros() -> dict[str, Any]:
    """The macros registered on the router by default."""
    return {"model_rpo": model_rpo}
