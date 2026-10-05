"""Maneuver propagation engine — GMAT when available, two-body fallback always.

Given a maneuver plan (orchestrator/maneuver.py) and a body's starting orbit,
this realizes the *maneuvered trajectory* and writes a CCSDS OEM the globe reads.

Two engines behind one call:
  - Fallback (always present): the analytical two-body propagator
    (orchestrator/twobody.py). No external anything — so the RPO exemplar runs on
    a clean clone with no GMAT install.
  - GMAT (when GMAT_BIN/GMAT_HOME points at a real install): a subprocess seam —
    gmat_engine hands a Cartesian state to orchestrator/gmat_runner.py, which runs
    inside a GMAT-env child (PYTHONPATH/LD_LIBRARY_PATH set at launch, heavy native
    libs isolated from the orchestrator venv), propagates with a real force model,
    and writes the OEM. Mirrors how satsim_adapter isolates its TensorFlow venv.

The GMAT branch is gated by gmat_available(); until an install is wired and its
gmatpy recipe validated on the host, propagate_maneuver() runs the fallback. The
geometry handed to the globe is the same shape either way — GMAT is the fidelity
upgrade, not a requirement (the repo's standing doctrine).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import numpy as np

from . import propagate
from . import twobody
from . import maneuver

GMAT_BIN = os.environ.get("GMAT_BIN", "")     # a GmatConsole path OR the bin dir
GMAT_HOME = os.environ.get("GMAT_HOME", "")   # optional: the GMAT bin dir directly
EPOCH = propagate.EPOCH


def gmat_bin_dir() -> str | None:
    """Resolve the GMAT bin directory that contains gmatpy/. Accepts either
    GMAT_HOME (the bin dir) or GMAT_BIN (a dir, or a GmatConsole file whose parent
    is the bin dir). Returns None if it cannot be located with a gmatpy package."""
    for cand in (GMAT_HOME, GMAT_BIN):
        if not cand:
            continue
        d = cand if os.path.isdir(cand) else os.path.dirname(cand)
        if d and os.path.isdir(os.path.join(d, "gmatpy")):
            return d
    return None


def gmat_available() -> bool:
    return gmat_bin_dir() is not None


def propagate_maneuver(radius_km: float, phase_deg: float, inc_deg: float,
                       plan: dict[str, Any], ephemeris_dir: str,
                       name: str = "servicer", step_s: float = 300.0,
                       duration_s: float | None = None,
                       engine: str = "auto") -> dict[str, Any]:
    """Realize a maneuvered trajectory and write its OEM.

    The body starts on a circular orbit of `radius_km` at true anomaly
    `-phase_deg` (i.e. `phase_deg` behind the reference at anomaly 0), drops into
    the plan's drift orbit, and coasts for the phasing time — the visible
    'close the gap and rendezvous' arc. `engine`: 'auto' uses GMAT if available
    else fallback; 'fallback' forces two-body; 'gmat' requires GMAT.
    """
    if duration_s is None:
        duration_s = float(plan.get("phasing", {}).get("transfer_time_s") or propagate_period(radius_km))
    drift_sma = float(plan.get("phasing", {}).get("drift_sma_km") or radius_km)

    # Servicer state entering the drift orbit: circular at -phase, then the
    # along-track burn that lowers it to the drift orbit (apogee at radius_km).
    r0, v0 = twobody.circular_state(radius_km, -phase_deg, inc_deg)
    v_drift = maneuver.vis_viva(radius_km, drift_sma)
    v0 = twobody.apply_dv_tangential(v0, v_drift)

    use_gmat = engine == "gmat" or (engine == "auto" and gmat_available())
    if use_gmat:
        try:
            res = _run_gmat(r0, v0, duration_s, step_s, ephemeris_dir, name)
            res["engine"] = "GMAT R2025a"
            return res
        except Exception as e:
            if engine == "gmat":
                return {"error": f"GMAT engine failed: {e}"}
            # auto: fall through to the analytical fallback

    samples = twobody.sample_states(r0, v0, duration_s, step_s)
    oem = propagate._write_oem(samples, {"name": name, "frame": "INERTIAL"}, ephemeris_dir)
    return {"oem": oem, "engine": "analytical (two-body)", "samples": len(samples), "name": name}


def propagate_period(radius_km: float) -> float:
    return maneuver.period(radius_km)


def propagate_circular(radius_km: float, phase_deg: float, inc_deg: float,
                       duration_s: float, ephemeris_dir: str, name: str,
                       step_s: float = 300.0) -> dict[str, Any]:
    """Write a named, phase-controlled circular-orbit OEM (via two-body). Used to
    place a target/observer at a chosen true anomaly so multiple bodies share one
    consistent inertial frame on the globe. General — not scenario-specific."""
    r0, v0 = twobody.circular_state(radius_km, phase_deg, inc_deg)
    samples = twobody.sample_states(r0, v0, duration_s, step_s)
    oem = propagate._write_oem(samples, {"name": name, "frame": "INERTIAL"}, ephemeris_dir)
    return {"oem": oem, "samples": len(samples), "name": name}


# -- GMAT subprocess seam (validated once an install is wired) ---------------

def _run_gmat(r0: np.ndarray, v0: np.ndarray, duration_s: float, step_s: float,
              ephemeris_dir: str, name: str) -> dict[str, Any]:
    """Hand a Cartesian state to the GMAT-env child, which propagates with a real
    force model and writes the OEM. Raises on any failure so 'auto' can fall back."""
    bin_dir = gmat_bin_dir()
    if not bin_dir:
        raise RuntimeError("no GMAT install (set GMAT_HOME or GMAT_BIN)")
    spec = {
        "name": name, "ephemeris_dir": os.path.abspath(ephemeris_dir),
        "epoch": EPOCH,
        "r_km": [float(x) for x in r0], "v_kms": [float(x) for x in v0],
        "duration_s": float(duration_s), "step_s": float(step_s),
    }
    env = dict(os.environ)
    env["PYTHONPATH"] = bin_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = bin_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    runner = os.path.join(os.path.dirname(__file__), "gmat_runner.py")
    proc = subprocess.run([sys.executable, runner, json.dumps(spec)],
                          cwd=bin_dir, env=env, capture_output=True,
                          text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"gmat_runner exited {proc.returncode}: {proc.stderr[-500:]}")
    # The runner prints one JSON line with the written OEM path.
    line = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    if not line:
        raise RuntimeError(f"gmat_runner produced no result: {proc.stdout[-300:]}")
    out = json.loads(line[-1])
    if "error" in out:
        raise RuntimeError(out["error"])
    return out
