"""Closed-form maneuver planning — deterministic astrodynamics, no GMAT needed.

This module computes the Δv budget and burn schedule for a rendezvous/proximity
operation from first-principles orbital mechanics. It is deliberately GMAT-free:
the numbers are pure Python (numpy) so they are identical every run and are
unit-testable against textbook values. GMAT (orchestrator/gmat_engine.py) then
PROPAGATES the resulting maneuvered trajectory at higher fidelity; the two
concerns are separate.

Nothing here is specific to any scenario. It works in terms of orbit radius (any
regime), phase/separation angle, standoff and hold distances, and timing. A GEO
refueling approach is one *instance* of these general routines, not a special
case in the code.

Two building blocks:
  - Coplanar phasing (`phasing_transfer`): close an along-track / longitude gap
    between two objects on the same circular orbit by dropping into a drift orbit
    for k revolutions, then circularizing — two impulses. Vis-viva throughout.
  - Clohessy-Wiltshire proximity (`cw_two_impulse`): a two-impulse transfer in
    the Hill (LVLH) frame from one relative position to another over a fixed
    time, the standard model for close-in approach (V-bar / R-bar).

`plan_rendezvous` composes them into a full plan: a phasing leg to close the
big gap, then a CW leg from a standoff point to a hold point.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MU = 398600.4418      # Earth GM, km^3/s^2
RE = 6378.137         # Earth equatorial radius, km


# -- basic two-body relations ----------------------------------------------

def circular_speed(r_km: float) -> float:
    """Speed on a circular orbit of radius r (km) -> km/s."""
    return math.sqrt(MU / r_km)


def vis_viva(r_km: float, a_km: float) -> float:
    """Speed at radius r on an orbit of semi-major axis a (km) -> km/s."""
    return math.sqrt(MU * (2.0 / r_km - 1.0 / a_km))


def period(a_km: float) -> float:
    """Orbital period (s) for semi-major axis a (km)."""
    return 2.0 * math.pi * math.sqrt(a_km ** 3 / MU)


def mean_motion(a_km: float) -> float:
    """Mean motion n (rad/s) for semi-major axis a (km)."""
    return math.sqrt(MU / a_km ** 3)


# -- coplanar phasing (close an along-track / longitude gap) ----------------

def phasing_transfer(r_km: float, phase_deg: float, revs: int = 1) -> dict[str, Any]:
    """Two-impulse phasing to close a phase gap on a circular orbit.

    To catch up (or drop back) by `phase_deg` degrees relative to another object
    on the same circular orbit of radius `r_km`, the chaser enters a phasing
    (drift) orbit whose period differs so that it gains exactly the phase over
    `revs` of its own revolutions, then re-circularizes. Positive phase_deg = the
    chaser must gain angle (target is ahead) -> shorter period -> lower orbit.

    Returns dv1/dv2 (km/s), total_dv, transfer_time (s), and the drift SMA. This
    is regime-agnostic: GEO relocation is r_km≈42164 with a small phase; a LEO
    catch-up is the same math at a smaller radius.
    """
    a_circ = r_km
    t_circ = period(a_circ)
    dtheta = math.radians(phase_deg)
    # Phasing-orbit period so the chaser gains `dtheta` over `revs` revs.
    # It must traverse (2*pi*revs - dtheta) while the target does 2*pi*revs
    # (if gaining), i.e. the phasing period is scaled accordingly.
    t_phase = t_circ * (1.0 - dtheta / (2.0 * math.pi * revs))
    if t_phase <= 0:
        return {"error": f"phase {phase_deg} deg too large to close in {revs} rev(s)"}
    a_phase = (MU * (t_phase / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)
    v_circ = circular_speed(r_km)
    v_phase = vis_viva(r_km, a_phase)          # speed at the shared radius (burn point)
    dv1 = abs(v_phase - v_circ)                # enter drift orbit
    dv2 = abs(v_circ - v_phase)                # circularize back
    return {
        "dv1_kms": dv1, "dv2_kms": dv2, "total_dv_kms": dv1 + dv2,
        "transfer_time_s": t_phase * revs,
        "drift_sma_km": a_phase, "circ_sma_km": a_circ,
    }


# -- Clohessy-Wiltshire relative motion (proximity) -------------------------

def cw_stm(n: float, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clohessy-Wiltshire state-transition sub-matrices for a circular reference
    orbit of mean motion n over time t. Frame: x=radial, y=along-track, z=cross.
    Returns (Phi_rr, Phi_rv, Phi_vr, Phi_vv), each 3x3."""
    s, c, nt = math.sin(n * t), math.cos(n * t), n * t
    phi_rr = np.array([[4 - 3 * c, 0, 0],
                       [6 * (s - nt), 1, 0],
                       [0, 0, c]])
    phi_rv = np.array([[s / n, 2 * (1 - c) / n, 0],
                       [-2 * (1 - c) / n, (4 * s - 3 * nt) / n, 0],
                       [0, 0, s / n]])
    phi_vr = np.array([[3 * n * s, 0, 0],
                       [-6 * n * (1 - c), 0, 0],
                       [0, 0, -n * s]])
    phi_vv = np.array([[c, 2 * s, 0],
                       [-2 * s, 4 * c - 3, 0],
                       [0, 0, c]])
    return phi_rr, phi_rv, phi_vr, phi_vv


def cw_two_impulse(r0: np.ndarray, rf: np.ndarray, n: float, t: float,
                   v0: np.ndarray | None = None,
                   vf: np.ndarray | None = None) -> dict[str, Any]:
    """Two-impulse CW transfer from relative position r0 to rf over time t.

    r0, rf are Hill-frame positions (km). v0 is the chaser's current relative
    velocity (km/s, default 0 = co-orbiting); vf the desired final relative
    velocity (default 0 = station-keep at rf). Returns dv1/dv2 vectors + mags and
    total. Raises via 'error' if the transfer geometry is singular for this t.
    """
    v0 = np.zeros(3) if v0 is None else np.asarray(v0, float)
    vf = np.zeros(3) if vf is None else np.asarray(vf, float)
    r0 = np.asarray(r0, float); rf = np.asarray(rf, float)
    phi_rr, phi_rv, phi_vr, phi_vv = cw_stm(n, t)
    if abs(np.linalg.det(phi_rv)) < 1e-12:
        return {"error": f"CW transfer singular at t={t:.1f}s (Phi_rv not invertible); choose a different time"}
    # Required initial velocity to arrive at rf after t.
    v0_plus = np.linalg.solve(phi_rv, rf - phi_rr @ r0)
    dv1 = v0_plus - v0
    # Velocity on arrival, and the burn to match the desired final velocity.
    vf_minus = phi_vr @ r0 + phi_vv @ v0_plus
    dv2 = vf - vf_minus
    return {
        "dv1_vec_kms": dv1.tolist(), "dv2_vec_kms": dv2.tolist(),
        "dv1_kms": float(np.linalg.norm(dv1)), "dv2_kms": float(np.linalg.norm(dv2)),
        "total_dv_kms": float(np.linalg.norm(dv1) + np.linalg.norm(dv2)),
        "transfer_time_s": t,
    }


# -- high-level composition -------------------------------------------------

def plan_rendezvous(target_radius_km: float, phase_deg: float,
                    standoff_km: float = 20.0, hold_km: float = 0.5,
                    approach: str = "v-bar", phasing_revs: int = 1,
                    approach_time_s: float | None = None) -> dict[str, Any]:
    """A general two-leg rendezvous plan for a chaser approaching a target on a
    circular orbit of radius `target_radius_km`, separated by `phase_deg`.

    Leg 1 (phasing): close the phase gap via a drift orbit -> arrives at a
      standoff point `standoff_km` behind (v-bar) or below (r-bar) the target.
    Leg 2 (CW approach): transfer from the standoff point to a hold point
      `hold_km` from the target and station-keep there.

    `approach` selects the Hill-frame axis: 'v-bar' = along-track (y),
    'r-bar' = radial (x). Returns the per-burn schedule + total Δv. All inputs
    are general; a GEO refueling approach is target_radius_km≈42164, small
    phase, v-bar approach — just an instance.
    """
    n = mean_motion(target_radius_km)
    if approach_time_s is None:
        approach_time_s = 0.25 * period(target_radius_km)   # a quarter-rev default

    axis = 1 if approach.lower().startswith("v") else 0     # y (v-bar) or x (r-bar)
    r_standoff = np.zeros(3); r_standoff[axis] = -standoff_km
    r_hold = np.zeros(3); r_hold[axis] = -hold_km

    phasing = phasing_transfer(target_radius_km, phase_deg, revs=phasing_revs)
    if "error" in phasing:
        return phasing
    cw = cw_two_impulse(r_standoff, r_hold, n, approach_time_s)
    if "error" in cw:
        return cw

    burns = [
        {"t_s": 0.0, "dv_kms": phasing["dv1_kms"], "leg": "phasing-enter",
         "note": f"enter drift orbit (SMA {phasing['drift_sma_km']:.0f} km)"},
        {"t_s": phasing["transfer_time_s"], "dv_kms": phasing["dv2_kms"], "leg": "phasing-circularize",
         "note": f"circularize at standoff {standoff_km:g} km ({approach})"},
        {"t_s": phasing["transfer_time_s"], "dv_kms": cw["dv1_kms"], "leg": "approach-depart",
         "note": f"depart standoff toward hold {hold_km:g} km"},
        {"t_s": phasing["transfer_time_s"] + cw["transfer_time_s"], "dv_kms": cw["dv2_kms"],
         "leg": "approach-arrive", "note": "null relative velocity, station-keep"},
    ]
    total = phasing["total_dv_kms"] + cw["total_dv_kms"]
    return {
        "target_radius_km": target_radius_km, "phase_deg": phase_deg,
        "approach": approach, "standoff_km": standoff_km, "hold_km": hold_km,
        "phasing": phasing, "cw": cw, "burns": burns,
        "total_dv_kms": total, "total_dv_ms": total * 1000.0,
        "total_time_s": burns[-1]["t_s"],
    }
