"""General two-body state propagation — the analytical fallback engine.

The baseline propagator (orchestrator/propagate.py) only does *circular* orbits
by altitude. A maneuver puts a body on an *elliptical* drift orbit, so to render
a maneuvered trajectory with no GMAT we need to propagate an arbitrary state
(r, v) forward under two-body gravity. This module does that with the universal-
variable formulation (Curtis, Algorithm 3.4), which is robust for near-circular
orbits where classical-element propagation goes singular.

It is deliberately general: build a state, apply an impulsive Δv (just add to the
velocity), and sample the coast. GMAT (orchestrator/gmat_engine.py, when present)
replaces this with a real force model; the geometry it feeds the globe is the
same shape.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MU = 398600.4418      # Earth GM, km^3/s^2


def _stumpff_c(z: float) -> float:
    if z > 1e-9:
        return (1 - math.cos(math.sqrt(z))) / z
    if z < -1e-9:
        return (math.cosh(math.sqrt(-z)) - 1) / (-z)
    return 0.5


def _stumpff_s(z: float) -> float:
    if z > 1e-9:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / sz ** 3
    if z < -1e-9:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / sz ** 3
    return 1.0 / 6.0


def kepler_propagate(r0: np.ndarray, v0: np.ndarray, dt: float,
                     mu: float = MU) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a Cartesian state (km, km/s) forward by dt seconds under two-body
    gravity. Universal-variable Kepler; returns (r, v)."""
    r0 = np.asarray(r0, float); v0 = np.asarray(v0, float)
    r0n = float(np.linalg.norm(r0)); v0n = float(np.linalg.norm(v0))
    vr0 = float(np.dot(r0, v0)) / r0n
    alpha = 2.0 / r0n - v0n ** 2 / mu           # 1/a
    sqmu = math.sqrt(mu)
    chi = sqmu * abs(alpha) * dt                # initial guess
    for _ in range(200):
        z = alpha * chi * chi
        c, s = _stumpff_c(z), _stumpff_s(z)
        f = (r0n * vr0 / sqmu * chi * chi * c
             + (1 - alpha * r0n) * chi ** 3 * s
             + r0n * chi - sqmu * dt)
        df = (r0n * vr0 / sqmu * chi * (1 - alpha * chi * chi * s)
              + (1 - alpha * r0n) * chi * chi * c + r0n)
        ratio = f / df
        chi -= ratio
        if abs(ratio) < 1e-8:
            break
    z = alpha * chi * chi
    c, s = _stumpff_c(z), _stumpff_s(z)
    ff = 1 - chi * chi / r0n * c
    gg = dt - 1.0 / sqmu * chi ** 3 * s
    r = ff * r0 + gg * v0
    rn = float(np.linalg.norm(r))
    fdot = sqmu / (rn * r0n) * (alpha * chi ** 3 * s - chi)
    gdot = 1 - chi * chi / rn * c
    v = fdot * r0 + gdot * v0
    return r, v


def circular_state(radius_km: float, phase_deg: float,
                   inc_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian state on a circular orbit, matching propagate.py's plane
    convention (plane inclined about the x-axis), at true-anomaly `phase_deg`.
    Returns (r, v) so two bodies at different phases are consistent on the globe."""
    a = radius_km
    w = math.sqrt(MU / a ** 3)
    th, inc = math.radians(phase_deg), math.radians(inc_deg)
    r = a * np.array([math.cos(th), math.sin(th) * math.cos(inc), math.sin(th) * math.sin(inc)])
    v = a * w * np.array([-math.sin(th), math.cos(th) * math.cos(inc), math.cos(th) * math.sin(inc)])
    return r, v


def sample_states(r0: np.ndarray, v0: np.ndarray, duration_s: float,
                  step_s: float = 60.0) -> list[tuple[float, float, float, float]]:
    """Sample a coasting trajectory as [(t, x, y, z)] in km — the exact tuple shape
    propagate._write_oem consumes, so the globe renders it with no new plumbing."""
    n = max(2, int(duration_s / step_s))
    out = []
    for i in range(n + 1):
        t = i * step_s
        r, _ = kepler_propagate(r0, v0, t)
        out.append((t, float(r[0]), float(r[1]), float(r[2])))
    return out


def apply_dv_tangential(v: np.ndarray, new_speed_kms: float) -> np.ndarray:
    """Return a velocity along the same direction as `v` but with a new magnitude
    (an along-track impulsive burn). Used to drop into / out of a drift orbit."""
    v = np.asarray(v, float)
    vhat = v / float(np.linalg.norm(v))
    return vhat * new_speed_kms
