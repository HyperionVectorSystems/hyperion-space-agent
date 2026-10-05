"""twobody.py + gmat_engine.py (fallback) tests. No GMAT, no network.
Run: ./.venv/bin/python -m pytest tests/test_engine.py -q
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from orchestrator import twobody as TB
from orchestrator import maneuver as M
from orchestrator import gmat_engine as E


# -- two-body propagator ----------------------------------------------------

def test_circular_state_is_circular():
    r, v = TB.circular_state(42164.0, 30.0, 0.0)
    assert abs(np.linalg.norm(r) - 42164.0) < 1e-6
    assert abs(np.linalg.norm(v) - M.circular_speed(42164.0)) < 1e-9


def test_kepler_returns_to_start_after_one_period():
    # A circular orbit propagated one full period returns to its start point.
    r0, v0 = TB.circular_state(42164.0, 0.0, 0.0)
    T = M.period(42164.0)
    r, v = TB.kepler_propagate(r0, v0, T)
    assert np.linalg.norm(r - r0) < 1.0          # < 1 km after a full GEO rev
    assert np.linalg.norm(v - v0) < 1e-3


def test_kepler_conserves_energy_on_ellipse():
    # Lower the speed at apogee -> ellipse; specific energy is conserved.
    r0, v0 = TB.circular_state(42164.0, 0.0, 0.0)
    v0 = TB.apply_dv_tangential(v0, M.vis_viva(42164.0, 41694.0))
    def energy(r, v):
        return 0.5 * np.dot(v, v) - TB.MU / np.linalg.norm(r)
    e0 = energy(r0, v0)
    r1, v1 = TB.kepler_propagate(r0, v0, 20000.0)
    assert abs(energy(r1, v1) - e0) < 1e-6


# -- engine fallback --------------------------------------------------------

@pytest.mark.skipif(E.gmat_available(), reason="GMAT is wired here (GMAT_HOME set); this asserts the absent case")
def test_gmat_not_available_without_env():
    # With no GMAT_HOME/GMAT_BIN pointing at a gmatpy install, fallback is used.
    assert E.gmat_available() in (False,)          # no env in the test process


def _radii_from_oem(path):
    radii = []
    with open(path) as f:
        past = False
        for line in f:
            line = line.strip()
            if line == "META_STOP":
                past = True; continue
            if past and line and line[0].isdigit():
                _, x, y, z = (float(p) for p in line.split())
                radii.append(math.sqrt(x * x + y * y + z * z))
    return radii


def test_propagate_maneuver_fallback_writes_oem_and_drifts():
    plan = M.plan_rendezvous(42164.0, phase_deg=6.0, standoff_km=20.0, hold_km=0.5)
    d = tempfile.mkdtemp()
    res = E.propagate_maneuver(42164.0, phase_deg=6.0, inc_deg=0.1, plan=plan,
                               ephemeris_dir=d, name="servicer", step_s=300.0,
                               engine="fallback")
    assert res["engine"].startswith("analytical")
    assert os.path.exists(res["oem"]) and res["samples"] > 10
    radii = _radii_from_oem(res["oem"])
    # The drift orbit dips below GEO (perigee) then returns toward apogee=42164.
    assert min(radii) < 42000.0                    # dipped below GEO
    assert abs(max(radii) - 42164.0) < 5.0         # apogee at the burn radius


def test_engine_mode_gmat_without_install_errors():
    if E.gmat_available():
        pytest.skip("GMAT is wired on this host; the no-install error path can't be exercised")
    plan = M.plan_rendezvous(42164.0, 6.0)
    d = tempfile.mkdtemp()
    res = E.propagate_maneuver(42164.0, 6.0, 0.1, plan, d, engine="gmat")
    assert "error" in res                          # 'gmat' forced but none installed


@pytest.mark.skipif(not E.gmat_available(), reason="GMAT not wired (set GMAT_HOME)")
def test_gmat_branch_matches_fallback(tmp_path):
    # On a GMAT-equipped host: the real-force-model branch must agree with the
    # analytic two-body fallback for a point-mass run (cross-validates the seam).
    plan = M.plan_rendezvous(42164.0, 6.0)
    g = E.propagate_maneuver(42164.0, 6.0, 0.1, plan, str(tmp_path), name="s_g", engine="gmat")
    f = E.propagate_maneuver(42164.0, 6.0, 0.1, plan, str(tmp_path), name="s_f", engine="fallback")
    assert not g.get("error") and g["engine"].startswith("GMAT")

    def last_xyz(oem):
        xyz = None
        for line in open(oem):
            p = line.split()
            if len(p) == 4 and p[0].replace('.', '', 1).isdigit():
                xyz = tuple(float(v) for v in p[1:])
        return xyz
    gx, fx = last_xyz(g["oem"]), last_xyz(f["oem"])
    assert math.sqrt(sum((a - b) ** 2 for a, b in zip(gx, fx))) < 1.0     # < 1 km
