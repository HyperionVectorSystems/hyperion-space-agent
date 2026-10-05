"""maneuver.py unit tests — validate the closed-form planning against textbook
values and internal consistency. No GMAT, no network; pure math.
Run: ./.venv/bin/python -m pytest tests/test_maneuver.py -q
"""

from __future__ import annotations

import math
import numpy as np

from orchestrator import maneuver as M


# -- two-body basics vs textbook -------------------------------------------

def test_geo_circular_speed_and_period():
    # GEO radius 42164 km: v ~ 3.0747 km/s, period ~ one sidereal day (86164 s).
    assert abs(M.circular_speed(42164.0) - 3.0747) < 1e-3
    assert abs(M.period(42164.0) - 86164.0) < 60.0        # within a minute


def test_leo_circular_speed():
    # ~400 km LEO (r=6778): ~7.67 km/s.
    assert abs(M.circular_speed(6778.0) - 7.6686) < 2e-3


def test_vis_viva_reduces_to_circular():
    r = 7000.0
    assert abs(M.vis_viva(r, r) - M.circular_speed(r)) < 1e-9


# -- phasing transfer -------------------------------------------------------

def test_phasing_zero_phase_is_zero_dv():
    p = M.phasing_transfer(42164.0, 0.0, revs=1)
    assert p["total_dv_kms"] < 1e-6


def test_phasing_symmetric_burns():
    p = M.phasing_transfer(42164.0, 8.0, revs=1)
    assert abs(p["dv1_kms"] - p["dv2_kms"]) < 1e-9          # enter == exit
    assert p["total_dv_kms"] > 0


def test_phasing_monotonic_in_phase():
    small = M.phasing_transfer(42164.0, 4.0, revs=1)["total_dv_kms"]
    big = M.phasing_transfer(42164.0, 12.0, revs=1)["total_dv_kms"]
    assert big > small                                      # larger gap costs more


def test_phasing_geo_relocation_is_modest_dv():
    # Closing a few degrees at GEO over one rev is a small maneuver (tens of m/s).
    p = M.phasing_transfer(42164.0, 6.0, revs=1)
    assert 0.0 < p["total_dv_kms"] < 0.1                    # < 100 m/s


# -- CW state transition + two-impulse --------------------------------------

def test_cw_stm_identity_at_small_time():
    n = M.mean_motion(42164.0)
    rr, rv, vr, vv = M.cw_stm(n, 1e-6)
    assert np.allclose(rr, np.eye(3), atol=1e-3)
    assert np.allclose(vv, np.eye(3), atol=1e-3)
    assert np.allclose(rv, np.zeros((3, 3)), atol=1e-3)
    assert np.allclose(vr, np.zeros((3, 3)), atol=1e-3)


def test_cw_two_impulse_vbar_approach_is_finite_positive():
    n = M.mean_motion(42164.0)
    t = 0.25 * M.period(42164.0)
    r0 = np.array([0.0, -20.0, 0.0])       # 20 km behind (v-bar)
    rf = np.array([0.0, -0.5, 0.0])        # hold 0.5 km behind
    res = M.cw_two_impulse(r0, rf, n, t)
    assert "error" not in res
    assert res["total_dv_kms"] > 0 and math.isfinite(res["total_dv_kms"])


def test_cw_singular_time_reports_error():
    # At t = one full period, Phi_rv is singular (cross-track s/n -> 0).
    n = M.mean_motion(42164.0)
    t = M.period(42164.0)
    res = M.cw_two_impulse([0, -20, 0], [0, -0.5, 0], n, t)
    assert "error" in res


# -- high-level plan --------------------------------------------------------

def test_plan_rendezvous_structure_and_budget():
    plan = M.plan_rendezvous(42164.0, phase_deg=6.0, standoff_km=20.0, hold_km=0.5,
                             approach="v-bar", phasing_revs=1)
    assert "error" not in plan
    assert len(plan["burns"]) == 4
    # burns are time-ordered and each has a positive dv
    ts = [b["t_s"] for b in plan["burns"]]
    assert ts == sorted(ts)
    assert all(b["dv_kms"] >= 0 for b in plan["burns"])
    # total is the sum of the two legs, and reported in m/s too
    assert abs(plan["total_dv_kms"] - (plan["phasing"]["total_dv_kms"] + plan["cw"]["total_dv_kms"])) < 1e-9
    assert abs(plan["total_dv_ms"] - plan["total_dv_kms"] * 1000.0) < 1e-6


def test_plan_rbar_vs_vbar_both_valid():
    v = M.plan_rendezvous(42164.0, 6.0, approach="v-bar")
    r = M.plan_rendezvous(42164.0, 6.0, approach="r-bar")
    assert "error" not in v and "error" not in r
    assert v["total_dv_kms"] > 0 and r["total_dv_kms"] > 0
