"""Tests for the Phase 3 observer synthetic view — the LOS->RA/Dec projection, the
deterministic render, the FITS WCS, and the tool's default observer/target
selection. No LLM, no GMAT (the two-body fallback writes the OEMs the tool reads).
"""

from __future__ import annotations

import math
import os

import numpy as np
from astropy.wcs import WCS

from orchestrator import observer_render as orr
from orchestrator.tools import build_registry, gmat_maneuver, gmat_adapter, observer_view
from orchestrator.macros import default_macros
from orchestrator.router import Router


# -- pure geometry ----------------------------------------------------------

def test_los_radec_known_axes():
    obs = np.zeros(3)
    ra, dec, rng = orr.los_radec(obs, np.array([42000., 0., 0.]))     # +x
    assert (round(ra, 6), round(dec, 6)) == (0.0, 0.0) and round(rng) == 42000
    ra, _, _ = orr.los_radec(obs, np.array([0., 42000., 0.]))          # +y -> RA 90
    assert round(ra, 6) == 90.0
    _, dec, _ = orr.los_radec(obs, np.array([0., 0., 42000.]))         # +z -> Dec 90
    assert round(dec, 6) == 90.0


def test_los_radec_is_relative_to_observer():
    # A target coincident in direction from a shifted observer still reads correctly.
    obs = np.array([1000., 0., 0.])
    ra, dec, rng = orr.los_radec(obs, np.array([43000., 0., 0.]))
    assert round(ra, 6) == 0.0 and round(dec, 6) == 0.0 and round(rng) == 42000


def test_mean_radec_wraps_around_zero():
    # points straddling the 0/360 seam must average to ~0, not ~180
    ra, dec = orr.mean_radec([(359.0, 0.0), (1.0, 0.0)])
    assert min(ra, 360.0 - ra) < 1e-6 and abs(dec) < 1e-6


def test_angular_sep_deg():
    assert abs(orr.angular_sep_deg((0, 0), (0, 1)) - 1.0) < 1e-9
    assert abs(orr.angular_sep_deg((0, 0), (90, 0)) - 90.0) < 1e-9


# -- render -----------------------------------------------------------------

def _close_pair():
    """Observer at origin, two targets a few arcsec apart in RA (a near-unresolved
    point pair, the exemplar geometry)."""
    obs = np.zeros(3)
    t1 = {"name": "client", "pos_km": np.array([42000., 0., 0.])}
    t2 = {"name": "servicer", "pos_km": np.array([42000., 4., 0.])}
    return obs, [t1, t2]


def test_render_scene_targets_on_frame_and_centered():
    obs, targets = _close_pair()
    img, wcs, ov = orr.render_scene(obs, targets, None, 512, 42, 60, 1000., 8., 4000., 20000.)
    assert img.shape == (512, 512)
    assert len(ov["targets"]) == 2
    for m in ov["targets"]:
        assert 0 <= m["x_pixel"] < 512 and 0 <= m["y_pixel"] < 512     # on-frame
    # a tight pair -> a small auto-FOV (floored at 0.25 deg) and a few-arcsec split
    assert ov["fov_deg"] <= 0.25 + 1e-9
    assert 0 < ov["separation_arcsec"] < 60


def test_render_is_deterministic_under_seed():
    obs, targets = _close_pair()
    a, _, ova = orr.render_scene(obs, targets, 0.5, 256, 7, 40, 1000., 8., 4000., 20000.)
    b, _, ovb = orr.render_scene(obs, targets, 0.5, 256, 7, 40, 1000., 8., 4000., 20000.)
    assert np.array_equal(a, b)                                         # same seed -> same frame
    assert [t["x_pixel"] for t in ova["targets"]] == [t["x_pixel"] for t in ovb["targets"]]


def test_explicit_fov_is_honored():
    obs, targets = _close_pair()
    _, _, ov = orr.render_scene(obs, targets, 1.5, 256, 1, 10, 1000., 8., 4000., 20000.)
    assert abs(ov["fov_deg"] - 1.5) < 1e-9


def test_fits_written_with_roundtrip_wcs(tmp_path):
    obs, targets = _close_pair()
    out = orr.render_observer_view(obs, targets, str(tmp_path), basename="obs_test",
                                   fov_deg=0.5, npix=128)
    fits_path = out["fits"]
    assert os.path.exists(fits_path) and os.path.exists(out["overlay_path"])
    from astropy.io import fits
    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header
        assert hdul[0].data.shape == (128, 128)
    w = WCS(hdr)                                                        # header round-trips to a WCS
    assert list(w.wcs.ctype) == ["RA---TAN", "DEC--TAN"]
    # CRVAL sits at the target-pair center the overlay reports
    assert abs(w.wcs.crval[0] - out["overlay"]["center"]["ra"]) < 1e-6


# -- the tool (default observer/target selection over real OEMs) ------------

def test_tool_needs_two_objects(tmp_path, monkeypatch):
    monkeypatch.setattr(observer_view, "EPHEMERIS_DIR", str(tmp_path))
    monkeypatch.setattr(observer_view, "SCENE_DIR", str(tmp_path / "scenes"))
    res = observer_view.ObserverView().run({})
    assert res.error is not None and "two objects" in res.error


def test_tool_picks_lowest_orbit_as_observer(tmp_path, monkeypatch):
    scenes = tmp_path / "scenes"
    monkeypatch.setattr(gmat_maneuver, "EPHEMERIS_DIR", str(tmp_path))
    monkeypatch.setattr(gmat_adapter, "EPHEMERIS_DIR", str(tmp_path))
    monkeypatch.setattr(observer_view, "EPHEMERIS_DIR", str(tmp_path))
    monkeypatch.setattr(observer_view, "SCENE_DIR", str(scenes))
    r = Router(build_registry(), macros=default_macros())
    r.dispatch("model_rpo", {"target_altitude_km": 35786, "phase_deg": 2,
                             "observer_altitude_km": 550})
    res = r.dispatch("render_observer_view", {})
    assert res.error is None
    assert res.data["observer"].startswith("circular_550km")            # the LEO body, not GEO
    names = {t["name"] for t in res.data["targets"]}
    assert {"client", "servicer"} <= names
    assert res.artifact["kind"] == "image"
    assert res.artifact["fits"].endswith(".fits")
    assert os.path.exists(scenes / "observer_view.fits")


def test_tool_reads_targets_at_common_epoch_unresolved_pair(tmp_path, monkeypatch):
    # client and servicer end their OEMs at different times; reading both at the
    # common epoch must leave them as a near-unresolved pair (a few arcsec), not
    # tens of degrees apart (the each-own-last-sample bug).
    scenes = tmp_path / "scenes"
    monkeypatch.setattr(gmat_maneuver, "EPHEMERIS_DIR", str(tmp_path))
    monkeypatch.setattr(gmat_adapter, "EPHEMERIS_DIR", str(tmp_path))
    monkeypatch.setattr(observer_view, "EPHEMERIS_DIR", str(tmp_path))
    monkeypatch.setattr(observer_view, "SCENE_DIR", str(scenes))
    r = Router(build_registry(), macros=default_macros())
    r.dispatch("model_rpo", {"target_altitude_km": 35786, "phase_deg": 2,
                             "observer_altitude_km": 550})
    res = r.dispatch("render_observer_view", {})
    assert res.error is None
    assert res.data["separation_arcsec"] < 300                          # arcmin-scale at worst, not degrees


def test_dispatch_drops_model_injected_junk_args(tmp_path, monkeypatch):
    # Live testing: the small model injected observer='Earth', fov_deg=10, npix=1024
    # from a request that named none of them. The schema advertises no args, so
    # validate_args strips them on dispatch and the robust defaults win: the lowest
    # orbit is the observer (NOT the bogus 'Earth'), and the FOV auto-frames the pair
    # (NOT the hallucinated 10 degrees).
    scenes = tmp_path / "scenes"
    monkeypatch.setattr(gmat_maneuver, "EPHEMERIS_DIR", str(tmp_path))
    monkeypatch.setattr(gmat_adapter, "EPHEMERIS_DIR", str(tmp_path))
    monkeypatch.setattr(observer_view, "EPHEMERIS_DIR", str(tmp_path))
    monkeypatch.setattr(observer_view, "SCENE_DIR", str(scenes))
    r = Router(build_registry(), macros=default_macros())
    r.dispatch("model_rpo", {"target_altitude_km": 35786, "phase_deg": 2,
                             "observer_altitude_km": 550})
    res = r.dispatch("render_observer_view",
                     {"observer": "Earth", "fov_deg": 10, "npix": 1024})
    assert res.error is None
    assert res.data["observer"].startswith("circular_550km")            # not 'Earth'
    assert res.data["fov_deg"] <= 0.25 + 1e-9                           # auto-framed, not 10


def test_read_state_clamps_out_of_range(tmp_path):
    # a time past the OEM's span returns the last sample, not an extrapolation
    from orchestrator import propagate
    propagate.propagate("altitude=550", duration_min=90, ephemeris_dir=str(tmp_path))
    oem = propagate.newest_oem(str(tmp_path))
    _n, pos, total = propagate._parse_oem(oem)
    last = np.asarray(pos).reshape(-1, 4)[-1, 1:] / 1000.0
    got = orr.read_state_at(oem, total + 1e6)                           # way past the end
    assert np.allclose(got, last)
