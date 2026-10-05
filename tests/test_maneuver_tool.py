"""Regression tests for degenerate-argument handling in the maneuver tool and the
RPO macro — the exact junk args the small model produced in live voice testing
(phase=0, standoff=0, hold=0, empty names, observer placed at the target altitude).
These assert the sensibility guards fire so a bad arg can't collapse the scene or
write an empty-named OEM. No LLM, no GMAT.
"""

from __future__ import annotations

import os

from orchestrator.tools import build_registry, gmat_maneuver, gmat_adapter, clear_view
from orchestrator.tools.gmat_maneuver import PlanManeuver, _approach
from orchestrator.tools.clear_view import ClearView
from orchestrator.router import Router, _spell_to_digits
from orchestrator.macros import default_macros


def test_tool_floors_degenerate_args(tmp_path, monkeypatch):
    monkeypatch.setattr(gmat_maneuver, "EPHEMERIS_DIR", str(tmp_path))
    res = PlanManeuver().run({"target_altitude_km": 35786, "phase_deg": 0,
                              "standoff_km": 0, "hold_km": 0,
                              "target_name": "", "chaser_name": "", "approach": ""})
    assert res.error is None
    # empty names must not survive -> defaults, and NO empty-named OEM on disk
    assert res.data["target"] == "client" and res.data["chaser"] == "servicer"
    names = os.listdir(tmp_path)
    assert ".oem" not in names
    assert "client.oem" in names and "servicer.oem" in names
    # zeroed phase/standoff/hold must not collapse the maneuver
    assert res.data["total_dv_ms"] > 1.0


def test_macro_observer_not_placed_at_target(tmp_path, monkeypatch):
    monkeypatch.setattr(gmat_maneuver, "EPHEMERIS_DIR", str(tmp_path))
    monkeypatch.setattr(gmat_adapter, "EPHEMERIS_DIR", str(tmp_path))
    r = Router(build_registry(), macros=default_macros())
    res = r.dispatch("model_rpo", {"target_altitude_km": 35786,
                                   "observer_altitude_km": 35786, "phase_deg": 0})
    assert res.error is None
    names = os.listdir(tmp_path)
    # observer must fall back to LEO (550 km), not be drawn at the GEO target
    assert any(n.startswith("circular_550km") for n in names)
    assert not any(n.startswith("circular_35786km") for n in names)
    assert "client.oem" in names and "servicer.oem" in names


def test_dispatch_drops_model_supplied_names(tmp_path, monkeypatch):
    # The model grabbed prompt nouns as names ("GEO", "rendezvous"). Since the
    # schema no longer exposes name params, validate_args drops them on dispatch
    # and the entities keep their safe defaults — no mislabeled OEMs.
    monkeypatch.setattr(gmat_maneuver, "EPHEMERIS_DIR", str(tmp_path))
    r = Router(build_registry(), macros=default_macros())
    res = r.dispatch("plan_maneuver", {"target_altitude_km": 35786, "phase_deg": 6,
                                       "target_name": "GEO", "chaser_name": "rendezvous"})
    assert res.error is None
    assert res.data["target"] == "client" and res.data["chaser"] == "servicer"
    names = os.listdir(tmp_path)
    assert "GEO.oem" not in names and "rendezvous.oem" not in names
    assert "client.oem" in names and "servicer.oem" in names


def test_router_recovers_phase_from_text():
    # The model dropped '12-degree' -> phase_deg 0; the schema's x_recover hint lets
    # the router backfill it from the raw text. A valid value is left untouched.
    r = Router(build_registry(), macros=default_macros())
    assert r._recover("plan a rendezvous with 12-degree separation",
                      "plan_maneuver", {"phase_deg": 0})["phase_deg"] == 12.0
    assert r._recover("something at 12 degrees",
                      "plan_maneuver", {"phase_deg": 8})["phase_deg"] == 8      # not overridden
    assert r._recover("model an RPO with 20 degree separation",
                      "model_rpo", {"phase_deg": 0})["phase_deg"] == 20.0        # macro param too


def test_approach_normalization():
    # a word grabbed from the sentence must NOT flip the approach to r-bar
    assert _approach("rendezvous") == "v-bar"
    assert _approach("") == "v-bar"
    assert _approach(None) == "v-bar"
    assert _approach("R-R") == "v-bar"          # whisper garble -> safe default
    # real tokens are honored
    assert _approach("radial") == "r-bar"
    assert _approach("r-bar") == "r-bar"
    assert _approach("use a v-bar approach") == "v-bar"


def test_spelled_number_recovery():
    assert _spell_to_digits("six degrees") == "6 degrees"
    assert _spell_to_digits("fifteen deg behind") == "15 deg behind"
    r = Router(build_registry(), macros=default_macros())
    # model dropped the spelled number -> phase_deg 0; recovery reads "six"
    assert r._recover("servicer six degrees behind", "model_rpo",
                      {"phase_deg": 0})["phase_deg"] == 6.0
    assert r._recover("with a twelve degree separation", "plan_maneuver",
                      {"phase_deg": 0})["phase_deg"] == 12.0


def test_clear_view_tool(tmp_path, monkeypatch):
    monkeypatch.setattr(clear_view, "EPHEMERIS_DIR", str(tmp_path))
    (tmp_path / "x.oem").write_text("CCSDS_OEM_VERS = 2.0\nMETA_STOP\n0.000 1 2 3\n")
    res = ClearView().run({})
    assert res.data["cleared"] == 1
    assert "x.oem" not in os.listdir(tmp_path)       # moved aside, globe resets
    assert res.artifact.get("kind") == "viz"


def test_plan_maneuver_artifact_has_orbit_and_burn_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(gmat_maneuver, "EPHEMERIS_DIR", str(tmp_path))
    r = Router(build_registry(), macros=default_macros())
    a = r.dispatch("plan_maneuver", {"target_altitude_km": 35786, "phase_deg": 6}).artifact
    assert a["kind"] == "viz" and "Rendezvous" in a["title"]
    assert len(a["tables"]) == 2
    od, bp = a["tables"]
    assert od["title"] == "Orbit details"
    assert od["rows"][0] == ["Satellite", "Altitude (km)", "Period (min)", "Incl (deg)", "Note"]
    assert len(od["rows"]) == 3                                  # header + client + servicer
    assert bp["title"] == "Burn plan" and bp["rows"][0] == ["Burn", "Time", "Δv", "Note"]
    assert len(bp["rows"]) >= 5                                  # header + 4 burns


def test_model_rpo_orbit_table_includes_observer(tmp_path, monkeypatch):
    monkeypatch.setattr(gmat_maneuver, "EPHEMERIS_DIR", str(tmp_path))
    monkeypatch.setattr(gmat_adapter, "EPHEMERIS_DIR", str(tmp_path))
    r = Router(build_registry(), macros=default_macros())
    a = r.dispatch("model_rpo", {"target_altitude_km": 35786, "phase_deg": 8}).artifact
    od = a["tables"][0]
    assert len(od["rows"]) == 4                                  # header + client + servicer + observer
    assert od["rows"][-1][0] == "observer"
    assert a["tables"][1]["rows"][0][0] == "Burn"               # burn plan carried through


def test_propagate_orbit_artifact_has_orbit_table(tmp_path, monkeypatch):
    monkeypatch.setattr(gmat_adapter, "EPHEMERIS_DIR", str(tmp_path))
    r = Router(build_registry(), macros=default_macros())
    a = r.dispatch("propagate_orbit", {"altitude_km": 35786}).artifact
    assert a["tables"][0]["rows"][0] == ["Field", "Value"]


def test_clear_view_clears_artifact_pane():
    a = ClearView().run({}).artifact
    assert a["kind"] == "viz" and a["tables"] == []             # empty list clears the pane


def test_tool_honors_explicit_valid_args(tmp_path, monkeypatch):
    # A genuine non-degenerate request is passed through unchanged.
    monkeypatch.setattr(gmat_maneuver, "EPHEMERIS_DIR", str(tmp_path))
    res = PlanManeuver().run({"target_altitude_km": 35786, "phase_deg": 12,
                              "standoff_km": 30, "hold_km": 1.0, "approach": "r-bar",
                              "target_name": "sat-A", "chaser_name": "sat-B"})
    assert res.error is None
    assert res.data["phase_deg"] == 12 and res.data["approach"] == "r-bar"
    assert res.data["target"] == "sat-A" and res.data["chaser"] == "sat-B"
    assert "sat-A.oem" in os.listdir(tmp_path) and "sat-B.oem" in os.listdir(tmp_path)
