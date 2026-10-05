"""Analytical orbit propagation — the baseline that makes the globe work with no
external tool.

This is a real, self-contained propagator so "ask for an orbit, see it on the
globe" works immediately after the harness venv is built, no GMAT download. Two
paths:

  - a named object with a bundled TLE (e.g. "ISS"): SGP4 via the `sgp4` package.
  - "altitude=<km>": a circular two-body orbit from classical elements.

It writes a CCSDS OEM (the same artifact GMAT would) and converts the newest OEM
to Cesium CZML in the INERTIAL frame (Cesium handles inertial-to-fixed), so the
globe draws the orbit. The GMAT adapter uses this by default; a real GMAT
install is a higher-fidelity swap, not a requirement.

Bundled TLEs go stale (elements drift); they are here to make the demo run, not
for operational accuracy. Replace them, or wire a TLE source, for real work.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any

MU = 398600.4418          # Earth GM, km^3/s^2
RE = 6378.137             # Earth equatorial radius, km
DEFAULT_INC_DEG = 51.6    # circular-orbit inclination when the caller gives none

# A tiny bundled catalog so a name like "ISS" resolves offline. Public TLEs.
# These WILL be out of date; refresh from celestrak/space-track for real use.
SAMPLE_TLES: dict[str, tuple[str, str]] = {
    "ISS": (
        "1 25544U 98067A   26001.50000000  .00016717  00000-0  10270-3 0  9005",
        "2 25544  51.6400 208.9163 0006317  69.9862 290.2000 15.49500000    05",
    ),
}


def _parse_km(text: str) -> float | None:
    """Extract an altitude in km from a loose string. Tolerates thousands
    separators and unit suffixes the model will naturally emit, e.g.
    '550', '550km', '35,786 km', '400 kilometers', 'alt 35786 km'."""
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", text or "")
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def resolve(obj: str) -> dict[str, Any]:
    """Return a plan: a TLE-backed object or an altitude-backed circular orbit."""
    o = (obj or "").strip()
    if o.lower().startswith("altitude="):
        alt = _parse_km(o.split("=", 1)[1])
        if alt is None:
            return {"error": f"could not parse altitude from {o!r}"}
        return {"mode": "circular", "alt_km": alt, "name": f"circular_{int(alt)}km"}
    for name, tle in SAMPLE_TLES.items():
        if o and (name.lower() in o.lower() or o.lower() in name.lower()):
            return {"mode": "tle", "name": name, "tle": tle}
    # A bare altitude phrasing without the 'altitude=' prefix, e.g. '35,786 km'.
    if "km" in o.lower() and re.search(r"\d", o):
        alt = _parse_km(o)
        if alt is not None:
            return {"mode": "circular", "alt_km": alt, "name": f"circular_{int(alt)}km"}
    return {"error": f"unknown object {o!r}. Try a bundled name (ISS) or 'altitude=<km>'."}


def propagate(obj: str, duration_min: float, step_s: float = 60.0,
              ephemeris_dir: str = "./data/ephemeris",
              inclination_deg: float | None = None) -> dict[str, Any]:
    """Propagate, write an OEM, return {name, period_min, inclination_deg, oem,
    samples}. `inclination_deg` sets the plane for a circular 'altitude=' orbit
    (default 51.6); it is ignored for a TLE object, whose inclination comes from
    the TLE."""
    plan = resolve(obj)
    if "error" in plan:
        return plan
    os.makedirs(ephemeris_dir, exist_ok=True)
    n = max(2, int(duration_min * 60 / step_s))

    if plan["mode"] == "tle":
        try:
            from sgp4.api import Satrec, jday
        except Exception:
            return {"error": "the sgp4 package is required for named objects (pip install sgp4)."}
        l1, l2 = plan["tle"]
        sat = Satrec.twoline2rv(l1, l2)
        period_min = (2 * math.pi / sat.no_kozai) if sat.no_kozai else 0.0  # no_kozai in rad/min
        inc_deg = math.degrees(sat.inclo)     # inclination is intrinsic to the TLE
        # Start at the TLE epoch; sample forward. TEME position in km.
        jd, fr = sat.jdsatepoch, sat.jdsatepochF
        samples = []
        for i in range(n):
            t = i * step_s
            e, r, _v = sat.sgp4_tsince(t / 60.0)  # tsince in minutes
            if e == 0:
                samples.append((t, r[0], r[1], r[2]))
        meta = {"name": plan["name"], "frame": "TEME"}
    else:
        alt = plan["alt_km"]
        inc_deg = DEFAULT_INC_DEG if inclination_deg is None else float(inclination_deg)
        a = RE + alt
        period_s = 2 * math.pi * math.sqrt(a ** 3 / MU)
        period_min = period_s / 60.0
        inc = math.radians(inc_deg)
        w = 2 * math.pi / period_s        # mean motion, rad/s
        samples = []
        for i in range(n):
            t = i * step_s
            th = w * t
            # circular orbit in a plane inclined by `inc` about the x-axis
            x = a * math.cos(th)
            y = a * math.sin(th) * math.cos(inc)
            z = a * math.sin(th) * math.sin(inc)
            samples.append((t, x, y, z))
        # Encode inclination in the name so two orbits at the same altitude but
        # different inclinations are distinct entities (distinct OEM files).
        meta = {"name": f"circular_{int(alt)}km_{inc_deg:g}deg", "frame": "INERTIAL"}

    if len(samples) < 2:
        return {"error": "propagation produced no samples (check the object/TLE)."}
    oem = _write_oem(samples, meta, ephemeris_dir)
    return {"name": meta["name"], "period_min": round(period_min, 2),
            "inclination_deg": round(inc_deg, 2), "oem": oem, "samples": len(samples)}


def _write_oem(samples: list[tuple], meta: dict[str, Any], ephemeris_dir: str) -> str:
    """Write a minimal CCSDS OEM (km, km/s omitted). Positions only, good enough
    for a viz path."""
    path = os.path.join(ephemeris_dir, f"{meta['name']}.oem")
    lines = [
        "CCSDS_OEM_VERS = 2.0",
        f"OBJECT_NAME = {meta['name']}",
        f"REF_FRAME = {meta['frame']}",
        "TIME_SYSTEM = UTC",
        "META_STOP",
    ]
    for t, x, y, z in samples:
        lines.append(f"{t:.3f} {x:.6f} {y:.6f} {z:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def newest_oem(ephemeris_dir: str) -> str | None:
    try:
        files = [os.path.join(ephemeris_dir, n) for n in os.listdir(ephemeris_dir) if n.endswith(".oem")]
    except FileNotFoundError:
        return None
    return max(files, key=os.path.getmtime) if files else None


EPOCH = "2026-01-01T00:00:00Z"
# One color per entity, cycled by index, so multiple satellites are distinguishable.
_PALETTE = [
    [0, 255, 200], [255, 196, 0], [80, 150, 255], [255, 120, 80],
    [190, 130, 255], [120, 220, 120], [255, 90, 160], [120, 220, 235],
]


def list_oems(ephemeris_dir: str) -> list[str]:
    """All top-level OEM files (cleared ones live in a subdir, excluded)."""
    try:
        names = [n for n in os.listdir(ephemeris_dir) if n.endswith(".oem")]
    except FileNotFoundError:
        return []
    paths = [os.path.join(ephemeris_dir, n) for n in names if os.path.isfile(os.path.join(ephemeris_dir, n))]
    return sorted(paths, key=os.path.getmtime)


def _parse_oem(oem_path: str) -> tuple[str, list[float], float]:
    """Return (name, cartesian [t,x,y,z,...] in METERS, total_seconds)."""
    name = os.path.splitext(os.path.basename(oem_path))[0]
    positions: list[float] = []
    with open(oem_path) as f:
        past_meta = False
        for line in f:
            line = line.strip()
            if line == "META_STOP":
                past_meta = True
                continue
            if line.startswith("OBJECT_NAME"):
                name = line.split("=", 1)[1].strip()
            if past_meta and line and not line.startswith(("CCSDS", "OBJECT", "REF", "TIME")):
                parts = line.split()
                if len(parts) == 4:
                    t, x, y, z = (float(p) for p in parts)
                    positions += [t, x * 1000, y * 1000, z * 1000]
    total = positions[-4] if len(positions) >= 4 else 0.0
    return name, positions, total


def _entity(name: str, positions: list[float], total: float, idx: int) -> dict[str, Any]:
    rgb = _PALETTE[idx % len(_PALETTE)]
    return {
        "id": name, "name": name,
        "label": {"text": name, "fillColor": {"rgba": rgb + [255]}, "font": "12px sans-serif",
                  "pixelOffset": {"cartesian2": [10, 0]}, "showBackground": True},
        "path": {"leadTime": total, "trailTime": total, "width": 2,
                 "material": {"solidColor": {"color": {"rgba": rgb + [200]}}}},
        "point": {"pixelSize": 8, "color": {"rgba": rgb + [255]}},
        "position": {"referenceFrame": "INERTIAL", "epoch": EPOCH, "cartesian": positions,
                     "interpolationAlgorithm": "LAGRANGE", "interpolationDegree": 5},
    }


def oems_to_czml(ephemeris_dir: str) -> list[dict[str, Any]]:
    """Merge ALL current OEMs into one CZML document, one entity per object, so
    the globe shows every propagated satellite at once."""
    oems = list_oems(ephemeris_dir)
    doc = [{"id": "document", "name": "orbits", "version": "1.0",
            "clock": {"interval": f"{EPOCH}/{EPOCH}", "currentTime": EPOCH, "multiplier": 60}}]
    for i, path in enumerate(oems):
        name, positions, total = _parse_oem(path)
        if positions:
            doc.append(_entity(name, positions, total, i))
    return doc


def oem_to_czml(oem_path: str) -> list[dict[str, Any]]:
    """Single-OEM CZML (kept for callers that want one orbit)."""
    name, positions, total = _parse_oem(oem_path)
    if not positions:
        return [{"id": "document", "version": "1.0"}]
    return [
        {"id": "document", "name": name, "version": "1.0",
         "clock": {"interval": f"{EPOCH}/{EPOCH}", "currentTime": EPOCH, "multiplier": 60}},
        _entity(name, positions, total, 0),
    ]


def clear_oems(ephemeris_dir: str) -> int:
    """Move all current OEMs into a `_cleared/` subdir (NOT a delete), so the
    globe resets. Returns how many were moved."""
    oems = list_oems(ephemeris_dir)
    if not oems:
        return 0
    dest = os.path.join(ephemeris_dir, "_cleared")
    os.makedirs(dest, exist_ok=True)
    moved = 0
    for path in oems:
        base = os.path.basename(path)
        target = os.path.join(dest, f"{moved}_{base}")
        os.rename(path, target)     # move, not delete
        moved += 1
    return moved
