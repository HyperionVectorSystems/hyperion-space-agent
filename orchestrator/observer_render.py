"""Observer synthetic-view renderer — "what the observer sees" looking at a target.

Phase 3 of the reference harness. The maneuver tools (Phase 2) put an observer and
one or more targets on the globe as OEM ephemerides. This module renders the SCENE
FROM THE OBSERVER'S EYE: the line of sight from the observer to each target becomes
a right-ascension / declination, and the targets are drawn as point sources on a
synthetic star field, written to a FITS with a valid WCS so the sky viewer places
them correctly.

It is deliberately general — any observer, any list of targets, any epoch, any
field of view. The GEO servicing exemplar (a LEO observer watching a client +
servicer ~36,000 km away, a near-unresolved point pair) is one instance; nothing
here branches on it.

Everything is offline: no star catalog fetch, no network. The star field is
synthetic noise + random point sources, seeded for reproducibility, so the same
geometry renders the same frame every run (which is what makes it unit-testable).

Firewall note: this is generic astrometry (LOS geometry + a tangent-plane
projection) and a synthetic image. No detection method, nothing proprietary.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from . import propagate

# The inertial frame the OEMs are written in (INERTIAL for circular orbits, TEME
# for TLEs) is treated as an equatorial frame: +x toward the vernal equinox, +z
# toward the north celestial pole. That is the standard convention for turning a
# Cartesian LOS into RA/Dec for a teaching view, and it is exact enough that the
# two targets land in the right relative place on the frame.


def read_state_at(oem_path: str, t_s: float | None) -> np.ndarray | None:
    """Return the target's Cartesian position (km) at time `t_s` seconds after the
    OEM epoch, linearly interpolated between the bracketing samples. `t_s=None`
    means the last sample. `t_s` beyond the OEM's span is clamped to its ends (no
    extrapolation), so mixing OEMs of different durations is safe.

    OEMs are position-only (propagate._write_oem), which is all a line-of-sight
    needs — direction, not velocity.
    """
    _name, positions_m, _total = propagate._parse_oem(oem_path)
    if len(positions_m) < 4:
        return None
    arr = np.asarray(positions_m, float).reshape(-1, 4)     # rows of [t, x, y, z] (meters)
    times = arr[:, 0]
    xyz_km = arr[:, 1:] / 1000.0                            # meters -> km
    if t_s is None:
        return xyz_km[-1]
    t = float(min(max(t_s, times[0]), times[-1]))           # clamp into range
    # np.interp per axis handles the bracketing + linear blend, and returns the
    # endpoint value outside the range (already clamped, so this is exact).
    return np.array([np.interp(t, times, xyz_km[:, k]) for k in range(3)])


def mean_radius_km(oem_path: str) -> float:
    """Mean orbital radius of an OEM's samples (km). Used to pick the observer: the
    body on the lowest orbit is the natural viewer looking up at higher targets."""
    _name, positions_m, _total = propagate._parse_oem(oem_path)
    if len(positions_m) < 4:
        return float("inf")
    arr = np.asarray(positions_m, float).reshape(-1, 4)[:, 1:] / 1000.0
    return float(np.linalg.norm(arr, axis=1).mean())


def los_radec(observer_km: np.ndarray, target_km: np.ndarray) -> tuple[float, float, float]:
    """Line of sight observer->target as (ra_deg, dec_deg, range_km).

    RA is measured in the x-y plane from +x, Dec above/below it. Units cancel in
    the direction, so any consistent length unit works; range is reported in km.
    """
    d = np.asarray(target_km, float) - np.asarray(observer_km, float)
    rng = float(np.linalg.norm(d))
    if rng == 0.0:
        return 0.0, 0.0, 0.0
    ra = math.degrees(math.atan2(d[1], d[0])) % 360.0
    dec = math.degrees(math.asin(max(-1.0, min(1.0, d[2] / rng))))
    return ra, dec, rng


def angular_sep_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle separation between two (ra, dec) points, degrees."""
    ra1, dec1 = math.radians(a[0]), math.radians(a[1])
    ra2, dec2 = math.radians(b[0]), math.radians(b[1])
    cos_sep = (math.sin(dec1) * math.sin(dec2)
               + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))


def mean_radec(radecs: list[tuple[float, float]]) -> tuple[float, float]:
    """Direction-averaged center of a set of (ra, dec), robust to the 0/360 wrap
    (average the unit vectors, not the angles)."""
    vecs = []
    for ra, dec in radecs:
        r, d = math.radians(ra), math.radians(dec)
        vecs.append([math.cos(d) * math.cos(r), math.cos(d) * math.sin(r), math.sin(d)])
    v = np.asarray(vecs, float).mean(axis=0)
    n = np.linalg.norm(v)
    if n == 0:
        return radecs[0]
    v /= n
    ra = math.degrees(math.atan2(v[1], v[0])) % 360.0
    dec = math.degrees(math.asin(max(-1.0, min(1.0, v[2]))))
    return ra, dec


def build_wcs(center_ra: float, center_dec: float, fov_deg: float, npix: int) -> WCS:
    """A tangent-plane (gnomonic) WCS centered on (center_ra, center_dec) spanning
    fov_deg across npix pixels. RA increases to the left (CDELT1 < 0), the sky
    convention, so the frame reads correctly in the viewer."""
    scale = fov_deg / npix                          # degrees per pixel
    w = WCS(naxis=2)
    w.wcs.crpix = [npix / 2.0 + 0.5, npix / 2.0 + 0.5]   # FITS 1-indexed center
    w.wcs.cdelt = [-scale, scale]
    w.wcs.crval = [center_ra, center_dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _add_psf(img: np.ndarray, x: float, y: float, flux: float, sigma: float = 1.6) -> None:
    """Add a 2-D Gaussian point-spread function centered at pixel (x, y). x, y are
    0-indexed array coordinates (col, row). Renders into a small stamp for speed."""
    ny, nx = img.shape
    rad = int(math.ceil(4 * sigma))
    x0, x1 = max(0, int(x) - rad), min(nx, int(x) + rad + 1)
    y0, y1 = max(0, int(y) - rad), min(ny, int(y) + rad + 1)
    if x0 >= x1 or y0 >= y1:
        return                                      # target lands off-frame
    ys, xs = np.mgrid[y0:y1, x0:x1]
    g = flux * np.exp(-(((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2)))
    img[y0:y1, x0:x1] += g


def render_scene(observer_km: np.ndarray, targets: list[dict[str, Any]],
                 fov_deg: float | None, npix: int, seed: int,
                 n_stars: int, bias: float, read_noise: float,
                 star_flux: float, target_flux: float) -> tuple[np.ndarray, WCS, dict[str, Any]]:
    """Compute LOS RA/Dec for every target, build a WCS framing them, and render a
    synthetic frame (noise + star field + a PSF per target). Returns (image, wcs,
    overlay-dict). Pure function of its inputs given `seed`, hence testable."""
    # 1) line of sight to each target
    marks: list[dict[str, Any]] = []
    for tgt in targets:
        ra, dec, rng = los_radec(observer_km, np.asarray(tgt["pos_km"], float))
        marks.append({"name": tgt["name"], "ra": ra, "dec": dec, "range_km": round(rng, 3)})

    center = mean_radec([(m["ra"], m["dec"]) for m in marks])

    # 2) field of view: honor an explicit FOV, else auto-frame the target spread
    #    (4x the widest pairwise separation) with a sane floor and ceiling.
    max_sep = 0.0
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            max_sep = max(max_sep, angular_sep_deg(
                (marks[i]["ra"], marks[i]["dec"]), (marks[j]["ra"], marks[j]["dec"])))
    if fov_deg is None or fov_deg <= 0:
        fov_deg = min(5.0, max(0.25, 4.0 * max_sep)) if max_sep > 0 else 1.0

    wcs = build_wcs(center[0], center[1], fov_deg, npix)

    # 3) synthetic background + star field (deterministic under `seed`)
    rng_gen = np.random.default_rng(seed)
    img = rng_gen.normal(bias, read_noise, size=(npix, npix)).astype(np.float32)
    for _ in range(n_stars):
        sx, sy = rng_gen.uniform(0, npix), rng_gen.uniform(0, npix)
        flux = star_flux * rng_gen.uniform(0.3, 1.0)     # a spread of stellar brightness
        _add_psf(img, sx, sy, flux, sigma=1.4)

    # 4) place each target and record its pixel coordinate in the overlay
    for m in marks:
        px, py = wcs.world_to_pixel_values(m["ra"], m["dec"])   # 0-indexed (col, row)
        _add_psf(img, float(px), float(py), target_flux, sigma=1.7)
        m["x_pixel"], m["y_pixel"] = round(float(px), 2), round(float(py), 2)

    overlay = {
        "center": {"ra": round(center[0], 6), "dec": round(center[1], 6)},
        "fov_deg": round(fov_deg, 6),
        "separation_arcsec": round(max_sep * 3600.0, 3),
        "npix": npix,
        "targets": marks,
    }
    return img, wcs, overlay


def render_observer_view(observer_km: np.ndarray, targets: list[dict[str, Any]],
                         scene_dir: str, basename: str = "observer_view",
                         fov_deg: float | None = None, npix: int = 512,
                         seed: int = 42, n_stars: int = 120, bias: float = 1000.0,
                         read_noise: float = 8.0, star_flux: float = 4000.0,
                         target_flux: float = 20000.0) -> dict[str, Any]:
    """Render and write the observer frame. Writes `<basename>.fits` (with WCS) and
    `<basename>.overlay.json` (target RA/Dec + pixels) into `scene_dir`. Returns a
    dict of the file paths + the overlay, for the tool to serve and speak."""
    os.makedirs(scene_dir, exist_ok=True)
    img, wcs, overlay = render_scene(observer_km, targets, fov_deg, npix, seed,
                                     n_stars, bias, read_noise, star_flux, target_flux)

    fits_path = os.path.join(scene_dir, f"{basename}.fits")
    hdr = wcs.to_header()
    hdr["BUNIT"] = "count"
    hdr["OBJECT"] = "observer synthetic view"
    hdr["HISTORY"] = "synthetic observer view (AMOS reference harness, Phase 3)"
    fits.PrimaryHDU(data=img, header=hdr).writeto(fits_path, overwrite=True)

    overlay_path = os.path.join(scene_dir, f"{basename}.overlay.json")
    with open(overlay_path, "w") as f:
        json.dump(overlay, f, indent=2)

    return {"fits": fits_path, "overlay_path": overlay_path, "overlay": overlay}
