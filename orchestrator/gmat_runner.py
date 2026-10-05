"""GMAT-env child process: propagate a Cartesian state with a real force model.

Run indirectly by orchestrator/gmat_engine.py inside a subprocess whose
PYTHONPATH/LD_LIBRARY_PATH point at the GMAT bin (so `import gmatpy` and its
native libs resolve). Reads one JSON spec from argv[1], seeds a GMAT spacecraft
with the given Cartesian state, propagates for duration_s sampling every step_s,
and writes a CCSDS OEM (positions, km, inertial) — the same format the globe
reads. Prints one JSON line: {"oem","samples","name"} or {"error"}.

Self-contained by design: it does NOT import the orchestrator package (it runs
in a different interpreter env), so the OEM writer is inlined here.

NOTE: the gmatpy propagation recipe below is best-effort and PENDING on-host
VALIDATION (the spike). gmat_engine gates this off via gmat_available() until an
install is wired, so the harness runs the two-body fallback until this is proven.
"""

from __future__ import annotations

import json
import os
import sys


def _write_oem(samples, name, ephemeris_dir):
    os.makedirs(ephemeris_dir, exist_ok=True)
    path = os.path.join(ephemeris_dir, f"{name}.oem")
    lines = ["CCSDS_OEM_VERS = 2.0", f"OBJECT_NAME = {name}",
             "REF_FRAME = INERTIAL", "TIME_SYSTEM = UTC", "META_STOP"]
    for t, x, y, z in samples:
        lines.append(f"{t:.3f} {x:.6f} {y:.6f} {z:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def run(spec):
    import gmatpy as gmat
    gmat.Setup("gmat_startup_file.txt")

    rx, ry, rz = spec["r_km"]
    vx, vy, vz = spec["v_kms"]
    name = spec["name"]

    sat = gmat.Construct("Spacecraft", "ManSat")
    sat.SetField("CoordinateSystem", "EarthMJ2000Eq")
    sat.SetField("DisplayStateType", "Cartesian")
    for field, val in (("X", rx), ("Y", ry), ("Z", rz),
                       ("VX", vx), ("VY", vy), ("VZ", vz)):
        sat.SetField(field, val)

    fm = gmat.Construct("ForceModel", "FM")
    # Point-mass Earth: needs no gravity potential file (validated on-host
    # 2026-09-25). For oblateness/perturbations, swap to a GravityField and set a
    # PotentialFile (e.g. JGM2.cof) — PrepareInternals requires that file to exist:
    #   egrav = gmat.Construct("GravityField"); egrav.SetField("BodyName","Earth")
    #   egrav.SetField("Degree",4); egrav.SetField("Order",4)
    #   egrav.SetField("PotentialFile", "<GMAT>/data/gravity/earth/JGM2.cof")
    pmf = gmat.Construct("PointMassForce", "EarthPM")
    try:
        pmf.SetField("BodyName", "Earth")
    except Exception:
        pass                              # defaults to Earth if the field is fixed
    fm.AddForce(pmf)

    prop = gmat.Construct("Propagator", "Prop")
    gator = gmat.Construct("RungeKutta89", "Gator")
    prop.SetReference(gator)
    prop.SetReference(fm)

    gmat.Initialize()
    prop.AddPropObject(sat)
    prop.PrepareInternals()
    gator = prop.GetPropagator()

    duration = float(spec["duration_s"])
    step = float(spec["step_s"])
    n = max(2, int(duration / step))
    samples = [(0.0, rx, ry, rz)]
    t = 0.0
    for _ in range(n):
        gator.Step(step)
        gator.UpdateSpaceObject()
        st = gator.GetState()                 # [x,y,z,vx,vy,vz] in the prop frame
        t += step
        samples.append((t, float(st[0]), float(st[1]), float(st[2])))

    oem = _write_oem(samples, name, spec["ephemeris_dir"])
    return {"oem": oem, "samples": len(samples), "name": name}


def main():
    try:
        spec = json.loads(sys.argv[1])
    except Exception as e:
        print(json.dumps({"error": f"bad spec: {e}"}))
        return 1
    try:
        print(json.dumps(run(spec)))
        return 0
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
