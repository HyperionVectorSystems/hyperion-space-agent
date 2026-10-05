"""Example detector adapter (STUB — teaching example only).

This is a deliberately simple matched-filter / threshold detector, included so
the toolchain has a working detection step end to end. It is NOT a research
detector and makes no performance claim. Replace it with your own detector by
implementing the same contract; the orchestrator does not care which detector
is registered.

Any real detection or estimation method is intentionally out of scope for this
public repository.
"""

from __future__ import annotations

from typing import Any

from .contract import Tool, ToolResult

try:
    import numpy as np
    from astropy.io import fits
    _HAVE_STACK = True
except Exception:
    _HAVE_STACK = False


class ExampleDetector(Tool):
    name = "detect_example"

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Run a simple example detector on a FITS frame and return "
                "candidate marks (teaching stub, not a research detector)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fits": {"type": "string", "description": "path to a FITS frame"},
                    "sigma": {"type": "number", "description": "threshold in MAD sigmas", "default": 5.0},
                },
                "required": ["fits"],
            },
        }

    def run(self, args: dict[str, Any]) -> ToolResult:
        if not _HAVE_STACK:
            return ToolResult(error="numpy + astropy required for the example detector.")
        path = str(args.get("fits", ""))
        sigma = float(args.get("sigma", 5.0))
        try:
            img = fits.getdata(path).astype("float32")
        except Exception as e:
            return ToolResult(error=f"could not read FITS: {e}")
        # Simplest possible detector: robust threshold on MAD sigma.
        med = float(np.median(img))
        mad = float(np.median(np.abs(img - med))) or 1.0
        thresh = med + sigma * 1.4826 * mad
        ys, xs = np.where(img > thresh)
        marks = [{"x": int(x), "y": int(y), "class": "candidate"} for x, y in zip(xs[:200], ys[:200])]
        return ToolResult(
            text=f"Example detector flagged {len(marks)} pixels above {sigma:.0f} sigma.",
            data={"count": len(marks), "threshold": thresh},
            artifact={"kind": "overlay", "marks": marks, "color": "cyan", "label": "detections"},
        )

    def render(self, result: ToolResult) -> dict[str, Any]:
        return result.artifact or {}
