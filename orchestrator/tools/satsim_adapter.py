"""SatSim scene-generation adapter.

SatSim is a TensorFlow package, so it runs in its OWN environment
(.venv-satsim, created by scripts/fetch_tools.sh), not the harness venv. This
adapter shells out to that venv to generate a synthetic SDA frame plus SatNet
truth, then returns an "image" artifact the Aladin pane opens.

SatSim was not integrated in our own demo stack, so treat the pinned version as
"verify on first run": generate one scene and confirm the FITS + annotation
paths before relying on it.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from .contract import Tool, ToolResult

SATSIM_VENV = os.environ.get("SATSIM_VENV", "./.venv-satsim/bin/python")
SATSIM_CONFIG = os.environ.get("SATSIM_CONFIG", "")   # a satsim yaml/json config
SCENE_DIR = os.environ.get("SCENE_DIR", "./data/scenes")


class SatsimScene(Tool):
    name = "satsim_scene"

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Generate a synthetic SDA image (star field with satellite "
                "streaks) and its SatNet truth annotations, for display and detection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {"type": "integer", "description": "number of synthetic objects", "default": 3},
                    "snr": {"type": "number", "description": "target signal-to-noise", "default": 15},
                },
                "required": [],
            },
        }

    def run(self, args: dict[str, Any]) -> ToolResult:
        if not os.path.exists(SATSIM_VENV):
            return ToolResult(error="SatSim venv not found. Run scripts/fetch_tools.sh to create .venv-satsim.")
        if not SATSIM_CONFIG or not os.path.exists(SATSIM_CONFIG):
            return ToolResult(error="Set SATSIM_CONFIG in .env to a satsim scene config (see fetch_tools.sh notes).")
        os.makedirs(SCENE_DIR, exist_ok=True)
        try:
            # SatSim's CLI generates into an output dir; capture its FITS + annotation.
            subprocess.run([SATSIM_VENV, "-m", "satsim", "run", "--output_dir", SCENE_DIR, SATSIM_CONFIG],
                           check=True, capture_output=True, timeout=600)
        except Exception as e:
            return ToolResult(error=f"SatSim run failed: {e}")
        fits_path = _find(SCENE_DIR, ".fits")
        annot_path = _find(SCENE_DIR, ".json")
        if not fits_path:
            return ToolResult(error="SatSim produced no FITS; check the config and the venv.")
        return ToolResult(
            text=f"Generated a synthetic scene ({args.get('targets', 3)} targets).",
            data={"fits": fits_path, "annotations": annot_path},
            artifact={"kind": "image", "fits": _served(fits_path), "overlay": _served(annot_path)},
        )

    def render(self, result: ToolResult) -> dict[str, Any]:
        return result.artifact or {}


def _find(d: str, ext: str) -> str:
    hits = [os.path.join(d, n) for n in os.listdir(d) if n.endswith(ext)] if os.path.isdir(d) else []
    return max(hits, key=os.path.getmtime) if hits else ""


def _served(path: str) -> str:
    """Map a scene-dir path to the browser-served /data/scenes URL."""
    return "/data/scenes/" + os.path.basename(path) if path else ""
