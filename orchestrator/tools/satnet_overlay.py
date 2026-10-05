"""SatNet annotation overlay adapter.

Reads a SatNet-format annotation file (from SatSim, or your own labels) and
turns it into overlay marks the FITS viewer draws on top of a frame. This is a
render-heavy, compute-light adapter: it does not detect anything, it displays
known truth boxes so a viewer can compare truth against detections.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .contract import Tool, ToolResult


class SatnetOverlay(Tool):
    name = "satnet_overlay"

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": "Overlay SatNet truth annotations on the current FITS frame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "annotations": {"type": "string", "description": "path to a SatNet-format JSON"},
                },
                "required": ["annotations"],
            },
        }

    def run(self, args: dict[str, Any]) -> ToolResult:
        path = str(args.get("annotations", ""))
        if not path or not os.path.exists(path):
            return ToolResult(error=f"annotation file not found: {path}")
        try:
            with open(path) as f:
                ann = json.load(f)
        except Exception as e:  # malformed labels are an expected failure
            return ToolResult(error=f"could not read annotations: {e}")
        marks = _to_marks(ann)
        return ToolResult(
            text=f"Overlaid {len(marks)} truth annotations.",
            data={"count": len(marks)},
            artifact={"kind": "overlay", "marks": marks, "color": "green", "label": "truth"},
        )

    def render(self, result: ToolResult) -> dict[str, Any]:
        return result.artifact or {}


def _to_marks(ann: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a SatNet annotation record into simple viewer marks.
    Adjust the field names to your SatNet schema variant."""
    marks: list[dict[str, Any]] = []
    for obj in ann.get("objects", []):
        # SatNet commonly carries per-object pixel coords; pass them through.
        marks.append({
            "x": obj.get("x_center"),
            "y": obj.get("y_center"),
            "class": obj.get("class_name", "object"),
        })
    return marks
