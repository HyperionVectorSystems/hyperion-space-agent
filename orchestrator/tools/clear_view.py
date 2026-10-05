"""Clear-the-globe adapter.

'Clear the globe/scene' is a natural spoken command, but there was no matching
intent, so the model mis-routed it to the nearest tool (satsim_scene). This gives
the router a deterministic target: it moves the current OEMs aside (via
propagate.clear_oems — a move to a _cleared/ subdir, NOT a delete) so the globe
resets. Returns the count; a viz artifact nudges the hub to reload the (now empty)
scene.
"""

from __future__ import annotations

import os
from typing import Any

from .contract import Tool, ToolResult
from .. import propagate

EPHEMERIS_DIR = os.environ.get("EPHEMERIS_DIR", "./data/ephemeris")


class ClearView(Tool):
    name = "clear_view"

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Clear the globe / reset the scene — remove all currently displayed "
                "orbits. Use for 'clear the globe', 'reset the view', 'start over'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

    def run(self, args: dict[str, Any]) -> ToolResult:
        n = propagate.clear_oems(EPHEMERIS_DIR)
        return ToolResult(
            text=(f"Cleared {n} object{'s' if n != 1 else ''} from the globe."
                  if n else "The globe is already clear."),
            data={"cleared": n},
            # empty tables -> the hub clears the artifact pane too (no stale data)
            artifact={"kind": "viz", "czml_source": "newest", "tables": []},
        )

    def render(self, result: ToolResult) -> dict[str, Any]:
        return result.artifact or {"kind": "viz", "czml_source": "newest", "tables": []}
