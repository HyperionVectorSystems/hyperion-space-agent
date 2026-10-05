"""Tool interface contract.

This is the core of the harness: one small contract every tool adapter
implements, so the orchestrator can offer the tool to the model, run it, and
render its result without knowing anything tool-specific. Add a tool by
implementing this class and registering it; the orchestrator discovers it.

Three methods:
  - schema():  the JSON-schema description the model sees (name, purpose, args).
  - run():     execute the tool for a set of arguments, return a ToolResult.
  - render():  turn a ToolResult into what the hub shows (a viz update, a table,
               a chart spec, or an image + overlay). Pure formatting, no compute.

Keeping run() (compute) separate from render() (presentation) is deliberate:
the same result can be spoken, tabulated, or drawn without re-running the tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """What an adapter's run() returns.

    text:     a short, speakable summary the model grounds its answer on.
    data:     structured result (numbers, rows) for the artifact pane.
    artifact: optional render hint, e.g. {"kind": "table"|"chart"|"image"|"viz",
              ...}. render() fills this; the hub reads "kind" to place it.
    error:    set on failure; the orchestrator speaks it plainly instead of
              hallucinating a success.
    """

    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifact: dict[str, Any] | None = None
    error: str | None = None


class Tool:
    """Base class for a tool adapter. Subclass and implement the three methods.

    name must be a stable, lowercase identifier (the model calls the tool by it).
    """

    name: str = "unnamed_tool"

    def schema(self) -> dict[str, Any]:
        """Return the OpenAI-style function schema the model is offered.

        Example:
            return {
                "name": self.name,
                "description": "Propagate an orbit and return period + ground track.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object": {"type": "string", "description": "NORAD id or name"},
                        "duration_min": {"type": "number", "default": 90},
                    },
                    "required": ["object"],
                },
            }
        """
        raise NotImplementedError

    def run(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool. Validate args, call the external tool, return a
        ToolResult. Never raise for expected failure; set ToolResult.error."""
        raise NotImplementedError

    def render(self, result: ToolResult) -> dict[str, Any]:
        """Turn a ToolResult into a hub artifact dict. Default: no artifact
        (text-only). Override to emit a table / chart / image / viz update."""
        return result.artifact or {}


class ToolRegistry:
    """Holds the active adapters. The orchestrator asks it for the schema list
    (to offer the model) and dispatches by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)
