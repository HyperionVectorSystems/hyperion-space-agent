"""Default tool registry.

Registers the baseline adapters. Add your own by importing it here and calling
registry.register(...) — the orchestrator offers whatever is registered to the
model automatically. See docs/interface-contract.md.
"""

from .contract import Tool, ToolRegistry, ToolResult
from .gmat_adapter import GmatPropagate
from .gmat_maneuver import PlanManeuver
from .satsim_adapter import SatsimScene
from .satnet_overlay import SatnetOverlay
from .detector_example import ExampleDetector
from .clear_view import ClearView
from .observer_view import ObserverView


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(GmatPropagate())
    reg.register(PlanManeuver())
    reg.register(SatsimScene())
    reg.register(SatnetOverlay())
    reg.register(ExampleDetector())
    reg.register(ClearView())
    reg.register(ObserverView())
    return reg


__all__ = ["Tool", "ToolResult", "ToolRegistry", "build_registry"]
