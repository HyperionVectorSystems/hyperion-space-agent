# Adding a tool: the interface contract

A tool adapter is one class implementing three methods. The orchestrator offers
whatever is registered to the model and dispatches by name, so adding a tool
needs no orchestrator edits.

## The three methods (`orchestrator/tools/contract.py`)

- `schema()` returns the function description the model sees: `name`,
  `description`, and a JSON-schema `parameters` block. Write the description for
  a model deciding when to call it, and keep argument names obvious.
- `run(args)` executes the tool and returns a `ToolResult`. Validate `args`,
  call the external tool, normalize the output. Never raise for an expected
  failure; set `ToolResult.error` so the orchestrator speaks it plainly.
- `render(result)` turns a `ToolResult` into a hub artifact dict. Pure
  formatting, no compute, so the same result can be spoken, tabulated, or drawn.

`ToolResult` carries `text` (a short speakable summary the model grounds on),
`data` (structured numbers/rows), `artifact` (a render hint), and `error`.

## Artifact kinds the hub understands

- `{"kind": "viz", "czml_source": "newest"}` — a new orbit was written; the
  globe reloads on its next poll.
- `{"kind": "table", "rows": [[...], ...]}` — rendered in the artifact pane.
- `{"kind": "chart", ...}` — RESERVED. The hub has no chart branch yet; it
  falls through to a JSON dump. Wire your chart lib in
  `web/index.html:renderArtifact` before using this kind.
- `{"kind": "image", "fits": path, "overlay": path}` — open a FITS frame in the
  viewer.
- `{"kind": "overlay", "marks": [...], "color": ..., "label": ...}` — draw marks
  on the current frame (truth in green, detections in cyan, by convention).

## Minimal example

```python
from .contract import Tool, ToolResult

class Hello(Tool):
    name = "hello"

    def schema(self):
        return {
            "name": self.name,
            "description": "Return a friendly greeting for a named object.",
            "parameters": {
                "type": "object",
                "properties": {"who": {"type": "string"}},
                "required": ["who"],
            },
        }

    def run(self, args):
        who = args.get("who", "world")
        return ToolResult(text=f"Hello, {who}.", data={"who": who})
```

## Register it

Add it to `orchestrator/tools/__init__.py:build_registry()`:

```python
from .hello import Hello
...
reg.register(Hello())
```

Restart the orchestrator. The model is now offered `hello` and the loop
dispatches to it automatically. No `server.py` change.

## Conventions

- One stable lowercase `name` per tool; the model calls it by that string.
- Keep `run()` (compute) and `render()` (presentation) separate.
- Point tool binaries and data dirs through environment variables (see
  `.env.example`), never hardcode paths, so a user re-points without editing code.
