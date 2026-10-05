"""Router unit tests — prove control flow is deterministic and Python-owned.

These tests stub the LLM entirely: the point is that the router, not the model,
decides what runs, so the model's reply is just data we feed in. Run from the
repo root:  ./.venv/bin/python -m pytest tests/ -q   (or: python -m pytest).
"""

from __future__ import annotations

import json

from orchestrator import router as R
from orchestrator.router import Router, CHAT_INTENT, validate_args
from orchestrator.tools import ToolRegistry, Tool, ToolResult


class FakeTool(Tool):
    """A minimal tool with one required and one defaulted+typed arg."""

    name = "propagate_orbit"

    def __init__(self):
        self.calls = []

    def schema(self):
        return {
            "name": self.name,
            "description": "propagate an orbit",
            "parameters": {
                "type": "object",
                "properties": {
                    "object": {"type": "string", "description": "name or altitude"},
                    "duration_min": {"type": "number", "default": 95},
                },
                "required": ["object"],
            },
        }

    def run(self, args):
        self.calls.append(args)                       # record exactly what ran
        return ToolResult(text=f"ran with {args}", data=args)

    def render(self, result):
        return {"kind": "viz", "czml_source": "newest"}


def _registry():
    reg = ToolRegistry()
    tool = FakeTool()
    reg.register(tool)
    return reg, tool


def _stub_llm(monkeypatch, content):
    """Make llm_client.chat return a canned assistant message `content`."""
    def fake_chat(messages, tools=None, max_tokens=1024, temperature=0.2):
        return {"choices": [{"message": {"content": content}}]}
    monkeypatch.setattr(R.llm_client, "chat", fake_chat)


# -- validate_args (pure) ---------------------------------------------------

def test_validate_coerces_types_and_fills_defaults():
    params = {"properties": {"object": {"type": "string"},
                             "duration_min": {"type": "number", "default": 95}},
              "required": ["object"]}
    clean, missing = validate_args(params, {"object": 550, "duration_min": "120"})
    assert clean == {"object": "550", "duration_min": 120.0}   # coerced
    assert missing == []


def test_validate_reports_missing_required():
    params = {"properties": {"object": {"type": "string"}}, "required": ["object"]}
    clean, missing = validate_args(params, {})
    assert missing == ["object"]


def test_validate_drops_unknown_keys():
    params = {"properties": {"object": {"type": "string"}}, "required": []}
    clean, _ = validate_args(params, {"object": "ISS", "evil": "rm -rf"})
    assert clean == {"object": "ISS"}                # smuggled key dropped


# -- classify (LLM as a bounded transducer) ---------------------------------

def test_classify_routes_valid_intent(monkeypatch):
    reg, _ = _registry()
    _stub_llm(monkeypatch, json.dumps({"intent": "propagate_orbit", "args": {"object": "ISS"}}))
    intent, args = Router(reg).classify("propagate the ISS")
    assert intent == "propagate_orbit"
    assert args == {"object": "ISS"}


def test_classify_offset_intent_never_escapes_closed_set(monkeypatch):
    # The model proposes a tool that does NOT exist. It must not be dispatched;
    # the router falls back (keyword) or to chat — never the bogus name.
    reg, _ = _registry()
    _stub_llm(monkeypatch, json.dumps({"intent": "launch_missiles", "args": {}}))
    intent, _ = Router(reg).classify("do something unlisted and dangerous")
    assert intent in (CHAT_INTENT, "propagate_orbit")   # never 'launch_missiles'
    assert intent != "launch_missiles"


def test_classify_garbage_reply_is_chat(monkeypatch):
    reg, _ = _registry()
    _stub_llm(monkeypatch, "I think maybe you want... (no json here)")
    intent, _ = Router(reg).classify("hello there")
    assert intent == CHAT_INTENT


def test_classify_deliberate_chat_not_overridden_by_keyword(monkeypatch):
    # The model deliberately chooses chat for a QUESTION that happens to contain a
    # tool-name token ('orbit'). The keyword fallback must NOT override it.
    reg, tool = _registry()
    _stub_llm(monkeypatch, json.dumps({"intent": "chat", "args": {}}))
    intent, _ = Router(reg).classify("tell me about orbit mechanics")
    assert intent == CHAT_INTENT
    assert tool.calls == []


def test_classify_tolerates_code_fences(monkeypatch):
    reg, _ = _registry()
    fenced = "```json\n{\"intent\":\"propagate_orbit\",\"args\":{\"object\":\"ISS\"}}\n```"
    _stub_llm(monkeypatch, fenced)
    intent, args = Router(reg).classify("propagate ISS")
    assert intent == "propagate_orbit" and args["object"] == "ISS"


def test_classify_model_down_falls_back_to_keyword(monkeypatch):
    reg, _ = _registry()
    def boom(*a, **k):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(R.llm_client, "chat", boom)
    # "propagate" matches only propagate_orbit's tokens -> single-hit keyword route
    intent, _ = Router(reg).classify("please propagate this")
    assert intent == "propagate_orbit"


# -- dispatch (deterministic; validates before running) ---------------------

def test_dispatch_runs_tool_with_validated_args():
    reg, tool = _registry()
    res = Router(reg).dispatch("propagate_orbit", {"object": "ISS", "duration_min": "30", "junk": 1})
    assert res.error is None
    assert tool.calls == [{"object": "ISS", "duration_min": 30.0}]   # coerced, junk dropped
    assert res.artifact == {"kind": "viz", "czml_source": "newest"}   # rendered by dispatch


def test_dispatch_missing_required_is_plain_error_not_a_run():
    reg, tool = _registry()
    res = Router(reg).dispatch("propagate_orbit", {})
    assert res.error and "needs" in res.error
    assert tool.calls == []                          # nothing executed


def test_dispatch_unknown_intent_errors():
    reg, _ = _registry()
    res = Router(reg).dispatch("nope", {})
    assert res.error and "unknown intent" in res.error


def test_dispatch_macro_runs_python_sequence():
    reg, tool = _registry()
    seen = {}

    def my_macro(router, args):
        # a macro is a fixed Python sequence of dispatches
        router.dispatch("propagate_orbit", {"object": "clientsat"})
        router.dispatch("propagate_orbit", {"object": "servicersat"})
        seen["ran"] = True
        return ToolResult(text="macro done", data={"steps": 2})

    r = Router(reg, macros={"model_rpo": my_macro})
    assert "model_rpo" in r.closed_set()
    res = r.dispatch("model_rpo", {})
    assert seen.get("ran") and res.text == "macro done"
    assert len(tool.calls) == 2                       # the macro drove two runs


def test_determinism_same_input_same_dispatch(monkeypatch):
    reg, tool = _registry()
    _stub_llm(monkeypatch, json.dumps({"intent": "propagate_orbit", "args": {"object": "ISS"}}))
    r = Router(reg)
    for _ in range(3):
        intent, args = r.classify("propagate the ISS")
        r.dispatch(intent, args)
    assert tool.calls == [{"object": "ISS", "duration_min": 95}] * 3   # identical every run
