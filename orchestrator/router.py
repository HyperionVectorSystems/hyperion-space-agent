"""Deterministic router — the orchestrator's control plane.

This is the piece that makes the harness an *instrument* rather than a chatbot:
control flow lives in Python, not in the language model. The model is used only
as a transducer — it turns a natural-language request into a structured
{intent, args} proposal drawn from a CLOSED SET — and the router then validates
that proposal against each tool's own schema and decides, deterministically,
what runs. The model never chooses *whether* a tool runs, *which* tool runs, or
in what *sequence*; the router does. This matches docs/architecture-spec.json
(Gate 1: "parse input (closed set) → tool-command → tool; else → LLM").

Why not native OpenAI tool-calling (`tool_choice=auto`)? Because it hands the
control decision to the model. Small local models route unreliably (a "thinking"
model may return reasoning instead of a call), and a multi-step workflow becomes
non-deterministic. Here the same request produces the same dispatch every run,
and a multi-step procedure is a Python *macro*, not an improvisation.

The closed set is derived from whatever tools are registered plus any registered
macros — it is not a hardcoded list, so adding a tool extends the router with no
change here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from . import llm_client
from .tools import ToolRegistry, ToolResult

# The one non-tool intent: answer conversationally with no tool.
CHAT_INTENT = "chat"

# A macro is a named composite: given the router and validated args, it runs a
# fixed Python sequence of tool calls and returns a single ToolResult. Macros
# are how a multi-step workflow stays deterministic (see Phase 2's model_rpo).
Macro = Callable[["Router", dict[str, Any]], ToolResult]


class Router:
    """Owns the route table and all control flow.

    The route table is dynamic: `closed_set()` = registered tool names + macro
    names + `chat`. Nothing about any specific domain or scenario is encoded.
    """

    def __init__(self, registry: ToolRegistry, macros: dict[str, Macro] | None = None) -> None:
        self.registry = registry
        self.macros: dict[str, Macro] = dict(macros or {})

    # -- the closed set -----------------------------------------------------

    def closed_set(self) -> list[str]:
        return list(self.registry.names()) + list(self.macros) + [CHAT_INTENT]

    def catalog(self) -> list[dict[str, Any]]:
        """Compact tool descriptions the slot-filler is shown. Macros are offered
        by name + a generic note (their args mirror the tools they compose)."""
        items = [t.schema() for t in (self.registry.get(n) for n in self.registry.names()) if t]
        for name, fn in self.macros.items():
            items.append({"name": name,
                          "description": getattr(fn, "description", f"composite procedure: {name}"),
                          "parameters": getattr(fn, "parameters", {"type": "object", "properties": {}})})
        return items

    # -- classify (LLM as a bounded transducer) -----------------------------

    def classify(self, text: str) -> tuple[str, dict[str, Any]]:
        """Return (intent, args) with intent guaranteed to be in the closed set.

        Primary path: the model proposes {intent, args} constrained to the closed
        set. A DELIBERATE 'chat' choice is respected — we never override a valid
        chat with a keyword guess (that was a latent bug). The keyword fallback
        fires only when routing was unavailable (model down / unparseable) or the
        model proposed an off-set name. Either way dispatch() validates; classify
        never executes anything.
        """
        intent, args = self._llm_route(text)          # intent is None if routing was unavailable
        if intent == CHAT_INTENT:
            return CHAT_INTENT, {}                      # deliberate chat: respect it, do not override
        if intent in self.closed_set():
            args = args if isinstance(args, dict) else {}
            return intent, self._recover(text, intent, args)
        # Unavailable (None) or off-set proposal: conservative keyword match, else chat.
        kw = self._keyword_fallback(text)
        return (kw, self._recover(text, kw, {})) if kw is not None else (CHAT_INTENT, {})

    def _llm_route(self, text: str) -> tuple[str | None, dict[str, Any]]:
        """Ask the model for {intent, args}. Returns (None, {}) if the model is
        unavailable or its reply is unparseable — the caller then keyword-falls-back."""
        names = [n for n in self.closed_set() if n != CHAT_INTENT]
        catalog_lines = []
        for item in self.catalog():
            props = (item.get("parameters") or {}).get("properties", {}) or {}
            params = ", ".join(f"{k}:{(v or {}).get('type','any')}" for k, v in props.items())
            catalog_lines.append(f"- {item['name']}: {item.get('description','')} | args: {params or 'none'}")
        system = (
            "You route requests for a Space Domain Awareness toolchain. First decide: "
            "is the user ISSUING A COMMAND to perform an action, or ASKING A QUESTION?\n"
            "- A factual or informational question (what / where / how many / why / "
            "is / are ...) is intent 'chat'. Do NOT run a tool to answer a question.\n"
            "- Choose a tool ONLY for an imperative command to perform that tool's action.\n"
            "When extracting a numeric argument, keep the FULL number — write 20000 for "
            "'20,000', never truncate at a comma.\n"
            "Reply with ONLY minified JSON, nothing else: {\"intent\":\"<name>\",\"args\":{...}}.\n"
            f"Valid intents: {', '.join(names + [CHAT_INTENT])}.\n"
            "Examples:\n"
            "  'What is the altitude of GEO?' -> {\"intent\":\"chat\",\"args\":{}}\n"
            "  'Where is Hubble?' -> {\"intent\":\"chat\",\"args\":{}}\n"
            "  'Propagate the ISS for 120 minutes' -> {\"intent\":\"propagate_orbit\",\"args\":{\"object\":\"ISS\",\"duration_min\":120}}\n"
            "  'Propagate a satellite at 20,000 km, 35 deg inclination' -> {\"intent\":\"propagate_orbit\",\"args\":{\"altitude_km\":20000,\"inclination\":35}}\n"
            "  'Put something in GEO' -> {\"intent\":\"propagate_orbit\",\"args\":{\"altitude_km\":35786}}\n"
            "Tool catalog:\n" + "\n".join(catalog_lines)
        )
        try:
            resp = llm_client.chat(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
                max_tokens=200, temperature=0.0,
            )
        except Exception:
            return None, {}
        content = llm_client.first_message(resp).get("content", "") or ""
        return self._parse_route(content)

    @staticmethod
    def _parse_route(content: str) -> tuple[str | None, dict[str, Any]]:
        """Pull the first JSON object out of the model's reply, tolerating code
        fences and stray prose. Returns (None, {}) on anything unparseable so the
        caller can keyword-fall-back; returns ('chat', {}) only for a deliberate
        chat choice the model actually emitted."""
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None, {}
        try:
            obj = json.loads(m.group())
        except Exception:
            return None, {}
        intent = obj.get("intent")
        if not intent or not isinstance(intent, str):
            return None, {}
        args = obj.get("args", {})
        return intent.strip(), (args if isinstance(args, dict) else {})

    def _keyword_fallback(self, text: str) -> str | None:
        """A general, name-derived shortcut: split each intent name on '_' into
        tokens and match against the request words. Returns an intent only when
        exactly one candidate matches, so it never guesses between tools. Args are
        left empty; dispatch() will report any missing required field plainly."""
        words = set(re.findall(r"[a-z]+", (text or "").lower()))
        hits = []
        for name in self.closed_set():
            if name == CHAT_INTENT:
                continue
            tokens = {t for t in name.lower().split("_") if len(t) > 3}
            if tokens & words:
                hits.append(name)
        return hits[0] if len(hits) == 1 else None

    # -- deterministic text recovery (schema-declared, domain-agnostic) ------

    def _params_for(self, intent: str) -> dict[str, Any]:
        """The parameter schema for an intent — a macro's declared parameters or a
        tool's schema()['parameters']."""
        if intent in self.macros:
            return getattr(self.macros[intent], "parameters", {}) or {}
        tool = self.registry.get(intent)
        return (tool.schema().get("parameters", {}) if tool else {}) or {}

    def _recover(self, text: str, intent: str, args: dict[str, Any]) -> dict[str, Any]:
        """Backfill a param from the raw request when the model omitted it or gave a
        degenerate value. A tool/macro opts in per-param via an 'x_recover' hint
        {pattern, nonpositive}; the router stays domain-agnostic — it only applies
        whatever the tool declared. Guards the small model dropping e.g. '12 deg'."""
        props = (self._params_for(intent).get("properties") or {})
        for key, spec in props.items():
            rec = (spec or {}).get("x_recover")
            if not rec:
                continue
            cur = args.get(key)
            degenerate = cur is None
            if not degenerate and rec.get("nonpositive"):
                try:
                    degenerate = float(cur) <= 0
                except (TypeError, ValueError):
                    degenerate = True
            if degenerate:
                m = re.search(rec["pattern"], _spell_to_digits(text or ""), re.I)
                if m:
                    try:
                        args[key] = float(m.group(1))
                    except (TypeError, ValueError):
                        pass
        return args

    # -- dispatch (deterministic; validates before running) -----------------

    def dispatch(self, intent: str, args: dict[str, Any]) -> ToolResult:
        """Validate and execute. Never runs anything the closed set / schema does
        not allow. Renders the tool's artifact into the final hub dict so the
        caller reads a single, uniform ToolResult."""
        if intent == CHAT_INTENT:
            return ToolResult(error="no tool for this request")  # caller uses chat path
        if intent in self.macros:
            return self.macros[intent](self, args)
        tool = self.registry.get(intent)
        if tool is None:
            return ToolResult(error=f"unknown intent {intent!r}")
        params = tool.schema().get("parameters", {}) or {}
        clean, missing = validate_args(params, args or {})
        if missing:
            return ToolResult(error=f"{intent} needs: {', '.join(missing)}.")
        result = tool.run(clean)
        if not result.error:
            rendered = tool.render(result) or None
            result.artifact = rendered or None
        return result


# -- argument validation (pure, deterministic) ------------------------------

def validate_args(schema_params: dict[str, Any], raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Coerce and validate `raw` against a tool's JSON-schema `parameters`.

    Keeps only declared properties, coerces to the declared type, fills defaults,
    and reports any missing required fields. Unknown keys are dropped (the model
    does not get to smuggle in behavior). Returns (clean_args, missing_required).
    """
    props = schema_params.get("properties", {}) or {}
    required = schema_params.get("required", []) or []
    out: dict[str, Any] = {}
    for key, spec in props.items():
        spec = spec or {}
        if key in raw and raw[key] is not None:
            coerced = _coerce(raw[key], spec.get("type"))
            if coerced is not None:
                out[key] = coerced
        elif "default" in spec:
            out[key] = spec["default"]
    missing = [k for k in required if k not in out]
    return out, missing


_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORD_NUM_RE = re.compile(r"\b(" + "|".join(_WORD_NUM) + r")\b", re.I)


def _spell_to_digits(text: str) -> str:
    """Replace standalone spelled-out numbers with digits so the numeric recovery
    patterns match when whisper writes 'six degrees' instead of '6 degrees'."""
    return _WORD_NUM_RE.sub(lambda m: str(_WORD_NUM[m.group(0).lower()]), text or "")


def _coerce(value: Any, typ: str | None) -> Any:
    """Best-effort coercion to a JSON-schema scalar type. Returns None if the
    value cannot be coerced (so validate_args drops it rather than passing junk)."""
    try:
        if typ == "number":
            return float(value)
        if typ == "integer":
            return int(float(value))
        if typ == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "1", "yes", "on")
        if typ == "string":
            return str(value)
    except (TypeError, ValueError):
        return None
    return value  # unknown/unspecified type: pass through
