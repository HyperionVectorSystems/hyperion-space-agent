"""LLM client — OpenAI-compatible.

The orchestrator talks to the model over the OpenAI chat-completions API. That
means the same code drives a local llama.cpp `llama-server`, a local Ollama
(OpenAI-compat endpoint), or a remote hosted API: only LLM_BASE_URL changes.
This is the seam that lets the hosted version swap the local model for a remote
one with no rearchitecture.
"""

from __future__ import annotations

import os
from typing import Any

import requests

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "local")   # label; llama-server serves the loaded -m
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")    # only needed for a remote API
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "120"))


def chat(messages: list[dict[str, Any]],
         tools: list[dict[str, Any]] | None = None,
         max_tokens: int = 1024,
         temperature: float = 0.2) -> dict[str, Any]:
    """One chat-completions call. Returns the raw response dict. Pass `tools`
    (a list of function schemas) on the routing pass; omit it on the grounding
    pass. Raises requests.HTTPError on a transport error so the caller can
    surface a plain failure rather than a hallucinated success."""
    payload: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        payload["tools"] = [{"type": "function", "function": t} for t in tools]
        payload["tool_choice"] = "auto"
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    r = requests.post(f"{LLM_BASE_URL}/chat/completions", json=payload,
                      headers=headers, timeout=LLM_TIMEOUT)
    r.raise_for_status()
    return r.json()


def first_message(resp: dict[str, Any]) -> dict[str, Any]:
    """Pull the assistant message from a completion response."""
    return resp.get("choices", [{}])[0].get("message", {}) or {}
