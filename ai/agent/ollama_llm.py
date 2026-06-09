#!/usr/bin/env python3
"""LLM answer strategy (W3.3, strategy A) backed by a local Ollama model.

The LLM performs the *routing* — it picks one tool operation and its params as
JSON — and the tool executes deterministically, so answers stay grounded (the
model never invents numbers). This isolates the LLM's contribution (intent
classification) and keeps the eval checks meaningful.

Availability-gated: if the Ollama server or a usable model is absent, the
strategy reports itself unavailable and the comparison harness skips it (so CI
and offline runs stay green). Uses only the stdlib (urllib) — no extra deps.

The model is auto-selected from installed models (override with OLLAMA_MODEL),
preferring those best suited to structured JSON tool-routing.
"""

import inspect
import json
import os
import urllib.request

from ai.agent.base import Answer, AnswerStrategy

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Preference order for auto-selection (best-suited for JSON tool-routing first).
# deepseek-r1 is excluded by default — its reasoning tokens complicate JSON output.
_PREFERENCE = ["qwen2.5-coder", "llama3.2", "mistral", "gemma3", "qwen2.5vl", "gemma4", "phi3"]

_SYSTEM = """You route an aviation-operations question to exactly ONE tool call.

Analytics operations (use tool "analytics"):
- count_emergencies(squawk?)         number of emergency squawk events; optional squawk "7500"/"7600"/"7700"
- list_emergency_aircraft(squawk?)   icao24 ids of aircraft that squawked an emergency code
- airport_congestion(airport_icao)   aircraft count near a 4-letter airport ICAO
- count_active_airports()            number of distinct airports with traffic
- count_active_sectors()             number of distinct H3 sectors active
- total_flights_tracked()            number of distinct flights
- flight_summary(icao24)             altitude/speed/pings for a 6-hex aircraft id
- highest_altitude_flight()          the flight with the highest altitude

For METAR/weather decoding, squawk-code meaning, or incident/report lookups use
tool "retrieval" with operation "search" and params {"query": "<the question>"}.

Respond with ONLY a JSON object, no prose:
{"tool": "analytics"|"retrieval", "operation": "<name or search>", "params": {...}}"""


def _list_models(host=DEFAULT_HOST, timeout=3):
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            data = json.load(r)
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def pick_model(names):
    """Choose the most suitable installed model (exact base match, then prefix)."""
    if not names:
        return None
    for pref in _PREFERENCE:
        for n in names:
            if n.split(":")[0] == pref:
                return n
    for pref in _PREFERENCE:
        for n in names:
            if n.startswith(pref):
                return n
    return names[0]


class OllamaStrategy(AnswerStrategy):
    def __init__(self, analytics, retrieval, model=None, host=DEFAULT_HOST, timeout=120):
        self.analytics = analytics
        self.retrieval = retrieval
        self.host = host
        self.timeout = timeout
        self.model = model or os.environ.get("OLLAMA_MODEL") or pick_model(_list_models(host))
        self.name = f"ollama:{self.model}" if self.model else "ollama:unavailable"

    @classmethod
    def available(cls, host=DEFAULT_HOST):
        """Return (available, model, installed_names)."""
        names = _list_models(host)
        chosen = os.environ.get("OLLAMA_MODEL") or pick_model(names)
        return (bool(names and chosen), chosen, names)

    def _chat(self, question):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.load(r)
        content = resp.get("message", {}).get("content", "")
        tokens = resp.get("prompt_eval_count", 0) + resp.get("eval_count", 0)
        return content, tokens

    def answer(self, question: str) -> Answer:
        content, tokens = self._chat(question)
        meta = {"model": self.model, "tokens": tokens, "llm_calls": 1}
        try:
            call = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            call = {}

        tool = call.get("tool")
        operation = call.get("operation")
        params = call.get("params") or {}
        if "squawk" in params and params["squawk"] is not None:
            params["squawk"] = str(params["squawk"])

        if tool == "analytics" and operation in self.analytics.operations:
            sig = inspect.signature(self.analytics.operations[operation])
            allowed = {k: v for k, v in params.items() if k in sig.parameters}
            try:
                result = self.analytics.call(operation, **allowed)
                route = f"analytics:{operation}"
            except Exception:
                result = self.retrieval.search(question)
                route = "retrieval:search"
        else:
            query = params.get("query") or question
            result = self.retrieval.search(query)
            route = "retrieval:search"

        cites = ", ".join(result.get("sources", [])) or "none"
        return Answer(question=question,
                      answer_text=f"{result.get('answer')} (source: {cites})",
                      result=result, route=route, strategy=self.name, meta=meta)
