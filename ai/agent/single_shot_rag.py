#!/usr/bin/env python3
"""Single-shot RAG answer strategy (ai-04, architecture A).

Control flow
------------
1. ONE LLM call with the user question.
   - The system prompt presents both tool schemas.
   - The LLM returns a JSON routing decision:
       {"tool": "analytics"|"retrieval", "operation": "<op>", "params": {...}}
2. The routing decision is validated against the tool whitelist.
   - If "analytics": the named operation is called with validated params;
     the tool computes the answer deterministically and returns {answer, sources}.
   - If "retrieval": RetrievalTool.search() retrieves context from the corpus
     and returns {answer, sources, results}.
3. For retrieval questions a second LLM call synthesises a concise answer
   grounded in the retrieved snippets (single-shot RAG).
   - The final answer_text always appends "(source: <ids>)" so citation is
     visible; sources come from the tool, never invented.
4. Answer.sources is populated from result["sources"] — the tool's own
   citation list — so grounding is 100% preserved.

LLM client
----------
Injectable via the ``llm`` constructor argument (any callable that accepts
``messages: list[dict]`` and returns ``(content_str, token_count_int)``).
When ``llm`` is None the class builds a default client from the environment:
  - OPENAI_API_KEY present -> OpenAI via stdlib urllib (no openai package).
  - OLLAMA_HOST present (or default reachable) and model available -> Ollama.
  - Neither available -> strategy is unavailable; answer() raises RuntimeError.

No sockets are opened at import time; the default client is built lazily on
the first ``answer()`` call (or can be probed with ``is_available()``).
"""

import inspect
import json
import os
import urllib.request

from ai.agent.base import Answer, AnswerStrategy

# ---------------------------------------------------------------------------
# Shared routing system prompt (identical schema to ollama_llm for consistency)
# ---------------------------------------------------------------------------

_ROUTING_SYSTEM = """You route an aviation-operations question to exactly ONE tool call.

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

_SYNTHESIS_SYSTEM = """You are an aviation-operations assistant.
The user asked a question. You are given retrieved document snippets from a
trusted corpus as context. Answer concisely and factually, drawing ONLY on
the provided context. Do not invent numbers or facts not present in the context.
Keep the answer under 3 sentences."""


# ---------------------------------------------------------------------------
# Provider helpers (stdlib urllib only — no openai package)
# ---------------------------------------------------------------------------

def _openai_chat(messages, api_key, model="gpt-4o-mini", timeout=60):
    """Call the OpenAI chat-completions endpoint via stdlib urllib."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 256,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    return content, tokens


def _ollama_chat(messages, host, model, timeout=120):
    """Call the Ollama chat endpoint via stdlib urllib."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    content = resp.get("message", {}).get("content", "")
    tokens = resp.get("prompt_eval_count", 0) + resp.get("eval_count", 0)
    return content, tokens


def _build_default_llm():
    """Return (callable, provider_name) or (None, None) if no provider available."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        def _call(messages):
            return _openai_chat(messages, api_key, model=model)
        return _call, f"openai:{model}"

    # Try Ollama
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        from ai.agent.ollama_llm import _list_models, pick_model  # type: ignore
        names = _list_models(host, timeout=2)
        model = os.environ.get("OLLAMA_MODEL") or pick_model(names)
        if model:
            def _call(messages):
                return _ollama_chat(messages, host, model)
            return _call, f"ollama:{model}"
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class SingleShotRAGStrategy(AnswerStrategy):
    """Single-shot RAG: one LLM routing call + optional grounded synthesis.

    Parameters
    ----------
    analytics:
        AnalyticsTool instance.
    retrieval:
        RetrievalTool instance.
    llm:
        Optional callable ``(messages: list[dict]) -> (str, int)``.
        If None, a provider is selected from the environment at first use.
    """

    name = "single_shot_rag"

    def __init__(self, analytics, retrieval, llm=None):
        self.analytics = analytics
        self.retrieval = retrieval
        self._llm_override = llm   # injected; may be None (lazy default)
        self._llm = None           # resolved on first use
        self._provider = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return True if at least one LLM provider is reachable."""
        fn, _ = _build_default_llm()
        return fn is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_llm(self):
        if self._llm_override is not None:
            return self._llm_override, "stub"
        if self._llm is None:
            fn, name = _build_default_llm()
            if fn is None:
                raise RuntimeError(
                    "SingleShotRAGStrategy: no LLM provider available. "
                    "Set OPENAI_API_KEY or OLLAMA_HOST."
                )
            self._llm = fn
            self._provider = name
        return self._llm, self._provider

    @staticmethod
    def _coerce_params(raw) -> dict:
        """Coerce an LLM-supplied params value to a plain dict.

        The LLM is untrusted and occasionally emits params as a JSON string,
        a bare scalar, or None instead of an object.  We normalise defensively:
          - dict           → returned as-is (common case, zero overhead).
          - str            → attempt json.loads; use result only if it is a dict.
          - anything else  → empty dict (let the tool's own validation handle it).
        A malformed params value degrades the step to an empty-params call rather
        than crashing the whole answer() invocation.
        """
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return {}

    def _route(self, question: str):
        """Ask the LLM to pick a tool. Returns (call_dict, token_count)."""
        llm, _ = self._get_llm()
        messages = [
            {"role": "system", "content": _ROUTING_SYSTEM},
            {"role": "user", "content": question},
        ]
        content, tokens = llm(messages)
        try:
            call = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            call = {}
        return call, tokens

    def _synthesise(self, question: str, snippets: str) -> tuple[str, int]:
        """Single grounded synthesis call for retrieval answers."""
        llm, _ = self._get_llm()
        messages = [
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"<context>\n{snippets}\n</context>\n\n"
                    f"Question: {question}"
                ),
            },
        ]
        return llm(messages)

    # ------------------------------------------------------------------
    # answer()
    # ------------------------------------------------------------------

    def answer(self, question: str) -> Answer:
        _, provider = self._get_llm()  # raises if unavailable
        total_tokens = 0
        llm_calls = 0

        # Step 1: LLM routing call
        call, tok = self._route(question)
        total_tokens += tok
        llm_calls += 1

        tool = call.get("tool")
        operation = call.get("operation")
        params = self._coerce_params(call.get("params"))
        if "squawk" in params and params["squawk"] is not None:
            params["squawk"] = str(params["squawk"])

        # Step 2: validated tool execution
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

        # Step 3: for retrieval, synthesise a grounded answer
        if route == "retrieval:search":
            snippets = result.get("answer", "")
            if snippets:
                try:
                    synthesis, tok2 = self._synthesise(question, snippets)
                    total_tokens += tok2
                    llm_calls += 1
                    answer_body = synthesis.strip()
                except Exception:
                    answer_body = snippets
            else:
                answer_body = "No relevant documents found."
        else:
            answer_body = str(result.get("answer", ""))

        cites = ", ".join(result.get("sources", [])) or "none"
        _, prov = self._get_llm()

        return Answer(
            question=question,
            answer_text=f"{answer_body} (source: {cites})",
            result=result,
            route=route,
            strategy=self.name,
            meta={
                "provider": provider,
                "tokens": total_tokens,
                "llm_calls": llm_calls,
            },
        )
