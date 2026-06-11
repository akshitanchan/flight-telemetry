#!/usr/bin/env python3
"""Plan-Execute answer strategy (ai-04, architecture C).

Control flow
------------
Phase 1 — PLAN (one LLM call)
  The LLM receives the question and produces a JSON plan: an ordered list of
  tool steps. Each step has ``tool``, ``operation``, and ``params`` fields.
  The plan is validated before any execution begins:
    - Unknown operations are removed; an error is logged.
    - An empty plan (or one reduced to nothing by validation) triggers a
      retrieval fallback so the strategy always produces a grounded answer.
  Step count is capped at ``max_steps`` to bound cost.

Phase 2 — EXECUTE (one tool call per validated step, no further LLM calls)
  Each step is dispatched through the whitelist (AnalyticsTool.call or
  RetrievalTool.search). Tool results accumulate in a "context" list. If a
  step fails, an error observation is recorded and execution continues.

Phase 3 — SYNTHESISE (one LLM call)
  A final LLM call receives the question and ALL accumulated tool results
  (as structured context) and produces a concise natural-language answer.
  Crucially: the prompt instructs the LLM to answer only from the provided
  context (no number invention). The answer_text is then appended with the
  union of all sources from all executed steps.

Sources
-------
All sources gathered from every executed step are union-merged into
result["sources"] for citation-coverage checks. The strategy ALWAYS produces
at least one source (even the retrieval fallback produces one).

LLM client
----------
Same injectable pattern as the other architectures: constructor arg ``llm``
accepts any callable ``(messages) -> (str, int)``; None falls back to
env-based provider resolution (OPENAI_API_KEY or OLLAMA_HOST).
"""

import inspect
import json
import logging
import os
import urllib.request

from ai.agent.base import Answer, AnswerStrategy

logger = logging.getLogger(__name__)

_MAX_STEPS_DEFAULT = 4

_PLAN_SYSTEM = """You are an aviation-operations planning agent.
Given a question, produce an ordered plan of tool calls needed to answer it.

Available tools:
  analytics:
    - count_emergencies(squawk?)         number of emergency squawk events
    - list_emergency_aircraft(squawk?)   list of icao24 ids with an emergency code
    - airport_congestion(airport_icao)   aircraft count at a 4-letter ICAO airport
    - count_active_airports()            number of distinct airports with traffic
    - count_active_sectors()             number of distinct H3 sectors active
    - total_flights_tracked()            number of distinct flights
    - flight_summary(icao24)             altitude/speed/pings for a 6-hex aircraft id
    - highest_altitude_flight()          the flight with the highest altitude
  retrieval:
    - search(query)                      find documents in the corpus

Respond with ONLY a JSON object containing a "steps" array:
{
  "steps": [
    {"tool": "analytics"|"retrieval", "operation": "<name or search>", "params": {...}},
    ...
  ]
}

Rules:
- Produce the MINIMUM number of steps needed (usually 1-2).
- squawk params must be strings: "7500", "7600", or "7700".
- Never include steps that compute data you will invent — only real operations.
- For multi-part questions that need two different operations, include both."""

_SYNTHESIS_SYSTEM = """You are an aviation-operations assistant.
You are given the results of one or more tool calls made to answer a question.
Synthesise a concise, factual answer (2-3 sentences maximum) using ONLY the
provided tool results. Do not invent numbers, aircraft IDs, or names not present
in the results. If the results are insufficient, say so."""


# ---------------------------------------------------------------------------
# Shared LLM plumbing
# ---------------------------------------------------------------------------

def _openai_chat(messages, api_key, model="gpt-4o-mini", timeout=60):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 512,
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
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
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
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        def _call(messages):
            return _openai_chat(messages, api_key, model=model)
        return _call, f"openai:{model}"
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

class PlanExecuteStrategy(AnswerStrategy):
    """Plan-execute: LLM plans → validated tool execution → LLM synthesis.

    Parameters
    ----------
    analytics:
        AnalyticsTool instance.
    retrieval:
        RetrievalTool instance.
    llm:
        Optional injectable callable ``(messages) -> (str, int)``.
    max_steps:
        Maximum number of plan steps to execute (default 4).
    """

    name = "plan_execute"

    def __init__(self, analytics, retrieval, llm=None, max_steps=_MAX_STEPS_DEFAULT):
        self.analytics = analytics
        self.retrieval = retrieval
        self._llm_override = llm
        self._llm = None
        self._provider = None
        self.max_steps = max_steps

    @classmethod
    def is_available(cls) -> bool:
        fn, _ = _build_default_llm()
        return fn is not None

    def _get_llm(self):
        if self._llm_override is not None:
            return self._llm_override, "stub"
        if self._llm is None:
            fn, name = _build_default_llm()
            if fn is None:
                raise RuntimeError(
                    "PlanExecuteStrategy: no LLM provider available. "
                    "Set OPENAI_API_KEY or OLLAMA_HOST."
                )
            self._llm = fn
            self._provider = name
        return self._llm, self._provider

    # ------------------------------------------------------------------
    # Phase 1: Plan
    # ------------------------------------------------------------------

    def _plan(self, question: str) -> tuple[list[dict], int]:
        """Ask the LLM to produce a plan. Returns (steps, tokens)."""
        llm, _ = self._get_llm()
        messages = [
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": question},
        ]
        content, tokens = llm(messages)
        try:
            plan = json.loads(content)
            steps = plan.get("steps", [])
            if not isinstance(steps, list):
                steps = []
        except (json.JSONDecodeError, TypeError):
            steps = []
        return steps, tokens

    def _validate_steps(self, steps: list[dict]) -> list[dict]:
        """Filter steps to only those with valid tool/operation combinations."""
        valid = []
        for step in steps:
            tool = step.get("tool", "")
            operation = step.get("operation", "")
            if tool == "analytics" and operation in self.analytics.operations:
                valid.append(step)
            elif tool == "retrieval" and operation in ("search",):
                valid.append(step)
            else:
                logger.warning("plan_execute: dropping invalid step %r", step)
        return valid

    # ------------------------------------------------------------------
    # Phase 2: Execute
    # ------------------------------------------------------------------

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

    def _execute_step(self, step: dict) -> tuple[dict, str]:
        """Execute one validated step. Returns (result_dict, route_str)."""
        tool = step.get("tool", "")
        operation = step.get("operation", "")
        params = self._coerce_params(step.get("params"))
        if "squawk" in params and params["squawk"] is not None:
            params["squawk"] = str(params["squawk"])

        if tool == "analytics":
            sig = inspect.signature(self.analytics.operations[operation])
            allowed = {k: v for k, v in params.items() if k in sig.parameters}
            try:
                result = self.analytics.call(operation, **allowed)
                return result, f"analytics:{operation}"
            except Exception as exc:
                return {
                    "answer": f"error: {exc}",
                    "sources": [],
                    "rows": [],
                }, f"analytics:{operation}:error"
        else:  # retrieval
            query = params.get("query", "")
            result = self.retrieval.search(query)
            return result, "retrieval:search"

    # ------------------------------------------------------------------
    # Phase 3: Synthesise
    # ------------------------------------------------------------------

    def _synthesise(self, question: str, context_blocks: list[dict]) -> tuple[str, int]:
        """Final LLM call: synthesise a grounded answer from all tool results."""
        llm, _ = self._get_llm()
        context_text = "\n\n".join(
            f"Step {i + 1} ({b['route']}):\n{json.dumps(b['result'].get('answer'), indent=2)}"
            for i, b in enumerate(context_blocks)
        )
        messages = [
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"<tool_results>\n{context_text}\n</tool_results>\n\n"
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

        # Phase 1: plan
        raw_steps, tok = self._plan(question)
        total_tokens += tok
        llm_calls += 1

        valid_steps = self._validate_steps(raw_steps)[: self.max_steps]

        # Fallback: if no valid steps, retrieve directly
        if not valid_steps:
            valid_steps = [
                {"tool": "retrieval", "operation": "search",
                 "params": {"query": question}}
            ]

        # Phase 2: execute all steps
        all_sources: list[str] = []
        context_blocks: list[dict] = []
        last_result: dict = {"answer": None, "sources": [], "rows": []}
        last_route = "retrieval:search"

        for step in valid_steps:
            result, route = self._execute_step(step)
            last_result = result
            last_route = route
            for s in result.get("sources", []):
                if s not in all_sources:
                    all_sources.append(s)
            context_blocks.append({"route": route, "result": result})

        # Phase 3: synthesise
        if context_blocks:
            try:
                synthesis, tok2 = self._synthesise(question, context_blocks)
                total_tokens += tok2
                llm_calls += 1
                answer_body = synthesis.strip()
            except Exception:
                answer_body = str(last_result.get("answer", "No answer found."))
        else:
            answer_body = str(last_result.get("answer", "No answer found."))

        # Merge all sources into final result for citation checks.
        final_result = dict(last_result)
        final_result["sources"] = all_sources if all_sources else last_result.get("sources", [])

        cites = ", ".join(final_result["sources"]) or "none"
        return Answer(
            question=question,
            answer_text=f"{answer_body} (source: {cites})",
            result=final_result,
            route=last_route,
            strategy=self.name,
            meta={
                "provider": provider,
                "tokens": total_tokens,
                "llm_calls": llm_calls,
                "plan_steps": len(valid_steps),
            },
        )
