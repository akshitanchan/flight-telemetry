#!/usr/bin/env python3
"""ReAct (Reason + Act) answer strategy (ai-04, architecture B).

Control flow (bounded loop, max ``max_steps`` iterations)
---------------------------------------------------------
Each iteration:
  Thought  — LLM receives the question + the full conversation history
             (previous thoughts, actions, and observations) and emits:
               {"thought": "...", "action": {"tool": ..., "operation": ..., "params": ...}}
             OR the terminal marker:
               {"thought": "...", "action": {"tool": "finish", "answer": "..."}}
             (The LLM is NEVER allowed to emit a final numeric answer without
              a preceding tool call — if it tries to use "finish" before any
              observation, the loop inserts one forced tool step.)

  Act      — The action is validated against the tool whitelist. Any tool
             not on the whitelist is rejected and an error observation is fed
             back so the LLM can correct itself.

  Observe  — The tool result is appended to the history as an "observation".
             Observations carry the tool's own sources list.

Terminal condition
------------------
The loop exits when:
  a) The LLM emits action.tool == "finish"; OR
  b) ``max_steps`` iterations are exhausted (hard cap; prevents runaway cost).

In case (b) the last tool observation is used as the final answer.

Sources
-------
All sources accumulated from every tool call are union-merged into the final
result's "sources" list so the Answer always carries a citation even when
multiple tools were consulted.

LLM client
----------
Same injectable pattern as SingleShotRAGStrategy: constructor arg ``llm``
accepts any callable ``(messages) -> (str, int)``; None falls back to
``ai.providers.build_default`` for env-based provider resolution.
"""

import inspect
import json

from ai.agent.base import Answer, AnswerStrategy
from ai.providers import build_default

_MAX_STEPS_DEFAULT = 4

_SYSTEM = """You are an aviation-operations reasoning agent that uses tools to answer questions.

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

At each step respond with ONLY a JSON object:
  {"thought": "<your reasoning>", "action": {"tool": "<tool>", "operation": "<op>", "params": {...}}}

When you have enough information from the tool observations to answer, emit the finish action:
  {"thought": "<final reasoning>", "action": {"tool": "finish", "answer": "<concise answer>"}}

Rules:
- NEVER invent numbers. Always call a tool to obtain data first.
- You MUST call at least one tool before finishing.
- If the tool returns an error, try a different operation or parameter.
- squawk params must be strings: "7500", "7600", or "7700"."""


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class ReActStrategy(AnswerStrategy):
    """ReAct loop: LLM reasons → calls a validated tool → observes → repeats.

    Parameters
    ----------
    analytics:
        AnalyticsTool instance.
    retrieval:
        RetrievalTool instance.
    llm:
        Optional injectable callable ``(messages) -> (str, int)``.
    max_steps:
        Hard cap on the number of reason-act-observe iterations (default 4).
    """

    name = "react"

    def __init__(self, analytics, retrieval, llm=None, max_steps=_MAX_STEPS_DEFAULT):
        self.analytics = analytics
        self.retrieval = retrieval
        self._llm_override = llm
        self._llm = None
        self._provider = None
        self.max_steps = max_steps

    @classmethod
    def is_available(cls) -> bool:
        fn, _ = build_default()
        return fn is not None

    def _get_llm(self):
        if self._llm_override is not None:
            return self._llm_override, "stub"
        if self._llm is None:
            fn, name = build_default()
            if fn is None:
                raise RuntimeError(
                    "ReActStrategy: no LLM provider available. "
                    "Set OPENAI_API_KEY or OLLAMA_HOST."
                )
            self._llm = fn
            self._provider = name
        return self._llm, self._provider

    # ------------------------------------------------------------------
    # Tool execution (validate + dispatch)
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

    def _execute_tool(self, action: dict) -> tuple[dict, str]:
        """Execute a validated tool action. Returns (result_dict, route_str)."""
        tool = action.get("tool", "")
        operation = action.get("operation", "")
        params = self._coerce_params(action.get("params"))
        if "squawk" in params and params["squawk"] is not None:
            params["squawk"] = str(params["squawk"])

        if tool == "analytics" and operation in self.analytics.operations:
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
        elif tool == "retrieval":
            query = params.get("query") or ""
            result = self.retrieval.search(query)
            return result, "retrieval:search"
        else:
            return {
                "answer": f"unknown tool/operation: {tool}/{operation}",
                "sources": [],
                "rows": [],
            }, "unknown"

    # ------------------------------------------------------------------
    # answer()
    # ------------------------------------------------------------------

    def answer(self, question: str) -> Answer:
        llm, provider = self._get_llm()
        total_tokens = 0
        llm_calls = 0

        # Conversation history fed to the LLM each step.
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": question},
        ]

        # Accumulate state across steps.
        all_sources: list[str] = []
        last_result: dict = {"answer": None, "sources": [], "rows": []}
        last_route = "retrieval:search"   # safe fallback
        finish_answer: str | None = None
        tool_called = False

        for step in range(self.max_steps):
            # -- Thought + Action --
            content, tok = llm(messages)
            total_tokens += tok
            llm_calls += 1

            try:
                step_json = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                # Non-JSON response: treat as a retrieval fallback.
                step_json = {
                    "thought": "could not parse LLM response",
                    "action": {"tool": "retrieval", "operation": "search",
                               "params": {"query": question}},
                }

            action = step_json.get("action", {})
            thought = step_json.get("thought", "")

            # Enforce: LLM must call a tool before finishing.
            if action.get("tool") == "finish" and not tool_called:
                action = {"tool": "retrieval", "operation": "search",
                          "params": {"query": question}}

            if action.get("tool") == "finish":
                finish_answer = str(action.get("answer", ""))
                break

            # -- Act: execute the validated tool --
            result, route = self._execute_tool(action)
            tool_called = True
            last_result = result
            last_route = route

            new_sources = result.get("sources", [])
            for s in new_sources:
                if s not in all_sources:
                    all_sources.append(s)

            # -- Observe: append observation to conversation --
            observation_text = (
                f"Tool result: {json.dumps(result.get('answer'))}\n"
                f"Sources: {json.dumps(new_sources)}"
            )
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation_text}",
            })

        # If the loop exhausted steps without a finish, use the last observation.
        if finish_answer is None:
            finish_answer = str(last_result.get("answer", "No answer found."))

        # Merge sources back into result dict for citation checks.
        final_result = dict(last_result)
        final_result["sources"] = all_sources if all_sources else last_result.get("sources", [])

        cites = ", ".join(final_result["sources"]) or "none"
        return Answer(
            question=question,
            answer_text=f"{finish_answer} (source: {cites})",
            result=final_result,
            route=last_route,
            strategy=self.name,
            meta={
                "provider": provider,
                "tokens": total_tokens,
                "llm_calls": llm_calls,
                "steps": llm_calls,
            },
        )
