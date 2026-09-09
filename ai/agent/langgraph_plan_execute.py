#!/usr/bin/env python3
"""Plan-execute strategy expressed as an explicit LangGraph state graph.

Same three-phase architecture as :mod:`ai.agent.plan_execute` (plan, execute,
synthesise) but with the control flow made explicit as graph nodes and edges
instead of a Python loop:

    START -> planner -> select_tool -> execute_tool -> route_next
                              ^                             |
                              +----- (more steps left) ------+
                                             |
                                     (steps exhausted)
                                             v
                                        synthesise -> END

``planner`` and ``synthesise`` are the only nodes that call the LLM; the LLM
only ever picks tools, never produces a number. ``select_tool``/``execute_tool``
validate and dispatch through the same whitelisted tools as every other
strategy, so the only place this architecture can diverge from plan-execute's
answers is in how steps are sequenced, not in what the tools compute.
"""

import inspect
import json
import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from ai.agent.base import Answer, AnswerStrategy
from ai.agent.plan_execute import _PLAN_SYSTEM, _SYNTHESIS_SYSTEM, _valid_step, _coerce_params
from ai.providers import build_default

logger = logging.getLogger(__name__)

_MAX_STEPS_DEFAULT = 4


class _State(TypedDict):
    question: str
    plan: list[dict]
    cursor: int
    context: Annotated[list[dict], operator.add]
    sources: Annotated[list[str], operator.add]
    tokens: Annotated[int, operator.add]
    llm_calls: Annotated[int, operator.add]
    route: str
    answer: str


class LangGraphPlanExecuteStrategy(AnswerStrategy):
    """Plan-execute as a LangGraph state graph (same tools, same prompts).

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

    name = "langgraph_plan_execute"

    def __init__(self, analytics, retrieval, llm=None, max_steps=_MAX_STEPS_DEFAULT):
        self.analytics = analytics
        self.retrieval = retrieval
        self._llm_override = llm
        self._llm = None
        self._provider = None
        self.max_steps = max_steps
        self._graph = self._build_graph()

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
                    "LangGraphPlanExecuteStrategy: no LLM provider available. "
                    "Set OPENAI_API_KEY or OLLAMA_HOST."
                )
            self._llm = fn
            self._provider = name
        return self._llm, self._provider


    def _build_graph(self):
        graph = StateGraph(_State)
        graph.add_node("planner", self._planner)
        graph.add_node("select_tool", self._select_tool)
        graph.add_node("execute_tool", self._execute_tool)
        graph.add_node("synthesise", self._synthesise)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "select_tool")
        graph.add_edge("select_tool", "execute_tool")
        graph.add_conditional_edges(
            "execute_tool",
            self._route_next,
            {"select_tool": "select_tool", "synthesise": "synthesise"},
        )
        graph.add_edge("synthesise", END)
        return graph.compile()


    def _planner(self, state: _State) -> dict:
        # malformed llm output degrades to an empty plan; select_tool falls back to retrieval
        llm, _ = self._get_llm()
        messages = [
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": state["question"]},
        ]
        content, tokens = llm(messages)
        try:
            parsed = json.loads(content)
            steps = parsed.get("steps", [])
            if not isinstance(steps, list):
                steps = []
        except (json.JSONDecodeError, TypeError):
            steps = []
        return {"plan": steps, "cursor": 0, "tokens": tokens, "llm_calls": 1}

    def _select_tool(self, state: _State) -> dict:
        # no llm/tool calls; just walks the plan to find the next valid step
        plan = list(state["plan"])
        cursor = state["cursor"]
        limit = min(len(plan), self.max_steps)
        while cursor < limit:
            if _valid_step(plan[cursor], self.analytics):
                break
            cursor += 1
        else:
            if state["context"]:
                # remaining steps were all invalid but we already have grounded
                # results from earlier steps, so there is nothing left to run
                return {"cursor": limit}
            plan = plan + [
                {"tool": "retrieval", "operation": "search",
                 "params": {"query": state["question"]}}
            ]
            cursor = len(plan) - 1
        plan[cursor] = self._normalize(plan[cursor])
        return {"plan": plan, "cursor": cursor}

    def _execute_tool(self, state: _State) -> dict:
        plan = state["plan"]
        cursor = state["cursor"]
        limit = min(len(plan), self.max_steps)
        if cursor >= limit:
            # select_tool already parked the cursor at the limit; nothing to run
            return {}
        step = plan[cursor]
        tool = step["tool"]
        operation = step["operation"]
        params = step["params"]
        if tool == "analytics":
            try:
                result = self.analytics.call(operation, **params)
                route = f"analytics:{operation}"
            except Exception as exc:
                result = {"answer": f"error: {exc}", "sources": [], "rows": []}
                route = f"analytics:{operation}:error"
        else:  # retrieval
            result = self.retrieval.search(**params)
            route = "retrieval:search"
        return {
            "context": [{"route": route, "result": result}],
            "sources": list(result.get("sources", [])),
            "route": route,
            "cursor": cursor + 1,
        }

    def _route_next(self, state: _State) -> str:
        limit = min(len(state["plan"]), self.max_steps)
        return "select_tool" if state["cursor"] < limit else "synthesise"

    def _synthesise(self, state: _State) -> dict:
        llm, _ = self._get_llm()
        context_blocks = state["context"]
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
                    f"Question: {state['question']}"
                ),
            },
        ]
        try:
            content, tokens = llm(messages)
            return {"answer": content.strip(), "tokens": tokens, "llm_calls": 1}
        except Exception:
            # synthesis failure still yields an answer, using the last tool result verbatim
            last = context_blocks[-1]["result"] if context_blocks else {"answer": None}
            return {"answer": str(last.get("answer", "No answer found."))}


    def _normalize(self, step: dict) -> dict:
        tool = step.get("tool", "")
        operation = step.get("operation", "")
        params = _coerce_params(step.get("params"))
        sig = inspect.signature(
            self.analytics.operations[operation] if tool == "analytics" else self.retrieval.search
        )
        # drop params the target operation doesn't accept; retrieval always needs a query key
        allowed = {k: v for k, v in params.items() if k in sig.parameters}
        if tool == "retrieval":
            allowed.setdefault("query", "")
        return {"tool": tool, "operation": operation, "params": allowed}


    def answer(self, question: str) -> Answer:
        _, provider = self._get_llm()  # raises if unavailable
        initial_state: _State = {
            "question": question,
            "plan": [],
            "cursor": 0,
            "context": [],
            "sources": [],
            "tokens": 0,
            "llm_calls": 0,
            "route": "retrieval:search",
            "answer": "",
        }
        final = self._graph.invoke(
            initial_state, config={"recursion_limit": 2 * self.max_steps + 4}
        )

        sources: list[str] = []
        for s in final["sources"]:
            if s not in sources:
                sources.append(s)

        context_blocks = final["context"]
        last_result = context_blocks[-1]["result"] if context_blocks \
            else {"answer": None, "sources": [], "rows": []}
        result = dict(last_result)
        result["sources"] = sources if sources else last_result.get("sources", [])

        cites = ", ".join(result["sources"]) or "none"
        return Answer(
            question=question,
            answer_text=f"{final['answer']} (source: {cites})",
            result=result,
            route=final["route"],
            strategy=self.name,
            meta={
                "provider": provider,
                "tokens": final["tokens"],
                "llm_calls": final["llm_calls"],
                "plan_steps": len(context_blocks),
            },
        )
