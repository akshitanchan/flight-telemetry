#!/usr/bin/env python3
"""Tests for the LangGraph plan-execute architecture (ai-04, LangGraph variant).

Same stub-LLM approach as ai/tests/test_architectures.py: a canned callable
injected via the ``llm`` constructor argument stands in for a real model, so
the graph's control flow (planning, validation, dispatch, synthesis) is
exercised entirely offline. The important test here is parity with
:class:`ai.agent.plan_execute.PlanExecuteStrategy`: given the same stub
responses, the state-graph control flow must produce the same answer as the
Python-loop control flow, since both wrap the same validated tools.
"""

import json
import socket
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.tools.analytics import AnalyticsTool
from ai.tools.retrieval import RetrievalTool
from ai.agent.langgraph_plan_execute import LangGraphPlanExecuteStrategy
from ai.agent.plan_execute import PlanExecuteStrategy

GOLD = PROJECT_ROOT / "ai" / "fixtures" / "gold"
CORPUS = PROJECT_ROOT / "ai" / "fixtures" / "corpus" / "corpus.json"


def _make_stub(responses):
    """Stub LLM callable that yields canned responses in order (repeats the last)."""
    if isinstance(responses, str):
        responses = [responses]
    state = {"idx": 0}

    def _stub(messages):  # noqa: ARG001
        idx = min(state["idx"], len(responses) - 1)
        content = responses[idx]
        state["idx"] += 1
        return content, len(content)

    return _stub


def _plan(steps):
    return json.dumps({"steps": steps})


class ArchTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analytics = AnalyticsTool(GOLD)
        cls.retrieval = RetrievalTool(CORPUS)


class TestLangGraphPlanExecute(ArchTestBase):
    def setUp(self):
        self.Cls = LangGraphPlanExecuteStrategy

    def _make(self, stub_responses, max_steps=4):
        return self.Cls(self.analytics, self.retrieval,
                         llm=_make_stub(stub_responses), max_steps=max_steps)

    # -- Parity with PlanExecuteStrategy (the important test) --

    def _assert_parity(self, stub_responses, question, max_steps=4):
        lg = self.Cls(self.analytics, self.retrieval,
                       llm=_make_stub(list(stub_responses)), max_steps=max_steps)
        pe = PlanExecuteStrategy(self.analytics, self.retrieval,
                                  llm=_make_stub(list(stub_responses)), max_steps=max_steps)
        a_lg = lg.answer(question)
        a_pe = pe.answer(question)
        self.assertEqual(a_lg.route, a_pe.route)
        self.assertEqual(a_lg.answer_text, a_pe.answer_text)
        self.assertEqual(a_lg.meta["tokens"], a_pe.meta["tokens"])
        return a_lg, a_pe

    def test_parity_single_step_analytics_plan(self):
        self._assert_parity(
            [
                _plan([{"tool": "analytics", "operation": "count_emergencies", "params": {}}]),
                "There were 3 emergency events.",
            ],
            "How many emergency events?",
        )

    def test_parity_single_step_retrieval_plan(self):
        a_lg, a_pe = self._assert_parity(
            [
                _plan([{"tool": "retrieval", "operation": "search",
                        "params": {"query": "METAR EHAM"}}]),
                "METAR decoded.",
            ],
            "Decode the METAR for EHAM.",
        )
        self.assertEqual(a_lg.route, "retrieval:search")
        self.assertIn("metar-eham", a_lg.sources)

    def test_parity_multi_step_plan(self):
        self._assert_parity(
            [
                _plan([
                    {"tool": "analytics", "operation": "count_emergencies", "params": {}},
                    {"tool": "analytics", "operation": "total_flights_tracked", "params": {}},
                ]),
                "3 emergencies out of 10 flights.",
            ],
            "How many flights declared emergency?",
        )

    # -- Whitelist enforcement --

    def test_step_outside_whitelist_is_dropped(self):
        strat = self._make([
            _plan([
                {"tool": "analytics", "operation": "drop_all_tables", "params": {}},
                {"tool": "analytics", "operation": "count_emergencies", "params": {}},
            ]),
            "3 emergencies.",
        ])
        a = strat.answer("How many emergencies?")
        self.assertEqual(a.route, "analytics:count_emergencies")
        self.assertEqual(a.result["answer"], 3)
        self.assertIn("gold_emergency_events", a.sources)

    # -- Step cap --

    def test_plan_longer_than_max_steps_is_truncated(self):
        steps = [
            {"tool": "analytics", "operation": "count_emergencies", "params": {}},
            {"tool": "analytics", "operation": "count_active_airports", "params": {}},
            {"tool": "analytics", "operation": "count_active_sectors", "params": {}},
            {"tool": "analytics", "operation": "total_flights_tracked", "params": {}},
            {"tool": "analytics", "operation": "highest_altitude_flight", "params": {}},
        ]
        strat = self._make([_plan(steps), "summary"], max_steps=2)
        a = strat.answer("Summarize everything.")
        self.assertEqual(a.meta["plan_steps"], 2)
        self.assertTrue(a.sources)
        self.assertIn("source:", a.answer_text)

    # -- Fallback --

    def test_empty_plan_falls_back_to_retrieval(self):
        strat = self._make([_plan([]), "No relevant data found."])
        a = strat.answer("What is the meaning of life?")
        self.assertEqual(a.route, "retrieval:search")
        self.assertIn("source:", a.answer_text)

    def test_unparsable_plan_falls_back_to_retrieval(self):
        strat = self._make(["not json at all", "fallback answer"])
        a = strat.answer("Something?")
        self.assertEqual(a.route, "retrieval:search")
        self.assertIn("source:", a.answer_text)

    # -- Source merging --

    def test_multi_step_sources_merged_without_duplicates(self):
        strat = self._make([
            _plan([
                {"tool": "analytics", "operation": "count_emergencies", "params": {}},
                {"tool": "analytics", "operation": "list_emergency_aircraft", "params": {}},
            ]),
            "3 emergencies across the same events.",
        ])
        a = strat.answer("Summarize the emergency events.")
        self.assertEqual(a.sources.count("gold_emergency_events"), 1)

    # -- Offline --

    def test_runs_with_no_network(self):
        strat = self._make([
            _plan([{"tool": "analytics", "operation": "count_emergencies", "params": {}}]),
            "3 emergencies.",
        ])

        def _no_socket(*args, **kwargs):
            raise AssertionError("socket.socket was called; graph must not touch the network")

        orig_socket = socket.socket
        socket.socket = _no_socket
        try:
            a = strat.answer("How many emergencies?")
        finally:
            socket.socket = orig_socket
        self.assertEqual(a.result["answer"], 3)
        self.assertIn("source:", a.answer_text)

    # -- Grounding invariant --

    def test_sources_always_populated(self):
        strat = self._make([
            _plan([{"tool": "analytics", "operation": "count_active_airports", "params": {}}]),
            "11 airports.",
        ])
        a = strat.answer("How many airports?")
        self.assertTrue(a.sources)
        self.assertIn("source:", a.answer_text)

    # -- Strategy name / meta --

    def test_strategy_name(self):
        strat = self._make([_plan([{"tool": "analytics", "operation": "count_emergencies",
                                    "params": {}}]), "3."])
        a = strat.answer("test")
        self.assertEqual(a.strategy, "langgraph_plan_execute")

    def test_meta_has_plan_steps_and_llm_calls(self):
        strat = self._make([_plan([{"tool": "analytics", "operation": "count_emergencies",
                                    "params": {}}]), "3."])
        a = strat.answer("test")
        self.assertEqual(a.meta["plan_steps"], 1)
        self.assertEqual(a.meta["llm_calls"], 2)  # plan call + synthesis call

    def test_is_available_returns_bool(self):
        import os
        if os.environ.get("OPENAI_API_KEY", ""):
            self.skipTest("OPENAI_API_KEY set")
        self.assertIsInstance(LangGraphPlanExecuteStrategy.is_available(), bool)


if __name__ == "__main__":
    unittest.main()
