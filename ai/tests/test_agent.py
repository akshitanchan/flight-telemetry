#!/usr/bin/env python3
"""Tests for the rule-based answer strategy and the answer-path eval."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.tools.analytics import AnalyticsTool
from ai.tools.retrieval import RetrievalTool
from ai.agent.rule_based import RuleBasedStrategy
from ai.eval.run_answers import evaluate

GOLD = PROJECT_ROOT / "ai" / "fixtures" / "gold"
CORPUS = PROJECT_ROOT / "ai" / "fixtures" / "corpus" / "corpus.json"


class TestRuleBasedStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strat = RuleBasedStrategy(AnalyticsTool(GOLD), RetrievalTool(CORPUS))

    def test_routes_emergency_count(self):
        a = self.strat.answer("How many emergency squawk events were recorded?")
        self.assertEqual(a.route, "analytics:count_emergencies")
        self.assertEqual(a.result["answer"], 3)

    def test_routes_emergency_by_squawk(self):
        a = self.strat.answer("How many flights squawked 7700?")
        self.assertEqual(a.route, "analytics:count_emergencies")
        self.assertEqual(a.result["answer"], 2)

    def test_routes_which_aircraft(self):
        a = self.strat.answer("Which aircraft squawked 7500?")
        self.assertEqual(a.route, "analytics:list_emergency_aircraft")
        self.assertEqual(set(a.result["answer"]), {"400a30"})

    def test_routes_airport_congestion(self):
        a = self.strat.answer("How many aircraft were associated with airport LOIR?")
        self.assertEqual(a.route, "analytics:airport_congestion")
        self.assertEqual(a.result["answer"]["aircraft_count"], 1)

    def test_routes_flight_summary(self):
        a = self.strat.answer("What was the maximum altitude reached by flight 3c6751?")
        self.assertEqual(a.route, "analytics:flight_summary")

    def test_routes_retrieval_metar(self):
        a = self.strat.answer("Decode the METAR for EHAM.")
        self.assertEqual(a.route, "retrieval:search")
        self.assertIn("metar-eham", a.sources)

    def test_answers_carry_citation(self):
        a = self.strat.answer("How many emergency squawk events were recorded?")
        self.assertTrue(a.sources)
        self.assertIn("source:", a.answer_text)


class TestAnswerEval(unittest.TestCase):
    def test_all_golden_questions_answered_correctly(self):
        results, metrics = evaluate()
        failed = [r["id"] for r in results if not r["passed"]]
        self.assertEqual(failed, [], f"failing: {failed}")
        self.assertEqual(metrics["citation_coverage"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
