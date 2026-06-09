#!/usr/bin/env python3
"""Tests for the second deterministic strategy and the comparison harness.

The Ollama LLM strategy is intentionally excluded here (include_llm=False) so the
suite stays offline and CI-safe; ai-compare-small exercises it when a server is up.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.tools.analytics import AnalyticsTool
from ai.tools.retrieval import RetrievalTool
from ai.agent.keyword_router import KeywordScoreStrategy
from ai.eval.compare import compare

GOLD = PROJECT_ROOT / "ai" / "fixtures" / "gold"
CORPUS = PROJECT_ROOT / "ai" / "fixtures" / "corpus" / "corpus.json"


class TestKeywordScoreStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strat = KeywordScoreStrategy(AnalyticsTool(GOLD), RetrievalTool(CORPUS))

    def _route(self, q):
        return self.strat.answer(q).route

    def test_emergency_count(self):
        self.assertEqual(self._route("How many emergency squawk events were recorded?"),
                         "analytics:count_emergencies")

    def test_which_aircraft(self):
        self.assertEqual(self._route("Which aircraft squawked 7700?"),
                         "analytics:list_emergency_aircraft")

    def test_airport_congestion(self):
        self.assertEqual(self._route("How many aircraft were associated with airport LOIR?"),
                         "analytics:airport_congestion")

    def test_flight_summary_vs_highest(self):
        self.assertEqual(self._route("Maximum altitude reached by flight 3c6751?"),
                         "analytics:flight_summary")
        self.assertEqual(self._route("Which flight reached the highest altitude?"),
                         "analytics:highest_altitude_flight")

    def test_retrieval(self):
        self.assertEqual(self._route("Decode the METAR for EHAM."), "retrieval:search")


class TestComparison(unittest.TestCase):
    def test_deterministic_strategies_perfect(self):
        summaries, _ = compare(gold_dir=GOLD, corpus_path=CORPUS, include_llm=False)
        names = {s["strategy"] for s in summaries}
        self.assertEqual(names, {"rule_based_v1", "keyword_score_v1"})
        for s in summaries:
            self.assertEqual(s["passed"], s["total"], f"{s['strategy']} failed some questions")
            self.assertEqual(s["accuracy"], 1.0)
            self.assertEqual(s["citation_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
