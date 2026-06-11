#!/usr/bin/env python3
"""Tests for dashboard.agent_box — offline, no Streamlit, no LLM keys.

Conventions mirror dashboard/tests/test_data.py:
  - unittest.TestCase subclasses
  - PROJECT_ROOT path insertion for runability without an editable install
  - Reads real gold files from data/processed/ and the real corpus
  - NEVER imports streamlit; NEVER requires OPENAI_API_KEY / OLLAMA_HOST

Run with:
    python -m pytest dashboard/tests/test_agent_box.py -v
    python -m unittest dashboard.tests.test_agent_box -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# These imports must succeed with no Streamlit, no LLM, no services.
from dashboard.agent_box import (  # noqa: E402
    AnswerResult,
    answer_question,
    build_default_strategy,
)

REAL_GOLD_DIR = PROJECT_ROOT / "data" / "processed"
REAL_CORPUS = PROJECT_ROOT / "ai" / "fixtures" / "corpus" / "corpus.json"


class TestImportNoStreamlit(unittest.TestCase):
    """Guard: dashboard.agent_box must never import streamlit."""

    def test_streamlit_not_in_source(self):
        import re

        import dashboard.agent_box as mod

        source = Path(mod.__file__).read_text()
        pattern = re.compile(r"^\s*(import streamlit|from streamlit\b)", re.MULTILINE)
        matches = pattern.findall(source)
        self.assertEqual(
            matches,
            [],
            f"dashboard/agent_box.py must not contain a streamlit import; found: {matches}",
        )

    def test_streamlit_not_in_sys_modules_after_import(self):
        # Streamlit should not have been loaded as a side-effect of the import.
        self.assertNotIn("streamlit", sys.modules)


class TestAnswerResultDataclass(unittest.TestCase):
    """AnswerResult is a plain dataclass — serialisable, no Streamlit types."""

    def test_direct_construction(self):
        r = AnswerResult(
            answer_text="42 flights.",
            sources=["gold_routing_stats"],
            route="analytics:total_flights_tracked",
            strategy="rule_based_v1",
        )
        self.assertEqual(r.answer_text, "42 flights.")
        self.assertEqual(r.sources, ["gold_routing_stats"])

    def test_defaults(self):
        r = AnswerResult(answer_text="ok")
        self.assertEqual(r.sources, [])
        self.assertEqual(r.route, "")
        self.assertEqual(r.strategy, "")


class TestBuildDefaultStrategy(unittest.TestCase):
    """build_default_strategy returns a RuleBasedStrategy pointing at real data."""

    def test_returns_strategy_with_correct_type(self):
        from ai.agent.rule_based import RuleBasedStrategy

        s = build_default_strategy(REAL_GOLD_DIR, REAL_CORPUS)
        self.assertIsInstance(s, RuleBasedStrategy)

    def test_analytics_tool_loaded(self):
        s = build_default_strategy(REAL_GOLD_DIR, REAL_CORPUS)
        # AnalyticsTool.call is the public dispatch method
        self.assertTrue(callable(s.analytics.call))

    def test_retrieval_tool_loaded(self):
        s = build_default_strategy(REAL_GOLD_DIR, REAL_CORPUS)
        self.assertTrue(callable(s.retrieval.search))

    def test_custom_paths_accepted(self):
        """Passing explicit paths should not raise."""
        build_default_strategy(
            gold_dir=str(REAL_GOLD_DIR),
            corpus_path=str(REAL_CORPUS),
        )


class TestAnswerQuestionEmergencyCount(unittest.TestCase):
    """Core offline acceptance test: emergency count returns grounded answer."""

    @classmethod
    def setUpClass(cls):
        cls.result = answer_question(
            "How many emergency squawk events were there?",
            gold_dir=REAL_GOLD_DIR,
            corpus_path=REAL_CORPUS,
        )

    def test_returns_answer_result(self):
        self.assertIsInstance(self.result, AnswerResult)

    def test_answer_text_non_empty(self):
        self.assertTrue(
            self.result.answer_text.strip(),
            "answer_text must not be empty",
        )

    def test_at_least_one_source(self):
        self.assertGreaterEqual(
            len(self.result.sources),
            1,
            f"Expected ≥1 source entry; got: {self.result.sources}",
        )

    def test_source_is_emergency_table(self):
        self.assertIn(
            "gold_emergency_events",
            self.result.sources,
            f"Expected gold_emergency_events in sources; got: {self.result.sources}",
        )

    def test_route_is_analytics(self):
        self.assertTrue(
            self.result.route.startswith("analytics:"),
            f"Expected analytics route; got: {self.result.route}",
        )

    def test_strategy_name(self):
        self.assertEqual(self.result.strategy, "rule_based_v1")


class TestAnswerQuestionSquawk7700(unittest.TestCase):
    """Squawk 7700 count is grounded in gold_emergency_events."""

    @classmethod
    def setUpClass(cls):
        cls.result = answer_question(
            "How many flights squawked 7700?",
            gold_dir=REAL_GOLD_DIR,
            corpus_path=REAL_CORPUS,
        )

    def test_answer_text_non_empty(self):
        self.assertTrue(self.result.answer_text.strip())

    def test_at_least_one_source(self):
        self.assertGreaterEqual(len(self.result.sources), 1)

    def test_source_is_emergency_table(self):
        self.assertIn("gold_emergency_events", self.result.sources)


class TestAnswerQuestionRetrieval(unittest.TestCase):
    """Retrieval path: METAR question produces a corpus-sourced answer."""

    @classmethod
    def setUpClass(cls):
        cls.result = answer_question(
            "What does a METAR weather report contain?",
            gold_dir=REAL_GOLD_DIR,
            corpus_path=REAL_CORPUS,
        )

    def test_answer_text_non_empty(self):
        self.assertTrue(self.result.answer_text.strip())

    def test_at_least_one_source(self):
        self.assertGreaterEqual(len(self.result.sources), 1)

    def test_route_is_retrieval(self):
        self.assertEqual(
            self.result.route,
            "retrieval:search",
            f"Expected retrieval:search route; got: {self.result.route}",
        )


class TestAnswerQuestionTotalFlights(unittest.TestCase):
    """Total flights tracked — grounded in gold_routing_stats."""

    @classmethod
    def setUpClass(cls):
        cls.result = answer_question(
            "How many flights were tracked?",
            gold_dir=REAL_GOLD_DIR,
            corpus_path=REAL_CORPUS,
        )

    def test_answer_text_non_empty(self):
        self.assertTrue(self.result.answer_text.strip())

    def test_at_least_one_source(self):
        self.assertGreaterEqual(len(self.result.sources), 1)

    def test_source_is_routing_table(self):
        self.assertIn("gold_routing_stats", self.result.sources)


class TestCustomStrategyInjection(unittest.TestCase):
    """Injecting a strategy= bypasses the default build path."""

    def test_injected_strategy_is_used(self):
        from ai.agent.base import Answer, AnswerStrategy

        class _Stub(AnswerStrategy):
            name = "stub"

            def answer(self, question: str) -> Answer:
                return Answer(
                    question=question,
                    answer_text="stub answer",
                    result={"answer": "stub", "sources": ["stub_source"]},
                    route="stub:route",
                    strategy=self.name,
                )

        result = answer_question(
            "any question",
            gold_dir=REAL_GOLD_DIR,
            corpus_path=REAL_CORPUS,
            strategy=_Stub(),
        )
        self.assertEqual(result.answer_text, "stub answer")
        self.assertEqual(result.sources, ["stub_source"])
        self.assertEqual(result.strategy, "stub")


if __name__ == "__main__":
    unittest.main()
