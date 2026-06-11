#!/usr/bin/env python3
"""Tests for the second deterministic strategy and the comparison harness.

The Ollama LLM strategy is intentionally excluded here (include_llm=False) so the
suite stays offline and CI-safe; ai-compare-small exercises it when a server is up.

ai-07: compare() now returns (core_summaries, all_summaries, injection_result, notes).
  - core_summaries: deterministic strategies on CORE tier — the AUTHORITATIVE gate.
  - all_summaries:  all strategies on the full golden set — for ADVISORY metrics.
  - injection_result: block-rate dict from the injection suite — ADVISORY.
  - notes: list of human-readable notes (skipped strategies, etc.).
"""

import json
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
GOLDEN = PROJECT_ROOT / "ai" / "eval" / "golden_set.json"
INJECTIONS = PROJECT_ROOT / "ai" / "fixtures" / "injections.json"


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
        """Deterministic strategies must score accuracy==1.0 and citation==1.0 on the CORE tier.

        The golden set is split into tier='core' (authoritative CI gate, all questions
        deterministically routable by rule_based_v1 / keyword_score_v1) and
        tier='extended' (harder/long-tail questions for LLM architecture comparison in
        ai-04 where deterministic routers are not required to succeed).  This gate
        evaluates CORE questions only so that the bar is both meaningful and achievable
        by frozen deterministic strategies.
        """
        with open(GOLDEN) as f:
            all_questions = json.load(f).get("questions", [])
        core_questions = [q for q in all_questions if q.get("tier") == "core"]
        self.assertGreater(len(core_questions), 0, "golden set has no core-tier questions")

        import tempfile, json as _json
        core_golden = {"version": "1.1", "questions": core_questions}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            _json.dump(core_golden, tmp)
            tmp_path = Path(tmp.name)

        try:
            # compare() returns (core_summaries, all_summaries, injection_result, notes)
            # For this gate test we pass a core-only golden and use core_summaries.
            core_summaries, _, _, _ = compare(
                gold_dir=GOLD, corpus_path=CORPUS,
                golden_path=tmp_path, include_llm=False,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        names = {s["strategy"] for s in core_summaries}
        self.assertEqual(names, {"rule_based_v1", "keyword_score_v1"})
        for s in core_summaries:
            self.assertEqual(
                s["accuracy"], 1.0,
                f"{s['strategy']} accuracy={s['accuracy']} on CORE tier "
                f"(passed={s['passed']}/{s['total']})"
            )
            self.assertEqual(
                s["citation_coverage"], 1.0,
                f"{s['strategy']} citation_coverage={s['citation_coverage']} on CORE tier"
            )


class TestCompareReturnShape(unittest.TestCase):
    """Verify the ai-07 extended return shape of compare() (offline, no LLM)."""

    @classmethod
    def setUpClass(cls):
        cls.result = compare(
            gold_dir=GOLD,
            corpus_path=CORPUS,
            golden_path=GOLDEN,
            injections_path=INJECTIONS,
            include_llm=False,
        )

    def test_returns_four_tuple(self):
        self.assertEqual(len(self.result), 4,
                         "compare() must return a 4-tuple (core, all, injection, notes)")

    def test_core_summaries_deterministic_only(self):
        core_summaries, _, _, _ = self.result
        names = {s["strategy"] for s in core_summaries}
        self.assertEqual(names, {"rule_based_v1", "keyword_score_v1"})

    def test_all_summaries_includes_deterministic(self):
        _, all_summaries, _, _ = self.result
        names = {s["strategy"] for s in all_summaries}
        self.assertIn("rule_based_v1", names)
        self.assertIn("keyword_score_v1", names)

    def test_advisory_columns_present(self):
        """Every summary must have the full set of metric columns including ai-07 additions."""
        _, all_summaries, _, _ = self.result
        required = {
            "strategy", "total", "passed", "accuracy", "citation_coverage",
            "faithfulness_mean", "p50_latency_ms", "p95_latency_ms",
            "mean_latency_ms", "llm_calls", "tokens", "cost_usd", "per_question",
        }
        for s in all_summaries:
            missing = required - set(s.keys())
            self.assertFalse(
                missing,
                f"summary for '{s['strategy']}' is missing columns: {missing}"
            )

    def test_faithfulness_range(self):
        """faithfulness_mean must be in [0.0, 1.0]."""
        _, all_summaries, _, _ = self.result
        for s in all_summaries:
            self.assertGreaterEqual(
                s["faithfulness_mean"], 0.0,
                f"{s['strategy']} faithfulness_mean < 0"
            )
            self.assertLessEqual(
                s["faithfulness_mean"], 1.0,
                f"{s['strategy']} faithfulness_mean > 1"
            )

    def test_injection_block_rate_present(self):
        """injection_result must have block_rate and be in [0, 1]."""
        _, _, injection_result, _ = self.result
        self.assertIn("block_rate", injection_result,
                      "injection_result missing block_rate key")
        rate = injection_result["block_rate"]
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_injection_block_rate_nonzero(self):
        """The deterministic guardrail should block at least some injections."""
        _, _, injection_result, _ = self.result
        self.assertGreater(
            injection_result.get("blocked", 0), 0,
            "GuardedStrategy blocked 0 injections — input guardrail may be broken"
        )

    def test_notes_is_list(self):
        _, _, _, notes = self.result
        self.assertIsInstance(notes, list)

    def test_llm_skipped_in_notes_when_no_provider(self):
        """With include_llm=False, notes must explain LLM strategies were skipped."""
        _, _, _, notes = self.result
        combined = " ".join(notes).lower()
        self.assertIn("skipped", combined,
                      "notes should mention that LLM strategies were skipped")

    def test_cost_usd_nonnegative(self):
        """cost_usd must be >= 0 for every strategy."""
        _, all_summaries, _, _ = self.result
        for s in all_summaries:
            self.assertGreaterEqual(
                s["cost_usd"], 0.0,
                f"{s['strategy']} cost_usd is negative"
            )

    def test_core_tier_counts(self):
        """core_summaries must evaluate only the core-tier questions (not extended)."""
        with open(GOLDEN) as f:
            all_qs = json.load(f).get("questions", [])
        core_count = sum(1 for q in all_qs if q.get("tier") == "core")
        core_summaries, _, _, _ = self.result
        for s in core_summaries:
            self.assertEqual(
                s["total"], core_count,
                f"core_summaries[{s['strategy']}].total={s['total']} "
                f"but there are {core_count} core-tier questions"
            )


if __name__ == "__main__":
    unittest.main()
