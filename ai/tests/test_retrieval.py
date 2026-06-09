#!/usr/bin/env python3
"""Tests for the retrieval tool against the bundled corpus fixture."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.tools.retrieval import RetrievalTool

CORPUS = PROJECT_ROOT / "ai" / "fixtures" / "corpus" / "corpus.json"


class TestRetrievalTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = RetrievalTool(CORPUS)

    def test_metar_search_top_hit(self):
        r = self.tool.search("METAR EHAM Amsterdam Schiphol decode")
        self.assertEqual(r["sources"][0], "metar-eham")

    def test_squawk_reference_retrieved(self):
        r = self.tool.search("what does squawk 7700 transponder code indicate")
        self.assertIn("ref-squawk-codes", r["sources"])

    def test_report_retrieved(self):
        r = self.tool.search("engine failure during climb incident report")
        self.assertIn("ntsb-eng-failure", r["sources"])

    def test_get(self):
        self.assertIsNotNone(self.tool.get("metar-eddf"))
        self.assertIsNone(self.tool.get("does-not-exist"))

    def test_no_match_returns_empty(self):
        r = self.tool.search("zzzz qqqq nonexistenttoken")
        self.assertEqual(r["sources"], [])


if __name__ == "__main__":
    unittest.main()
