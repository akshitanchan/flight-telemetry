#!/usr/bin/env python3
"""Tests that the golden question set is well-formed and passes on fixtures."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.eval.run_fixtures import evaluate


class TestGoldenSet(unittest.TestCase):
    def test_all_golden_questions_pass(self):
        results, all_passed = evaluate()
        failed = [r["id"] for r in results if not r["passed"]]
        self.assertTrue(all_passed, f"failing questions: {failed}")

    def test_golden_set_has_reasonable_coverage(self):
        results, _ = evaluate()
        self.assertGreaterEqual(len(results), 10)


if __name__ == "__main__":
    unittest.main()
