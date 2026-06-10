#!/usr/bin/env python3
"""
Unit tests for data/cloud/databricks/experiments/harness.py

Tests confirm:
1. run_experiment_scale() returns a dict with the expected keys and correct types.
2. Numeric invariants hold: base_rows + inc_rows == target_size, speedup_factor > 0.
3. run_experiment() (the multi-scale driver) returns one result per requested size.
4. The harness is importable without Spark (offline-safe).

These tests do NOT collide with ds-04's test_gold_logic.py — they exercise the
experiment harness only, not the gold aggregation logic.

Run:
    pytest data/cloud/databricks/tests/test_harness.py -v
"""

import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.cloud.databricks.experiments.harness import (  # noqa: E402
    run_experiment,
    run_experiment_scale,
    _setup_seed,
)
import duckdb  # noqa: E402

# ---------------------------------------------------------------------------
# Expected result dict keys (contract between harness and notebook/template)
# ---------------------------------------------------------------------------
_EXPECTED_KEYS = {
    "target_size",
    "base_rows",
    "inc_rows",
    "full_recompute_s",
    "incremental_merge_s",
    "speedup_factor",
}


class TestHarnessResultShape(unittest.TestCase):
    """run_experiment_scale returns a well-formed result dict."""

    def setUp(self) -> None:
        """Create a fresh in-memory DuckDB connection with the silver seed."""
        self.con = duckdb.connect()
        _setup_seed(self.con)

    def tearDown(self) -> None:
        self.con.close()

    def test_result_has_expected_keys(self) -> None:
        result = run_experiment_scale(self.con, target_size=5000, inc_pct=0.10)
        self.assertEqual(set(result.keys()), _EXPECTED_KEYS)

    def test_target_size_preserved(self) -> None:
        result = run_experiment_scale(self.con, target_size=5000, inc_pct=0.10)
        self.assertEqual(result["target_size"], 5000)

    def test_base_plus_inc_equals_target(self) -> None:
        """base_rows + inc_rows must equal target_size (exact split)."""
        result = run_experiment_scale(self.con, target_size=5000, inc_pct=0.10)
        self.assertEqual(
            result["base_rows"] + result["inc_rows"],
            result["target_size"],
        )

    def test_latencies_are_positive_floats(self) -> None:
        result = run_experiment_scale(self.con, target_size=5000, inc_pct=0.10)
        self.assertIsInstance(result["full_recompute_s"], float)
        self.assertIsInstance(result["incremental_merge_s"], float)
        self.assertGreater(result["full_recompute_s"], 0.0)
        self.assertGreater(result["incremental_merge_s"], 0.0)

    def test_speedup_is_positive(self) -> None:
        result = run_experiment_scale(self.con, target_size=5000, inc_pct=0.10)
        self.assertGreater(result["speedup_factor"], 0.0)

    def test_different_inc_pct(self) -> None:
        """Harness works with a non-default incremental percentage."""
        result = run_experiment_scale(self.con, target_size=4000, inc_pct=0.20)
        self.assertEqual(result["inc_rows"], 800)   # 20% of 4000
        self.assertEqual(result["base_rows"], 3200)

    def test_small_scale_returns_valid_result(self) -> None:
        """Even a very small scale (1000 rows) should produce a valid result."""
        result = run_experiment_scale(self.con, target_size=1000, inc_pct=0.10)
        self.assertEqual(set(result.keys()), _EXPECTED_KEYS)
        self.assertGreater(result["speedup_factor"], 0.0)


class TestHarnessMultiScale(unittest.TestCase):
    """run_experiment() returns one result per requested scale."""

    def test_multi_scale_length(self) -> None:
        sizes = [2000, 5000]
        results = run_experiment(sizes, inc_pct=0.10, save_json=False)
        self.assertEqual(len(results), 2)

    def test_multi_scale_target_sizes_match(self) -> None:
        sizes = [2000, 5000]
        results = run_experiment(sizes, inc_pct=0.10, save_json=False)
        returned_sizes = [r["target_size"] for r in results]
        self.assertEqual(returned_sizes, sizes)

    def test_each_result_has_expected_keys(self) -> None:
        results = run_experiment([3000], inc_pct=0.10, save_json=False)
        for r in results:
            self.assertEqual(set(r.keys()), _EXPECTED_KEYS)


if __name__ == "__main__":
    unittest.main()
