#!/usr/bin/env python3
"""Tests for dashboard.data — gold-table loader.

Conventions mirror ai/tests/test_analytics.py:
* unittest.TestCase subclasses
* PROJECT_ROOT path insertion so the test is runnable without an editable
  install: ``python -m pytest dashboard/tests/ -v`` or
  ``python -m unittest discover -s dashboard/tests -v``
* No streamlit import anywhere in this file or in dashboard.data
* Offline-safe: reads the real gold files from data/processed/ and exercises
  the missing-file degrade path against a temp directory
"""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# dashboard.data must be importable with NO streamlit and NO services running
from dashboard.data import (  # noqa: E402
    GOLD_COLUMNS,
    GOLD_FILES,
    GoldTables,
    load_gold,
)

REAL_GOLD_DIR = PROJECT_ROOT / "data" / "processed"


class TestGoldLoaderColumns(unittest.TestCase):
    """Verify column contracts on all four tables from data/processed/."""

    @classmethod
    def setUpClass(cls):
        cls.gold = load_gold(REAL_GOLD_DIR)

    # ---- return type ----

    def test_returns_gold_tables_instance(self):
        self.assertIsInstance(self.gold, GoldTables)

    # ---- congestion ----

    def test_congestion_columns(self):
        self.assertEqual(list(self.gold.congestion.columns), GOLD_COLUMNS["congestion"])

    def test_congestion_not_empty(self):
        self.assertGreater(len(self.gold.congestion), 0)

    def test_congestion_airport_icao_present(self):
        self.assertIn("airport_icao", self.gold.congestion.columns)

    # ---- sector ----

    def test_sector_columns(self):
        self.assertEqual(list(self.gold.sector.columns), GOLD_COLUMNS["sector"])

    def test_sector_not_empty(self):
        self.assertGreater(len(self.gold.sector), 0)

    def test_sector_h3_r4_present(self):
        self.assertIn("h3_r4", self.gold.sector.columns)

    # ---- emergency ----

    def test_emergency_columns(self):
        self.assertEqual(list(self.gold.emergency.columns), GOLD_COLUMNS["emergency"])

    def test_emergency_not_empty(self):
        self.assertGreater(len(self.gold.emergency), 0)

    def test_emergency_squawk_values(self):
        valid = {"7500", "7600", "7700"}
        actual = set(self.gold.emergency["squawk"].dropna().unique())
        self.assertTrue(actual.issubset(valid), f"Unexpected squawk codes: {actual - valid}")

    # ---- routing ----

    def test_routing_columns(self):
        self.assertEqual(list(self.gold.routing.columns), GOLD_COLUMNS["routing"])

    def test_routing_not_empty(self):
        self.assertGreater(len(self.gold.routing), 0)

    def test_routing_ping_count_positive(self):
        self.assertTrue((self.gold.routing["ping_count"] >= 1).all())


class TestGoldLoaderDegrade(unittest.TestCase):
    """Verify that missing files produce empty DataFrames — no crash."""

    def test_all_missing_returns_empty_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold = load_gold(tmpdir)
            self.assertIsInstance(gold, GoldTables)
            self.assertTrue(gold.congestion.empty)
            self.assertTrue(gold.sector.empty)
            self.assertTrue(gold.emergency.empty)
            self.assertTrue(gold.routing.empty)

    def test_empty_tables_have_correct_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold = load_gold(tmpdir)
            for table in ("congestion", "sector", "emergency", "routing"):
                df = getattr(gold, table)
                self.assertEqual(
                    list(df.columns),
                    GOLD_COLUMNS[table],
                    msg=f"Column mismatch for empty {table} table",
                )

    def test_partial_missing_does_not_crash(self):
        """One missing file should not affect the other three tables."""
        import json
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Copy all files except routing
            for table, filename in GOLD_FILES.items():
                src = REAL_GOLD_DIR / filename
                if table != "routing" and src.exists():
                    shutil.copy(src, tmppath / filename)

            gold = load_gold(tmpdir)
            # routing should be empty
            self.assertTrue(gold.routing.empty)
            self.assertEqual(list(gold.routing.columns), GOLD_COLUMNS["routing"])
            # others should have data
            self.assertGreater(len(gold.congestion), 0)
            self.assertGreater(len(gold.sector), 0)
            self.assertGreater(len(gold.emergency), 0)


class TestGoldLoaderEnvVar(unittest.TestCase):
    """Verify GOLD_DIR environment variable is respected."""

    def test_env_var_overrides_default(self):
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            original = os.environ.get("GOLD_DIR")
            try:
                os.environ["GOLD_DIR"] = tmpdir
                # Passing gold_dir=None forces the env-var code path
                gold = load_gold(None)
                self.assertTrue(gold.congestion.empty)
            finally:
                if original is None:
                    os.environ.pop("GOLD_DIR", None)
                else:
                    os.environ["GOLD_DIR"] = original


class TestNoStreamlitImport(unittest.TestCase):
    """Guard: dashboard.data must never import streamlit."""

    def test_streamlit_not_imported_by_data_module(self):
        import re

        import dashboard.data as data_module

        source_file = Path(data_module.__file__).read_text()
        # Match actual import statements only: "import streamlit" or
        # "from streamlit" at the start of a line (ignoring comments/docstrings).
        import_pattern = re.compile(r"^\s*(import streamlit|from streamlit\b)", re.MULTILINE)
        matches = import_pattern.findall(source_file)
        self.assertEqual(
            matches,
            [],
            f"dashboard/data.py must not contain a streamlit import statement; found: {matches}",
        )


if __name__ == "__main__":
    unittest.main()
