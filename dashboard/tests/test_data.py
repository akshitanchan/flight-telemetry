#!/usr/bin/env python3
"""Tests for dashboard.data — gold-table loader.

Conventions mirror ai/tests/test_analytics.py:
* unittest.TestCase subclasses
* PROJECT_ROOT path insertion so the test is runnable without an editable
  install: ``python -m pytest dashboard/tests/ -v`` or
  ``python -m unittest discover -s dashboard/tests -v``
* No streamlit import anywhere in this file or in dashboard.data
* Offline-safe: all tests that require non-empty gold data use small,
  schema-valid synthetic JSONL fixtures written into a temp directory.
  No generated artifact under data/processed/ is required.
"""

import json
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

# ---------------------------------------------------------------------------
# Synthetic gold fixtures
#
# Records are derived from shared/contracts/fixtures/gold_*_valid.json and
# satisfy the full column contract defined in GOLD_COLUMNS.  Extra meta-keys
# (e.g. _description) are stripped so only schema fields are present.
# These fixtures make every test self-contained and CI-safe: no generated
# artifact from data/processed/ is required.
# ---------------------------------------------------------------------------

_SYNTHETIC_GOLD_ROWS: dict[str, list[dict]] = {
    "congestion": [
        {
            "airport_icao": "EHAM",
            "window_start": "2024-06-03T14:00:00Z",
            "window_end": "2024-06-03T15:00:00Z",
            "aircraft_count": 42,
            "avg_altitude_m": 3200.5,
            "ground_count": 12,
            "airborne_count": 30,
        },
        {
            "airport_icao": "EDDF",
            "window_start": "2024-06-03T15:00:00Z",
            "window_end": "2024-06-03T16:00:00Z",
            "aircraft_count": 27,
            "avg_altitude_m": 2800.0,
            "ground_count": 8,
            "airborne_count": 19,
        },
    ],
    "sector": [
        {
            "h3_r4": "841ea45ffffffff",
            "window_start": "2024-06-03T12:00:00+00:00",
            "window_end": "2024-06-03T12:05:00+00:00",
            "aircraft_count": 2,
        },
        {
            "h3_r4": "841fa23ffffffff",
            "window_start": "2024-06-03T12:05:00+00:00",
            "window_end": "2024-06-03T12:10:00+00:00",
            "aircraft_count": 1,
        },
    ],
    "emergency": [
        {
            "icao24": "3c6751",
            "callsign": "DLH42N",
            "squawk": "7700",
            "first_seen_ts": "2024-06-03T15:00:00Z",
            "last_seen_ts": "2024-06-03T15:12:30Z",
            "lat": 50.0379,
            "lon": 8.5706,
            "origin_country": "Germany",
            "nearest_airport": "EDDF",
            "duration_s": 750,
        },
        {
            "icao24": "4b1806",
            "callsign": "SWR162",
            "squawk": "7600",
            "first_seen_ts": "2024-06-03T14:30:05Z",
            "last_seen_ts": "2024-06-03T14:45:00Z",
            "lat": 52.3080,
            "lon": 4.7638,
            "origin_country": "Switzerland",
            "nearest_airport": "EHAM",
            "duration_s": 895,
        },
    ],
    "routing": [
        {
            "icao24": "3c6751",
            "callsign": "DLH42N",
            "window_start": "2024-06-03T12:00:00+00:00",
            "window_end": "2024-06-03T12:00:40+00:00",
            "origin_lat": 49.535431,
            "origin_lon": 6.033023,
            "destination_lat": 49.6,
            "destination_lon": 6.1,
            "max_altitude_m": 11574.0,
            "avg_velocity_mps": 236.32,
            "ping_count": 5,
        },
        {
            "icao24": "30014c",
            "callsign": None,
            "window_start": "2024-06-03T12:00:00+00:00",
            "window_end": "2024-06-03T12:00:40+00:00",
            "origin_lat": 46.886562,
            "origin_lon": 17.485017,
            "destination_lat": 46.9,
            "destination_lon": 17.5,
            "max_altitude_m": None,
            "avg_velocity_mps": None,
            "ping_count": 3,
        },
    ],
}


def _write_synthetic_gold(dest_dir: Path, exclude: str | None = None) -> None:
    """Write synthetic gold JSONL files for all four tables into dest_dir.

    Parameters
    ----------
    dest_dir:
        Target directory (must already exist).
    exclude:
        Optional table key to skip (e.g. ``"routing"`` to test the partial-
        missing degrade path).
    """
    for table, filename in GOLD_FILES.items():
        if table == exclude:
            continue
        out = dest_dir / filename
        with out.open("w") as fh:
            for row in _SYNTHETIC_GOLD_ROWS[table]:
                fh.write(json.dumps(row) + "\n")


class TestGoldLoaderColumns(unittest.TestCase):
    """Verify column contracts on all four tables using synthetic fixtures.

    A small set of schema-valid rows is written to a temporary directory in
    setUpClass so the tests are self-contained and pass on a fresh CI checkout
    where data/processed/ does not exist.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._gold_dir = Path(cls._tmpdir.name)
        _write_synthetic_gold(cls._gold_dir)
        cls.gold = load_gold(cls._gold_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

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
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Write all synthetic gold files except routing to exercise the
            # partial-missing degrade path without needing data/processed/.
            _write_synthetic_gold(tmppath, exclude="routing")

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
