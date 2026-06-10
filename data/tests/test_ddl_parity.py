#!/usr/bin/env python3
"""
Tests that all 10 cloud DDL files (5 Delta + 5 BigQuery) are in byte-faithful
parity with the frozen JSON Schema contracts (contract version 1.0.0).

Invokes check_ddl_parity.check_parity() for each (table, dialect) pair and
fails with a descriptive assertion message on any violation.

Collected by `make test-data` alongside test_bronze_to_silver and
test_silver_to_gold.
"""

import sys
import unittest
from pathlib import Path

# Ensure the repo root is on sys.path so the checker module can be imported
# regardless of the working directory pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from data.cloud.ddl.check_ddl_parity import check_parity, TABLES, DIALECTS


class TestDDLParityDelta(unittest.TestCase):
    """Each Delta DDL file is in parity with its schema contract."""

    def _assert_parity(self, table: str, dialect: str) -> None:
        errors = check_parity(table, dialect, verbose=False)
        self.assertEqual(
            errors,
            [],
            msg=(
                f"DDL parity violations for [{dialect}] {table}:\n"
                + "\n".join(f"  - {e}" for e in errors)
            ),
        )

    def test_delta_silver_flight_state(self):
        self._assert_parity("silver_flight_state", "delta")

    def test_delta_gold_airport_congestion(self):
        self._assert_parity("gold_airport_congestion", "delta")

    def test_delta_gold_sector_load(self):
        self._assert_parity("gold_sector_load", "delta")

    def test_delta_gold_emergency_events(self):
        self._assert_parity("gold_emergency_events", "delta")

    def test_delta_gold_routing_stats(self):
        self._assert_parity("gold_routing_stats", "delta")


class TestDDLParityBigQuery(unittest.TestCase):
    """Each BigQuery DDL file is in parity with its schema contract."""

    def _assert_parity(self, table: str, dialect: str) -> None:
        errors = check_parity(table, dialect, verbose=False)
        self.assertEqual(
            errors,
            [],
            msg=(
                f"DDL parity violations for [{dialect}] {table}:\n"
                + "\n".join(f"  - {e}" for e in errors)
            ),
        )

    def test_bq_silver_flight_state(self):
        self._assert_parity("silver_flight_state", "bigquery")

    def test_bq_gold_airport_congestion(self):
        self._assert_parity("gold_airport_congestion", "bigquery")

    def test_bq_gold_sector_load(self):
        self._assert_parity("gold_sector_load", "bigquery")

    def test_bq_gold_emergency_events(self):
        self._assert_parity("gold_emergency_events", "bigquery")

    def test_bq_gold_routing_stats(self):
        self._assert_parity("gold_routing_stats", "bigquery")


if __name__ == "__main__":
    unittest.main()
