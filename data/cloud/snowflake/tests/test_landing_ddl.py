#!/usr/bin/env python3
"""
Tests that landing_ddl() in data/cloud/snowflake/load_landing.py stays in
parity with the frozen gold contracts, and that importing the module never
pulls in the (network-requiring) snowflake connector.

Collected by `make test-cloud` (pytest data/cloud/).
"""

import json
import sys
import unittest
from pathlib import Path

# Ensure the repo root is on sys.path so the module under test can be
# imported regardless of the working directory pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from data.cloud.snowflake.load_landing import landing_ddl
from data.scripts.export_gold_parquet import SCHEMAS

CONTRACTS_DIR = _REPO_ROOT / "shared" / "contracts"


def _parse_ddl_columns(ddl: str) -> list[tuple[str, str, bool]]:
    """Parse a `create or replace table ... (...)` statement into
    (column_name, column_type, not_null) tuples, in source order."""
    start = ddl.index("(\n") + len("(\n")
    assert ddl.endswith("\n)"), f"unexpected ddl tail:\n{ddl!r}"
    body = ddl[start : -len("\n)")]

    columns = []
    # lines are joined with ",\n" in landing_ddl, so this comma is always the
    # column separator, never a comma inside a type like number(38,0)
    for raw_line in body.split(",\n"):
        line = raw_line.strip()
        not_null = line.endswith(" not null")
        if not_null:
            line = line[: -len(" not null")]
        name, col_type = line.split(None, 1)
        columns.append((name, col_type, not_null))
    return columns


class TestLandingDDLContractParity(unittest.TestCase):
    """landing_ddl(table) matches the frozen gold contract for each landing table."""

    def _assert_matches_contract(self, table: str) -> None:
        base = table[: -len("_landing")]
        contract = json.loads(
            (CONTRACTS_DIR / f"{base}.schema.json").read_text(encoding="utf-8")
        )
        columns = _parse_ddl_columns(landing_ddl(table))
        column_names = [name for name, _, _ in columns]

        expected_order = list(SCHEMAS[base].names)
        self.assertEqual(
            column_names,
            expected_order,
            f"{table}: ddl column order {column_names} != "
            f"export_gold_parquet SCHEMAS order {expected_order}",
        )

        contract_columns = set(contract["properties"])
        self.assertEqual(
            set(column_names),
            contract_columns,
            f"{table}: ddl columns {set(column_names)} != "
            f"contract columns {contract_columns}",
        )

        required = set(contract.get("required", []))
        for name, col_type, not_null in columns:
            expected_not_null = name in required
            self.assertEqual(
                not_null,
                expected_not_null,
                f"{table}.{name}: ddl not null = {not_null}, "
                f"expected {expected_not_null} (required={name in required})",
            )
            if contract["properties"][name].get("format") == "date-time":
                self.assertEqual(
                    col_type,
                    "timestamp_tz",
                    f"{table}.{name}: date-time field typed {col_type!r}, "
                    f"expected timestamp_tz",
                )

    def test_airport_congestion_matches_contract(self):
        self._assert_matches_contract("gold_airport_congestion_landing")

    def test_sector_load_matches_contract(self):
        self._assert_matches_contract("gold_sector_load_landing")

    def test_emergency_events_matches_contract(self):
        self._assert_matches_contract("gold_emergency_events_landing")

    def test_routing_stats_matches_contract(self):
        self._assert_matches_contract("gold_routing_stats_landing")


class TestLandingDDLOfflineImport(unittest.TestCase):
    """Importing load_landing must never touch the snowflake connector."""

    def test_import_does_not_pull_in_snowflake_connector(self):
        for name in list(sys.modules):
            if name == "snowflake" or name.startswith("snowflake."):
                del sys.modules[name]
        sys.modules.pop("data.cloud.snowflake.load_landing", None)

        import data.cloud.snowflake.load_landing  # noqa: F401

        self.assertNotIn(
            "snowflake.connector",
            sys.modules,
            "importing load_landing pulled snowflake.connector into sys.modules",
        )


if __name__ == "__main__":
    unittest.main()
