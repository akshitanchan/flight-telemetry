#!/usr/bin/env python3
"""
data/cloud/databricks/dlt/coverage_check.py
W4.3 / ds-03 — Offline coverage gate: proves every schema rule has a DLT expectation.

Usage:
    python data/cloud/databricks/dlt/coverage_check.py          # exits 0 if all covered
    python data/cloud/databricks/dlt/coverage_check.py --verbose # show per-rule detail

Exit codes:
    0  all schema rules have a corresponding DLT expectation
    1  one or more rules are unguarded

Rule categories checked per schema:
    NOT_NULL  — every field listed in "required" must have a <prefix>_not_null expectation
    RANGE     — every property with minimum/maximum must have a range/min expectation
    ENUM      — every property with enum must have an enum expectation
    PATTERN   — every property with pattern must have a pattern/format expectation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup: allow running from project root or from this file's directory.
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
# dlt/ -> databricks/ -> cloud/ -> data/ -> flight-telemetry/  (4 hops from _FILE_DIR)
_PROJECT_ROOT = _FILE_DIR.parent.parent.parent.parent
_CONTRACTS_DIR = _PROJECT_ROOT / "shared" / "contracts"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import expectation dicts from sibling modules.
# These are plain Python dicts — no Spark/DLT runtime needed.
# ---------------------------------------------------------------------------
from data.cloud.databricks.dlt.expectations_silver import (  # noqa: E402
    SILVER_ALL_EXPECTATIONS,
)
from data.cloud.databricks.dlt.expectations_gold import (  # noqa: E402
    GOLD_CONGESTION_EXPECTATIONS,
    GOLD_SECTOR_EXPECTATIONS,
    GOLD_EMERGENCY_EXPECTATIONS,
    GOLD_ROUTING_EXPECTATIONS,
)

# ---------------------------------------------------------------------------
# Schema → expectation-dict mapping
# ---------------------------------------------------------------------------
SCHEMA_EXPECTATION_MAP: list[tuple[str, dict[str, str]]] = [
    ("silver_flight_state.schema.json",    SILVER_ALL_EXPECTATIONS),
    ("gold_airport_congestion.schema.json", GOLD_CONGESTION_EXPECTATIONS),
    ("gold_sector_load.schema.json",        GOLD_SECTOR_EXPECTATIONS),
    ("gold_emergency_events.schema.json",   GOLD_EMERGENCY_EXPECTATIONS),
    ("gold_routing_stats.schema.json",      GOLD_ROUTING_EXPECTATIONS),
]


# ---------------------------------------------------------------------------
# Rule extraction helpers
# ---------------------------------------------------------------------------

def _extract_rules(schema: dict) -> list[dict[str, Any]]:
    """
    Walk the schema and return a flat list of constraint rules.

    Each rule is a dict:
        {
            "field":    str,        # property name
            "kind":     str,        # NOT_NULL | RANGE_MIN | RANGE_MAX | ENUM | PATTERN
            "detail":   Any,        # the constraint value (number / list / str)
        }
    """
    rules: list[dict[str, Any]] = []
    required_fields: set[str] = set(schema.get("required", []))
    properties: dict = schema.get("properties", {})

    for field, prop in properties.items():
        # NOT_NULL: field is in "required"
        if field in required_fields:
            rules.append({"field": field, "kind": "NOT_NULL", "detail": None})

        # Unwrap nullable union types: ["number","null"] -> "number"
        prop_type = prop.get("type", "")
        if isinstance(prop_type, list):
            non_null_types = [t for t in prop_type if t != "null"]
            effective_type = non_null_types[0] if non_null_types else ""
        else:
            effective_type = prop_type

        # RANGE_MIN
        if "minimum" in prop:
            rules.append({"field": field, "kind": "RANGE_MIN", "detail": prop["minimum"]})

        # RANGE_MAX
        if "maximum" in prop:
            rules.append({"field": field, "kind": "RANGE_MAX", "detail": prop["maximum"]})

        # ENUM
        if "enum" in prop:
            rules.append({"field": field, "kind": "ENUM", "detail": prop["enum"]})

        # PATTERN
        if "pattern" in prop:
            rules.append({"field": field, "kind": "PATTERN", "detail": prop["pattern"]})

    return rules


# ---------------------------------------------------------------------------
# Coverage checker
# ---------------------------------------------------------------------------

def _expr_covers_field(expr: str, field: str) -> bool:
    """Return True if the SQL expression string references the given field name."""
    return field in expr


def _expectation_covers_rule(
    rule: dict[str, Any],
    expectations: dict[str, str],
) -> tuple[bool, str]:
    """
    Determine whether any expectation in *expectations* covers *rule*.

    Returns (covered: bool, matching_key: str).

    Coverage heuristics:
      NOT_NULL  — any expectation whose key ends in '_not_null' and whose
                  expression contains IS NOT NULL for the field.
      RANGE_MIN — any expectation whose expression references the field and
                  contains '>= <value>' or 'BETWEEN <value>'.
      RANGE_MAX — any expectation whose expression references the field and
                  contains '<= <value>' or 'BETWEEN ... AND <value>'.
      ENUM      — any expectation whose expression references the field and
                  contains 'IN ('.
      PATTERN   — any expectation whose expression references the field and
                  contains 'RLIKE' or 'LENGTH('.
    """
    field = rule["field"]
    kind = rule["kind"]

    for key, expr in expectations.items():
        if not _expr_covers_field(expr, field):
            continue

        expr_upper = expr.upper()

        if kind == "NOT_NULL" and "IS NOT NULL" in expr_upper:
            return True, key

        elif kind == "RANGE_MIN":
            # '>= <min>' or 'BETWEEN <min> AND'
            min_val = rule["detail"]
            if (
                f">= {min_val}" in expr
                or f">={min_val}" in expr
                or "BETWEEN" in expr_upper
            ):
                return True, key

        elif kind == "RANGE_MAX":
            # '<= <max>' or 'BETWEEN ... AND <max>'
            max_val = rule["detail"]
            if (
                f"<= {max_val}" in expr
                or f"<={max_val}" in expr
                or "BETWEEN" in expr_upper
            ):
                return True, key

        elif kind == "ENUM" and "IN (" in expr_upper:
            return True, key

        elif kind == "PATTERN" and (
            "RLIKE" in expr_upper or "LENGTH(" in expr_upper
        ):
            return True, key

    return False, ""


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------

def check_coverage(verbose: bool = False) -> bool:
    """
    For each schema, enumerate every rule and assert a DLT expectation covers it.

    Returns True if all rules are covered, False otherwise.
    """
    all_covered = True

    for schema_file, expectations in SCHEMA_EXPECTATION_MAP:
        schema_path = _CONTRACTS_DIR / schema_file
        if not schema_path.exists():
            print(f"ERROR: schema not found: {schema_path}")
            all_covered = False
            continue

        with open(schema_path) as f:
            schema = json.load(f)

        schema_name = schema_file.replace(".schema.json", "")
        rules = _extract_rules(schema)

        uncovered: list[dict] = []
        covered: list[tuple[dict, str]] = []

        for rule in rules:
            ok, matching_key = _expectation_covers_rule(rule, expectations)
            if ok:
                covered.append((rule, matching_key))
            else:
                uncovered.append(rule)

        if verbose or uncovered:
            print(f"\n{'='*64}")
            print(f"Schema: {schema_name}  ({len(rules)} rules)")
            print(f"{'='*64}")

        if verbose:
            for rule, key in covered:
                print(
                    f"  [COVERED]  {rule['field']:25s} {rule['kind']:12s}"
                    f"  -> {key}"
                )

        if uncovered:
            all_covered = False
            for rule in uncovered:
                print(
                    f"  [MISSING]  {rule['field']:25s} {rule['kind']:12s}"
                    f"  detail={rule['detail']!r}"
                )
            print(
                f"  {len(uncovered)} rule(s) NOT covered in {schema_name}!"
            )
        elif verbose:
            print(f"  All {len(rules)} rules covered.")

    return all_covered


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline coverage gate: verifies every schema rule has a "
            "corresponding DLT expectation. Exits 0 if all covered, 1 otherwise."
        )
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-rule coverage detail (both covered and missing)",
    )
    args = parser.parse_args()

    ok = check_coverage(verbose=args.verbose)

    if ok:
        print("\nCoverage check PASSED: every schema rule has a DLT expectation.")
        sys.exit(0)
    else:
        print(
            "\nCoverage check FAILED: one or more schema rules have no DLT expectation."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
