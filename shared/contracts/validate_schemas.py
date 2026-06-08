#!/usr/bin/env python3
"""
Contract schema validator for the Flight Telemetry Platform.

Validates JSON fixture files against their corresponding JSON Schema contracts.
Designed as the backing implementation for `make test-contracts`.

Usage:
    python validate_schemas.py                    # run all validations
    python validate_schemas.py --schema silver    # run only silver schema tests
    python validate_schemas.py --verbose          # show per-record detail

Exit codes:
    0  all validations passed
    1  one or more validations failed
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema
    from jsonschema import Draft202012Validator, ValidationError
except ImportError:
    print("ERROR: jsonschema package required. Install with: pip install jsonschema")
    sys.exit(1)


CONTRACTS_DIR = Path(__file__).parent
FIXTURES_DIR = CONTRACTS_DIR / "fixtures"

# Map of schema files to their fixture files.
# Each entry: (schema_file, valid_fixture, invalid_fixture_or_None)
SCHEMA_FIXTURE_MAP = [
    (
        "silver_flight_state.schema.json",
        "silver_flight_state_valid.json",
        "silver_flight_state_invalid.json",
    ),
    (
        "gold_airport_congestion.schema.json",
        "gold_airport_congestion_valid.json",
        None,
    ),
    (
        "gold_emergency_events.schema.json",
        "gold_emergency_events_valid.json",
        None,
    ),
]


def load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def validate_valid_fixtures(
    schema: dict, fixtures: list[dict], schema_name: str, verbose: bool
) -> int:
    """Validate that all records in the valid fixture pass schema validation.
    Returns count of failures."""
    validator = Draft202012Validator(schema)
    failures = 0
    for i, record in enumerate(fixtures):
        # Strip _description meta-field before validation
        record_clean = {k: v for k, v in record.items() if not k.startswith("_")}
        errors = list(validator.iter_errors(record_clean))
        if errors:
            failures += 1
            desc = record.get("_description", f"record {i}")
            print(f"  FAIL (should be valid): {desc}")
            for e in errors:
                print(f"    → {e.message}")
        elif verbose:
            desc = record.get("_description", f"record {i}")
            print(f"  OK: {desc}")
    return failures


def validate_invalid_fixtures(
    schema: dict, fixtures: list[dict], schema_name: str, verbose: bool
) -> int:
    """Validate that all records in the invalid fixture FAIL schema validation.
    Returns count of unexpected passes."""
    validator = Draft202012Validator(schema)
    failures = 0
    for i, record in enumerate(fixtures):
        record_clean = {k: v for k, v in record.items() if not k.startswith("_")}
        errors = list(validator.iter_errors(record_clean))
        if not errors:
            failures += 1
            desc = record.get("_description", f"record {i}")
            print(f"  FAIL (should be invalid but passed): {desc}")
        elif verbose:
            desc = record.get("_description", f"record {i}")
            expected = record.get("_expected_error", "unspecified")
            print(f"  OK (correctly rejected): {desc} [{expected}]")
    return failures


def run_tests(schema_filter: str | None, verbose: bool) -> bool:
    total_pass = 0
    total_fail = 0

    for schema_file, valid_file, invalid_file in SCHEMA_FIXTURE_MAP:
        schema_name = schema_file.replace(".schema.json", "")

        # Apply filter if provided
        if schema_filter and schema_filter not in schema_name:
            continue

        schema_path = CONTRACTS_DIR / schema_file
        if not schema_path.exists():
            print(f"SKIP: schema not found: {schema_path}")
            continue

        schema = load_json(schema_path)
        print(f"\n{'='*60}")
        print(f"Schema: {schema_name} (v{schema.get('_contract_version', '?')})")
        print(f"{'='*60}")

        # Check schema itself is valid
        try:
            Draft202012Validator.check_schema(schema)
            print(f"  Schema self-check: OK")
        except jsonschema.SchemaError as e:
            print(f"  Schema self-check: FAIL — {e.message}")
            total_fail += 1
            continue

        # Valid fixtures
        valid_path = FIXTURES_DIR / valid_file
        if valid_path.exists():
            valid_records = load_json(valid_path)
            print(f"\n  Valid fixtures ({len(valid_records)} records):")
            fails = validate_valid_fixtures(schema, valid_records, schema_name, verbose)
            total_fail += fails
            total_pass += len(valid_records) - fails
        else:
            print(f"  SKIP: valid fixtures not found: {valid_path}")

        # Invalid fixtures
        if invalid_file:
            invalid_path = FIXTURES_DIR / invalid_file
            if invalid_path.exists():
                invalid_records = load_json(invalid_path)
                print(f"\n  Invalid fixtures ({len(invalid_records)} records):")
                fails = validate_invalid_fixtures(
                    schema, invalid_records, schema_name, verbose
                )
                total_fail += fails
                total_pass += len(invalid_records) - fails
            else:
                print(f"  SKIP: invalid fixtures not found: {invalid_path}")

    print(f"\n{'='*60}")
    print(f"Results: {total_pass} passed, {total_fail} failed")
    print(f"{'='*60}")
    return total_fail == 0


def main():
    parser = argparse.ArgumentParser(description="Validate contract schemas")
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Filter to schemas containing this substring (e.g. 'silver', 'gold')",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show per-record detail"
    )
    args = parser.parse_args()

    success = run_tests(args.schema, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
