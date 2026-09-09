#!/usr/bin/env python3
"""
CLI entry point for local Gold table aggregation.

Usage:
  python -m data.transforms.cli_gold --input data/processed/silver_flight_state.jsonl
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure the project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms.silver_to_gold import (
    run_silver_to_gold,
    aggregate_airport_congestion,
    aggregate_sector_load,
    aggregate_emergency_events,
    aggregate_routing_stats,
    load_schema,
    write_gold_table,
)

TABLE_MAP = {
    "congestion": ("gold_airport_congestion", aggregate_airport_congestion),
    "sector": ("gold_sector_load", aggregate_sector_load),
    "emergency": ("gold_emergency_events", aggregate_emergency_events),
    "routing": ("gold_routing_stats", aggregate_routing_stats),
}

def run_single_gold_table(
    table: str,
    silver_input: Path,
    out_dir: Path,
    contracts_dir: Path,
    validate: bool = True
) -> dict:
    # mirrors run_silver_to_gold but for a single table
    start_ts = time.monotonic()

    records = []
    with open(silver_input) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    file_base, aggregate_fn = TABLE_MAP[table]
    gold_records = aggregate_fn(records)
    schema = load_schema(contracts_dir / f"{file_base}.schema.json")
    written = write_gold_table(out_dir / f"{file_base}.jsonl", gold_records, schema, validate)

    elapsed = time.monotonic() - start_ts
    stats = {
        "input_file": silver_input.name,
        "silver_records_read": len(records),
        f"gold_{table}_written": written,
        "elapsed_s": round(elapsed, 3),
    }
    logging.info("Silver->Gold (%s only) complete: %s", table, json.dumps(stats, indent=2))
    return stats

def main():
    parser = argparse.ArgumentParser(description="Run local silver-to-gold aggregations")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "silver_flight_state.jsonl",
        help="Path to input silver records (JSONL)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory to write gold tables (JSONL)",
    )
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=PROJECT_ROOT / "shared" / "contracts",
        help="Path to JSON schema contracts",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip jsonschema validation of outputs",
    )
    parser.add_argument(
        "--table",
        choices=["congestion", "sector", "emergency", "routing", "all"],
        default="all",
        help="Run only this gold aggregate",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )

    if not args.input.exists():
        logging.error("Input file not found: %s", args.input)
        sys.exit(1)

    if args.table == "all":
        stats = run_silver_to_gold(
            silver_input=args.input,
            out_dir=args.out_dir,
            contracts_dir=args.contracts_dir,
            validate=not args.no_validate
        )
    else:
        stats = run_single_gold_table(
            table=args.table,
            silver_input=args.input,
            out_dir=args.out_dir,
            contracts_dir=args.contracts_dir,
            validate=not args.no_validate
        )

    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
