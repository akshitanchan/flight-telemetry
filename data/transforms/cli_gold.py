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
from pathlib import Path

# Ensure the project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms.silver_to_gold import run_silver_to_gold

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

    stats = run_silver_to_gold(
        silver_input=args.input,
        out_dir=args.out_dir,
        contracts_dir=args.contracts_dir,
        validate=not args.no_validate
    )

    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
