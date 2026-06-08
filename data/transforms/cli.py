#!/usr/bin/env python3
"""
CLI for the bronze-to-silver transform.

Usage:
    python -m data.transforms.cli --input data/interim/landing.jsonl
    python -m data.transforms.cli --input data/interim/landing.jsonl --no-validate
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from data.transforms.bronze_to_silver import run_transform


def main():
    parser = argparse.ArgumentParser(
        description="Transform landing (bronze) records to silver format"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "landing.jsonl",
        help="Path to input landing JSONL (default: data/interim/landing.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "silver_flight_state.jsonl",
        help="Path to output silver JSONL (default: data/processed/silver_flight_state.jsonl)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip contract validation on output records",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        print("Hint: Run 'make systems-replay-sample' first.", file=sys.stderr)
        sys.exit(1)

    summary = run_transform(
        input_path=args.input,
        output_path=args.output,
        validate=not args.no_validate,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
