#!/usr/bin/env python3
"""
CLI for the spatiotemporal index benchmark.

Usage:
    python -m systems.index.cli --input data/processed/silver_flight_state.jsonl
    python -m systems.index.cli --input data/processed/silver_flight_state.jsonl --format markdown
    python -m systems.index.cli --input data/processed/silver_flight_state.jsonl --queries 200 --profile local
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from systems.index.geohash_index import GeohashPrefixIndex
from systems.index.h3_index import H3Index
from systems.index.workload import generate_workload
from systems.index.benchmark import (
    run_benchmark,
    format_results_markdown,
    format_results_json,
    format_results_csv,
)


def load_silver_records(path: Path) -> list[dict]:
    """Load silver records from JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Run spatiotemporal index benchmark"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "silver_flight_state.jsonl",
        help="Path to silver JSONL records",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Number of queries in workload (default: 100)",
    )
    parser.add_argument(
        "--profile",
        choices=["point", "local", "regional", "continental"],
        default="regional",
        help="Query size profile (default: regional)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "csv"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write results to file (default: stdout)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["geohash_p3", "geohash_p4", "geohash_p5", "h3_r3", "h3_r4", "h3_r5"],
        help="Index strategies to benchmark",
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
    logger = logging.getLogger("benchmark")

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        print("Hint: Run 'make data-local-silver' first.", file=sys.stderr)
        sys.exit(1)

    # Load data
    logger.info("Loading silver records from %s", args.input)
    records = load_silver_records(args.input)
    logger.info("Loaded %d records", len(records))

    # Generate workload
    logger.info("Generating %d %s queries", args.queries, args.profile)
    queries = generate_workload(
        num_queries=args.queries,
        profile=args.profile,
    )

    # Build index strategies
    strategy_map = {
        "geohash_p3": lambda: GeohashPrefixIndex(prefix_precision=3),
        "geohash_p4": lambda: GeohashPrefixIndex(prefix_precision=4),
        "geohash_p5": lambda: GeohashPrefixIndex(prefix_precision=5),
        "h3_r3": lambda: H3Index(resolution=3),
        "h3_r4": lambda: H3Index(resolution=4),
        "h3_r5": lambda: H3Index(resolution=5),
        "h3_r6": lambda: H3Index(resolution=6),
    }

    results = []
    for name in args.strategies:
        factory = strategy_map.get(name)
        if not factory:
            logger.warning("Unknown strategy: %s (skipping)", name)
            continue

        logger.info("Benchmarking strategy: %s", name)
        index = factory()
        result = run_benchmark(
            index=index,
            records=records,
            queries=queries,
            query_profile=args.profile,
        )
        results.append(result)
        logger.info("  %s: p50=%.1fµs p95=%.1fµs p99=%.1fµs",
                     name,
                     result.query_p50_s * 1e6,
                     result.query_p95_s * 1e6,
                     result.query_p99_s * 1e6)

    # Format output
    if args.format == "markdown":
        output = format_results_markdown(results)
    elif args.format == "json":
        output = format_results_json(results)
    else:
        output = format_results_csv(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
        logger.info("Results written to %s", args.output)
    else:
        print(output)


if __name__ == "__main__":
    main()
