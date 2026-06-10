#!/usr/bin/env python3
"""
CLI for the spatiotemporal index benchmark.

Modes
-----
offline (default)
    Uses seeded synthetic records — no database or data files required.
    All three backends (geohash, h3, postgis_gist) are benchmarked; the
    PostGIS strategy is skipped cleanly when no DB is reachable.

postgres
    Same as offline but explicitly requests the PostGIS backend.  Skips
    it (with a WARNING) when DATABASE_URL is unset or the DB is unreachable.

file
    Loads real silver records from --input (legacy behaviour).

Usage examples
--------------
# Offline 3-way comparison (default, fast, CI-safe):
    python -m systems.index.cli

# Offline with 1 million rows:
    python -m systems.index.cli --synthetic-rows 1000000

# Postgres mode (skips PostGIS quietly when no DB):
    python -m systems.index.cli --mode postgres --synthetic-rows 100000

# Legacy file mode:
    python -m systems.index.cli --mode file --input data/processed/silver_flight_state.jsonl

# Custom strategies:
    python -m systems.index.cli --strategies geohash_p4 h3_r4 --format json
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
from systems.index.postgis_index import PostGISIndex
from systems.index.workload import generate_workload, generate_synthetic_records
from systems.index.benchmark import (
    run_benchmark,
    format_results_markdown,
    format_results_json,
    format_results_csv,
)

# Default row count for --synthetic-rows: small enough to stay fast in CI,
# big enough to show meaningful differences between backends.
_DEFAULT_SYNTHETIC_ROWS = 10_000

# Default 3-way comparison strategies for the offline/postgres modes.
# Covers one representative from each backend family.
_DEFAULT_COMPARISON_STRATEGIES = ["geohash_p4", "h3_r4", "postgis_gist"]


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
        description="Run spatiotemporal index benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- Data source ---
    parser.add_argument(
        "--mode",
        choices=["offline", "postgres", "file"],
        default="offline",
        help=(
            "Data source / execution mode. "
            "'offline' (default): use seeded synthetic data, no DB needed. "
            "'postgres': synthetic data + PostGIS backend, skips when DB absent. "
            "'file': load real records from --input (legacy)."
        ),
    )
    parser.add_argument(
        "--synthetic-rows",
        "--rows",
        dest="synthetic_rows",
        type=int,
        default=_DEFAULT_SYNTHETIC_ROWS,
        metavar="N",
        help=(
            f"Number of synthetic records to generate (default: {_DEFAULT_SYNTHETIC_ROWS:,}). "
            "Pass ≥1000000 for the 1M-row benchmark. Only used in offline/postgres modes."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic record + query generation (default: 42).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "silver_flight_state.jsonl",
        help="Path to silver JSONL records (only used in --mode file).",
    )
    # --- Workload ---
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
    # --- Strategy selection ---
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=_DEFAULT_COMPARISON_STRATEGIES,
        help=(
            "Index strategies to benchmark. "
            "Default: geohash_p4 h3_r4 postgis_gist. "
            "Available: geohash_p3/p4/p5, h3_r3/r4/r5/r6, postgis_gist."
        ),
    )
    # --- Output ---
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

    # ------------------------------------------------------------------ #
    # Data loading — mode-dependent                                        #
    # ------------------------------------------------------------------ #
    if args.mode == "file":
        if not args.input.exists():
            print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
            print("Hint: Run 'make data-local-silver' first, or use --mode offline.", file=sys.stderr)
            sys.exit(1)
        logger.info("Loading silver records from %s", args.input)
        records = load_silver_records(args.input)
        logger.info("Loaded %d records", len(records))
    else:
        # offline / postgres — use seeded synthetic data
        logger.info(
            "Generating %d synthetic records (seed=%d)",
            args.synthetic_rows,
            args.seed,
        )
        records = generate_synthetic_records(n=args.synthetic_rows, seed=args.seed)
        logger.info("Generated %d synthetic records", len(records))

    # ------------------------------------------------------------------ #
    # Workload generation                                                  #
    # ------------------------------------------------------------------ #
    logger.info("Generating %d %s queries (seed=%d)", args.queries, args.profile, args.seed)
    queries = generate_workload(
        num_queries=args.queries,
        profile=args.profile,
        seed=args.seed,
    )

    # ------------------------------------------------------------------ #
    # Strategy map — all registered backends                              #
    # ------------------------------------------------------------------ #
    strategy_map = {
        "geohash_p3": lambda: GeohashPrefixIndex(prefix_precision=3),
        "geohash_p4": lambda: GeohashPrefixIndex(prefix_precision=4),
        "geohash_p5": lambda: GeohashPrefixIndex(prefix_precision=5),
        "h3_r3": lambda: H3Index(resolution=3),
        "h3_r4": lambda: H3Index(resolution=4),
        "h3_r5": lambda: H3Index(resolution=5),
        "h3_r6": lambda: H3Index(resolution=6),
        # PostGIS GIST backend — availability-gated: skipped when no DB is reachable.
        "postgis_gist": lambda: PostGISIndex(),
    }

    # ------------------------------------------------------------------ #
    # PostGIS availability gate                                           #
    # ------------------------------------------------------------------ #
    # Evaluate DB reachability once so we log one clear message rather than
    # one per strategy.  The check is intentionally lazy (only runs when
    # postgis_gist is in the requested strategy list) so the offline path
    # never touches the network.
    _postgis_available: bool | None = None

    def _is_postgis_available() -> bool:
        nonlocal _postgis_available
        if _postgis_available is None:
            _postgis_available = PostGISIndex.is_available()
            if not _postgis_available:
                logger.warning(
                    "Strategy postgis_gist requires a live database "
                    "(DATABASE_URL not set or Postgres unreachable) — skipping."
                )
        return _postgis_available

    # ------------------------------------------------------------------ #
    # Run benchmarks                                                       #
    # ------------------------------------------------------------------ #
    results = []
    for name in args.strategies:
        factory = strategy_map.get(name)
        if not factory:
            logger.warning("Unknown strategy: %s (skipping)", name)
            continue

        # Availability gate: only check when this strategy needs a DB.
        if name == "postgis_gist" and not _is_postgis_available():
            # Warning already logged inside _is_postgis_available().
            continue

        logger.info("Benchmarking strategy: %s (%d records)", name, len(records))
        index = factory()
        result = run_benchmark(
            index=index,
            records=records,
            queries=queries,
            query_profile=args.profile,
        )
        results.append(result)
        logger.info(
            "  %s: build=%.3fs p50=%.1fµs p95=%.1fµs p99=%.1fµs",
            name,
            result.build_time_s,
            result.query_p50_s * 1e6,
            result.query_p95_s * 1e6,
            result.query_p99_s * 1e6,
        )

    if not results:
        logger.warning("No strategies produced results — nothing to format.")
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # Format and emit output                                               #
    # ------------------------------------------------------------------ #
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
