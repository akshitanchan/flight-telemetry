#!/usr/bin/env python3
"""
Replay CLI — command-line entry point for historical replay ingestion.

Reads sample telemetry JSONL, normalizes records, deduplicates on
(icao24, event_ts), and writes to a landing file.

Usage:
    python -m systems.replay.cli --input data/raw/sample_state_vectors.jsonl
    python -m systems.replay.cli --input data/raw/sample_state_vectors.jsonl --dry-run
    python -m systems.replay.cli --input data/raw/sample_state_vectors.jsonl --output data/interim/landing.jsonl
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from systems.replay.reader import read_snapshots
from systems.replay.normalizer import normalize_batch
from systems.replay.dedup import IdempotencyStore
from systems.replay.writer import LandingWriter


def configure_logging(level: str = "INFO") -> None:
    """Configure structured logging to stderr."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )


def run_replay(
    input_path: Path,
    output_path: Path,
    journal_path: Path | None,
    dry_run: bool,
) -> dict:
    """Execute the replay ingestion pipeline.

    Returns a summary dict with stats.
    """
    logger = logging.getLogger("replay")
    start = time.monotonic()

    # Initialize components
    dedup = IdempotencyStore(journal_path=journal_path)
    writer = LandingWriter(output_path=output_path, dry_run=dry_run)

    total_snapshots = 0
    total_vectors = 0
    total_normalized = 0
    total_written = 0

    with writer:
        for snapshot in read_snapshots(input_path):
            total_snapshots += 1
            ts = snapshot["time"]
            states = snapshot.get("states") or []
            total_vectors += len(states)

            # Normalize
            records = normalize_batch(ts, states)
            total_normalized += len(records)

            # Deduplicate and write
            for record in records:
                idem_key = record["idem_key"]
                if dedup.check_and_mark(idem_key):
                    writer.write(record)
                    total_written += 1

    # Persist journal for future runs
    if not dry_run:
        dedup.flush_journal()

    elapsed = time.monotonic() - start
    dedup_stats = dedup.stats

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "dry_run": dry_run,
        "snapshots_read": total_snapshots,
        "vectors_read": total_vectors,
        "vectors_normalized": total_normalized,
        "vectors_written": total_written,
        "duplicates_skipped": dedup_stats["duplicates_this_run"],
        "elapsed_s": round(elapsed, 3),
        "throughput_records_per_s": (
            round(total_vectors / elapsed, 1) if elapsed > 0 else 0
        ),
    }

    logger.info("Replay complete: %s", json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Replay historical telemetry through ingestion pipeline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSONL file (OpenSky /states/all format)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "landing.jsonl",
        help="Path to output landing JSONL file",
    )
    parser.add_argument(
        "--journal",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / ".idem_journal",
        help="Path to idempotency journal file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and normalize but do not write output",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    args = parser.parse_args()

    configure_logging(args.log_level)

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        print("Hint: Run 'make data-sample' first to generate sample data.", file=sys.stderr)
        sys.exit(1)

    summary = run_replay(
        input_path=args.input,
        output_path=args.output,
        journal_path=None if args.dry_run else args.journal,
        dry_run=args.dry_run,
    )

    # Print summary to stdout as JSON
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
