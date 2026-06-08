#!/usr/bin/env python3
"""
Data bootstrap — main entry point.

Orchestrates sample data generation and reference data downloads
for local development. All outputs go to gitignored directories.

Usage:
    python data/scripts/bootstrap.py                  # full sample mode
    python data/scripts/bootstrap.py --mode sample    # same as above
    python data/scripts/bootstrap.py --mode reference # download reference data only
    python data/scripts/bootstrap.py --mode all       # sample + reference
    python data/scripts/bootstrap.py --verify         # verify manifest checksums
"""

import argparse
import json
import sys
from pathlib import Path

# Resolve project root relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.scripts.generate_sample import generate_sample_data
from data.scripts.download_reference import download_reference_data
from data.scripts.manifest import generate_manifest, verify_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap sample and reference data for local development"
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "reference", "all"],
        default="sample",
        help="What to bootstrap (default: sample)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Output directory for generated/downloaded data",
    )
    parser.add_argument(
        "--num-aircraft",
        type=int,
        default=10,
        help="Number of distinct aircraft in sample (default: 10)",
    )
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=5,
        help="Number of time snapshots per aircraft (default: 5)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing data against manifest checksums",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verify:
        print("Verifying manifest checksums...")
        ok = verify_manifest(output_dir)
        sys.exit(0 if ok else 1)

    if args.mode in ("sample", "all"):
        print(f"Generating sample data → {output_dir}")
        sample_path = generate_sample_data(
            output_dir=output_dir,
            num_aircraft=args.num_aircraft,
            num_snapshots=args.num_snapshots,
        )
        print(f"  ✓ {sample_path.name} ({sample_path.stat().st_size:,} bytes)")

    if args.mode in ("reference", "all"):
        print(f"Downloading reference data → {output_dir}")
        ref_paths = download_reference_data(output_dir=output_dir)
        for p in ref_paths:
            print(f"  ✓ {p.name} ({p.stat().st_size:,} bytes)")

    # Generate manifest
    print("Generating manifest...")
    manifest_path = generate_manifest(output_dir)
    print(f"  ✓ {manifest_path.name}")

    print("\nBootstrap complete.")


if __name__ == "__main__":
    main()
