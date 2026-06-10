#!/usr/bin/env python3
"""
W4.3 · ds-05 — Incremental MERGE vs Full-Recompute Experiment Harness

Measures the latency of two refresh strategies for gold_airport_congestion at
multiple row-count scales, using DuckDB as the local columnar proxy.

This harness is the LOCAL SMOKE CHECK.  The headline numbers come from the
owner's Databricks / Delta cloud run (see notebooks/03_refresh_experiment.py
and results_template.md).  DuckDB results are labelled "proxy" throughout.

Design mirrors data/experiments/refresh_strategy.py with the following
upgrades:

1. Schema matches gold_airport_congestion exactly (airport_icao, window_start,
   window_end, aircraft_count, ground_count, airborne_count, avg_altitude_m,
   window_date) — not the icao24-keyed routing stats proxy.

2. The MERGE statement mirrors the production MERGE in
   notebooks/02_silver_to_gold.py Section B exactly (match on
   (airport_icao, window_start); full row replacement on MATCHED; INSERT on
   NOT MATCHED).

3. avg_altitude_m is always derived as alt_sum / alt_count, never as an
   average-of-averages, so the MERGE is idempotent.

4. Results are returned as a structured dict (not just printed) so the test
   in tests/test_harness.py can assert shape without parsing stdout.

Usage (smoke check):
    python data/cloud/databricks/experiments/harness.py --sizes 10000
    python data/cloud/databricks/experiments/harness.py --sizes 10000,50000,200000 --inc-pct 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import duckdb

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s INFO  [%(name)s] %(message)s",
)
logger = logging.getLogger("experiments.harness")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
SILVER_PATH = _PROJECT_ROOT / "data" / "processed" / "silver_flight_state.jsonl"
OUT_DIR = _PROJECT_ROOT / "data" / "cloud" / "databricks" / "experiments" / "outputs"

# ---------------------------------------------------------------------------
# Result type alias
# ---------------------------------------------------------------------------

ExperimentResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Data setup
# ---------------------------------------------------------------------------

def _setup_seed(con: duckdb.DuckDBPyConnection) -> None:
    """Load the silver JSONL into a seed DuckDB table called raw_silver.

    Mirrors setup_base_data() in data/experiments/refresh_strategy.py.
    The seed is loaded only once per connection; subsequent calls to
    generate_scale() inflate it via CROSS JOIN.
    """
    if not SILVER_PATH.exists():
        raise FileNotFoundError(
            f"Silver data not found at {SILVER_PATH}. "
            "Run 'make data-local-silver' first."
        )
    logger.info("Loading silver seed from %s", SILVER_PATH)
    con.execute(
        f"CREATE TABLE IF NOT EXISTS raw_silver "
        f"AS SELECT * FROM read_json_auto('{SILVER_PATH}')"
    )


def _generate_synthetic_silver(
    con: duckdb.DuckDBPyConnection,
    target_size: int,
    table_name: str,
) -> None:
    """Inflate raw_silver to target_size rows using CROSS JOIN.

    Synthetic rows get varied airport identifiers by cycling through a set of
    real ICAO airport codes so that the GROUP BY in the recompute step
    produces realistic cardinality.  airport values are derived from
    'nearest_airport' in the silver source where present; rows without
    nearest_airport are assigned one of five synthetic airports to ensure
    every row has a non-NULL airport_icao (the experiment measures MERGE
    performance, not NULL-filtering rate).

    Mirrors generate_scale() in data/experiments/refresh_strategy.py.
    """
    raw_count_row = con.execute("SELECT count(*) FROM raw_silver").fetchone()
    raw_count = raw_count_row[0] if raw_count_row else 1
    multiplier = (target_size // max(raw_count, 1)) + 1

    logger.info("Generating %d rows for '%s' (multiplier=%d)...", target_size, table_name, multiplier)

    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"""
        CREATE TABLE {table_name} AS
        SELECT
            -- Vary icao24 so GROUP BY cardinality grows with scale.
            concat(s.icao24, '_', m.seq::VARCHAR)                     AS icao24,
            s.event_ts,
            -- Assign an airport: use existing nearest_airport or cycle synthetics.
            CASE
                WHEN s.nearest_airport IS NOT NULL THEN s.nearest_airport
                ELSE (
                    CASE (m.seq % 5)
                        WHEN 0 THEN 'EHAM'
                        WHEN 1 THEN 'EGLL'
                        WHEN 2 THEN 'LFPG'
                        WHEN 3 THEN 'EDDF'
                        ELSE        'LEMD'
                    END
                )
            END                                                        AS airport_icao,
            -- Derive 5-minute window bucket (mirrors _floor_to_window).
            date_trunc('minute', s.event_ts::TIMESTAMP)
                - INTERVAL (CAST(
                      extract(minute FROM s.event_ts::TIMESTAMP)::INT % 5
                  AS INT) * 60) SECOND                                 AS window_start,
            date_trunc('minute', s.event_ts::TIMESTAMP)
                - INTERVAL (CAST(
                      extract(minute FROM s.event_ts::TIMESTAMP)::INT % 5
                  AS INT) * 60) SECOND
                + INTERVAL '5' MINUTE                                  AS window_end,
            -- Vary on_ground so both ground/airborne counts are populated.
            (m.seq % 3 = 0)                                           AS on_ground,
            s.baro_altitude_m
        FROM raw_silver s
        CROSS JOIN (
            SELECT generate_series AS seq
            FROM generate_series(1, {multiplier})
        ) m
        LIMIT {target_size}
    """)


# ---------------------------------------------------------------------------
# Core experiment: one scale point
# ---------------------------------------------------------------------------

def run_experiment_scale(
    con: duckdb.DuckDBPyConnection,
    target_size: int,
    inc_pct: float,
) -> ExperimentResult:
    """Time full GROUP BY recompute vs incremental MERGE at one scale.

    Parameters
    ----------
    con:
        Active DuckDB connection (in-memory or file-backed).
    target_size:
        Total number of synthetic silver rows to generate.
    inc_pct:
        Fraction of rows treated as the "new" incremental batch.
        Must be in (0, 1).  Default 0.10 = 10% incremental.

    Returns
    -------
    dict with keys:
        target_size, base_rows, inc_rows,
        full_recompute_s, incremental_merge_s, speedup_factor
    """
    base_size = int(target_size * (1.0 - inc_pct))
    inc_size = target_size - base_size

    logger.info(
        "--- Scale: %d rows (base: %d, incremental: %d) ---",
        target_size, base_size, inc_size,
    )

    # 1. Generate full synthetic silver, then split.
    _generate_synthetic_silver(con, target_size, "exp_silver")

    con.execute("DROP TABLE IF EXISTS exp_silver_base")
    con.execute("DROP TABLE IF EXISTS exp_silver_inc")

    con.execute(f"""
        CREATE TABLE exp_silver_base AS
        SELECT * FROM (
            SELECT *, row_number() OVER () AS _rn FROM exp_silver
        ) WHERE _rn <= {base_size}
    """)
    con.execute(f"""
        CREATE TABLE exp_silver_inc AS
        SELECT * FROM (
            SELECT *, row_number() OVER () AS _rn FROM exp_silver
        ) WHERE _rn > {base_size}
    """)

    # ------------------------------------------------------------------
    # Strategy A: Full Recompute — aggregate ALL silver rows into gold.
    # Mirrors Section A4 of notebooks/02_silver_to_gold.py.
    # ------------------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS gold_full_recompute")
    _t0 = time.monotonic()

    con.execute("""
        CREATE TABLE gold_full_recompute AS
        SELECT
            airport_icao,
            window_start,
            window_end,
            count(icao24)                                  AS aircraft_count,
            sum(CASE WHEN on_ground THEN 1 ELSE 0 END)    AS ground_count,
            sum(CASE WHEN NOT on_ground THEN 1 ELSE 0 END) AS airborne_count,
            CASE
                WHEN count(baro_altitude_m) > 0
                     THEN round(sum(baro_altitude_m) / count(baro_altitude_m), 2)
                ELSE NULL
            END                                            AS avg_altitude_m,
            window_start::DATE                             AS window_date
        FROM exp_silver
        GROUP BY airport_icao, window_start, window_end
    """)

    full_recompute_s = time.monotonic() - _t0

    # ------------------------------------------------------------------
    # Strategy B: Incremental MERGE
    #
    # Step 1 (un-timed): pre-build the base gold table from the base
    #   silver partition.  This simulates the "state from yesterday"
    #   that already exists in the target before the new batch arrives.
    #
    # Step 2 (timed): aggregate only the incremental silver batch and
    #   MERGE INTO the base gold table.
    #
    # MERGE statement mirrors the production MERGE in Section B2 of
    # notebooks/02_silver_to_gold.py exactly:
    #   - Match key: (airport_icao, window_start)
    #   - WHEN MATCHED: full row replacement (all columns updated)
    #   - WHEN NOT MATCHED: INSERT new window row
    #
    # avg_altitude_m is recomputed from alt_sum / alt_count in both the
    # base build and the incremental aggregate — never averaged-of-averages.
    # ------------------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS gold_incremental")

    # Pre-build base gold (un-timed).
    con.execute("""
        CREATE TABLE gold_incremental AS
        SELECT
            airport_icao,
            window_start,
            window_end,
            count(icao24)                                  AS aircraft_count,
            sum(CASE WHEN on_ground THEN 1 ELSE 0 END)    AS ground_count,
            sum(CASE WHEN NOT on_ground THEN 1 ELSE 0 END) AS airborne_count,
            CASE
                WHEN count(baro_altitude_m) > 0
                     THEN round(sum(baro_altitude_m) / count(baro_altitude_m), 2)
                ELSE NULL
            END                                            AS avg_altitude_m,
            window_start::DATE                             AS window_date
        FROM exp_silver_base
        GROUP BY airport_icao, window_start, window_end
    """)

    # Timed: aggregate incremental batch and MERGE.
    _t1 = time.monotonic()

    con.execute("""
        MERGE INTO gold_incremental AS target
        USING (
            SELECT
                airport_icao,
                window_start,
                window_end,
                count(icao24)                                  AS aircraft_count,
                sum(CASE WHEN on_ground THEN 1 ELSE 0 END)    AS ground_count,
                sum(CASE WHEN NOT on_ground THEN 1 ELSE 0 END) AS airborne_count,
                CASE
                    WHEN count(baro_altitude_m) > 0
                         THEN round(sum(baro_altitude_m) / count(baro_altitude_m), 2)
                    ELSE NULL
                END                                            AS avg_altitude_m,
                window_start::DATE                             AS window_date
            FROM exp_silver_inc
            GROUP BY airport_icao, window_start, window_end
        ) AS source
        ON  target.airport_icao = source.airport_icao
        AND target.window_start = source.window_start
        WHEN MATCHED THEN UPDATE SET
            window_end     = source.window_end,
            aircraft_count = source.aircraft_count,
            ground_count   = source.ground_count,
            airborne_count = source.airborne_count,
            avg_altitude_m = source.avg_altitude_m,
            window_date    = source.window_date
        WHEN NOT MATCHED THEN INSERT (
            airport_icao, window_start, window_end,
            aircraft_count, ground_count, airborne_count,
            avg_altitude_m, window_date
        ) VALUES (
            source.airport_icao, source.window_start, source.window_end,
            source.aircraft_count, source.ground_count, source.airborne_count,
            source.avg_altitude_m, source.window_date
        )
    """)

    incremental_merge_s = time.monotonic() - _t1

    speedup = round(
        full_recompute_s / max(incremental_merge_s, 1e-6), 2
    )

    logger.info(
        "Full Recompute: %.4fs | Incremental MERGE: %.4fs | Speedup: %.2fx",
        full_recompute_s, incremental_merge_s, speedup,
    )

    # Clean up per-scale temp tables so the next iteration starts clean.
    for tbl in ("exp_silver", "exp_silver_base", "exp_silver_inc",
                "gold_full_recompute", "gold_incremental"):
        con.execute(f"DROP TABLE IF EXISTS {tbl}")

    return {
        "target_size": target_size,
        "base_rows": base_size,
        "inc_rows": inc_size,
        "full_recompute_s": round(full_recompute_s, 4),
        "incremental_merge_s": round(incremental_merge_s, 4),
        "speedup_factor": speedup,
    }


# ---------------------------------------------------------------------------
# Multi-scale driver
# ---------------------------------------------------------------------------

def run_experiment(
    sizes: list[int],
    inc_pct: float = 0.10,
    *,
    save_json: bool = True,
) -> list[ExperimentResult]:
    """Run the experiment over multiple row-count scales.

    Parameters
    ----------
    sizes:
        List of target row counts (e.g. [10_000, 50_000, 200_000]).
    inc_pct:
        Fraction of rows treated as the incremental batch.
    save_json:
        If True, write results to experiments/outputs/harness_results.json.

    Returns
    -------
    List of ExperimentResult dicts, one per scale point.
    """
    con = duckdb.connect()
    _setup_seed(con)

    results: list[ExperimentResult] = []
    for size in sizes:
        res = run_experiment_scale(con, size, inc_pct)
        results.append(res)

    con.close()

    if save_json:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_file = OUT_DIR / "harness_results.json"
        with open(out_file, "w") as fh:
            json.dump(results, fh, indent=2)
        logger.info("Results written to %s", out_file)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_table(results: list[ExperimentResult]) -> None:
    """Print an ASCII results table to stdout."""
    print()
    print("=== DuckDB Proxy Results (LOCAL — not headline) ===")
    print(
        f"{'Total Rows':<14} | {'Base Rows':<12} | {'Inc Rows':<10} | "
        f"{'Full Recompute (s)':<20} | {'Incremental MERGE (s)':<23} | {'Speedup':<8}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r['target_size']:<14} | {r['base_rows']:<12} | {r['inc_rows']:<10} | "
            f"{r['full_recompute_s']:<20} | {r['incremental_merge_s']:<23} | {r['speedup_factor']}x"
        )
    print("-" * 100)
    print()
    print("NOTE: DuckDB numbers are a local columnar PROXY for Delta behaviour.")
    print("      Fill results_template.md with real Databricks cluster numbers.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "W4.3 ds-05 — Incremental MERGE vs Full-Recompute Experiment Harness "
            "(DuckDB local proxy). "
            "Real headline numbers come from the owner's Databricks run."
        )
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="10000,50000,200000",
        help="Comma-separated target row counts (default: 10000,50000,200000). "
             "Databricks Free Edition quota: keep max at 500000.",
    )
    parser.add_argument(
        "--inc-pct",
        type=float,
        default=0.10,
        help="Fraction of rows treated as the incremental batch (default: 0.10 = 10%%).",
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    results = run_experiment(sizes, inc_pct=args.inc_pct)
    _print_table(results)


if __name__ == "__main__":
    main()
