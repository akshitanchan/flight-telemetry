#!/usr/bin/env python3
"""
W2.4: Incremental vs Recompute Refresh Strategy Experiment

This script uses DuckDB to simulate the performance characteristics of an
Incremental MERGE vs a Full Recompute aggregation strategy on columnar data.
"""

import duckdb
import time
import json
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO  [%(name)s] %(message)s")
logger = logging.getLogger("experiments.refresh")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SILVER_PATH = PROJECT_ROOT / "data" / "processed" / "silver_flight_state.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "experiments" / "outputs"

def setup_base_data(con: duckdb.DuckDBPyConnection):
    """Load the silver JSONL into an initial duckdb table."""
    if not SILVER_PATH.exists():
        logger.error(f"Silver data not found at {SILVER_PATH}. Run 'make data-local-silver' first.")
        raise FileNotFoundError(SILVER_PATH)
        
    con.execute(f"CREATE TABLE raw_silver AS SELECT * FROM read_json_auto('{SILVER_PATH}')")
    
def generate_scale(con: duckdb.DuckDBPyConnection, target_size: int, table_name: str):
    """Inflate the raw_silver table using a cross join to reach target_size."""
    logger.info(f"Generating {target_size} rows for '{table_name}'...")
    
    # How many times do we need to duplicate the raw_silver table?
    raw_count = con.execute("SELECT count(*) FROM raw_silver").fetchone()[0]
    multiplier = (target_size // raw_count) + 1
    
    con.execute(f"""
        CREATE TABLE {table_name} AS 
        SELECT 
            -- Make icao24 somewhat unique so the group-by cardinality grows
            concat(s.icao24, '_', m.seq::VARCHAR) as icao24,
            s.callsign,
            s.event_ts,
            s.lat,
            s.lon,
            s.baro_altitude_m,
            s.velocity_ms
        FROM raw_silver s
        CROSS JOIN (SELECT generate_series AS seq FROM generate_series(1, {multiplier})) m
        LIMIT {target_size}
    """)

def run_experiment_scale(con: duckdb.DuckDBPyConnection, target_size: int, inc_pct: float) -> dict:
    base_size = int(target_size * (1.0 - inc_pct))
    inc_size = target_size - base_size
    
    logger.info(f"--- Scale: {target_size} rows (Base: {base_size}, Incremental: {inc_size}) ---")
    
    # 1. Setup tables
    con.execute("DROP TABLE IF EXISTS test_silver")
    generate_scale(con, target_size, "test_silver")
    
    # Split into base and incremental based on a synthetic row_number
    con.execute("DROP TABLE IF EXISTS silver_base")
    con.execute("DROP TABLE IF EXISTS silver_inc")
    
    con.execute(f"""
        CREATE TABLE silver_base AS 
        SELECT * FROM (
            SELECT *, row_number() over () as rn FROM test_silver
        ) WHERE rn <= {base_size}
    """)
    con.execute(f"""
        CREATE TABLE silver_inc AS 
        SELECT * FROM (
            SELECT *, row_number() over () as rn FROM test_silver
        ) WHERE rn > {base_size}
    """)
    
    # ---------------------------------------------------------
    # Strategy A: Full Recompute
    # ---------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS gold_full")
    start_time = time.monotonic()
    
    # Simulate routing stats grouping
    con.execute("""
        CREATE TABLE gold_full AS 
        SELECT 
            icao24,
            min(event_ts) as window_start,
            max(event_ts) as window_end,
            max(baro_altitude_m) as max_altitude_m,
            avg(velocity_ms) as avg_velocity_mps,
            count(*) as ping_count
        FROM test_silver
        GROUP BY icao24
    """)
    
    full_recompute_s = time.monotonic() - start_time
    
    # ---------------------------------------------------------
    # Strategy B: Incremental MERGE
    # ---------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS gold_inc")
    
    # Pre-build the base gold table (un-timed, simulates the state from yesterday)
    con.execute("""
        CREATE TABLE gold_inc AS 
        SELECT 
            icao24,
            min(event_ts) as window_start,
            max(event_ts) as window_end,
            max(baro_altitude_m) as max_altitude_m,
            sum(velocity_ms) as sum_velocity_ms,
            count(velocity_ms) as count_velocity_ms,
            count(*) as ping_count
        FROM silver_base
        GROUP BY icao24
    """)
    
    start_time = time.monotonic()
    
    # Execute the MERGE using the new incremental batch
    # In BigQuery/Databricks, this is a standard MERGE INTO statement
    con.execute("""
        MERGE INTO gold_inc t
        USING (
            SELECT 
                icao24,
                min(event_ts) as window_start,
                max(event_ts) as window_end,
                max(baro_altitude_m) as max_altitude_m,
                sum(velocity_ms) as sum_velocity_ms,
                count(velocity_ms) as count_velocity_ms,
                count(*) as ping_count
            FROM silver_inc
            GROUP BY icao24
        ) s ON t.icao24 = s.icao24
        WHEN MATCHED THEN UPDATE SET
            window_start = LEAST(t.window_start, s.window_start),
            window_end = GREATEST(t.window_end, s.window_end),
            max_altitude_m = GREATEST(t.max_altitude_m, s.max_altitude_m),
            sum_velocity_ms = t.sum_velocity_ms + s.sum_velocity_ms,
            count_velocity_ms = t.count_velocity_ms + s.count_velocity_ms,
            ping_count = t.ping_count + s.ping_count
        WHEN NOT MATCHED THEN INSERT (
            icao24, window_start, window_end, max_altitude_m, 
            sum_velocity_ms, count_velocity_ms, ping_count
        ) VALUES (
            s.icao24, s.window_start, s.window_end, s.max_altitude_m, 
            s.sum_velocity_ms, s.count_velocity_ms, s.ping_count
        )
    """)
    
    incremental_merge_s = time.monotonic() - start_time
    
    logger.info(f"Full Recompute: {full_recompute_s:.4f}s | Incremental MERGE: {incremental_merge_s:.4f}s")
    
    return {
        "target_size": target_size,
        "base_size": base_size,
        "incremental_size": inc_size,
        "full_recompute_s": round(full_recompute_s, 4),
        "incremental_merge_s": round(incremental_merge_s, 4),
        "speedup_factor": round(full_recompute_s / max(incremental_merge_s, 0.0001), 2)
    }

def main():
    parser = argparse.ArgumentParser(description="Incremental vs Recompute Experiment")
    parser.add_argument("--sizes", type=str, default="10000,100000,500000", help="Comma-separated target row counts")
    parser.add_argument("--inc-pct", type=float, default=0.1, help="Percentage of data considered incremental (default 0.1 for 10%%)")
    args = parser.parse_args()
    
    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    
    con = duckdb.connect()
    
    # 1. Load initial seed data
    setup_base_data(con)
    
    results = []
    
    # 2. Run experiment for each scale
    for size in sizes:
        res = run_experiment_scale(con, size, args.inc_pct)
        results.append(res)
        
    # 3. Output results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "refresh_experiment.json"
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Experiment complete. Results written to {out_file}")
    
    # Print ascii table summary
    print("\n=== Experiment Results: Incremental vs Full Recompute ===")
    print(f"{'Total Rows':<15} | {'Full Recompute (s)':<20} | {'Incremental MERGE (s)':<25} | {'Speedup':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['target_size']:<15} | {r['full_recompute_s']:<20} | {r['incremental_merge_s']:<25} | {r['speedup_factor']}x")
    print("-" * 80)

if __name__ == "__main__":
    main()
