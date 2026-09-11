#!/usr/bin/env python3
"""Port of data/experiments/refresh_strategy.py onto Snowflake: times a full-recompute
GROUP BY against a base-aggregate-plus-incremental-MERGE strategy at a given row count,
using working tables under a dedicated BENCH schema so it never touches landing or gold."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = THIS_FILE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.cloud.snowflake.load_landing import connect  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO  [%(name)s] %(message)s")
logger = logging.getLogger("cloud.snowflake.refresh_benchmark")

SILVER_PATH = _REPO_ROOT / "data" / "processed" / "silver_flight_state.jsonl"
OUT_PATH = _REPO_ROOT / "data" / "cloud" / "snowflake" / "outputs" / "refresh_benchmark.json"

# columns the two strategies actually touch (mirrors generate_scale's select
# list in the duckdb harness)
_RAW_SILVER_COLUMNS = ("icao24", "callsign", "event_ts", "lat", "lon", "baro_altitude_m", "velocity_ms")

_VALID_WAREHOUSE_SIZES = {
    "XSMALL", "SMALL", "MEDIUM", "LARGE", "XLARGE",
    "XXLARGE", "XXXLARGE", "X4LARGE", "X5LARGE", "X6LARGE",
}


def _normalize_warehouse_size(raw: str) -> str:
    token = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if token not in _VALID_WAREHOUSE_SIZES:
        raise ValueError(
            f"unrecognized warehouse size {raw!r}; expected one of {sorted(_VALID_WAREHOUSE_SIZES)}"
        )
    return token


def ensure_bench_schema(cur) -> None:
    """Create the BENCH working schema if it doesn't already exist."""
    cur.execute("create schema if not exists BENCH")


def configure_warehouse(cur, warehouse: str, size: str) -> str:
    """Resize the warehouse, warm it up, and disable result caching so both
    timed strategies see identical compute conditions. Returns the size
    Snowflake reports back, for the results artifact."""
    normalized = _normalize_warehouse_size(size)
    cur.execute(f"alter warehouse {warehouse} set warehouse_size = {normalized}")
    cur.execute("select 1")  # force the warehouse to resume at the new size before timing starts
    cur.fetchone()
    cur.execute("alter session set use_cached_result = false")

    cur.execute(f"show warehouses like '{warehouse}'")
    columns = [c[0].lower() for c in cur.description]
    row = dict(zip(columns, cur.fetchone()))
    return row["size"]


def load_raw_silver(cur, silver_path: Path) -> int:
    """Load the silver JSONL into BENCH.raw_silver. Uses executemany rather than
    PUT+COPY: the seed file is the small silver snapshot (thousands of rows at
    most, not the multiplied benchmark table), so a parameterized insert is
    simpler than staging a file and parsing JSON server-side for a load this small."""
    if not silver_path.exists():
        logger.error(f"Silver data not found at {silver_path}. Run 'make data-local-silver' first.")
        raise FileNotFoundError(silver_path)

    cur.execute("""
        create or replace table BENCH.raw_silver (
            icao24 varchar,
            callsign varchar,
            event_ts timestamp_tz,
            lat float,
            lon float,
            baro_altitude_m float,
            velocity_ms float
        )
    """)

    rows = []
    with silver_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(tuple(
                datetime.fromisoformat(rec[col]) if col == "event_ts" else rec.get(col)
                for col in _RAW_SILVER_COLUMNS
            ))

    cur.executemany(
        "insert into BENCH.raw_silver "
        "(icao24, callsign, event_ts, lat, lon, baro_altitude_m, velocity_ms) "
        "values (%s, %s, %s, %s, %s, %s, %s)",
        rows,
    )
    cur.execute("select count(*) from BENCH.raw_silver")
    return cur.fetchone()[0]


def generate_scale(cur, target_size: int, raw_count: int) -> None:
    """Inflate raw_silver to target_size rows via cross join, mirroring
    generate_scale() in the duckdb harness. Snowflake can't cross join a
    generator directly, so the row multiplier is materialised as its own
    table first."""
    multiplier = (target_size // raw_count) + 1
    logger.info(f"Generating {target_size} rows (multiplier={multiplier})...")

    cur.execute(f"""
        create or replace table BENCH.numbers as
        select seq4() + 1 as seq
        from table(generator(rowcount => {multiplier}))
    """)

    cur.execute(f"""
        create or replace table BENCH.test_silver as
        select
            concat(s.icao24, '_', m.seq::varchar) as icao24,
            s.callsign,
            s.event_ts,
            s.lat,
            s.lon,
            s.baro_altitude_m,
            s.velocity_ms
        from BENCH.raw_silver s
        cross join BENCH.numbers m
        limit {target_size}
    """)


def split_base_inc(cur, base_size: int) -> None:
    """Split test_silver into base/incremental by row_number, mirroring the
    duckdb harness's 90/10 split. Snowflake's row_number() requires an ORDER
    BY (duckdb's bare OVER () isn't valid here), and two independent
    OVER (ORDER BY seq4()) evaluations aren't guaranteed to agree across
    separate statements under Snowflake's parallel scan, so rn is computed
    once and both splits read the same stored column."""
    cur.execute("""
        create or replace table BENCH.test_silver_ranked as
        select *, row_number() over (order by seq4()) as rn
        from BENCH.test_silver
    """)
    cur.execute(f"""
        create or replace table BENCH.silver_base as
        select * from BENCH.test_silver_ranked where rn <= {base_size}
    """)
    cur.execute(f"""
        create or replace table BENCH.silver_inc as
        select * from BENCH.test_silver_ranked where rn > {base_size}
    """)


def run_strategy_a(cur) -> float:
    """Strategy A: timed full recompute."""
    start = time.monotonic()
    cur.execute("""
        create or replace table BENCH.gold_full as
        select
            icao24,
            min(event_ts) as window_start,
            max(event_ts) as window_end,
            max(baro_altitude_m) as max_altitude_m,
            avg(velocity_ms) as avg_velocity_mps,
            count(*) as ping_count
        from BENCH.test_silver
        group by icao24
    """)
    return time.monotonic() - start


def run_strategy_b(cur) -> float:
    """Strategy B: untimed base aggregate (simulates yesterday's gold table),
    then a timed incremental MERGE of the new batch."""
    cur.execute("""
        create or replace table BENCH.gold_inc as
        select
            icao24,
            min(event_ts) as window_start,
            max(event_ts) as window_end,
            max(baro_altitude_m) as max_altitude_m,
            sum(velocity_ms) as sum_velocity_ms,
            count(velocity_ms) as count_velocity_ms,
            count(*) as ping_count
        from BENCH.silver_base
        group by icao24
    """)

    start = time.monotonic()
    cur.execute("""
        merge into BENCH.gold_inc t
        using (
            select
                icao24,
                min(event_ts) as window_start,
                max(event_ts) as window_end,
                max(baro_altitude_m) as max_altitude_m,
                sum(velocity_ms) as sum_velocity_ms,
                count(velocity_ms) as count_velocity_ms,
                count(*) as ping_count
            from BENCH.silver_inc
            group by icao24
        ) s on t.icao24 = s.icao24
        when matched then update set
            window_start = least(t.window_start, s.window_start),
            window_end = greatest(t.window_end, s.window_end),
            max_altitude_m = greatest(t.max_altitude_m, s.max_altitude_m),
            sum_velocity_ms = t.sum_velocity_ms + s.sum_velocity_ms,
            count_velocity_ms = t.count_velocity_ms + s.count_velocity_ms,
            ping_count = t.ping_count + s.ping_count
        when not matched then insert (
            icao24, window_start, window_end, max_altitude_m,
            sum_velocity_ms, count_velocity_ms, ping_count
        ) values (
            s.icao24, s.window_start, s.window_end, s.max_altitude_m,
            s.sum_velocity_ms, s.count_velocity_ms, s.ping_count
        )
    """)
    return time.monotonic() - start


def cleanup(cur) -> None:
    """Drop the BENCH working tables, leaving the schema itself in place."""
    for table in (
        "gold_inc", "gold_full", "silver_inc", "silver_base",
        "test_silver_ranked", "test_silver", "numbers", "raw_silver",
    ):
        cur.execute(f"drop table if exists BENCH.{table}")


def run_benchmark(target_size: int, inc_pct: float, warehouse_size: str, silver_path: Path) -> dict:
    base_size = int(target_size * (1.0 - inc_pct))
    inc_size = target_size - base_size

    logger.info(f"--- Scale: {target_size} rows (base: {base_size}, incremental: {inc_size}) ---")

    conn = connect()
    try:
        cur = conn.cursor()
        try:
            ensure_bench_schema(cur)
            actual_size = configure_warehouse(cur, os.environ["SNOWFLAKE_WAREHOUSE"], warehouse_size)

            cur.execute("select current_region()")
            region = cur.fetchone()[0]

            raw_count = load_raw_silver(cur, silver_path)
            generate_scale(cur, target_size, raw_count)
            split_base_inc(cur, base_size)

            full_recompute_s = run_strategy_a(cur)
            incremental_merge_s = run_strategy_b(cur)
        finally:
            cleanup(cur)
            cur.close()
    finally:
        conn.close()

    logger.info(f"Full Recompute: {full_recompute_s:.4f}s | Incremental MERGE: {incremental_merge_s:.4f}s")

    return {
        "target_size": target_size,
        "base_size": base_size,
        "incremental_size": inc_size,
        "full_recompute_s": round(full_recompute_s, 4),
        "incremental_merge_s": round(incremental_merge_s, 4),
        "speedup_factor": round(full_recompute_s / max(incremental_merge_s, 0.0001), 2),
        "warehouse_size": actual_size,
        "region": region,
        "run_date": datetime.now(timezone.utc).date().isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Incremental vs Recompute Experiment (Snowflake)")
    parser.add_argument("--rows", type=int, default=1_000_000, help="Target row count")
    parser.add_argument("--inc-pct", type=float, default=0.1, help="Fraction of rows treated as incremental")
    parser.add_argument("--warehouse-size", type=str, default="XSMALL", help="Warehouse size for the run")
    parser.add_argument("--silver", type=Path, default=SILVER_PATH, help="Path to the silver JSONL seed file")
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="Path to write the JSON results artifact")
    args = parser.parse_args()

    result = run_benchmark(args.rows, args.inc_pct, args.warehouse_size, args.silver)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Benchmark complete. Results written to {args.out}")

    print("\n=== Experiment Results: Incremental vs Full Recompute (Snowflake) ===")
    print(f"{'Total Rows':<15} | {'Full Recompute (s)':<20} | {'Incremental MERGE (s)':<25} | {'Speedup':<10}")
    print("-" * 80)
    print(
        f"{result['target_size']:<15} | {result['full_recompute_s']:<20} | "
        f"{result['incremental_merge_s']:<25} | {result['speedup_factor']}x"
    )
    print("-" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
