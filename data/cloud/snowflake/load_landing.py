#!/usr/bin/env python3
"""Load local gold Parquet files into Snowflake landing tables that dbt's staging models read unchanged."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
CONTRACTS_DIR = THIS_FILE.parents[3] / "shared" / "contracts"

REQUIRED_ENV_VARS = [
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_ROLE",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
]

# table -> columns, in the order used by the SCHEMAS dict in
# data/scripts/export_gold_parquet.py (this is the Parquet column order).
TABLE_COLUMNS: dict[str, list[str]] = {
    "gold_airport_congestion_landing": [
        "airport_icao",
        "window_start",
        "window_end",
        "aircraft_count",
        "avg_altitude_m",
        "ground_count",
        "airborne_count",
    ],
    "gold_sector_load_landing": [
        "h3_r4",
        "window_start",
        "window_end",
        "aircraft_count",
    ],
    "gold_emergency_events_landing": [
        "icao24",
        "callsign",
        "squawk",
        "first_seen_ts",
        "last_seen_ts",
        "lat",
        "lon",
        "origin_country",
        "nearest_airport",
        "duration_s",
    ],
    "gold_routing_stats_landing": [
        "icao24",
        "callsign",
        "window_start",
        "window_end",
        "origin_lat",
        "origin_lon",
        "destination_lat",
        "destination_lon",
        "max_altitude_m",
        "avg_velocity_mps",
        "ping_count",
    ],
}


def _contract_path(table: str) -> Path:
    base = table[: -len("_landing")] if table.endswith("_landing") else table
    return CONTRACTS_DIR / f"{base}.schema.json"


def _sf_type(prop: dict) -> str:
    if prop.get("format") == "date-time":
        # unqualified timestamp on snowflake is timestamp_ntz and would
        # silently drop the utc offset the parquet columns carry
        return "timestamp_tz"
    raw_type = prop.get("type")
    base_type = raw_type
    if isinstance(raw_type, list):
        base_type = next((t for t in raw_type if t != "null"), "string")
    if base_type == "integer":
        return "number(38,0)"
    if base_type == "number":
        return "float"
    return "varchar"


def landing_ddl(table: str) -> str:
    contract = json.loads(_contract_path(table).read_text(encoding="utf-8"))
    required = set(contract.get("required", []))
    properties = contract["properties"]
    lines = []
    for column in TABLE_COLUMNS[table]:
        sf_type = _sf_type(properties[column])
        suffix = " not null" if column in required else ""
        lines.append(f"    {column} {sf_type}{suffix}")
    body = ",\n".join(lines)
    return f"create or replace table {table} (\n{body}\n)"


def connect():
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"missing required environment variable(s): {', '.join(missing)}"
        )
    import snowflake.connector

    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.environ["SNOWFLAKE_ROLE"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


def load(parquet_dir: Path) -> dict[str, int]:
    """Replace each Snowflake landing table with the current gold Parquet snapshot."""
    parquet_dir = Path(parquet_dir)
    counts: dict[str, int] = {}
    conn = connect()
    try:
        cursor = conn.cursor()
        try:
            for table in TABLE_COLUMNS:
                base = table[: -len("_landing")]
                parquet_path = (parquet_dir / f"{base}.parquet").resolve().as_posix()
                cursor.execute(landing_ddl(table))
                cursor.execute(
                    f"put file://{parquet_path} @%{table} "
                    f"auto_compress=false overwrite=true"
                )
                cursor.execute(
                    f"copy into {table} from @%{table} "
                    f"file_format = (type = parquet) "
                    f"match_by_column_name = case_insensitive purge = true"
                )
                rows = cursor.fetchall()
                counts[table] = sum(row[3] for row in rows)
        finally:
            cursor.close()
    finally:
        conn.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load local gold Parquet into Snowflake landing tables."
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("outputs/bigquery_landing"),
    )
    args = parser.parse_args()

    try:
        counts = load(args.parquet_dir)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for table, row_count in counts.items():
        print(f"{table}: {row_count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
