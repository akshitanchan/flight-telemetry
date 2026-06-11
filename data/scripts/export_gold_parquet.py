#!/usr/bin/env python3
"""Export the four local C3 gold JSONL tables as typed Parquet files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMAS: dict[str, pa.Schema] = {
    "gold_airport_congestion": pa.schema(
        [
            ("airport_icao", pa.string(), False),
            ("window_start", pa.timestamp("us", tz="UTC"), False),
            ("window_end", pa.timestamp("us", tz="UTC"), False),
            ("aircraft_count", pa.int64(), False),
            ("avg_altitude_m", pa.float64()),
            ("ground_count", pa.int64(), False),
            ("airborne_count", pa.int64(), False),
        ]
    ),
    "gold_sector_load": pa.schema(
        [
            ("h3_r4", pa.string(), False),
            ("window_start", pa.timestamp("us", tz="UTC"), False),
            ("window_end", pa.timestamp("us", tz="UTC"), False),
            ("aircraft_count", pa.int64(), False),
        ]
    ),
    "gold_emergency_events": pa.schema(
        [
            ("icao24", pa.string(), False),
            ("callsign", pa.string()),
            ("squawk", pa.string(), False),
            ("first_seen_ts", pa.timestamp("us", tz="UTC"), False),
            ("last_seen_ts", pa.timestamp("us", tz="UTC"), False),
            ("lat", pa.float64(), False),
            ("lon", pa.float64(), False),
            ("origin_country", pa.string(), False),
            ("nearest_airport", pa.string()),
            ("duration_s", pa.int64(), False),
        ]
    ),
    "gold_routing_stats": pa.schema(
        [
            ("icao24", pa.string(), False),
            ("callsign", pa.string()),
            ("window_start", pa.timestamp("us", tz="UTC"), False),
            ("window_end", pa.timestamp("us", tz="UTC"), False),
            ("origin_lat", pa.float64(), False),
            ("origin_lon", pa.float64(), False),
            ("destination_lat", pa.float64(), False),
            ("destination_lon", pa.float64(), False),
            ("max_altitude_m", pa.float64()),
            ("avg_velocity_mps", pa.float64()),
            ("ping_count", pa.int64(), False),
        ]
    ),
}

TIMESTAMP_COLUMNS = {
    "window_start",
    "window_end",
    "first_seen_ts",
    "last_seen_ts",
}


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def export_table(source: Path, destination: Path, schema: pa.Schema) -> int:
    records: list[dict] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            unknown = set(record) - set(schema.names)
            if unknown:
                raise ValueError(
                    f"{source}:{line_number} has unknown columns: {sorted(unknown)}"
                )
            for column in TIMESTAMP_COLUMNS & set(record):
                record[column] = _parse_timestamp(record[column])
            records.append(record)

    table = pa.Table.from_pylist(records, schema=schema)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="snappy")
    return table.num_rows


def export_all(input_dir: Path, output_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table_name, schema in SCHEMAS.items():
        source = input_dir / f"{table_name}.jsonl"
        if not source.exists():
            raise FileNotFoundError(
                f"Missing {source}. Run `make data-local-gold` first."
            )
        destination = output_dir / f"{table_name}.parquet"
        counts[table_name] = export_table(source, destination, schema)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export local gold JSONL tables as BigQuery-ready Parquet."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bigquery_landing"),
    )
    args = parser.parse_args()

    counts = export_all(args.input_dir, args.output_dir)
    for table_name, row_count in counts.items():
        print(
            f"{table_name}: {row_count} rows -> "
            f"{args.output_dir / f'{table_name}.parquet'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
