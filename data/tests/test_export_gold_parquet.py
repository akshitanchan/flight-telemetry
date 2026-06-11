from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from data.scripts.export_gold_parquet import SCHEMAS, export_all


ROWS = {
    "gold_airport_congestion": {
        "airport_icao": "EHAM",
        "window_start": "2024-06-03T12:00:00+00:00",
        "window_end": "2024-06-03T12:05:00+00:00",
        "aircraft_count": 2,
        "avg_altitude_m": 1000.5,
        "ground_count": 1,
        "airborne_count": 1,
    },
    "gold_sector_load": {
        "h3_r4": "841e033ffffffff",
        "window_start": "2024-06-03T12:00:00+00:00",
        "window_end": "2024-06-03T12:05:00+00:00",
        "aircraft_count": 2,
    },
    "gold_emergency_events": {
        "icao24": "400a30",
        "callsign": None,
        "squawk": "7500",
        "first_seen_ts": "2024-06-03T12:00:30+00:00",
        "last_seen_ts": "2024-06-03T12:00:30+00:00",
        "lat": 47.3,
        "lon": 10.8,
        "origin_country": "United Kingdom",
        "nearest_airport": None,
        "duration_s": 0,
    },
    "gold_routing_stats": {
        "icao24": "471f52",
        "callsign": "NOZ163",
        "window_start": "2024-06-03T12:00:00+00:00",
        "window_end": "2024-06-03T12:00:40+00:00",
        "origin_lat": 48.8,
        "origin_lon": 12.9,
        "destination_lat": 48.7,
        "destination_lon": 12.8,
        "max_altitude_m": None,
        "avg_velocity_mps": 106.6,
        "ping_count": 5,
    },
}


def test_export_all_writes_typed_parquet(tmp_path: Path):
    source_dir = tmp_path / "jsonl"
    output_dir = tmp_path / "parquet"
    source_dir.mkdir()

    for table_name, row in ROWS.items():
        (source_dir / f"{table_name}.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )

    counts = export_all(source_dir, output_dir)

    assert counts == {name: 1 for name in SCHEMAS}
    for table_name, expected_schema in SCHEMAS.items():
        table = pq.read_table(output_dir / f"{table_name}.parquet")
        assert table.num_rows == 1
        assert table.schema == expected_schema
        assert table.column_names == expected_schema.names
