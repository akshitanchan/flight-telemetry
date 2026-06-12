#!/usr/bin/env python3
"""
Unit tests for data/cloud/databricks/experiments/harness.py

Tests confirm:
1. run_experiment_scale() returns a dict with the expected keys and correct types.
2. Numeric invariants hold: base_rows + inc_rows == target_size, speedup_factor > 0.
3. run_experiment() (the multi-scale driver) returns one result per requested size.
4. The harness is importable without Spark (offline-safe).

These tests do NOT collide with ds-04's test_gold_logic.py — they exercise the
experiment harness only, not the gold aggregation logic.

Silver input is provided by the ``silver_seed`` pytest fixture, which writes a
small, schema-valid JSONL file into pytest's tmp_path.  No generated artifact
(data/processed/silver_flight_state.jsonl) is required.

Run:
    pytest data/cloud/databricks/tests/test_harness.py -v
"""

import json
import sys
from pathlib import Path

import duckdb
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.cloud.databricks.experiments.harness import (  # noqa: E402
    run_experiment,
    run_experiment_scale,
    _setup_seed,
)

# ---------------------------------------------------------------------------
# Expected result dict keys (contract between harness and notebook/template)
# ---------------------------------------------------------------------------
_EXPECTED_KEYS = {
    "target_size",
    "base_rows",
    "inc_rows",
    "full_recompute_s",
    "incremental_merge_s",
    "speedup_factor",
}

# ---------------------------------------------------------------------------
# Synthetic silver fixture
#
# The harness only reads these columns from the silver JSONL:
#   icao24, event_ts, nearest_airport, on_ground, baro_altitude_m
#
# The records below satisfy the full silver_flight_state schema contract
# (shared/contracts/silver_flight_state.schema.json) so they serve as
# regression-safe examples.  We write enough distinct rows that DuckDB's
# CROSS JOIN inflation produces valid results even at the smallest test
# scale (1 000 rows with multiplier=1).  10 rows is sufficient.
# ---------------------------------------------------------------------------

_SYNTHETIC_SILVER_ROWS = [
    {
        "icao24": "4b1806",
        "callsign": "SWR162",
        "event_ts": "2024-06-03T14:30:05Z",
        "lon": 4.7638,
        "lat": 52.3080,
        "baro_altitude_m": 11278.0,
        "velocity_ms": 234.5,
        "true_track_deg": 42.3,
        "vertical_rate_ms": 0.0,
        "on_ground": False,
        "squawk": "2536",
        "origin_country": "Switzerland",
        "nearest_airport": "EHAM",
        "geohash7": "u173zq6",
        "h3_r7": "872830b2fffffff",
        "metar_wind_kt": 12.0,
        "metar_vis_m": 9999.0,
        "metar_ceiling_ft": 3500.0,
    },
    {
        "icao24": "48414e",
        "callsign": None,
        "event_ts": "2024-06-03T14:30:10Z",
        "lon": 4.7640,
        "lat": 52.3105,
        "baro_altitude_m": None,
        "velocity_ms": 0.0,
        "true_track_deg": None,
        "vertical_rate_ms": None,
        "on_ground": True,
        "squawk": None,
        "origin_country": "Netherlands",
        "nearest_airport": "EHAM",
        "geohash7": "u173zq7",
        "h3_r7": "872830b2fffffff",
        "metar_wind_kt": None,
        "metar_vis_m": None,
        "metar_ceiling_ft": None,
    },
    {
        "icao24": "3c6751",
        "callsign": "DLH42N",
        "event_ts": "2024-06-03T15:00:00Z",
        "lon": 8.5706,
        "lat": 50.0379,
        "baro_altitude_m": 5486.0,
        "velocity_ms": 150.2,
        "true_track_deg": 180.0,
        "vertical_rate_ms": -5.0,
        "on_ground": False,
        "squawk": "7700",
        "origin_country": "Germany",
        "nearest_airport": "EDDF",
        "geohash7": "u0yj3s4",
        "h3_r7": "872830b3fffffff",
        "metar_wind_kt": 8.0,
        "metar_vis_m": 7000.0,
        "metar_ceiling_ft": 2000.0,
    },
    {
        "icao24": "a00001",
        "callsign": None,
        "event_ts": "2024-01-01T00:00:00Z",
        "lon": -73.7781,
        "lat": 40.6413,
        "baro_altitude_m": None,
        "velocity_ms": None,
        "true_track_deg": None,
        "vertical_rate_ms": None,
        "on_ground": True,
        "squawk": None,
        "origin_country": "United States",
        "nearest_airport": None,
        "geohash7": "dr5ru7b",
        "h3_r7": "872a1072fffffff",
        "metar_wind_kt": None,
        "metar_vis_m": None,
        "metar_ceiling_ft": None,
    },
    {
        "icao24": "7c4ee4",
        "callsign": "QFA1",
        "event_ts": "2024-06-03T06:15:00Z",
        "lon": 151.1772,
        "lat": -33.9461,
        "baro_altitude_m": 305.0,
        "velocity_ms": 80.0,
        "true_track_deg": 10.0,
        "vertical_rate_ms": 3.5,
        "on_ground": False,
        "squawk": "1200",
        "origin_country": "Australia",
        "nearest_airport": "YSSY",
        "geohash7": "r3gx2fj",
        "h3_r7": "872bef62fffffff",
        "metar_wind_kt": 5.0,
        "metar_vis_m": 10000.0,
        "metar_ceiling_ft": None,
    },
    {
        "icao24": "c06ab1",
        "callsign": "ACA872",
        "event_ts": "2024-06-03T18:45:00Z",
        "lon": -79.6306,
        "lat": 43.6777,
        "baro_altitude_m": 9144.0,
        "velocity_ms": 220.0,
        "true_track_deg": 270.0,
        "vertical_rate_ms": 0.0,
        "on_ground": False,
        "squawk": "3412",
        "origin_country": "Canada",
        "nearest_airport": "CYYZ",
        "geohash7": "dpz83q5",
        "h3_r7": "872a10a2fffffff",
        "metar_wind_kt": 15.0,
        "metar_vis_m": 8000.0,
        "metar_ceiling_ft": 5000.0,
    },
    {
        "icao24": "400f3a",
        "callsign": "BAW456",
        "event_ts": "2024-06-03T11:20:00Z",
        "lon": -0.4543,
        "lat": 51.4775,
        "baro_altitude_m": 762.0,
        "velocity_ms": 65.0,
        "true_track_deg": 90.0,
        "vertical_rate_ms": -2.0,
        "on_ground": False,
        "squawk": "5061",
        "origin_country": "United Kingdom",
        "nearest_airport": "EGLL",
        "geohash7": "gcpuve7",
        "h3_r7": "872195d4fffffff",
        "metar_wind_kt": 10.0,
        "metar_vis_m": 9999.0,
        "metar_ceiling_ft": 4000.0,
    },
    {
        "icao24": "34651e",
        "callsign": "AFR320",
        "event_ts": "2024-06-03T08:00:00Z",
        "lon": 2.5479,
        "lat": 49.0097,
        "baro_altitude_m": None,
        "velocity_ms": 5.0,
        "true_track_deg": 355.0,
        "vertical_rate_ms": 0.0,
        "on_ground": True,
        "squawk": None,
        "origin_country": "France",
        "nearest_airport": "LFPG",
        "geohash7": "u09tvp0",
        "h3_r7": "8728308afffffff",
        "metar_wind_kt": 7.0,
        "metar_vis_m": 6000.0,
        "metar_ceiling_ft": 1500.0,
    },
    {
        "icao24": "06a0a7",
        "callsign": "UAE211",
        "event_ts": "2024-06-03T21:00:00Z",
        "lon": 55.3644,
        "lat": 25.2532,
        "baro_altitude_m": 12192.0,
        "velocity_ms": 260.0,
        "true_track_deg": 315.0,
        "vertical_rate_ms": 1.0,
        "on_ground": False,
        "squawk": "2200",
        "origin_country": "United Arab Emirates",
        "nearest_airport": "OMDB",
        "geohash7": "thrsuqn",
        "h3_r7": "8743c3c2fffffff",
        "metar_wind_kt": 20.0,
        "metar_vis_m": 5000.0,
        "metar_ceiling_ft": None,
    },
    {
        "icao24": "899f05",
        "callsign": "SIA321",
        "event_ts": "2024-06-03T03:30:00Z",
        "lon": 103.9944,
        "lat": 1.3644,
        "baro_altitude_m": 152.0,
        "velocity_ms": 75.0,
        "true_track_deg": 200.0,
        "vertical_rate_ms": -1.5,
        "on_ground": False,
        "squawk": "4321",
        "origin_country": "Singapore",
        "nearest_airport": "WSSS",
        "geohash7": "w21zjfb",
        "h3_r7": "8765b1d4fffffff",
        "metar_wind_kt": 3.0,
        "metar_vis_m": 10000.0,
        "metar_ceiling_ft": None,
    },
]


@pytest.fixture()
def silver_seed(tmp_path: Path) -> Path:
    """Write a minimal, schema-valid silver JSONL into a temp directory.

    The file contains 10 records covering required and nullable fields as
    defined by shared/contracts/silver_flight_state.schema.json.  All rows
    that the harness needs (icao24, event_ts, nearest_airport, on_ground,
    baro_altitude_m) are present.  The file is discarded after each test.
    """
    silver_file = tmp_path / "silver_flight_state.jsonl"
    with silver_file.open("w") as fh:
        for row in _SYNTHETIC_SILVER_ROWS:
            fh.write(json.dumps(row) + "\n")
    return silver_file


# ---------------------------------------------------------------------------
# Shape and invariant tests — single scale point
# ---------------------------------------------------------------------------

class TestHarnessResultShape:
    """run_experiment_scale returns a well-formed result dict."""

    @pytest.fixture(autouse=True)
    def _con(self, silver_seed: Path) -> None:
        """Open an in-memory DuckDB connection seeded with synthetic silver."""
        self.con = duckdb.connect()
        _setup_seed(self.con, silver_path=silver_seed)
        yield
        self.con.close()

    def test_result_has_expected_keys(self) -> None:
        result = run_experiment_scale(self.con, target_size=5000, inc_pct=0.10)
        assert set(result.keys()) == _EXPECTED_KEYS

    def test_target_size_preserved(self) -> None:
        result = run_experiment_scale(self.con, target_size=5000, inc_pct=0.10)
        assert result["target_size"] == 5000

    def test_base_plus_inc_equals_target(self) -> None:
        """base_rows + inc_rows must equal target_size (exact split)."""
        result = run_experiment_scale(self.con, target_size=5000, inc_pct=0.10)
        assert result["base_rows"] + result["inc_rows"] == result["target_size"]

    def test_latencies_are_positive_floats(self) -> None:
        result = run_experiment_scale(self.con, target_size=5000, inc_pct=0.10)
        assert isinstance(result["full_recompute_s"], float)
        assert isinstance(result["incremental_merge_s"], float)
        assert result["full_recompute_s"] > 0.0
        assert result["incremental_merge_s"] > 0.0

    def test_speedup_is_positive(self) -> None:
        result = run_experiment_scale(self.con, target_size=5000, inc_pct=0.10)
        assert result["speedup_factor"] > 0.0

    def test_different_inc_pct(self) -> None:
        """Harness works with a non-default incremental percentage."""
        result = run_experiment_scale(self.con, target_size=4000, inc_pct=0.20)
        assert result["inc_rows"] == 800   # 20% of 4000
        assert result["base_rows"] == 3200

    def test_small_scale_returns_valid_result(self) -> None:
        """Even a very small scale (1000 rows) should produce a valid result."""
        result = run_experiment_scale(self.con, target_size=1000, inc_pct=0.10)
        assert set(result.keys()) == _EXPECTED_KEYS
        assert result["speedup_factor"] > 0.0


# ---------------------------------------------------------------------------
# Multi-scale driver tests
# ---------------------------------------------------------------------------

class TestHarnessMultiScale:
    """run_experiment() returns one result per requested scale."""

    def test_multi_scale_length(self, silver_seed: Path) -> None:
        sizes = [2000, 5000]
        results = run_experiment(sizes, inc_pct=0.10, save_json=False, silver_path=silver_seed)
        assert len(results) == 2

    def test_multi_scale_target_sizes_match(self, silver_seed: Path) -> None:
        sizes = [2000, 5000]
        results = run_experiment(sizes, inc_pct=0.10, save_json=False, silver_path=silver_seed)
        returned_sizes = [r["target_size"] for r in results]
        assert returned_sizes == sizes

    def test_each_result_has_expected_keys(self, silver_seed: Path) -> None:
        results = run_experiment([3000], inc_pct=0.10, save_json=False, silver_path=silver_seed)
        for r in results:
            assert set(r.keys()) == _EXPECTED_KEYS
