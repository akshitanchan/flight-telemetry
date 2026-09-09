#!/usr/bin/env python3
"""Tests for the --table flag of `python -m data.transforms.cli_gold`."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

ALL_GOLD_FILES = {
    "gold_airport_congestion.jsonl",
    "gold_sector_load.jsonl",
    "gold_emergency_events.jsonl",
    "gold_routing_stats.jsonl",
}

# a handful of records exercising all four aggregates: two aircraft in the
# same H3 r7 cell (a real cell, unlike the placeholder values in the shared
# fixtures) and window, one of them squawking an emergency code.
SILVER_RECORDS = [
    {
        "icao24": "111111",
        "callsign": "TST100",
        "event_ts": "2024-06-03T12:02:00+00:00",
        "lon": 4.70,
        "lat": 52.30,
        "squawk": "7700",
        "on_ground": False,
        "baro_altitude_m": 10000.0,
        "velocity_ms": 200.0,
        "h3_r7": "8719694b5ffffff",
        "nearest_airport": "EHAM",
        "origin_country": "Netherlands",
    },
    {
        "icao24": "222222",
        "callsign": "TST200",
        "event_ts": "2024-06-03T12:03:00+00:00",
        "lon": 4.75,
        "lat": 52.35,
        "squawk": "2000",
        "on_ground": True,
        "baro_altitude_m": None,
        "velocity_ms": 0.0,
        "h3_r7": "8719694b5ffffff",
        "nearest_airport": "EHAM",
        "origin_country": "Netherlands",
    },
]


class TestCliGoldSingleTable(unittest.TestCase):
    """`cli_gold --table <name>` writes only the requested gold table."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)

        self.silver_input = self.tmp_dir / "silver_flight_state.jsonl"
        with open(self.silver_input, "w") as f:
            for record in SILVER_RECORDS:
                f.write(json.dumps(record) + "\n")

    def tearDown(self):
        self._tmp.cleanup()

    def _run_cli(self, table: str, out_dir: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable, "-m", "data.transforms.cli_gold",
                "--input", str(self.silver_input),
                "--out-dir", str(out_dir),
                "--table", table,
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_table_sector_writes_only_sector_file(self):
        out_dir = self.tmp_dir / "out_sector"
        out_dir.mkdir()

        result = self._run_cli("sector", out_dir)

        self.assertEqual(
            result.returncode, 0,
            f"cli_gold --table sector failed:\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        written = {p.name for p in out_dir.iterdir()}
        self.assertEqual(
            written, {"gold_sector_load.jsonl"},
            f"expected only gold_sector_load.jsonl, got {written}"
        )
        lines = (out_dir / "gold_sector_load.jsonl").read_text().strip().splitlines()
        self.assertGreater(
            len(lines), 0, "gold_sector_load.jsonl was written but has no records"
        )
        record = json.loads(lines[0])
        self.assertEqual(
            set(record), {"h3_r4", "window_start", "window_end", "aircraft_count"}
        )

    def test_table_all_writes_all_four_files(self):
        out_dir = self.tmp_dir / "out_all"
        out_dir.mkdir()

        result = self._run_cli("all", out_dir)

        self.assertEqual(
            result.returncode, 0,
            f"cli_gold --table all failed:\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        written = {p.name for p in out_dir.iterdir()}
        self.assertEqual(
            written, ALL_GOLD_FILES,
            f"expected all four gold files, got {written}"
        )


if __name__ == "__main__":
    unittest.main()
