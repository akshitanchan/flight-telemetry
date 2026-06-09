#!/usr/bin/env python3
"""Tests for the bronze-to-silver transform."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.transforms.bronze_to_silver import transform_record, run_transform


class TestTransformRecord(unittest.TestCase):
    """Test individual record transformation."""

    def _make_landing(self, **overrides):
        """Build a valid landing record with optional overrides."""
        base = {
            "icao24": "4b1806",
            "callsign": "SWR162",
            "origin_country": "Switzerland",
            "event_ts": "2024-06-03T12:00:00+00:00",
            "snapshot_ts": "2024-06-03T12:00:00+00:00",
            "lon": 4.7638,
            "lat": 52.3080,
            "baro_altitude_m": 11278.0,
            "geo_altitude_m": 11290.5,
            "on_ground": False,
            "velocity_ms": 234.5,
            "true_track_deg": 42.3,
            "vertical_rate_ms": 0.0,
            "squawk": "2536",
            "spi": False,
            "position_source": 0,
            "last_contact": 1717416000,
            "idem_key": "4b1806:1717416000",
        }
        base.update(overrides)
        return base

    def test_basic_transform(self):
        """Valid landing record produces a silver record with all required fields."""
        record = self._make_landing()
        silver = transform_record(record)

        self.assertIsNotNone(silver)
        self.assertEqual(silver["icao24"], "4b1806")
        self.assertEqual(silver["callsign"], "SWR162")
        self.assertEqual(silver["event_ts"], "2024-06-03T12:00:00+00:00")
        self.assertEqual(silver["lon"], 4.7638)
        self.assertEqual(silver["lat"], 52.308)
        self.assertFalse(silver["on_ground"])
        self.assertEqual(silver["origin_country"], "Switzerland")

    def test_geohash_computed(self):
        """geohash7 is computed and is 7 characters."""
        silver = transform_record(self._make_landing())
        self.assertIsNotNone(silver)
        self.assertEqual(len(silver["geohash7"]), 7)
        self.assertRegex(silver["geohash7"], r"^[0-9b-hjkmnp-z]{7}$")

    def test_h3_computed(self):
        """h3_r7 is computed and is 15 hex characters."""
        silver = transform_record(self._make_landing())
        self.assertIsNotNone(silver)
        self.assertEqual(len(silver["h3_r7"]), 15)
        self.assertRegex(silver["h3_r7"], r"^[0-9a-f]{15}$")

    def test_enrichment_stubs_null(self):
        """Enrichment fields are stubbed as null."""
        silver = transform_record(self._make_landing())
        self.assertIsNone(silver["nearest_airport"])
        self.assertIsNone(silver["metar_wind_kt"])
        self.assertIsNone(silver["metar_vis_m"])
        self.assertIsNone(silver["metar_ceiling_ft"])

    def test_no_extra_fields(self):
        """Silver record has no extra fields beyond the schema."""
        silver = transform_record(self._make_landing())
        expected_fields = {
            "icao24", "callsign", "event_ts", "lon", "lat",
            "baro_altitude_m", "velocity_ms", "true_track_deg",
            "vertical_rate_ms", "on_ground", "squawk", "origin_country",
            "geohash7", "h3_r7", "nearest_airport",
            "metar_wind_kt", "metar_vis_m", "metar_ceiling_ft",
        }
        self.assertEqual(set(silver.keys()), expected_fields)

    def test_drops_missing_icao24(self):
        """Record without icao24 is dropped."""
        silver = transform_record(self._make_landing(icao24=None))
        self.assertIsNone(silver)

    def test_drops_short_icao24(self):
        """Record with too-short icao24 is dropped."""
        silver = transform_record(self._make_landing(icao24="abc"))
        self.assertIsNone(silver)

    def test_drops_missing_lon(self):
        """Record without lon is dropped."""
        silver = transform_record(self._make_landing(lon=None))
        self.assertIsNone(silver)

    def test_drops_missing_lat(self):
        """Record without lat is dropped."""
        silver = transform_record(self._make_landing(lat=None))
        self.assertIsNone(silver)

    def test_drops_out_of_range_lon(self):
        """Record with lon > 180 is dropped."""
        silver = transform_record(self._make_landing(lon=200.0))
        self.assertIsNone(silver)

    def test_drops_out_of_range_lat(self):
        """Record with lat < -90 is dropped."""
        silver = transform_record(self._make_landing(lat=-100.0))
        self.assertIsNone(silver)

    def test_drops_negative_velocity(self):
        """Record with negative velocity is dropped."""
        silver = transform_record(self._make_landing(velocity_ms=-5.0))
        self.assertIsNone(silver)

    def test_nullable_velocity_ok(self):
        """Null velocity passes range check."""
        silver = transform_record(self._make_landing(velocity_ms=None))
        self.assertIsNotNone(silver)
        self.assertIsNone(silver["velocity_ms"])

    def test_invalid_squawk_sanitized(self):
        """Squawk with non-octal digits is set to None."""
        silver = transform_record(self._make_landing(squawk="8888"))
        self.assertIsNotNone(silver)
        self.assertIsNone(silver["squawk"])

    def test_valid_squawk_preserved(self):
        """Valid octal squawk is preserved."""
        silver = transform_record(self._make_landing(squawk="7700"))
        self.assertIsNotNone(silver)
        self.assertEqual(silver["squawk"], "7700")

    def test_callsign_stripped(self):
        """Trailing-whitespace callsign is stripped (M15)."""
        silver = transform_record(self._make_landing(callsign="SWR162 "))
        self.assertIsNotNone(silver)
        self.assertEqual(silver["callsign"], "SWR162")

    def test_empty_callsign_becomes_none(self):
        """Whitespace-only callsign becomes None (M15)."""
        silver = transform_record(self._make_landing(callsign="   "))
        self.assertIsNotNone(silver)
        self.assertIsNone(silver["callsign"])

    def test_drops_impossible_altitude(self):
        """Physically impossible barometric altitude is dropped (M16)."""
        silver = transform_record(self._make_landing(baro_altitude_m=99999.0))
        self.assertIsNone(silver)

    def test_nullable_altitude_ok(self):
        """Null altitude passes the range check (M16)."""
        silver = transform_record(self._make_landing(baro_altitude_m=None))
        self.assertIsNotNone(silver)
        self.assertIsNone(silver["baro_altitude_m"])


class TestRunTransform(unittest.TestCase):
    """Integration test for the full transform pipeline."""

    def _make_landing_line(self, **overrides):
        """Build a landing JSONL line."""
        base = {
            "icao24": "4b1806",
            "callsign": "SWR162",
            "origin_country": "Switzerland",
            "event_ts": "2024-06-03T12:00:00+00:00",
            "snapshot_ts": "2024-06-03T12:00:00+00:00",
            "lon": 4.7638,
            "lat": 52.3080,
            "baro_altitude_m": 11278.0,
            "geo_altitude_m": 11290.5,
            "on_ground": False,
            "velocity_ms": 234.5,
            "true_track_deg": 42.3,
            "vertical_rate_ms": 0.0,
            "squawk": "2536",
            "spi": False,
            "position_source": 0,
            "last_contact": 1717416000,
            "idem_key": "4b1806:1717416000",
        }
        base.update(overrides)
        return json.dumps(base)

    def test_end_to_end(self):
        """Full pipeline reads bronze, writes silver, validates against contract."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fin:
            input_path = Path(fin.name)
            fin.write(self._make_landing_line() + "\n")
            fin.write(self._make_landing_line(icao24="3c6751", callsign="DLH42N",
                                               idem_key="3c6751:1717416000") + "\n")

        output_path = Path(tempfile.mktemp(suffix=".jsonl"))

        try:
            summary = run_transform(input_path, output_path, validate=True)

            self.assertEqual(summary["records_read"], 2)
            self.assertEqual(summary["records_written"], 2)
            self.assertEqual(summary["records_dropped"], 0)
            self.assertEqual(summary["validation_errors"], 0)

            # Verify output records
            with open(output_path) as f:
                records = [json.loads(line) for line in f]

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["icao24"], "4b1806")
            self.assertEqual(records[1]["icao24"], "3c6751")

            # Verify silver-required fields
            for r in records:
                self.assertIn("geohash7", r)
                self.assertIn("h3_r7", r)
                self.assertEqual(len(r["geohash7"]), 7)
                self.assertEqual(len(r["h3_r7"]), 15)

        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_dedup_in_transform(self):
        """Duplicate records in input are deduplicated."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fin:
            input_path = Path(fin.name)
            # Same idem_key written twice
            fin.write(self._make_landing_line() + "\n")
            fin.write(self._make_landing_line() + "\n")

        output_path = Path(tempfile.mktemp(suffix=".jsonl"))

        try:
            summary = run_transform(input_path, output_path, validate=True)

            self.assertEqual(summary["records_read"], 2)
            self.assertEqual(summary["records_written"], 1)
            self.assertEqual(summary["records_deduped"], 1)
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_drops_invalid_records(self):
        """Invalid records are dropped, valid ones pass through."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fin:
            input_path = Path(fin.name)
            fin.write(self._make_landing_line() + "\n")  # valid
            fin.write(self._make_landing_line(icao24=None, idem_key="null:1717416000") + "\n")  # invalid

        output_path = Path(tempfile.mktemp(suffix=".jsonl"))

        try:
            summary = run_transform(input_path, output_path, validate=True)

            self.assertEqual(summary["records_read"], 2)
            self.assertEqual(summary["records_written"], 1)
            self.assertEqual(summary["records_dropped"], 1)
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
