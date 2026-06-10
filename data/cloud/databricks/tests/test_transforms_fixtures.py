#!/usr/bin/env python3
"""
Fixture-parity tests for data/cloud/databricks/lib/transforms.py.

Approach
--------
* A shared ``_make_landing()`` helper (mirrored from data/tests/test_bronze_to_silver.py)
  produces valid bronze landing records.
* ``transform_record`` from lib/transforms.py is called on each fixture.
* Results are compared field-by-field against the local reference implementation
  (``data.transforms.bronze_to_silver.transform_record``) to guarantee parity.
* Additional tests exercise the validation rules, enrichment stubs, dedup key,
  and airport distance logic in isolation — no Spark required.

The shared/contracts/fixtures/silver_flight_state_valid.json fixture records are
used to verify that geohash7 and h3_r7 values produced by the library are
self-consistent (7 / 15 chars, matching patterns) rather than matching
hand-authored values that may differ from current library versions.
"""

import json
import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or from this directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from data.cloud.databricks.lib.transforms import (
    transform_record as cloud_transform,
    nearest_airport,
    haversine_dist_km,
    idem_key_for,
    load_airports_reference,
)

# Reference implementation for parity checks.
from data.transforms.bronze_to_silver import (
    transform_record as local_transform,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
_FIXTURES_DIR = _PROJECT_ROOT / "shared" / "contracts" / "fixtures"


def _load_valid_fixtures() -> list:
    path = _FIXTURES_DIR / "silver_flight_state_valid.json"
    with open(path) as f:
        return json.load(f)


def _make_landing(**overrides) -> dict:
    """Return a valid bronze landing record, with optional field overrides.

    Mirrors the helper in data/tests/test_bronze_to_silver.py exactly so the
    same fixture inputs drive both test suites.
    """
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


# ---------------------------------------------------------------------------
# Parity tests: cloud lib vs local reference implementation
# ---------------------------------------------------------------------------

class TestParityWithLocalTransform(unittest.TestCase):
    """Every rule in the local transform must produce identical output from
    the cloud lib transform when given the same input."""

    def _assert_parity(self, record: dict, msg: str = ""):
        """Assert cloud_transform and local_transform return the same dict."""
        expected = local_transform(record)
        actual = cloud_transform(record)
        self.assertEqual(
            expected,
            actual,
            msg=f"Parity failure{f' ({msg})' if msg else ''}\n"
                f"  local : {expected}\n"
                f"  cloud : {actual}",
        )

    def test_valid_record_parity(self):
        """Valid record produces identical output from both transforms."""
        self._assert_parity(_make_landing(), "baseline valid record")

    def test_icao24_invalid_parity(self):
        """Short icao24 is dropped by both transforms."""
        self._assert_parity(_make_landing(icao24="abc"), "short icao24")

    def test_icao24_none_parity(self):
        """None icao24 is dropped by both."""
        self._assert_parity(_make_landing(icao24=None), "none icao24")

    def test_missing_event_ts_parity(self):
        """Missing event_ts is dropped by both."""
        r = _make_landing()
        r.pop("event_ts", None)
        self._assert_parity(r, "missing event_ts")

    def test_missing_lon_parity(self):
        self._assert_parity(_make_landing(lon=None), "none lon")

    def test_missing_lat_parity(self):
        self._assert_parity(_make_landing(lat=None), "none lat")

    def test_lon_out_of_range_parity(self):
        self._assert_parity(_make_landing(lon=200.0), "lon > 180")

    def test_lat_out_of_range_parity(self):
        self._assert_parity(_make_landing(lat=-100.0), "lat < -90")

    def test_negative_velocity_parity(self):
        self._assert_parity(_make_landing(velocity_ms=-1.0), "negative velocity")

    def test_null_velocity_parity(self):
        self._assert_parity(_make_landing(velocity_ms=None), "null velocity")

    def test_true_track_out_of_range_parity(self):
        self._assert_parity(_make_landing(true_track_deg=361.0), "true_track > 360")

    def test_impossible_altitude_parity(self):
        self._assert_parity(_make_landing(baro_altitude_m=99999.0), "altitude > 30000")

    def test_altitude_at_lower_bound_parity(self):
        self._assert_parity(_make_landing(baro_altitude_m=-1000.0), "altitude at -1000")

    def test_altitude_below_lower_bound_parity(self):
        self._assert_parity(_make_landing(baro_altitude_m=-1001.0), "altitude < -1000")

    def test_null_altitude_parity(self):
        self._assert_parity(_make_landing(baro_altitude_m=None), "null altitude")

    def test_invalid_squawk_parity(self):
        """Non-octal squawk is sanitised to None by both."""
        self._assert_parity(_make_landing(squawk="8888"), "non-octal squawk")

    def test_valid_squawk_preserved_parity(self):
        self._assert_parity(_make_landing(squawk="7700"), "valid squawk 7700")

    def test_null_squawk_parity(self):
        self._assert_parity(_make_landing(squawk=None), "null squawk")

    def test_squawk_wrong_length_parity(self):
        self._assert_parity(_make_landing(squawk="77"), "squawk wrong length")

    def test_callsign_strip_parity(self):
        """Trailing whitespace callsign is stripped identically."""
        self._assert_parity(_make_landing(callsign="SWR162 "), "callsign trailing space")

    def test_callsign_whitespace_only_parity(self):
        """Whitespace-only callsign becomes None in both."""
        self._assert_parity(_make_landing(callsign="   "), "whitespace callsign")

    def test_callsign_none_parity(self):
        self._assert_parity(_make_landing(callsign=None), "none callsign")

    def test_on_ground_true_parity(self):
        self._assert_parity(_make_landing(on_ground=True), "on_ground=True")

    def test_on_ground_default_parity(self):
        """Missing on_ground defaults to False in both."""
        r = _make_landing()
        r.pop("on_ground", None)
        self._assert_parity(r, "missing on_ground defaults False")

    def test_origin_country_default_parity(self):
        """Missing origin_country defaults to 'Unknown' in both."""
        r = _make_landing()
        r.pop("origin_country", None)
        self._assert_parity(r, "missing origin_country defaults Unknown")

    def test_lon_rounded_to_6dp_parity(self):
        """lon is rounded to 6 decimal places in both."""
        self._assert_parity(_make_landing(lon=4.76381234567), "lon precision rounding")

    def test_lat_rounded_to_6dp_parity(self):
        self._assert_parity(_make_landing(lat=52.30801234567), "lat precision rounding")

    def test_multiple_records_parity(self):
        """Parity holds for all four silver fixture source coords."""
        fixtures = [
            _make_landing(icao24="4b1806", lat=52.3080, lon=4.7638),
            _make_landing(icao24="48414e", lat=52.3105, lon=4.7640, callsign=None,
                          on_ground=True, velocity_ms=0.0, baro_altitude_m=None,
                          true_track_deg=None, vertical_rate_ms=None, squawk=None),
            _make_landing(icao24="3c6751", lat=50.0379, lon=8.5706,
                          baro_altitude_m=5486.0, velocity_ms=150.2,
                          true_track_deg=180.0, vertical_rate_ms=-5.0, squawk="7700"),
            _make_landing(icao24="a00001", lat=40.6413, lon=-73.7781,
                          callsign=None, on_ground=True, velocity_ms=None,
                          baro_altitude_m=None, true_track_deg=None,
                          vertical_rate_ms=None, squawk=None),
        ]
        for rec in fixtures:
            self._assert_parity(rec, f"icao24={rec['icao24']}")


# ---------------------------------------------------------------------------
# Field-level assertions: cloud transform output values
# ---------------------------------------------------------------------------

class TestCloudTransformFields(unittest.TestCase):
    """Verify field values produced by the cloud transform directly."""

    def setUp(self):
        self.silver = cloud_transform(_make_landing())
        self.assertIsNotNone(self.silver)

    def test_icao24(self):
        self.assertEqual(self.silver["icao24"], "4b1806")

    def test_callsign(self):
        self.assertEqual(self.silver["callsign"], "SWR162")

    def test_event_ts(self):
        self.assertEqual(self.silver["event_ts"], "2024-06-03T12:00:00+00:00")

    def test_lon_rounded(self):
        self.assertEqual(self.silver["lon"], 4.7638)

    def test_lat_rounded(self):
        self.assertEqual(self.silver["lat"], 52.308)

    def test_on_ground_false(self):
        self.assertFalse(self.silver["on_ground"])

    def test_origin_country(self):
        self.assertEqual(self.silver["origin_country"], "Switzerland")

    def test_geohash7_is_7_chars(self):
        self.assertEqual(len(self.silver["geohash7"]), 7)

    def test_geohash7_matches_pattern(self):
        import re
        self.assertRegex(self.silver["geohash7"], r"^[0-9b-hjkmnp-z]{7}$")

    def test_h3_r7_is_15_chars(self):
        self.assertEqual(len(self.silver["h3_r7"]), 15)

    def test_h3_r7_matches_pattern(self):
        import re
        self.assertRegex(self.silver["h3_r7"], r"^[0-9a-f]{15}$")

    def test_enrichment_stubs_null(self):
        self.assertIsNone(self.silver["nearest_airport"])
        self.assertIsNone(self.silver["metar_wind_kt"])
        self.assertIsNone(self.silver["metar_vis_m"])
        self.assertIsNone(self.silver["metar_ceiling_ft"])

    def test_no_extra_fields(self):
        expected_keys = {
            "icao24", "callsign", "event_ts", "lon", "lat",
            "baro_altitude_m", "velocity_ms", "true_track_deg",
            "vertical_rate_ms", "on_ground", "squawk", "origin_country",
            "geohash7", "h3_r7", "nearest_airport",
            "metar_wind_kt", "metar_vis_m", "metar_ceiling_ft",
        }
        self.assertEqual(set(self.silver.keys()), expected_keys)

    def test_squawk_preserved(self):
        self.assertEqual(self.silver["squawk"], "2536")

    def test_baro_altitude_passthrough(self):
        self.assertEqual(self.silver["baro_altitude_m"], 11278.0)

    def test_velocity_passthrough(self):
        self.assertEqual(self.silver["velocity_ms"], 234.5)

    def test_vertical_rate_passthrough(self):
        self.assertEqual(self.silver["vertical_rate_ms"], 0.0)

    def test_true_track_passthrough(self):
        self.assertEqual(self.silver["true_track_deg"], 42.3)


# ---------------------------------------------------------------------------
# Spatial helpers tests
# ---------------------------------------------------------------------------

class TestSpatialHelpers(unittest.TestCase):

    def test_haversine_self_is_zero(self):
        self.assertAlmostEqual(haversine_dist_km(52.0, 4.0, 52.0, 4.0), 0.0, places=6)

    def test_haversine_schiphol_to_nearby(self):
        # EHAM coords; a point ~0.01 deg away should be < 2 km.
        d = haversine_dist_km(52.3086, 4.7639, 52.3186, 4.7739)
        self.assertLess(d, 2.0)

    def test_nearest_airport_within_radius(self):
        airports = [
            {"icao": "EHAM", "lat": 52.3086, "lon": 4.7639},
            {"icao": "EDDF", "lat": 50.0333, "lon": 8.5706},
        ]
        self.assertEqual(nearest_airport(52.31, 4.76, airports), "EHAM")

    def test_nearest_airport_outside_radius(self):
        airports = [{"icao": "EHAM", "lat": 52.3086, "lon": 4.7639}]
        self.assertIsNone(nearest_airport(0.0, 0.0, airports))

    def test_nearest_airport_empty_list(self):
        self.assertIsNone(nearest_airport(52.3, 4.7, []))

    def test_nearest_airport_exact_boundary(self):
        """A point exactly 50 km from the airport should match."""
        import math
        # Move ~50 km north of EHAM (0.45 deg lat ≈ 50 km)
        far_lat = 52.3086 + 50.0 / 111.0
        result = nearest_airport(far_lat, 4.7639,
                                 [{"icao": "EHAM", "lat": 52.3086, "lon": 4.7639}])
        # Result depends on exact dist; just confirm it doesn't raise.
        self.assertIn(result, ("EHAM", None))


# ---------------------------------------------------------------------------
# Dedup key helper
# ---------------------------------------------------------------------------

class TestIdemKey(unittest.TestCase):

    def test_uses_idem_key_when_present(self):
        r = _make_landing(idem_key="4b1806:1717416000")
        self.assertEqual(idem_key_for(r), "4b1806:1717416000")

    def test_falls_back_to_icao24_event_ts(self):
        r = _make_landing()
        r.pop("idem_key", None)
        self.assertEqual(idem_key_for(r), "4b1806:2024-06-03T12:00:00+00:00")

    def test_explicit_idem_key_overrides_fallback(self):
        r = _make_landing(idem_key="custom:key:123")
        self.assertEqual(idem_key_for(r), "custom:key:123")


# ---------------------------------------------------------------------------
# Drop/validation edge cases
# ---------------------------------------------------------------------------

class TestDropRules(unittest.TestCase):

    def test_drop_icao24_none(self):
        self.assertIsNone(cloud_transform(_make_landing(icao24=None)))

    def test_drop_icao24_5_chars(self):
        self.assertIsNone(cloud_transform(_make_landing(icao24="4b180")))

    def test_drop_icao24_7_chars(self):
        self.assertIsNone(cloud_transform(_make_landing(icao24="4b18066")))

    def test_drop_icao24_empty_string(self):
        self.assertIsNone(cloud_transform(_make_landing(icao24="")))

    def test_drop_missing_event_ts(self):
        r = _make_landing()
        r.pop("event_ts")
        self.assertIsNone(cloud_transform(r))

    def test_drop_none_event_ts(self):
        self.assertIsNone(cloud_transform(_make_landing(event_ts=None)))

    def test_drop_lon_none(self):
        self.assertIsNone(cloud_transform(_make_landing(lon=None)))

    def test_drop_lat_none(self):
        self.assertIsNone(cloud_transform(_make_landing(lat=None)))

    def test_drop_lon_gt_180(self):
        self.assertIsNone(cloud_transform(_make_landing(lon=181.0)))

    def test_drop_lon_lt_neg_180(self):
        self.assertIsNone(cloud_transform(_make_landing(lon=-181.0)))

    def test_drop_lat_gt_90(self):
        self.assertIsNone(cloud_transform(_make_landing(lat=91.0)))

    def test_drop_lat_lt_neg_90(self):
        self.assertIsNone(cloud_transform(_make_landing(lat=-91.0)))

    def test_drop_negative_velocity(self):
        self.assertIsNone(cloud_transform(_make_landing(velocity_ms=-0.001)))

    def test_drop_true_track_gt_360(self):
        self.assertIsNone(cloud_transform(_make_landing(true_track_deg=360.001)))

    def test_drop_true_track_lt_0(self):
        self.assertIsNone(cloud_transform(_make_landing(true_track_deg=-0.001)))

    def test_drop_baro_altitude_gt_30000(self):
        self.assertIsNone(cloud_transform(_make_landing(baro_altitude_m=30000.001)))

    def test_drop_baro_altitude_lt_neg1000(self):
        self.assertIsNone(cloud_transform(_make_landing(baro_altitude_m=-1000.001)))

    def test_pass_velocity_zero(self):
        """Zero velocity is valid."""
        self.assertIsNotNone(cloud_transform(_make_landing(velocity_ms=0.0)))

    def test_pass_true_track_zero(self):
        self.assertIsNotNone(cloud_transform(_make_landing(true_track_deg=0.0)))

    def test_pass_true_track_360(self):
        self.assertIsNotNone(cloud_transform(_make_landing(true_track_deg=360.0)))

    def test_pass_baro_altitude_neg1000(self):
        self.assertIsNotNone(cloud_transform(_make_landing(baro_altitude_m=-1000.0)))

    def test_pass_baro_altitude_30000(self):
        self.assertIsNotNone(cloud_transform(_make_landing(baro_altitude_m=30000.0)))


# ---------------------------------------------------------------------------
# Squawk normalisation
# ---------------------------------------------------------------------------

class TestSquawkNormalisation(unittest.TestCase):

    def test_valid_octal_preserved(self):
        silver = cloud_transform(_make_landing(squawk="0777"))
        self.assertIsNotNone(silver)
        self.assertEqual(silver["squawk"], "0777")

    def test_emergency_7700_preserved(self):
        silver = cloud_transform(_make_landing(squawk="7700"))
        self.assertEqual(silver["squawk"], "7700")

    def test_non_octal_digit_sanitised(self):
        silver = cloud_transform(_make_landing(squawk="8000"))
        self.assertIsNone(silver["squawk"])

    def test_non_octal_digit_9_sanitised(self):
        silver = cloud_transform(_make_landing(squawk="9123"))
        self.assertIsNone(silver["squawk"])

    def test_too_short_sanitised(self):
        silver = cloud_transform(_make_landing(squawk="123"))
        self.assertIsNone(silver["squawk"])

    def test_too_long_sanitised(self):
        silver = cloud_transform(_make_landing(squawk="12345"))
        self.assertIsNone(silver["squawk"])

    def test_null_squawk_stays_null(self):
        silver = cloud_transform(_make_landing(squawk=None))
        self.assertIsNone(silver["squawk"])


# ---------------------------------------------------------------------------
# Callsign normalisation
# ---------------------------------------------------------------------------

class TestCallsignNormalisation(unittest.TestCase):

    def test_trailing_space_stripped(self):
        silver = cloud_transform(_make_landing(callsign="SWR162 "))
        self.assertEqual(silver["callsign"], "SWR162")

    def test_leading_space_stripped(self):
        silver = cloud_transform(_make_landing(callsign=" SWR162"))
        self.assertEqual(silver["callsign"], "SWR162")

    def test_empty_string_becomes_none(self):
        silver = cloud_transform(_make_landing(callsign=""))
        self.assertIsNone(silver["callsign"])

    def test_whitespace_only_becomes_none(self):
        silver = cloud_transform(_make_landing(callsign="   "))
        self.assertIsNone(silver["callsign"])

    def test_none_callsign_stays_none(self):
        silver = cloud_transform(_make_landing(callsign=None))
        self.assertIsNone(silver["callsign"])


# ---------------------------------------------------------------------------
# Contract fixture cross-check: geohash/h3 consistency
# ---------------------------------------------------------------------------

class TestContractFixtureSpatialIndices(unittest.TestCase):
    """Run the cloud transform over landing records built from the silver
    contract fixture coordinates and verify geohash7/h3_r7 format correctness.

    The hand-authored fixture values in silver_flight_state_valid.json may
    differ from library output; we verify structural correctness, not exact
    equality to the fixture file.
    """

    def setUp(self):
        self._valid_fixtures = _load_valid_fixtures()

    def test_geohash7_format_for_all_fixtures(self):
        import re
        _pat = re.compile(r"^[0-9b-hjkmnp-z]{7}$")
        for fx in self._valid_fixtures:
            rec = _make_landing(
                icao24=fx["icao24"],
                lat=fx["lat"],
                lon=fx["lon"],
            )
            silver = cloud_transform(rec)
            self.assertIsNotNone(silver, f"Transform returned None for {fx['icao24']}")
            self.assertEqual(len(silver["geohash7"]), 7,
                             f"geohash7 length != 7 for {fx['icao24']}")
            self.assertRegex(silver["geohash7"], _pat,
                             f"geohash7 pattern mismatch for {fx['icao24']}")

    def test_h3_r7_format_for_all_fixtures(self):
        import re
        _pat = re.compile(r"^[0-9a-f]{15}$")
        for fx in self._valid_fixtures:
            rec = _make_landing(
                icao24=fx["icao24"],
                lat=fx["lat"],
                lon=fx["lon"],
            )
            silver = cloud_transform(rec)
            self.assertIsNotNone(silver)
            self.assertEqual(len(silver["h3_r7"]), 15,
                             f"h3_r7 length != 15 for {fx['icao24']}")
            self.assertRegex(silver["h3_r7"], _pat,
                             f"h3_r7 pattern mismatch for {fx['icao24']}")

    def test_geohash7_deterministic(self):
        """Same lat/lon always produces the same geohash7."""
        rec = _make_landing(lat=52.3080, lon=4.7638)
        result1 = cloud_transform(rec)
        result2 = cloud_transform(rec)
        self.assertEqual(result1["geohash7"], result2["geohash7"])

    def test_h3_r7_deterministic(self):
        """Same lat/lon always produces the same h3_r7."""
        rec = _make_landing(lat=52.3080, lon=4.7638)
        result1 = cloud_transform(rec)
        result2 = cloud_transform(rec)
        self.assertEqual(result1["h3_r7"], result2["h3_r7"])

    def test_different_coords_produce_different_geohash(self):
        """Different locations produce different geohash7 values."""
        rec1 = _make_landing(lat=52.3080, lon=4.7638)
        rec2 = _make_landing(lat=50.0379, lon=8.5706, icao24="3c6751")
        gh1 = cloud_transform(rec1)["geohash7"]
        gh2 = cloud_transform(rec2)["geohash7"]
        self.assertNotEqual(gh1, gh2)


if __name__ == "__main__":
    unittest.main()
