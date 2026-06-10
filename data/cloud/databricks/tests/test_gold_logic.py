#!/usr/bin/env python3
"""
Parity tests: data/cloud/databricks/lib/gold_logic.py vs
              data/transforms/silver_to_gold.py

Approach
--------
* A shared fixture silver input is defined once (identical to the fixture
  used in data/tests/test_silver_to_gold.py so both suites exercise the same
  records).
* Each of the four aggregation functions from gold_logic (cloud) is called on
  the fixture input and the result is compared field-by-field against the
  authoritative local implementation (silver_to_gold).
* Additional fixtures exercise edge cases: UNKNOWN exclusion (ADR-0006),
  emergency gap split (M17), callsign normalisation (M18), window boundary.
* All comparisons normalise to sorted lists of dicts so output ordering
  differences never cause spurious failures.

No Spark is required — this test runs fully offline.
"""

import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: allow running from project root or from this directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports: cloud implementation under test and local reference.
# ---------------------------------------------------------------------------
from data.cloud.databricks.lib.gold_logic import (
    WINDOW_MINUTES as CLOUD_WINDOW_MINUTES,
    EMERGENCY_GAP_S as CLOUD_EMERGENCY_GAP_S,
    aggregate_emergency_events as cloud_emergency,
    aggregate_sector_load as cloud_sector,
    aggregate_airport_congestion as cloud_congestion,
    aggregate_routing_stats as cloud_routing,
)
from data.transforms.silver_to_gold import (
    WINDOW_MINUTES as LOCAL_WINDOW_MINUTES,
    EMERGENCY_GAP_S as LOCAL_EMERGENCY_GAP_S,
    aggregate_emergency_events as local_emergency,
    aggregate_sector_load as local_sector,
    aggregate_airport_congestion as local_congestion,
    aggregate_routing_stats as local_routing,
)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _canonical(records: list[dict]) -> list[dict]:
    """Sort a list of dicts by their JSON-serialisable key for stable comparison.

    Uses the same sort key as write_gold_table in silver_to_gold.py so the
    comparison is order-independent.
    """
    import json
    return sorted(records, key=lambda r: json.dumps(r, sort_keys=True))


def _make_fixture() -> list[dict]:
    """Return the canonical fixture silver records.

    Mirrors the setUp fixture in data/tests/test_silver_to_gold.TestSilverToGold
    exactly so the same inputs drive both test suites.
    """
    return [
        # Normal flight — EHAM, airborne
        {
            "icao24": "111111",
            "callsign": "KLM123",
            "event_ts": "2024-06-03T12:02:00+00:00",
            "lon": 4.7,
            "lat": 52.3,
            "squawk": "2000",
            "on_ground": False,
            "baro_altitude_m": 10000.0,
            "h3_r7": "8719694b5ffffff",
            "nearest_airport": "EHAM",
        },
        # Emergency flight, first observation — EDDF, airborne
        {
            "icao24": "222222",
            "callsign": "EMG456",
            "event_ts": "2024-06-03T12:04:00+00:00",
            "lon": 8.5,
            "lat": 50.0,
            "squawk": "7700",
            "on_ground": False,
            "baro_altitude_m": 5000.0,
            "h3_r7": "871fa1b13ffffff",
            "nearest_airport": "EDDF",
        },
        # Emergency flight, second observation (same 5-min window, within gap)
        {
            "icao24": "222222",
            "callsign": "EMG456",
            "event_ts": "2024-06-03T12:04:30+00:00",
            "lon": 8.51,
            "lat": 50.01,
            "squawk": "7700",
            "on_ground": False,
            "baro_altitude_m": 4800.0,
            "h3_r7": "871fa1b13ffffff",
            "nearest_airport": "EDDF",
        },
        # Ground flight — EHAM
        {
            "icao24": "333333",
            "callsign": "GND789",
            "event_ts": "2024-06-03T12:01:00+00:00",
            "lon": 4.76,
            "lat": 52.3,
            "squawk": "1000",
            "on_ground": True,
            "baro_altitude_m": 0.0,
            "h3_r7": "8719694b5ffffff",
            "nearest_airport": "EHAM",
        },
    ]


def _make_gap_fixture() -> list[dict]:
    """Two 7700 observations for the same aircraft separated by >1800 s."""
    return [
        {
            "icao24": "777777", "callsign": "EMG1", "squawk": "7700",
            "event_ts": "2024-06-03T12:00:00+00:00", "lat": 50.0, "lon": 8.0,
            "on_ground": False, "baro_altitude_m": 5000.0,
            "h3_r7": "871fa1b13ffffff", "nearest_airport": "EDDF",
        },
        {
            "icao24": "777777", "callsign": "EMG1", "squawk": "7700",
            "event_ts": "2024-06-03T14:00:00+00:00", "lat": 50.1, "lon": 8.1,
            "on_ground": False, "baro_altitude_m": 5200.0,
            "h3_r7": "871fa1b13ffffff", "nearest_airport": "EDDF",
        },
    ]


def _make_callsign_normalise_fixture() -> list[dict]:
    """Two pings for the same aircraft with trailing-whitespace callsign variant."""
    return [
        {
            "icao24": "555555", "callsign": "ABC123 ",
            "event_ts": "2024-06-03T12:00:00+00:00", "lat": 50.0, "lon": 8.0,
            "on_ground": False, "baro_altitude_m": 9000.0, "velocity_ms": 200.0,
        },
        {
            "icao24": "555555", "callsign": "ABC123",
            "event_ts": "2024-06-03T12:01:00+00:00", "lat": 50.1, "lon": 8.1,
            "on_ground": False, "baro_altitude_m": 9100.0, "velocity_ms": 210.0,
        },
    ]


def _make_window_boundary_fixture() -> list[dict]:
    """Same h3_r7 cell, same aircraft, across a 5-minute boundary."""
    return [
        {"icao24": "111111", "h3_r7": "8719694b5ffffff",
         "event_ts": "2024-06-03T12:04:59+00:00"},
        {"icao24": "111111", "h3_r7": "8719694b5ffffff",
         "event_ts": "2024-06-03T12:05:00+00:00"},
    ]


def _make_no_airport_fixture() -> list[dict]:
    """Aircraft with nearest_airport=None must be excluded from congestion."""
    return [
        {
            "icao24": "999999", "callsign": "NONE1",
            "event_ts": "2024-06-03T12:00:00+00:00", "lon": 10.0, "lat": 48.0,
            "squawk": "1200", "on_ground": False, "baro_altitude_m": 9000.0,
            "h3_r7": "871fa1b13ffffff", "nearest_airport": None,
        },
    ]


# ---------------------------------------------------------------------------
# Parity tests: cloud == local for all four tables
# ---------------------------------------------------------------------------

class TestEmergencyEventsParity(unittest.TestCase):
    """gold_logic.aggregate_emergency_events matches silver_to_gold on all fixtures."""

    def _assert_parity(self, records: list[dict], msg: str = "") -> None:
        cloud_result = _canonical(cloud_emergency(records))
        local_result = _canonical(local_emergency(records))
        self.assertEqual(
            cloud_result, local_result,
            msg=f"Emergency parity failure{f' ({msg})' if msg else ''}\n"
                f"  cloud : {cloud_result}\n"
                f"  local : {local_result}",
        )

    def test_parity_baseline_fixture(self):
        self._assert_parity(_make_fixture(), "baseline 4-record fixture")

    def test_parity_gap_split(self):
        """Two events separated by >1800 s each produce duration_s=0 (M17)."""
        self._assert_parity(_make_gap_fixture(), "gap-split fixture")

    def test_parity_empty(self):
        self._assert_parity([], "empty input")

    def test_parity_non_emergency_squawk_only(self):
        """No emergency squawk → empty result from both implementations."""
        recs = [dict(_make_fixture()[0])]  # squawk "2000"
        self._assert_parity(recs, "non-emergency squawk only")

    def test_parity_all_three_emergency_codes(self):
        """7500, 7600, 7700 all produce events in both implementations."""
        recs = [
            {"icao24": "aaaaaa", "callsign": "A", "squawk": "7500",
             "event_ts": "2024-06-03T10:00:00+00:00", "lat": 50.0, "lon": 8.0,
             "on_ground": False, "baro_altitude_m": 5000.0,
             "h3_r7": "871fa1b13ffffff", "nearest_airport": "EDDF"},
            {"icao24": "bbbbbb", "callsign": "B", "squawk": "7600",
             "event_ts": "2024-06-03T10:01:00+00:00", "lat": 50.1, "lon": 8.1,
             "on_ground": False, "baro_altitude_m": 5100.0,
             "h3_r7": "871fa1b13ffffff", "nearest_airport": "EDDF"},
            {"icao24": "cccccc", "callsign": "C", "squawk": "7700",
             "event_ts": "2024-06-03T10:02:00+00:00", "lat": 50.2, "lon": 8.2,
             "on_ground": False, "baro_altitude_m": 5200.0,
             "h3_r7": "871fa1b13ffffff", "nearest_airport": "EDDF"},
        ]
        self._assert_parity(recs, "all three emergency squawk codes")

    def test_gap_split_produces_two_events(self):
        """Structural check: gap fixture yields exactly 2 events, both duration=0."""
        events = cloud_emergency(_make_gap_fixture())
        self.assertEqual(len(events), 2)
        for e in events:
            self.assertEqual(e["duration_s"], 0)

    def test_within_gap_merges_to_one_event(self):
        """Observations within EMERGENCY_GAP_S merge to one event."""
        result = cloud_emergency(_make_fixture())
        self.assertEqual(len(result), 1)
        e = result[0]
        self.assertEqual(e["icao24"], "222222")
        self.assertEqual(e["duration_s"], 30)


class TestSectorLoadParity(unittest.TestCase):
    """gold_logic.aggregate_sector_load matches silver_to_gold on all fixtures."""

    def _assert_parity(self, records: list[dict], msg: str = "") -> None:
        cloud_result = _canonical(cloud_sector(records))
        local_result = _canonical(local_sector(records))
        self.assertEqual(
            cloud_result, local_result,
            msg=f"Sector parity failure{f' ({msg})' if msg else ''}\n"
                f"  cloud : {cloud_result}\n"
                f"  local : {local_result}",
        )

    def test_parity_baseline_fixture(self):
        self._assert_parity(_make_fixture(), "baseline 4-record fixture")

    def test_parity_empty(self):
        self._assert_parity([], "empty input")

    def test_parity_window_boundary(self):
        """Same cell at 12:04:59 and 12:05:00 → two sector rows (M19)."""
        self._assert_parity(_make_window_boundary_fixture(), "window boundary")

    def test_parity_no_h3(self):
        """Records with no h3_r7 are silently skipped."""
        recs = [{"icao24": "aaaaaa", "event_ts": "2024-06-03T12:00:00+00:00"}]
        self._assert_parity(recs, "missing h3_r7")

    def test_window_boundary_produces_two_rows(self):
        """Structural check: one aircraft at window boundary produces 2 rows."""
        rows = cloud_sector(_make_window_boundary_fixture())
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["aircraft_count"], 1)

    def test_distinct_aircraft_count(self):
        """Two different aircraft in same cell/window → count=2."""
        recs = [
            {"icao24": "111111", "h3_r7": "8719694b5ffffff",
             "event_ts": "2024-06-03T12:02:00+00:00"},
            {"icao24": "222222", "h3_r7": "8719694b5ffffff",
             "event_ts": "2024-06-03T12:03:00+00:00"},
        ]
        rows = cloud_sector(recs)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["aircraft_count"], 2)


class TestAirportCongestionParity(unittest.TestCase):
    """gold_logic.aggregate_airport_congestion matches silver_to_gold on all fixtures."""

    def _assert_parity(self, records: list[dict], msg: str = "") -> None:
        cloud_result = _canonical(cloud_congestion(records))
        local_result = _canonical(local_congestion(records))
        self.assertEqual(
            cloud_result, local_result,
            msg=f"Congestion parity failure{f' ({msg})' if msg else ''}\n"
                f"  cloud : {cloud_result}\n"
                f"  local : {local_result}",
        )

    def test_parity_baseline_fixture(self):
        self._assert_parity(_make_fixture(), "baseline 4-record fixture")

    def test_parity_empty(self):
        self._assert_parity([], "empty input")

    def test_parity_no_airport_excluded(self):
        """ADR-0006: aircraft with nearest_airport=None excluded, not UNKNOWN."""
        self._assert_parity(_make_no_airport_fixture(), "no airport")

    def test_parity_mixed_airport_and_none(self):
        """Mixed fixture: only real airport rows in result, no UNKNOWN."""
        mixed = _make_no_airport_fixture() + [_make_fixture()[0]]
        self._assert_parity(mixed, "mixed airport and none")
        cloud_result = cloud_congestion(mixed)
        codes = {r["airport_icao"] for r in cloud_result}
        self.assertNotIn("UNKNOWN", codes)
        self.assertIn("EHAM", codes)

    def test_avg_altitude_from_components(self):
        """avg_altitude_m = alt_sum / alt_count, not averaged-of-averages.

        EHAM has aircraft 111111 (10000.0 m) and 333333 (0.0 m):
        alt_sum = 10000.0, alt_count = 2 → avg = 5000.0.
        """
        rows = cloud_congestion(_make_fixture())
        eham = next(r for r in rows if r["airport_icao"] == "EHAM")
        self.assertEqual(eham["avg_altitude_m"], 5000.0)

    def test_deduplicated_aircraft_count(self):
        """Each distinct icao24 counted once per airport/window regardless of ping count."""
        recs = [
            # Two pings from same aircraft — should count as 1
            {
                "icao24": "aaaaaa", "event_ts": "2024-06-03T12:00:00+00:00",
                "on_ground": False, "baro_altitude_m": 8000.0,
                "nearest_airport": "EHAM",
            },
            {
                "icao24": "aaaaaa", "event_ts": "2024-06-03T12:01:00+00:00",
                "on_ground": False, "baro_altitude_m": 8100.0,
                "nearest_airport": "EHAM",
            },
        ]
        rows = cloud_congestion(recs)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["aircraft_count"], 1)
        # avg_altitude uses only first observation (icao deduplicated on first ping)
        self.assertEqual(rows[0]["avg_altitude_m"], 8000.0)


class TestRoutingStatsParity(unittest.TestCase):
    """gold_logic.aggregate_routing_stats matches silver_to_gold on all fixtures."""

    def _assert_parity(self, records: list[dict], msg: str = "") -> None:
        cloud_result = _canonical(cloud_routing(records))
        local_result = _canonical(local_routing(records))
        self.assertEqual(
            cloud_result, local_result,
            msg=f"Routing parity failure{f' ({msg})' if msg else ''}\n"
                f"  cloud : {cloud_result}\n"
                f"  local : {local_result}",
        )

    def test_parity_baseline_fixture(self):
        self._assert_parity(_make_fixture(), "baseline 4-record fixture")

    def test_parity_empty(self):
        self._assert_parity([], "empty input")

    def test_parity_callsign_normalise(self):
        """Trailing-whitespace callsign variants merge to one route (M18)."""
        self._assert_parity(_make_callsign_normalise_fixture(), "callsign normalise")

    def test_parity_no_icao24(self):
        """Records with missing icao24 are skipped by both."""
        recs = [{"callsign": "X", "event_ts": "2024-06-03T12:00:00+00:00"}]
        self._assert_parity(recs, "missing icao24")

    def test_callsign_normalise_produces_one_route(self):
        """Structural check: two callsign variants → one route, ping_count=2."""
        routes = cloud_routing(_make_callsign_normalise_fixture())
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["callsign"], "ABC123")
        self.assertEqual(routes[0]["ping_count"], 2)

    def test_origin_is_first_destination_is_last(self):
        """origin=first lat/lon, destination=last lat/lon."""
        routes = cloud_routing(_make_callsign_normalise_fixture())
        r = routes[0]
        self.assertEqual(r["origin_lat"], 50.0)
        self.assertEqual(r["destination_lat"], 50.1)

    def test_max_altitude(self):
        """max_altitude_m tracks the highest observed baro_altitude_m."""
        routes = cloud_routing(_make_callsign_normalise_fixture())
        self.assertEqual(routes[0]["max_altitude_m"], 9100.0)

    def test_avg_velocity(self):
        """avg_velocity_mps = sum / count, rounded to 2 dp."""
        routes = cloud_routing(_make_callsign_normalise_fixture())
        # (200.0 + 210.0) / 2 = 205.0
        self.assertEqual(routes[0]["avg_velocity_mps"], 205.0)

    def test_emergency_route_two_pings(self):
        """Emergency aircraft with 2 pings in baseline fixture → ping_count=2."""
        routes = cloud_routing(_make_fixture())
        emg = next(r for r in routes if r["icao24"] == "222222")
        self.assertEqual(emg["ping_count"], 2)
        self.assertEqual(emg["origin_lat"], 50.0)
        self.assertEqual(emg["destination_lat"], 50.01)


# ---------------------------------------------------------------------------
# Constant parity
# ---------------------------------------------------------------------------

class TestConstantParity(unittest.TestCase):
    """WINDOW_MINUTES and EMERGENCY_GAP_S match between cloud and local."""

    def test_window_minutes_matches(self):
        self.assertEqual(CLOUD_WINDOW_MINUTES, LOCAL_WINDOW_MINUTES)

    def test_emergency_gap_s_matches(self):
        self.assertEqual(CLOUD_EMERGENCY_GAP_S, LOCAL_EMERGENCY_GAP_S)


if __name__ == "__main__":
    unittest.main()
