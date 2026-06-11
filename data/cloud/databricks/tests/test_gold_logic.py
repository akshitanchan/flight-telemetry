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
    incremental_merge_emergency,
    incremental_merge_routing,
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
# Overlapping-batch incremental merge tests
# ---------------------------------------------------------------------------

def _make_emg_silver_batch_a() -> list[dict]:
    """Emergency silver batch A: observations T+120 and T+150 for icao 'aabbcc'."""
    return [
        {
            "icao24": "aabbcc", "callsign": "EMG1", "squawk": "7700",
            "event_ts": "2024-06-03T12:02:00+00:00",
            "lat": 51.0, "lon": 9.0,
            "on_ground": False, "baro_altitude_m": 6000.0,
            "h3_r7": "871fa1b13ffffff", "nearest_airport": "EDDF",
            "origin_country": "Germany",
        },
        {
            "icao24": "aabbcc", "callsign": "EMG1", "squawk": "7700",
            "event_ts": "2024-06-03T12:02:30+00:00",
            "lat": 51.01, "lon": 9.01,
            "on_ground": False, "baro_altitude_m": 6100.0,
            "h3_r7": "871fa1b13ffffff", "nearest_airport": "EDDF",
            "origin_country": "Germany",
        },
    ]


def _make_emg_silver_batch_b_earlier() -> list[dict]:
    """Emergency silver batch B: an EARLIER observation of the same event at T+60."""
    return [
        {
            "icao24": "aabbcc", "callsign": "EMG1", "squawk": "7700",
            "event_ts": "2024-06-03T12:01:00+00:00",   # earlier than batch A
            "lat": 50.9, "lon": 8.9,
            "on_ground": False, "baro_altitude_m": 5800.0,
            "h3_r7": "871fa1b13ffffff", "nearest_airport": "EDDF",
            "origin_country": "Germany",
        },
    ]


def _make_routing_silver_batch_a() -> list[dict]:
    """Routing silver batch A: two pings for route 'ddeeff' / 'FLT1'."""
    return [
        {
            "icao24": "ddeeff", "callsign": "FLT1",
            "event_ts": "2024-06-03T13:05:00+00:00",
            "lat": 48.0, "lon": 11.0,
            "on_ground": False, "baro_altitude_m": 9000.0, "velocity_ms": 250.0,
        },
        {
            "icao24": "ddeeff", "callsign": "FLT1",
            "event_ts": "2024-06-03T13:10:00+00:00",
            "lat": 48.5, "lon": 11.5,
            "on_ground": False, "baro_altitude_m": 9200.0, "velocity_ms": 260.0,
        },
    ]


def _make_routing_silver_batch_b_earlier() -> list[dict]:
    """Routing silver batch B: an EARLIER ping for the same route arriving late."""
    return [
        {
            "icao24": "ddeeff", "callsign": "FLT1",
            "event_ts": "2024-06-03T13:00:00+00:00",   # earlier than batch A
            "lat": 47.5, "lon": 10.5,
            "on_ground": False, "baro_altitude_m": 8800.0, "velocity_ms": 240.0,
        },
    ]


class TestIncrementalMergeEmergencyOverlap(unittest.TestCase):
    """incremental_merge_emergency is correct for overlapping batches."""

    # --- helpers ---

    def _full_recompute(self, silver: list[dict]) -> list[dict]:
        return _canonical(cloud_emergency(silver))

    def _incremental(
        self,
        existing_gold: list[dict],
        existing_silver: list[dict],
        new_batch: list[dict],
    ) -> list[dict]:
        return _canonical(incremental_merge_emergency(existing_gold, existing_silver, new_batch))

    # --- tests ---

    def test_overlapping_batch_equals_full_recompute(self):
        """Batch B contains an earlier observation: incremental must equal full recompute.

        Scenario:
          Existing state: gold built from batch A (T+120, T+150).
          New batch B: observation at T+60 (earlier than batch A).
          Expected: gold has a single event whose first_seen_ts = T+60.
        """
        batch_a = _make_emg_silver_batch_a()
        batch_b = _make_emg_silver_batch_b_earlier()

        # Build existing gold from batch A.
        existing_gold = cloud_emergency(batch_a)
        # Apply batch B incrementally.
        result = self._incremental(existing_gold, batch_a, batch_b)

        # Full recompute over union.
        expected = self._full_recompute(batch_a + batch_b)

        self.assertEqual(
            result, expected,
            msg=f"Incremental(A then B) != full_recompute(A+B)\n"
                f"  incremental : {result}\n"
                f"  full        : {expected}",
        )

    def test_order_independent_a_then_b_equals_b_then_a(self):
        """Applying batch A then B yields same result as B then A."""
        batch_a = _make_emg_silver_batch_a()
        batch_b = _make_emg_silver_batch_b_earlier()

        # A then B
        gold_after_a = cloud_emergency(batch_a)
        result_ab = self._incremental(gold_after_a, batch_a, batch_b)

        # B then A
        gold_after_b = cloud_emergency(batch_b)
        result_ba = self._incremental(gold_after_b, batch_b, batch_a)

        self.assertEqual(
            result_ab, result_ba,
            msg=f"Order dependence detected for emergency:\n"
                f"  A then B : {result_ab}\n"
                f"  B then A : {result_ba}",
        )

    def test_order_independent_both_equal_full_recompute(self):
        """Both orderings equal full recompute over (A + B)."""
        batch_a = _make_emg_silver_batch_a()
        batch_b = _make_emg_silver_batch_b_earlier()
        expected = self._full_recompute(batch_a + batch_b)

        gold_after_a = cloud_emergency(batch_a)
        result_ab = self._incremental(gold_after_a, batch_a, batch_b)

        gold_after_b = cloud_emergency(batch_b)
        result_ba = self._incremental(gold_after_b, batch_b, batch_a)

        self.assertEqual(result_ab, expected)
        self.assertEqual(result_ba, expected)

    def test_first_seen_ts_is_earliest_across_both_batches(self):
        """The merged event's first_seen_ts must be the earliest observation."""
        batch_a = _make_emg_silver_batch_a()
        batch_b = _make_emg_silver_batch_b_earlier()

        gold_after_a = cloud_emergency(batch_a)
        result = self._incremental(gold_after_a, batch_a, batch_b)

        self.assertEqual(len(result), 1)
        event = result[0]
        # Earliest observation is in batch B at T+60.
        self.assertEqual(event["first_seen_ts"], "2024-06-03T12:01:00+00:00")
        # Latest observation is in batch A at T+150.
        self.assertEqual(event["last_seen_ts"], "2024-06-03T12:02:30+00:00")

    def test_idempotent_applying_same_batch_twice(self):
        """Applying the same new batch twice must produce no net change.

        In a real silver store, rows from batch_b are absorbed once (deduplicated
        by primary key).  On the second run of the same incremental job,
        existing_silver already contains batch_b rows, so passing new_batch=batch_b
        again with existing_silver=(batch_a + batch_b, deduplicated) must yield the
        same gold as the first run.
        """
        batch_a = _make_emg_silver_batch_a()
        batch_b = _make_emg_silver_batch_b_earlier()

        gold_after_a = cloud_emergency(batch_a)
        # First application of batch B.
        after_first = incremental_merge_emergency(gold_after_a, batch_a, batch_b)

        # Simulate silver store after absorbing batch_b: deduplicate by event_ts
        # (silver primary key includes event_ts, so duplicate rows are not stored).
        absorbed_silver = {r["event_ts"]: r for r in (batch_a + batch_b)}.values()
        existing_silver_after_absorb = list(absorbed_silver)

        # Second application of batch B against the already-updated silver store.
        after_second = self._incremental(after_first, existing_silver_after_absorb, batch_b)

        self.assertEqual(
            _canonical(after_first), _canonical(after_second),
            msg="Idempotency violated: second application of same batch changed result.",
        )

    def test_unaffected_entities_carried_forward_unchanged(self):
        """Gold rows for entities NOT in the new batch are untouched."""
        batch_a = _make_emg_silver_batch_a()
        # Add an unrelated emergency entity to existing gold.
        unrelated_gold_row = {
            "icao24": "ffffff", "callsign": "OTHER", "squawk": "7500",
            "first_seen_ts": "2024-06-03T10:00:00+00:00",
            "last_seen_ts": "2024-06-03T10:05:00+00:00",
            "lat": 48.0, "lon": 11.0, "origin_country": "Germany",
            "nearest_airport": "EDDM", "duration_s": 300,
        }
        existing_gold = cloud_emergency(batch_a) + [unrelated_gold_row]
        batch_b = _make_emg_silver_batch_b_earlier()

        result = incremental_merge_emergency(existing_gold, batch_a, batch_b)

        unrelated_in_result = [r for r in result if r["icao24"] == "ffffff"]
        self.assertEqual(len(unrelated_in_result), 1)
        self.assertEqual(unrelated_in_result[0], unrelated_gold_row)

    def test_empty_new_batch_returns_existing_gold_unchanged(self):
        """Empty new batch must return the existing gold row-for-row."""
        batch_a = _make_emg_silver_batch_a()
        existing_gold = cloud_emergency(batch_a)

        result = self._incremental(existing_gold, batch_a, [])
        self.assertEqual(_canonical(existing_gold), result)

    def test_no_duplicate_rows_after_overlapping_batches(self):
        """After overlapping batch application there must be exactly one event row."""
        batch_a = _make_emg_silver_batch_a()
        batch_b = _make_emg_silver_batch_b_earlier()

        gold_after_a = cloud_emergency(batch_a)
        result = self._incremental(gold_after_a, batch_a, batch_b)

        # Only one (icao24, squawk) entity present → should be exactly one event.
        self.assertEqual(len(result), 1)


class TestIncrementalMergeRoutingOverlap(unittest.TestCase):
    """incremental_merge_routing is correct for overlapping batches."""

    # --- helpers ---

    def _full_recompute(self, silver: list[dict]) -> list[dict]:
        return _canonical(cloud_routing(silver))

    def _incremental(
        self,
        existing_gold: list[dict],
        existing_silver: list[dict],
        new_batch: list[dict],
    ) -> list[dict]:
        return _canonical(incremental_merge_routing(existing_gold, existing_silver, new_batch))

    # --- tests ---

    def test_overlapping_batch_equals_full_recompute(self):
        """Batch B contains an earlier observation: incremental must equal full recompute."""
        batch_a = _make_routing_silver_batch_a()
        batch_b = _make_routing_silver_batch_b_earlier()

        existing_gold = cloud_routing(batch_a)
        result = self._incremental(existing_gold, batch_a, batch_b)
        expected = self._full_recompute(batch_a + batch_b)

        self.assertEqual(
            result, expected,
            msg=f"Incremental(A then B) != full_recompute(A+B)\n"
                f"  incremental : {result}\n"
                f"  full        : {expected}",
        )

    def test_order_independent_a_then_b_equals_b_then_a(self):
        """Applying batch A then B yields same result as B then A."""
        batch_a = _make_routing_silver_batch_a()
        batch_b = _make_routing_silver_batch_b_earlier()

        gold_after_a = cloud_routing(batch_a)
        result_ab = self._incremental(gold_after_a, batch_a, batch_b)

        gold_after_b = cloud_routing(batch_b)
        result_ba = self._incremental(gold_after_b, batch_b, batch_a)

        self.assertEqual(
            result_ab, result_ba,
            msg=f"Order dependence detected for routing:\n"
                f"  A then B : {result_ab}\n"
                f"  B then A : {result_ba}",
        )

    def test_order_independent_both_equal_full_recompute(self):
        """Both orderings equal full recompute over (A + B)."""
        batch_a = _make_routing_silver_batch_a()
        batch_b = _make_routing_silver_batch_b_earlier()
        expected = self._full_recompute(batch_a + batch_b)

        gold_after_a = cloud_routing(batch_a)
        result_ab = self._incremental(gold_after_a, batch_a, batch_b)

        gold_after_b = cloud_routing(batch_b)
        result_ba = self._incremental(gold_after_b, batch_b, batch_a)

        self.assertEqual(result_ab, expected)
        self.assertEqual(result_ba, expected)

    def test_window_start_is_earliest_across_both_batches(self):
        """After merging, window_start must be the earliest observation timestamp."""
        batch_a = _make_routing_silver_batch_a()
        batch_b = _make_routing_silver_batch_b_earlier()

        gold_after_a = cloud_routing(batch_a)
        result = self._incremental(gold_after_a, batch_a, batch_b)

        self.assertEqual(len(result), 1)
        route = result[0]
        # Earliest ping is in batch B at 13:00.
        self.assertEqual(route["window_start"], "2024-06-03T13:00:00+00:00")
        # Latest ping is batch A at 13:10.
        self.assertEqual(route["window_end"], "2024-06-03T13:10:00+00:00")

    def test_origin_reflects_earliest_observation(self):
        """origin_lat/lon must be from the earliest ping (batch B)."""
        batch_a = _make_routing_silver_batch_a()
        batch_b = _make_routing_silver_batch_b_earlier()

        gold_after_a = cloud_routing(batch_a)
        result = self._incremental(gold_after_a, batch_a, batch_b)
        route = result[0]

        self.assertEqual(route["origin_lat"], 47.5)
        self.assertEqual(route["origin_lon"], 10.5)

    def test_ping_count_includes_all_batches(self):
        """ping_count must count pings from all batches combined."""
        batch_a = _make_routing_silver_batch_a()   # 2 pings
        batch_b = _make_routing_silver_batch_b_earlier()  # 1 ping

        gold_after_a = cloud_routing(batch_a)
        result = self._incremental(gold_after_a, batch_a, batch_b)

        self.assertEqual(result[0]["ping_count"], 3)

    def test_avg_velocity_uses_all_batches(self):
        """avg_velocity_mps is the mean of all pings across batches."""
        batch_a = _make_routing_silver_batch_a()   # velocities 250, 260
        batch_b = _make_routing_silver_batch_b_earlier()  # velocity 240

        gold_after_a = cloud_routing(batch_a)
        result = self._incremental(gold_after_a, batch_a, batch_b)

        # (240 + 250 + 260) / 3 = 250.0
        self.assertAlmostEqual(result[0]["avg_velocity_mps"], 250.0, places=2)

    def test_idempotent_applying_same_batch_twice(self):
        """Applying the same new batch twice must produce no net change.

        After batch_b rows are absorbed into the silver store (deduplicated by
        event_ts), a second incremental run with the same new_batch must yield
        the same gold as the first run.
        """
        batch_a = _make_routing_silver_batch_a()
        batch_b = _make_routing_silver_batch_b_earlier()

        gold_after_a = cloud_routing(batch_a)
        # First application of batch B.
        after_first = incremental_merge_routing(gold_after_a, batch_a, batch_b)

        # Simulate silver store after absorbing batch_b: deduplicate by event_ts.
        absorbed_silver = {r["event_ts"]: r for r in (batch_a + batch_b)}.values()
        existing_silver_after_absorb = list(absorbed_silver)

        # Second application of batch B against the already-updated silver store.
        after_second = self._incremental(after_first, existing_silver_after_absorb, batch_b)

        self.assertEqual(
            _canonical(after_first), _canonical(after_second),
            msg="Idempotency violated: second application of same batch changed result.",
        )

    def test_unaffected_entities_carried_forward_unchanged(self):
        """Routing rows for entities NOT in the new batch are untouched."""
        batch_a = _make_routing_silver_batch_a()
        unrelated_gold_row = {
            "icao24": "112233", "callsign": "UNREL",
            "window_start": "2024-06-03T09:00:00+00:00",
            "window_end": "2024-06-03T09:30:00+00:00",
            "origin_lat": 52.0, "origin_lon": 4.0,
            "destination_lat": 52.5, "destination_lon": 4.5,
            "max_altitude_m": 8000.0, "avg_velocity_mps": 200.0, "ping_count": 5,
        }
        existing_gold = cloud_routing(batch_a) + [unrelated_gold_row]
        batch_b = _make_routing_silver_batch_b_earlier()

        result = incremental_merge_routing(existing_gold, batch_a, batch_b)

        unrelated_in_result = [r for r in result if r["icao24"] == "112233"]
        self.assertEqual(len(unrelated_in_result), 1)
        self.assertEqual(unrelated_in_result[0], unrelated_gold_row)

    def test_empty_new_batch_returns_existing_gold_unchanged(self):
        """Empty new batch must return the existing gold row-for-row."""
        batch_a = _make_routing_silver_batch_a()
        existing_gold = cloud_routing(batch_a)

        result = self._incremental(existing_gold, batch_a, [])
        self.assertEqual(_canonical(existing_gold), result)

    def test_no_duplicate_rows_after_overlapping_batches(self):
        """After overlapping batch application there must be exactly one route row."""
        batch_a = _make_routing_silver_batch_a()
        batch_b = _make_routing_silver_batch_b_earlier()

        gold_after_a = cloud_routing(batch_a)
        result = self._incremental(gold_after_a, batch_a, batch_b)

        self.assertEqual(len(result), 1)

    def test_callsign_normalisation_preserved_in_incremental(self):
        """Trailing-whitespace callsign variant in new batch is handled correctly."""
        batch_a_normal = [
            {
                "icao24": "ccddee", "callsign": "XYZ",
                "event_ts": "2024-06-03T14:05:00+00:00",
                "lat": 50.0, "lon": 10.0,
                "on_ground": False, "baro_altitude_m": 7000.0, "velocity_ms": 220.0,
            },
        ]
        batch_b_whitespace = [
            {
                "icao24": "ccddee", "callsign": "XYZ ",  # trailing space
                "event_ts": "2024-06-03T14:00:00+00:00",  # earlier
                "lat": 49.5, "lon": 9.5,
                "on_ground": False, "baro_altitude_m": 6800.0, "velocity_ms": 215.0,
            },
        ]

        gold_after_a = cloud_routing(batch_a_normal)
        result = self._incremental(gold_after_a, batch_a_normal, batch_b_whitespace)
        expected = self._full_recompute(batch_a_normal + batch_b_whitespace)

        self.assertEqual(result, expected)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ping_count"], 2)
        # origin from earliest ping (batch B)
        self.assertEqual(result[0]["origin_lat"], 49.5)


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
