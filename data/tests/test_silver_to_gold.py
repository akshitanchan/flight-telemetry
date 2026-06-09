#!/usr/bin/env python3
"""Tests for silver-to-gold aggregations."""

import unittest
from datetime import datetime, timezone

from data.transforms.silver_to_gold import (
    aggregate_emergency_events,
    aggregate_sector_load,
    aggregate_airport_congestion,
    aggregate_routing_stats,
    _floor_to_window
)

class TestSilverToGold(unittest.TestCase):
    def setUp(self):
        self.silver_records = [
            # Normal flight
            {
                "icao24": "111111",
                "callsign": "KLM123",
                "event_ts": "2024-06-03T12:02:00+00:00",
                "lon": 4.7,
                "lat": 52.3,
                "squawk": "2000",
                "on_ground": False,
                "baro_altitude_m": 10000.0,
                "h3_r7": "8719694b5ffffff", # parent at res 4 is 8419695ffffffff
                "nearest_airport": "EHAM"
            },
            # Emergency flight 1 (First observation)
            {
                "icao24": "222222",
                "callsign": "EMG456",
                "event_ts": "2024-06-03T12:04:00+00:00",
                "lon": 8.5,
                "lat": 50.0,
                "squawk": "7700",
                "on_ground": False,
                "baro_altitude_m": 5000.0,
                "h3_r7": "871fa1b13ffffff", # parent at res 4 is 841fa1bffffffff
                "nearest_airport": "EDDF"
            },
            # Emergency flight 1 (Second observation, same 5m window)
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
                "nearest_airport": "EDDF"
            },
            # Ground flight
            {
                "icao24": "333333",
                "callsign": "GND789",
                "event_ts": "2024-06-03T12:01:00+00:00",
                "lon": 4.76,
                "lat": 52.3,
                "squawk": "1000",
                "on_ground": True,
                "baro_altitude_m": 0.0,
                "h3_r7": "8719694b5ffffff", # same as first
                "nearest_airport": "EHAM"
            }
        ]

    def test_floor_to_window(self):
        dt = datetime(2024, 6, 3, 12, 14, 30, tzinfo=timezone.utc)
        floored = _floor_to_window(dt, 5)
        self.assertEqual(floored.minute, 10)
        self.assertEqual(floored.second, 0)
        self.assertEqual(floored.microsecond, 0)

    def test_aggregate_emergency_events(self):
        emg = aggregate_emergency_events(self.silver_records)
        self.assertEqual(len(emg), 1)
        record = emg[0]
        self.assertEqual(record["icao24"], "222222")
        self.assertEqual(record["squawk"], "7700")
        self.assertEqual(record["first_seen_ts"], "2024-06-03T12:04:00+00:00")
        self.assertEqual(record["last_seen_ts"], "2024-06-03T12:04:30+00:00")
        self.assertEqual(record["duration_s"], 30)

    def test_aggregate_sector_load(self):
        sectors = aggregate_sector_load(self.silver_records)
        self.assertEqual(len(sectors), 2)
        
        # Sort so we can predictably index
        sectors = sorted(sectors, key=lambda x: x["h3_r4"])
        
        # 8419695ffffffff should have 2 planes (111111, 333333)
        self.assertEqual(sectors[0]["h3_r4"], "8419695ffffffff")
        self.assertEqual(sectors[0]["window_start"], "2024-06-03T12:00:00+00:00")
        self.assertEqual(sectors[0]["window_end"], "2024-06-03T12:05:00+00:00")
        self.assertEqual(sectors[0]["aircraft_count"], 2)

        # 841fa1bffffffff should have 1 plane (222222 twice, but distinct is 1)
        self.assertEqual(sectors[1]["h3_r4"], "841fa1bffffffff")
        self.assertEqual(sectors[1]["aircraft_count"], 1)

    def test_aggregate_airport_congestion(self):
        congestion = aggregate_airport_congestion(self.silver_records)
        self.assertEqual(len(congestion), 2)
        
        # Sort by airport code
        congestion = sorted(congestion, key=lambda x: x["airport_icao"])
        
        # EDDF (Frankfurt) has 1 distinct plane, airborne
        eddf = congestion[0]
        self.assertEqual(eddf["airport_icao"], "EDDF")
        self.assertEqual(eddf["aircraft_count"], 1)
        self.assertEqual(eddf["ground_count"], 0)
        self.assertEqual(eddf["airborne_count"], 1) # Only distinct aircraft counted
        self.assertEqual(eddf["avg_altitude_m"], 5000.0) # Altitude of first observation

        # EHAM (Amsterdam) has 2 planes (1 airborne, 1 ground)
        eham = congestion[1]
        self.assertEqual(eham["airport_icao"], "EHAM")
        self.assertEqual(eham["aircraft_count"], 2)
        self.assertEqual(eham["ground_count"], 1)
        self.assertEqual(eham["airborne_count"], 1)
        self.assertEqual(eham["avg_altitude_m"], 5000.0) # (10000 + 0) / 2

    def test_aggregate_routing_stats(self):
        routes = aggregate_routing_stats(self.silver_records)
        # We have 3 distinct planes: 111111, 222222, 333333
        self.assertEqual(len(routes), 3)
        
        # Check emergency flight which has 2 pings
        emg_route = next(r for r in routes if r["icao24"] == "222222")
        self.assertEqual(emg_route["callsign"], "EMG456")
        self.assertEqual(emg_route["window_start"], "2024-06-03T12:04:00+00:00")
        self.assertEqual(emg_route["window_end"], "2024-06-03T12:04:30+00:00")
        self.assertEqual(emg_route["origin_lat"], 50.0)
        self.assertEqual(emg_route["destination_lat"], 50.01)
        self.assertEqual(emg_route["max_altitude_m"], 5000.0)
        self.assertEqual(emg_route["ping_count"], 2)

    def test_congestion_excludes_unassociated(self):
        """Records with no nearest_airport are excluded, not bucketed as UNKNOWN (C7)."""
        unassociated = {
            "icao24": "999999", "callsign": "NONE1",
            "event_ts": "2024-06-03T12:00:00+00:00", "lon": 10.0, "lat": 48.0,
            "squawk": "1200", "on_ground": False, "baro_altitude_m": 9000.0,
            "h3_r7": "871fa1b13ffffff", "nearest_airport": None,
        }
        self.assertEqual(aggregate_airport_congestion([unassociated]), [])

        # Mixed with an associated record: only the real airport appears, no UNKNOWN.
        mixed = [unassociated, self.silver_records[0]]  # second is EHAM
        codes = {c["airport_icao"] for c in aggregate_airport_congestion(mixed)}
        self.assertNotIn("UNKNOWN", codes)
        self.assertIn("EHAM", codes)

    def test_emergency_gap_splits_events(self):
        """Two 7700 observations far apart become two events, not one (M17)."""
        recs = [
            {"icao24": "777777", "callsign": "EMG1", "squawk": "7700",
             "event_ts": "2024-06-03T12:00:00+00:00", "lat": 50.0, "lon": 8.0,
             "on_ground": False, "baro_altitude_m": 5000.0,
             "h3_r7": "871fa1b13ffffff", "nearest_airport": "EDDF"},
            # 2 hours later — beyond EMERGENCY_GAP_S — should start a new event
            {"icao24": "777777", "callsign": "EMG1", "squawk": "7700",
             "event_ts": "2024-06-03T14:00:00+00:00", "lat": 50.1, "lon": 8.1,
             "on_ground": False, "baro_altitude_m": 5200.0,
             "h3_r7": "871fa1b13ffffff", "nearest_airport": "EDDF"},
        ]
        events = aggregate_emergency_events(recs)
        self.assertEqual(len(events), 2)
        for e in events:
            self.assertEqual(e["duration_s"], 0)

    def test_routing_callsign_normalized(self):
        """Whitespace callsign variants don't fan out into duplicate routes (M18)."""
        recs = [
            {"icao24": "555555", "callsign": "ABC123 ",
             "event_ts": "2024-06-03T12:00:00+00:00", "lat": 50.0, "lon": 8.0,
             "on_ground": False, "baro_altitude_m": 9000.0, "velocity_ms": 200.0},
            {"icao24": "555555", "callsign": "ABC123",
             "event_ts": "2024-06-03T12:01:00+00:00", "lat": 50.1, "lon": 8.1,
             "on_ground": False, "baro_altitude_m": 9100.0, "velocity_ms": 210.0},
        ]
        routes = aggregate_routing_stats(recs)
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["callsign"], "ABC123")
        self.assertEqual(routes[0]["ping_count"], 2)

if __name__ == "__main__":
    unittest.main()
