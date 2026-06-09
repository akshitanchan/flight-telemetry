#!/usr/bin/env python3
"""Tests for the structured analytics tool against the frozen gold fixtures."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.tools.analytics import AnalyticsTool, AnalyticsError

GOLD = PROJECT_ROOT / "ai" / "fixtures" / "gold"


class TestAnalyticsTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = AnalyticsTool(GOLD)

    def test_count_emergencies(self):
        self.assertEqual(self.tool.count_emergencies()["answer"], 3)
        self.assertEqual(self.tool.count_emergencies(squawk="7700")["answer"], 2)
        self.assertEqual(self.tool.count_emergencies(squawk="7500")["answer"], 1)

    def test_list_emergency_aircraft(self):
        r = self.tool.list_emergency_aircraft(squawk="7700")
        self.assertEqual(set(r["answer"]), {"3c6751", "a0b1c2"})
        self.assertEqual(r["sources"], ["gold_emergency_events"])

    def test_airport_congestion(self):
        r = self.tool.airport_congestion(airport_icao="LOIR")
        self.assertEqual(r["answer"]["aircraft_count"], 1)
        self.assertEqual(r["sources"], ["gold_airport_congestion"])

    def test_counts(self):
        self.assertEqual(self.tool.count_active_airports()["answer"], 11)
        self.assertEqual(self.tool.count_active_sectors()["answer"], 13)
        self.assertEqual(self.tool.total_flights_tracked()["answer"], 10)

    def test_flight_summary(self):
        r = self.tool.flight_summary(icao24="3c6751")
        self.assertAlmostEqual(r["answer"]["max_altitude_m"], 11574.0, places=1)
        self.assertEqual(r["answer"]["ping_count"], 5)

    def test_highest_altitude_flight(self):
        r = self.tool.highest_altitude_flight()
        self.assertEqual(r["answer"]["icao24"], "3c6751")

    def test_call_dispatch(self):
        self.assertEqual(self.tool.call("count_emergencies")["answer"], 3)

    def test_unknown_operation_rejected(self):
        # No free-form / destructive operations are reachable (plan §5.4).
        with self.assertRaises(AnalyticsError):
            self.tool.call("drop_table")

    def test_missing_required_param(self):
        with self.assertRaises(AnalyticsError):
            self.tool.airport_congestion(airport_icao=None)


if __name__ == "__main__":
    unittest.main()
