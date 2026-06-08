#!/usr/bin/env python3
"""Tests for the state-vector normalizer."""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import unittest
from systems.replay.normalizer import normalize_state_vector, normalize_batch


class TestNormalizeStateVector(unittest.TestCase):
    """Test individual state vector normalization."""

    def _make_sv(self, **overrides):
        """Build a valid 17-element OpenSky state vector with optional overrides."""
        base = [
            "4b1806",       # 0: icao24
            "SWR162 ",      # 1: callsign (note trailing space)
            "Switzerland",  # 2: origin_country
            1717416000,     # 3: time_position
            1717416000,     # 4: last_contact
            4.7638,         # 5: longitude
            52.3080,        # 6: latitude
            11278.0,        # 7: baro_altitude
            False,          # 8: on_ground
            234.5,          # 9: velocity
            42.3,           # 10: true_track
            0.0,            # 11: vertical_rate
            None,           # 12: sensors
            11290.5,        # 13: geo_altitude
            "2536",         # 14: squawk
            False,          # 15: spi
            0,              # 16: position_source
        ]
        for k, v in overrides.items():
            idx = int(k) if k.isdigit() else None
            if idx is not None:
                base[idx] = v
        return base

    def test_basic_normalization(self):
        """Normal state vector produces correct landing record."""
        sv = self._make_sv()
        result = normalize_state_vector(1717416000, sv)

        self.assertIsNotNone(result)
        self.assertEqual(result["icao24"], "4b1806")
        self.assertEqual(result["callsign"], "SWR162")  # trimmed
        self.assertEqual(result["origin_country"], "Switzerland")
        self.assertEqual(result["lon"], 4.7638)
        self.assertEqual(result["lat"], 52.3080)
        self.assertEqual(result["baro_altitude_m"], 11278.0)
        self.assertFalse(result["on_ground"])
        self.assertEqual(result["velocity_ms"], 234.5)
        self.assertEqual(result["squawk"], "2536")
        self.assertEqual(result["idem_key"], "4b1806:1717416000")

    def test_callsign_whitespace_stripped(self):
        """Callsign with trailing spaces is trimmed."""
        sv = self._make_sv()
        sv[1] = "  ABC123  "
        result = normalize_state_vector(1717416000, sv)
        self.assertEqual(result["callsign"], "ABC123")

    def test_callsign_empty_becomes_none(self):
        """Empty callsign (all whitespace) becomes None."""
        sv = self._make_sv()
        sv[1] = "      "
        result = normalize_state_vector(1717416000, sv)
        self.assertIsNone(result["callsign"])

    def test_callsign_null(self):
        """Null callsign stays None."""
        sv = self._make_sv()
        sv[1] = None
        result = normalize_state_vector(1717416000, sv)
        self.assertIsNone(result["callsign"])

    def test_icao24_lowercased(self):
        """ICAO24 is lowercased."""
        sv = self._make_sv()
        sv[0] = "4B1806"
        result = normalize_state_vector(1717416000, sv)
        self.assertEqual(result["icao24"], "4b1806")

    def test_event_ts_from_time_position(self):
        """event_ts uses time_position when available."""
        sv = self._make_sv()
        sv[3] = 1717416005  # different from snapshot_time
        result = normalize_state_vector(1717416000, sv)
        self.assertIn("2024-06-03T12:00:05", result["event_ts"])

    def test_event_ts_fallback_to_snapshot(self):
        """event_ts falls back to snapshot_time when time_position is None."""
        sv = self._make_sv()
        sv[3] = None
        result = normalize_state_vector(1717416000, sv)
        self.assertIn("2024-06-03T12:00:00", result["event_ts"])

    def test_idem_key_format(self):
        """Idempotency key is icao24:event_ts."""
        sv = self._make_sv()
        result = normalize_state_vector(1717416000, sv)
        self.assertEqual(result["idem_key"], "4b1806:1717416000")

    def test_malformed_too_short(self):
        """State vector with too few fields returns None."""
        sv = ["4b1806", "SWR162"]  # only 2 fields
        result = normalize_state_vector(1717416000, sv)
        self.assertIsNone(result)

    def test_malformed_empty(self):
        """Empty state vector returns None."""
        result = normalize_state_vector(1717416000, [])
        self.assertIsNone(result)

    def test_malformed_none(self):
        """None state vector returns None."""
        result = normalize_state_vector(1717416000, None)
        self.assertIsNone(result)

    def test_malformed_no_icao24(self):
        """State vector with None icao24 returns None."""
        sv = self._make_sv()
        sv[0] = None
        result = normalize_state_vector(1717416000, sv)
        self.assertIsNone(result)

    def test_on_ground_truthy(self):
        """on_ground is always boolean."""
        sv = self._make_sv()
        sv[8] = True
        result = normalize_state_vector(1717416000, sv)
        self.assertTrue(result["on_ground"])


class TestNormalizeBatch(unittest.TestCase):
    """Test batch normalization."""

    def test_batch_skips_malformed(self):
        """Batch normalization skips malformed records."""
        valid_sv = [
            "4b1806", "SWR162 ", "Switzerland", 1717416000, 1717416000,
            4.7638, 52.3080, 11278.0, False, 234.5, 42.3, 0.0,
            None, 11290.5, "2536", False, 0,
        ]
        malformed_sv = ["bad"]
        results = normalize_batch(1717416000, [valid_sv, malformed_sv, valid_sv])
        self.assertEqual(len(results), 2)

    def test_batch_empty(self):
        """Empty batch returns empty list."""
        results = normalize_batch(1717416000, [])
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
