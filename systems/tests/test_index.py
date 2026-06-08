#!/usr/bin/env python3
"""Tests for the spatiotemporal index benchmark components."""

import sys
from pathlib import Path
import unittest
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from systems.index.base import BBox, TimeWindow, SpatiotemporalQuery
from systems.index.geohash_index import GeohashPrefixIndex, _geohash_prefixes_for_bbox
from systems.index.h3_index import H3Index, _h3_cells_for_bbox
from systems.index.workload import generate_workload


class TestGeohashIndex(unittest.TestCase):
    def setUp(self):
        # A few mock silver records
        self.records = [
            {
                "icao24": "111111",
                "lon": 4.7638,
                "lat": 52.3080, # Amsterdam
                "event_ts": "2024-06-03T12:00:00+00:00",
                "geohash7": "u173s6m"
            },
            {
                "icao24": "222222",
                "lon": 8.5706,
                "lat": 50.0333, # Frankfurt
                "event_ts": "2024-06-03T12:05:00+00:00",
                "geohash7": "u0y1tep"
            },
            {
                "icao24": "333333",
                "lon": -0.4619,
                "lat": 51.4706, # London Heathrow
                "event_ts": "2024-06-03T11:55:00+00:00",
                "geohash7": "gcpuzm8"
            }
        ]

    def test_prefixes_for_bbox(self):
        # Small bbox covering Amsterdam
        bbox = BBox(4.7, 4.8, 52.3, 52.4)
        prefixes_p3 = _geohash_prefixes_for_bbox(bbox, 3)
        self.assertTrue(len(prefixes_p3) >= 1)
        self.assertIn("u17", prefixes_p3)
        
        prefixes_p5 = _geohash_prefixes_for_bbox(bbox, 5)
        self.assertTrue(len(prefixes_p5) >= 1)

    def test_build_and_query(self):
        index = GeohashPrefixIndex(prefix_precision=3)
        index.build(self.records)

        stats = index.stats()
        self.assertEqual(stats["record_count"], 3)
        self.assertEqual(stats["bucket_count"], 3) # u17, u0y, gcp

        # Query for Amsterdam flight
        q = SpatiotemporalQuery(
            bbox=BBox(4.7, 4.8, 52.3, 52.4),
            time_window=TimeWindow("2024-06-03T11:00:00+00:00", "2024-06-03T13:00:00+00:00")
        )
        res = index.query(q)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["icao24"], "111111")

        # Query outside time window
        q_time = SpatiotemporalQuery(
            bbox=BBox(4.7, 4.8, 52.3, 52.4),
            time_window=TimeWindow("2024-06-03T13:00:00+00:00", "2024-06-03T14:00:00+00:00")
        )
        res_time = index.query(q_time)
        self.assertEqual(len(res_time), 0)

        # Query outside bbox
        q_bbox = SpatiotemporalQuery(
            bbox=BBox(10.0, 11.0, 52.3, 52.4),
            time_window=TimeWindow("2024-06-03T11:00:00+00:00", "2024-06-03T13:00:00+00:00")
        )
        res_bbox = index.query(q_bbox)
        self.assertEqual(len(res_bbox), 0)

        # Broad query covering both Amsterdam and Frankfurt
        q_broad = SpatiotemporalQuery(
            bbox=BBox(4.0, 10.0, 49.0, 53.0),
            time_window=TimeWindow("2024-06-03T11:00:00+00:00", "2024-06-03T13:00:00+00:00")
        )
        res_broad = index.query(q_broad)
        self.assertEqual(len(res_broad), 2)
        icao_set = {r["icao24"] for r in res_broad}
        self.assertEqual(icao_set, {"111111", "222222"})


class TestH3Index(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "icao24": "111111",
                "lon": 4.7638,
                "lat": 52.3080, # Amsterdam
                "event_ts": "2024-06-03T12:00:00+00:00",
                "h3_r7": "8719694b5ffffff" 
            },
            {
                "icao24": "222222",
                "lon": 8.5706,
                "lat": 50.0333, # Frankfurt
                "event_ts": "2024-06-03T12:05:00+00:00",
                "h3_r7": "871fa1b13ffffff"
            }
        ]

    def test_h3_cells_for_bbox(self):
        bbox = BBox(4.7, 4.8, 52.3, 52.4)
        cells_r4 = _h3_cells_for_bbox(bbox, 4)
        self.assertTrue(len(cells_r4) >= 1)

    def test_build_and_query(self):
        index = H3Index(resolution=4)
        index.build(self.records)

        stats = index.stats()
        self.assertEqual(stats["record_count"], 2)

        q = SpatiotemporalQuery(
            bbox=BBox(4.7, 4.8, 52.3, 52.4),
            time_window=TimeWindow("2024-06-03T11:00:00+00:00", "2024-06-03T13:00:00+00:00")
        )
        res = index.query(q)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["icao24"], "111111")


class TestWorkloadGenerator(unittest.TestCase):
    def test_generate_workload(self):
        queries = generate_workload(num_queries=10, profile="local", seed=123)
        self.assertEqual(len(queries), 10)
        
        q = queries[0]
        self.assertIsInstance(q, SpatiotemporalQuery)
        
        # Local profile should have 2.0 bbox width/height
        lon_diff = q.bbox.lon_max - q.bbox.lon_min
        lat_diff = q.bbox.lat_max - q.bbox.lat_min
        self.assertAlmostEqual(lon_diff, 2.0)
        self.assertAlmostEqual(lat_diff, 2.0)

        # Time window should be 300 seconds (5 minutes)
        start_ts = datetime.fromisoformat(q.time_window.start)
        end_ts = datetime.fromisoformat(q.time_window.end)
        self.assertEqual((end_ts - start_ts).total_seconds(), 300)

if __name__ == "__main__":
    unittest.main()
