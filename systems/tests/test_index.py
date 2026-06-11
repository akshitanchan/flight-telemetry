#!/usr/bin/env python3
"""Tests for the spatiotemporal index benchmark components."""

import sys
from pathlib import Path
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from systems.index.base import BBox, TimeWindow, SpatiotemporalQuery
from systems.index.geohash_index import GeohashPrefixIndex, _geohash_prefixes_for_bbox
from systems.index.h3_index import H3Index, _h3_cells_for_bbox
from systems.index.postgis_index import PostGISIndex
from systems.index.workload import generate_workload
from systems.index.benchmark import run_benchmark

# Evaluate DB availability once at module load time so every PostGIS test
# gets a consistent skip reason without re-running the healthcheck per test.
_DB_AVAILABLE = PostGISIndex.is_available()


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


    def test_mixed_tz_suffix_time_filter(self):
        """A record stored with a 'Z' suffix matches a '+00:00' window (M12)."""
        index = GeohashPrefixIndex(prefix_precision=3)
        index.build([{
            "icao24": "555555", "lon": 4.7638, "lat": 52.3080,
            "event_ts": "2024-06-03T12:00:00Z", "geohash7": "u173s6m",
        }])
        q = SpatiotemporalQuery(
            bbox=BBox(4.7, 4.8, 52.3, 52.4),
            time_window=TimeWindow("2024-06-03T11:00:00+00:00", "2024-06-03T13:00:00+00:00"),
        )
        self.assertEqual(len(index.query(q)), 1)


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

class TestBenchmark(unittest.TestCase):
    def test_warmup_excluded_from_measurement(self):
        """Warmup queries are not part of the measured set (M14)."""
        records = [{
            "icao24": "111111", "lon": 4.7638, "lat": 52.3080,
            "event_ts": "2024-06-03T12:00:00+00:00",
            "geohash7": "u173s6m", "h3_r7": "8719694b5ffffff",
        }]
        index = GeohashPrefixIndex(prefix_precision=3)
        queries = generate_workload(num_queries=10, profile="regional", seed=7)
        result = run_benchmark(index, records, queries,
                               query_profile="regional", warmup_queries=3)
        # 10 generated - 3 warmup = 7 measured
        self.assertEqual(result.num_queries, 7)


class TestIndexEdgeCases(unittest.TestCase):
    def _query(self):
        return SpatiotemporalQuery(
            bbox=BBox(0.0, 1.0, 0.0, 1.0),
            time_window=TimeWindow("2024-06-03T11:00:00+00:00", "2024-06-03T13:00:00+00:00"))

    def test_empty_records(self):
        for index in (GeohashPrefixIndex(prefix_precision=3), H3Index(resolution=4)):
            index.build([])
            self.assertEqual(index.stats()["record_count"], 0)
            self.assertEqual(index.query(self._query()), [])

    def test_missing_spatial_fields_skipped(self):
        recs = [{"icao24": "111111", "event_ts": "2024-06-03T12:00:00+00:00"}]  # no geohash7/h3_r7
        g = GeohashPrefixIndex(prefix_precision=3)
        g.build(recs)
        self.assertEqual(g.stats()["record_count"], 0)
        h = H3Index(resolution=4)
        h.build(recs)
        self.assertEqual(h.stats()["record_count"], 0)


@unittest.skipUnless(_DB_AVAILABLE, "PostGIS tests require a live database (DATABASE_URL must be set)")
class TestPostGISIndex(unittest.TestCase):
    """Integration tests for the PostGIS GIST backend.

    Guarded with skipUnless so offline CI (no DATABASE_URL / no Postgres)
    stays green.  When a live DB is present these tests exercise the full
    build → query → stats round-trip against silver_flight_state.
    """

    # Three silver records that cover two European cities.
    _RECORDS = [
        {
            "icao24": "aaa001",
            "callsign": "TEST01",
            "event_ts": "2024-06-03T12:00:00+00:00",
            "lon": 4.7638,
            "lat": 52.3080,  # Amsterdam
            "baro_altitude_m": 10000.0,
            "velocity_ms": 250.0,
            "true_track_deg": 90.0,
            "vertical_rate_ms": 0.0,
            "on_ground": False,
            "squawk": "1234",
            "origin_country": "Netherlands",
            "nearest_airport": "EHAM",
            "geohash7": "u173s6m",
            "h3_r7": "8719694b5ffffff",
            "metar_wind_kt": None,
            "metar_vis_m": None,
            "metar_ceiling_ft": None,
        },
        {
            "icao24": "bbb002",
            "callsign": "TEST02",
            "event_ts": "2024-06-03T12:05:00+00:00",
            "lon": 8.5706,
            "lat": 50.0333,  # Frankfurt
            "baro_altitude_m": 9500.0,
            "velocity_ms": 240.0,
            "true_track_deg": 180.0,
            "vertical_rate_ms": -2.0,
            "on_ground": False,
            "squawk": "5678",
            "origin_country": "Germany",
            "nearest_airport": "EDDF",
            "geohash7": "u0y1tep",
            "h3_r7": "871fa1b13ffffff",
            "metar_wind_kt": None,
            "metar_vis_m": None,
            "metar_ceiling_ft": None,
        },
        {
            "icao24": "ccc003",
            "callsign": "TEST03",
            "event_ts": "2024-06-03T11:55:00+00:00",
            "lon": -0.4619,
            "lat": 51.4706,  # London Heathrow
            "baro_altitude_m": 300.0,
            "velocity_ms": 80.0,
            "true_track_deg": 270.0,
            "vertical_rate_ms": 5.0,
            "on_ground": False,
            "squawk": "9012",
            "origin_country": "United Kingdom",
            "nearest_airport": "EGLL",
            "geohash7": "gcpuzm8",
            "h3_r7": "871f1a640ffffff",
            "metar_wind_kt": None,
            "metar_vis_m": None,
            "metar_ceiling_ft": None,
        },
    ]

    def setUp(self):
        self.index = PostGISIndex(batch_size=100)

    def test_is_available(self):
        """is_available() returns True when the DB is reachable."""
        self.assertTrue(PostGISIndex.is_available())

    def test_build_upserts_records(self):
        """build() upserts records without error and reports correct count."""
        self.index.build(self._RECORDS)
        stats = self.index.stats()
        # record_count from stats() re-queries the full table; it must be >= 3
        # (other tests may have inserted rows already).
        self.assertGreaterEqual(stats["record_count"], 3)
        self.assertGreater(stats["build_time_s"], 0.0)
        self.assertGreater(stats["index_size_bytes"], 0)

    def test_build_is_idempotent(self):
        """Calling build() twice on the same records does not raise or duplicate."""
        self.index.build(self._RECORDS)
        count_after_first = self.index.stats()["record_count"]
        self.index.build(self._RECORDS)
        count_after_second = self.index.stats()["record_count"]
        # Row count must be stable; UPSERT must not insert duplicates.
        self.assertEqual(count_after_first, count_after_second)

    def test_query_spatial_filter(self):
        """query() returns only records inside the bbox."""
        self.index.build(self._RECORDS)

        # Tight bbox around Amsterdam — should return exactly the Amsterdam record.
        q = SpatiotemporalQuery(
            bbox=BBox(lon_min=4.7, lon_max=4.9, lat_min=52.2, lat_max=52.4),
            time_window=TimeWindow(
                "2024-06-03T11:00:00+00:00",
                "2024-06-03T13:00:00+00:00",
            ),
        )
        results = self.index.query(q)
        icao_set = {r["icao24"] for r in results}
        self.assertIn("aaa001", icao_set)
        self.assertNotIn("bbb002", icao_set)
        self.assertNotIn("ccc003", icao_set)

    def test_query_time_filter(self):
        """query() excludes records outside the time window."""
        self.index.build(self._RECORDS)

        # Window ends before any record — expect zero results in Amsterdam bbox.
        q = SpatiotemporalQuery(
            bbox=BBox(lon_min=4.7, lon_max=4.9, lat_min=52.2, lat_max=52.4),
            time_window=TimeWindow(
                "2024-06-03T10:00:00+00:00",
                "2024-06-03T11:30:00+00:00",
            ),
        )
        results = self.index.query(q)
        icao_set = {r["icao24"] for r in results}
        self.assertNotIn("aaa001", icao_set)

    def test_query_returns_dict_with_expected_fields(self):
        """Results have the same fields as geohash/h3 backends."""
        self.index.build(self._RECORDS)
        q = SpatiotemporalQuery(
            bbox=BBox(lon_min=4.7, lon_max=4.9, lat_min=52.2, lat_max=52.4),
            time_window=TimeWindow(
                "2024-06-03T11:00:00+00:00",
                "2024-06-03T13:00:00+00:00",
            ),
        )
        results = self.index.query(q)
        self.assertGreater(len(results), 0)
        row = results[0]
        for field in ("icao24", "event_ts", "lon", "lat", "geohash7", "h3_r7"):
            self.assertIn(field, row, f"Expected field '{field}' missing from result")

    def test_stats_structure(self):
        """stats() returns a dict with the required keys."""
        self.index.build(self._RECORDS)
        stats = self.index.stats()
        for key in ("record_count", "build_time_s", "index_size_bytes"):
            self.assertIn(key, stats, f"stats() missing required key '{key}'")
        self.assertIsInstance(stats["record_count"], int)
        self.assertIsInstance(stats["build_time_s"], float)
        self.assertIsInstance(stats["index_size_bytes"], int)

    def test_name_property(self):
        """name property returns the expected strategy identifier."""
        self.assertEqual(self.index.name, "postgis_gist")


class TestPostGISIndexUnavailable(unittest.TestCase):
    """Unit tests for the unavailability-degradation path.

    These run unconditionally (no DB needed) and verify that build() and
    query() raise a clear RuntimeError when the DB is unreachable.
    We patch is_available() to return False rather than requiring a missing DB.
    """

    def _make_query(self):
        return SpatiotemporalQuery(
            bbox=BBox(lon_min=0.0, lon_max=1.0, lat_min=0.0, lat_max=1.0),
            time_window=TimeWindow(
                "2024-06-03T11:00:00+00:00",
                "2024-06-03T13:00:00+00:00",
            ),
        )

    def test_build_raises_when_unavailable(self):
        """build() raises RuntimeError when DB is not reachable."""
        index = PostGISIndex()
        # Temporarily patch is_available on the class.
        original = PostGISIndex.is_available
        PostGISIndex.is_available = classmethod(lambda cls: False)
        try:
            with self.assertRaises(RuntimeError):
                index.build([{"icao24": "x", "event_ts": "2024-01-01T00:00:00+00:00",
                               "lon": 1.0, "lat": 1.0}])
        finally:
            PostGISIndex.is_available = original

    def test_query_raises_when_unavailable(self):
        """query() raises RuntimeError when DB is not reachable."""
        index = PostGISIndex()
        original = PostGISIndex.is_available
        PostGISIndex.is_available = classmethod(lambda cls: False)
        try:
            with self.assertRaises(RuntimeError):
                index.query(self._make_query())
        finally:
            PostGISIndex.is_available = original

    def test_stats_does_not_raise_when_unavailable(self):
        """stats() never raises — returns cached values when DB is gone."""
        index = PostGISIndex()
        original = PostGISIndex.is_available
        PostGISIndex.is_available = classmethod(lambda cls: False)
        try:
            stats = index.stats()
            self.assertIn("record_count", stats)
            self.assertIn("build_time_s", stats)
            self.assertIn("index_size_bytes", stats)
        finally:
            PostGISIndex.is_available = original


class TestPostGISIndexBuildUnit(unittest.TestCase):
    """Offline coverage for the psycopg3 cursor-based batch write path."""

    def test_build_uses_cursor_executemany(self):
        conn_context = MagicMock()
        conn = conn_context.__enter__.return_value
        cursor = conn.cursor.return_value.__enter__.return_value

        def execute(sql):
            result = MagicMock()
            result.fetchone.return_value = (
                (3,) if "COUNT" in sql else (4096,)
            )
            return result

        conn.execute.side_effect = execute
        records = [
            {
                "icao24": f"abc00{i}",
                "event_ts": f"2024-06-03T12:0{i}:00+00:00",
                "lon": 4.7 + i,
                "lat": 52.3,
            }
            for i in range(3)
        ]

        index = PostGISIndex(batch_size=2)
        with (
            patch.object(PostGISIndex, "is_available", return_value=True),
            patch("shared.store.pg.get_conn", return_value=conn_context),
        ):
            index.build(records)
            stats = index.stats()

        self.assertEqual(cursor.executemany.call_count, 2)
        self.assertEqual(stats["record_count"], 3)
        self.assertEqual(stats["index_size_bytes"], 4096)
        conn.commit.assert_called_once_with()


class TestSyntheticRecordGenerator(unittest.TestCase):
    """Tests for generate_synthetic_records (offline benchmark data source)."""

    def setUp(self):
        from systems.index.workload import generate_synthetic_records
        self._gen = generate_synthetic_records

    def test_correct_count(self):
        """Generator produces exactly the requested number of records."""
        records = self._gen(n=100, seed=42)
        self.assertEqual(len(records), 100)

    def test_required_fields_present(self):
        """Every record contains the fields required by all three backends."""
        required = {
            "icao24", "event_ts", "lon", "lat", "geohash7", "h3_r7", "category",
        }
        for rec in self._gen(n=50, seed=7):
            for field in required:
                self.assertIn(field, rec, f"Field '{field}' missing from synthetic record")

    def test_spatial_bounds(self):
        """Coordinates are within DEFAULT_SPACE bounds."""
        from systems.index.workload import DEFAULT_SPACE
        sp = DEFAULT_SPACE
        for rec in self._gen(n=200, seed=1):
            self.assertGreaterEqual(rec["lon"], sp["lon_min"])
            self.assertLessEqual(rec["lon"], sp["lon_max"])
            self.assertGreaterEqual(rec["lat"], sp["lat_min"])
            self.assertLessEqual(rec["lat"], sp["lat_max"])

    def test_determinism(self):
        """Same seed produces identical records across two calls."""
        r1 = self._gen(n=500, seed=99)
        r2 = self._gen(n=500, seed=99)
        # Compare a sample of records field by field to avoid dict-ordering edge cases
        for i in range(0, len(r1), 50):
            self.assertEqual(r1[i], r2[i], f"Record {i} differs between runs with same seed")

    def test_different_seeds_differ(self):
        """Different seeds produce different records."""
        r_a = self._gen(n=100, seed=1)
        r_b = self._gen(n=100, seed=2)
        # At minimum the coordinates should differ
        lons_a = [r["lon"] for r in r_a]
        lons_b = [r["lon"] for r in r_b]
        self.assertNotEqual(lons_a, lons_b)

    def test_geohash7_matches_coordinates(self):
        """geohash7 field decodes to coordinates within the record's location."""
        import pygeohash
        for rec in self._gen(n=20, seed=5):
            dec = pygeohash.decode(rec["geohash7"])
            # decode returns (lat, lon); allow ~0.1 degree tolerance for p7 precision
            self.assertAlmostEqual(dec[0], rec["lat"], delta=0.1)
            self.assertAlmostEqual(dec[1], rec["lon"], delta=0.1)

    def test_h3_r7_matches_coordinates(self):
        """h3_r7 field is a valid resolution-7 cell containing the record's point."""
        import h3
        for rec in self._gen(n=20, seed=5):
            expected = h3.latlng_to_cell(rec["lat"], rec["lon"], 7)
            self.assertEqual(rec["h3_r7"], expected)

    def test_geohash_index_accepts_synthetic_records(self):
        """GeohashPrefixIndex can build and query over synthetic records."""
        from systems.index.geohash_index import GeohashPrefixIndex
        from systems.index.base import BBox, TimeWindow, SpatiotemporalQuery
        records = self._gen(n=1_000, seed=42)
        index = GeohashPrefixIndex(prefix_precision=4)
        index.build(records)
        self.assertEqual(index.stats()["record_count"], 1_000)
        q = SpatiotemporalQuery(
            bbox=BBox(lon_min=-5.0, lon_max=25.0, lat_min=40.0, lat_max=58.0),
            time_window=TimeWindow("2024-06-03T11:00:00+00:00", "2024-06-03T13:00:00+00:00"),
        )
        results = index.query(q)
        self.assertGreater(len(results), 0)

    def test_h3_index_accepts_synthetic_records(self):
        """H3Index can build and query over synthetic records."""
        from systems.index.h3_index import H3Index
        from systems.index.base import BBox, TimeWindow, SpatiotemporalQuery
        records = self._gen(n=1_000, seed=42)
        index = H3Index(resolution=4)
        index.build(records)
        self.assertEqual(index.stats()["record_count"], 1_000)
        q = SpatiotemporalQuery(
            bbox=BBox(lon_min=-5.0, lon_max=25.0, lat_min=40.0, lat_max=58.0),
            time_window=TimeWindow("2024-06-03T11:00:00+00:00", "2024-06-03T13:00:00+00:00"),
        )
        results = index.query(q)
        self.assertGreater(len(results), 0)

    def test_large_synthetic_run_accepted(self):
        """Generator completes for n=1_000_000 in finite time and returns correct count."""
        import time
        t0 = time.monotonic()
        records = self._gen(n=1_000_000, seed=42)
        elapsed = time.monotonic() - t0
        self.assertEqual(len(records), 1_000_000)
        # Sanity: should finish well under 60 seconds on any modern machine
        self.assertLess(elapsed, 60, f"1M record generation took too long: {elapsed:.1f}s")


class TestOfflineCLIMode(unittest.TestCase):
    """Smoke tests for the offline CLI mode (no DB needed)."""

    def _run_cli(self, extra_args: list[str]) -> tuple[int, str, str]:
        """Run the CLI module in a subprocess, returning (returncode, stdout, stderr)."""
        import subprocess
        cmd = [
            sys.executable, "-m", "systems.index.cli",
            "--mode", "offline",
            "--synthetic-rows", "500",
            "--queries", "20",
            "--profile", "regional",
        ] + extra_args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        return result.returncode, result.stdout, result.stderr

    def test_offline_markdown_output(self):
        """CLI offline mode exits 0 and emits a Markdown table."""
        rc, stdout, stderr = self._run_cli(["--format", "markdown", "--strategies", "geohash_p4", "h3_r4"])
        self.assertEqual(rc, 0, f"CLI exited with {rc}. stderr:\n{stderr}")
        self.assertIn("geohash_prefix_p4", stdout)
        self.assertIn("h3_r4", stdout)

    def test_offline_json_output(self):
        """CLI offline mode emits valid JSON when --format json."""
        import json as _json
        rc, stdout, stderr = self._run_cli(["--format", "json", "--strategies", "geohash_p4", "h3_r4"])
        self.assertEqual(rc, 0, f"CLI exited with {rc}. stderr:\n{stderr}")
        data = _json.loads(stdout)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        for entry in data:
            self.assertIn("strategy", entry)
            self.assertIn("build_time_s", entry)

    def test_offline_csv_output(self):
        """CLI offline mode emits CSV when --format csv."""
        rc, stdout, stderr = self._run_cli(["--format", "csv", "--strategies", "geohash_p4"])
        self.assertEqual(rc, 0, f"CLI exited with {rc}. stderr:\n{stderr}")
        lines = [l for l in stdout.strip().splitlines() if l]
        self.assertGreaterEqual(len(lines), 2)  # header + at least 1 data row

    def test_offline_deterministic(self):
        """Two CLI runs with same seed produce identical JSON output."""
        import json as _json
        rc1, out1, _ = self._run_cli(["--format", "json", "--strategies", "geohash_p4", "--seed", "77"])
        rc2, out2, _ = self._run_cli(["--format", "json", "--strategies", "geohash_p4", "--seed", "77"])
        self.assertEqual(rc1, 0)
        self.assertEqual(rc2, 0)
        d1 = _json.loads(out1)
        d2 = _json.loads(out2)
        # Build times are wall-clock and will differ slightly; compare everything else
        for r1, r2 in zip(d1, d2):
            self.assertEqual(r1["strategy"], r2["strategy"])
            self.assertEqual(r1["record_count"], r2["record_count"])
            self.assertEqual(r1["result_count_total"], r2["result_count_total"])

    def test_offline_postgis_skip_when_no_db(self):
        """CLI offline mode skips postgis_gist (with WARNING) when DB absent."""
        import subprocess, os
        env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
        cmd = [
            sys.executable, "-m", "systems.index.cli",
            "--mode", "offline",
            "--synthetic-rows", "200",
            "--queries", "10",
            "--strategies", "postgis_gist",
            "--format", "json",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            env=env,
        )
        # Should exit 0 (clean skip), not non-zero
        self.assertEqual(result.returncode, 0, f"Expected clean exit. stderr:\n{result.stderr}")
        # Should warn about skipping
        self.assertIn("skipping", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
