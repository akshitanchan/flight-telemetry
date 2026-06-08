#!/usr/bin/env python3
"""
Geohash prefix index — first spatiotemporal index strategy.

Uses geohash7 prefixes for spatial bucketing and linear time filtering
within matching buckets. This is the simplest strategy and serves as
the baseline for comparison.

Approach:
  - Build: group records by geohash7 prefix (configurable precision 1–7)
  - Query: compute all geohash prefixes that overlap the bbox,
           then scan matching buckets with time filter
  - Tradeoff: fast build, simple, but bbox→geohash coverage can
              over-select (scans more records than needed)
"""

import sys
import time as time_mod
from collections import defaultdict
from typing import Any

import pygeohash

from systems.index.base import SpatiotemporalIndex, SpatiotemporalQuery, BBox


# Base32 alphabet used by geohash
GEOHASH_CHARS = "0123456789bcdefghjkmnpqrstuvwxyz"


def _geohash_prefixes_for_bbox(bbox: BBox, precision: int) -> set[str]:
    """Compute the set of geohash prefixes that overlap a bounding box.

    Strategy: sample a grid of points within the bbox and collect
    their geohash prefixes. For small bboxes this is fast and
    accurate; for very large bboxes we add boundary samples.
    """
    prefixes = set()

    # Adaptive grid: more samples for larger areas
    lon_range = bbox.lon_max - bbox.lon_min
    lat_range = bbox.lat_max - bbox.lat_min

    # Step size based on geohash precision to ensure we don't skip cells
    # p3 is ~1.4 deg, p4 is ~0.3 deg, p5 is ~0.04 deg
    precision_steps = {1: 10.0, 2: 2.0, 3: 0.5, 4: 0.1, 5: 0.02, 6: 0.005, 7: 0.001}
    step_lon = precision_steps.get(precision, 0.001)
    step_lat = step_lon

    lat = bbox.lat_min
    while lat <= bbox.lat_max:
        lon = bbox.lon_min
        while lon <= bbox.lon_max:
            gh = pygeohash.encode(lat, lon, precision=precision)
            prefixes.add(gh[:precision])
            lon += step_lon
        lat += step_lat

    # Always include corners and center
    for lat, lon in [
        (bbox.lat_min, bbox.lon_min),
        (bbox.lat_min, bbox.lon_max),
        (bbox.lat_max, bbox.lon_min),
        (bbox.lat_max, bbox.lon_max),
        ((bbox.lat_min + bbox.lat_max) / 2, (bbox.lon_min + bbox.lon_max) / 2),
    ]:
        gh = pygeohash.encode(lat, lon, precision=precision)
        prefixes.add(gh[:precision])

    return prefixes


class GeohashPrefixIndex(SpatiotemporalIndex):
    """Spatiotemporal index using geohash prefix bucketing.

    Records are grouped by their geohash7 prefix at a configurable
    precision level. Queries find overlapping prefixes and scan
    matching buckets with exact bbox + time filtering.
    """

    def __init__(self, prefix_precision: int = 4):
        """
        Args:
            prefix_precision: Geohash prefix length for bucketing (1–7).
                Lower = fewer, larger buckets (faster build, more scan).
                Higher = more, smaller buckets (slower build, less scan).
        """
        self._precision = prefix_precision
        self._buckets: dict[str, list[dict]] = defaultdict(list)
        self._record_count = 0
        self._build_time = 0.0

    @property
    def name(self) -> str:
        return f"geohash_prefix_p{self._precision}"

    def build(self, records: list[dict]) -> None:
        """Build geohash prefix buckets from silver records."""
        start = time_mod.monotonic()
        self._buckets.clear()
        self._record_count = 0

        for rec in records:
            gh = rec.get("geohash7", "")
            if len(gh) >= self._precision:
                prefix = gh[:self._precision]
                self._buckets[prefix].append(rec)
                self._record_count += 1

        self._build_time = time_mod.monotonic() - start

    def query(self, q: SpatiotemporalQuery) -> list[dict]:
        """Query records within bbox and time window."""
        # Find overlapping geohash prefixes
        prefixes = _geohash_prefixes_for_bbox(q.bbox, self._precision)

        results = []
        for prefix in prefixes:
            bucket = self._buckets.get(prefix, [])
            for rec in bucket:
                # Exact spatial check
                lon = rec.get("lon")
                lat = rec.get("lat")
                if lon is None or lat is None:
                    continue
                if not q.bbox.contains(lon, lat):
                    continue

                # Time check
                event_ts = rec.get("event_ts", "")
                if q.time_window.start <= event_ts <= q.time_window.end:
                    results.append(rec)

        return results

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        bucket_sizes = [len(b) for b in self._buckets.values()]
        size_estimate = (
            sys.getsizeof(self._buckets)
            + sum(sys.getsizeof(k) for k in self._buckets)
            + sum(sys.getsizeof(v) for v in self._buckets.values())
        )

        return {
            "strategy": self.name,
            "record_count": self._record_count,
            "build_time_s": round(self._build_time, 6),
            "index_size_bytes": size_estimate,
            "bucket_count": len(self._buckets),
            "bucket_size_avg": round(sum(bucket_sizes) / max(1, len(bucket_sizes)), 1),
            "bucket_size_max": max(bucket_sizes) if bucket_sizes else 0,
            "prefix_precision": self._precision,
        }
