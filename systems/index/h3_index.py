#!/usr/bin/env python3
"""
H3 index — second spatiotemporal index strategy.

Uses Uber's H3 hierarchical hexagonal grid system for spatial bucketing.
Records have a precomputed `h3_r7` resolution 7 cell in the silver layer.
This strategy groups them by H3 cell at a configurable resolution (<= 7).

Approach:
  - Build: group records by their H3 cell at the chosen resolution.
           Uses `h3.cell_to_parent` if resolution < 7.
  - Query: Convert the BBox to a polygon, use `h3.polygon_to_cells` to 
           get all overlapping cells at the chosen resolution, and scan
           matching buckets with a time filter.
  - Tradeoff: H3 polygon filling is much more accurate for arbitrary
              shapes than geohash grids, and hexagons have uniform
              adjacency, reducing boundary edge cases.
"""

import sys
import time as time_mod
from collections import defaultdict
from typing import Any

import h3

from systems.index.base import SpatiotemporalIndex, SpatiotemporalQuery, BBox, parse_iso_ts


def _h3_cells_for_bbox(bbox: BBox, resolution: int) -> set[str]:
    """Compute the set of H3 cells that overlap a bounding box.

    Strategy: Convert the BBox into a GeoJSON-like polygon (list of 
    lat/lon coordinates for the outer loop) and use H3's polyfill.
    """
    # H3 polygon format: list of (lat, lon) for the outer boundary
    polygon = [
        (bbox.lat_min, bbox.lon_min),
        (bbox.lat_min, bbox.lon_max),
        (bbox.lat_max, bbox.lon_max),
        (bbox.lat_max, bbox.lon_min),
    ]
    
    # polygon_to_cells returns a list of cell strings (at the given resolution)
    cells = set(h3.polygon_to_cells(h3.LatLngPoly(polygon), resolution))
    
    # Depending on the H3 version, polyfill might miss the exact boundary if the 
    # polygon is very small or falls between cell centers. 
    # As a fallback for very small bboxes, also add the center point's cell.
    if not cells:
        center_lat = (bbox.lat_min + bbox.lat_max) / 2
        center_lon = (bbox.lon_min + bbox.lon_max) / 2
        cells.add(h3.latlng_to_cell(center_lat, center_lon, resolution))

    return cells


class H3Index(SpatiotemporalIndex):
    """Spatiotemporal index using H3 hexagonal bucketing.

    Records are grouped by their H3 cell at a configurable resolution.
    Queries find overlapping cells using H3's native polygon fill.
    """

    def __init__(self, resolution: int = 4):
        """
        Args:
            resolution: H3 resolution for bucketing (1–7).
                Lower = fewer, larger hexagons (faster build, more scan).
                Higher = more, smaller hexagons (slower build, less scan).
        """
        if not (1 <= resolution <= 7):
            raise ValueError("Resolution must be between 1 and 7")
            
        self._resolution = resolution
        self._buckets: dict[str, list[dict]] = defaultdict(list)
        self._record_count = 0
        self._build_time = 0.0

    @property
    def name(self) -> str:
        return f"h3_r{self._resolution}"

    def build(self, records: list[dict]) -> None:
        """Build H3 buckets from silver records."""
        start = time_mod.monotonic()
        self._buckets.clear()
        self._record_count = 0

        for rec in records:
            h3_r7 = rec.get("h3_r7")
            if not h3_r7:
                continue
                
            # If target resolution is 7, use directly. Otherwise, get parent.
            if self._resolution == 7:
                cell = h3_r7
            else:
                try:
                    cell = h3.cell_to_parent(h3_r7, self._resolution)
                except Exception:
                    # In case of invalid h3_r7 strings
                    continue

            self._buckets[cell].append(rec)
            self._record_count += 1

        self._build_time = time_mod.monotonic() - start

    def query(self, q: SpatiotemporalQuery) -> list[dict]:
        """Query records within bbox and time window."""
        # Find overlapping H3 cells
        cells = _h3_cells_for_bbox(q.bbox, self._resolution)

        # Parse window bounds once — datetime comparison, not fragile string
        # comparison that breaks on mixed 'Z' vs '+00:00' suffixes (audit M13).
        t_start = parse_iso_ts(q.time_window.start)
        t_end = parse_iso_ts(q.time_window.end)

        results = []
        for cell in cells:
            bucket = self._buckets.get(cell, [])
            for rec in bucket:
                # Exact spatial check
                lon = rec.get("lon")
                lat = rec.get("lat")
                if lon is None or lat is None:
                    continue
                if not q.bbox.contains(lon, lat):
                    continue

                # Time check
                event_ts = rec.get("event_ts")
                if not event_ts:
                    continue
                if t_start <= parse_iso_ts(event_ts) <= t_end:
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
            "resolution": self._resolution,
        }
