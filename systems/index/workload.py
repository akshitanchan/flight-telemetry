#!/usr/bin/env python3
"""
Query workload generator for spatiotemporal index benchmarking.

Generates random spatiotemporal range queries ("all aircraft in bbox B
during window W") with configurable parameters. Deterministic via seed
for reproducible benchmarks.

Plan §5.1: "A query workload generator issuing spatiotemporal range queries."
"""

import random
from datetime import datetime, timedelta, timezone

from systems.index.base import BBox, TimeWindow, SpatiotemporalQuery


# Default query space — European airspace matching the sample data
DEFAULT_SPACE = {
    "lon_min": -5.0,
    "lon_max": 25.0,
    "lat_min": 40.0,
    "lat_max": 58.0,
    "time_start": "2024-06-03T11:00:00+00:00",
    "time_end": "2024-06-03T13:00:00+00:00",
}

# Query size profiles (bbox width/height in degrees, time window in seconds)
QUERY_PROFILES = {
    "point": {"bbox_size": 0.5, "time_window_s": 60},
    "local": {"bbox_size": 2.0, "time_window_s": 300},
    "regional": {"bbox_size": 5.0, "time_window_s": 600},
    "continental": {"bbox_size": 15.0, "time_window_s": 1800},
}


def generate_workload(
    num_queries: int = 100,
    profile: str = "regional",
    seed: int = 42,
    space: dict | None = None,
) -> list[SpatiotemporalQuery]:
    """Generate a reproducible workload of spatiotemporal range queries.

    Args:
        num_queries: Number of queries to generate.
        profile: Query size profile (point, local, regional, continental).
        seed: Random seed for reproducibility.
        space: Override the default query space bounds.

    Returns:
        List of SpatiotemporalQuery objects.
    """
    rng = random.Random(seed)
    sp = space or DEFAULT_SPACE
    prof = QUERY_PROFILES.get(profile, QUERY_PROFILES["regional"])
    bbox_size = prof["bbox_size"]
    time_window_s = prof["time_window_s"]

    queries = []
    for i in range(num_queries):
        # Random bbox center within space
        center_lon = rng.uniform(sp["lon_min"] + bbox_size / 2,
                                  sp["lon_max"] - bbox_size / 2)
        center_lat = rng.uniform(sp["lat_min"] + bbox_size / 2,
                                  sp["lat_max"] - bbox_size / 2)

        bbox = BBox(
            lon_min=round(center_lon - bbox_size / 2, 4),
            lon_max=round(center_lon + bbox_size / 2, 4),
            lat_min=round(center_lat - bbox_size / 2, 4),
            lat_max=round(center_lat + bbox_size / 2, 4),
        )

        # Random time window within space
        t_start_str = sp["time_start"]
        t_end_str = sp["time_end"]
        # Parse to compute random start within range
        t_start = datetime.fromisoformat(t_start_str)
        t_end = datetime.fromisoformat(t_end_str)
        range_s = (t_end - t_start).total_seconds()

        offset_s = rng.uniform(0, max(0, range_s - time_window_s))
        q_start = t_start + timedelta(seconds=offset_s)
        q_end = q_start + timedelta(seconds=time_window_s)

        tw = TimeWindow(
            start=q_start.isoformat(),
            end=q_end.isoformat(),
        )

        queries.append(SpatiotemporalQuery(bbox=bbox, time_window=tw, query_id=i))

    return queries
