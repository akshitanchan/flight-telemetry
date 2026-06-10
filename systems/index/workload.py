#!/usr/bin/env python3
"""
Query workload generator for spatiotemporal index benchmarking.

Generates random spatiotemporal range queries ("all aircraft in bbox B
during window W") with configurable parameters. Deterministic via seed
for reproducible benchmarks.

Also provides ``generate_synthetic_records``: a seeded silver-format record
generator used for the offline benchmark mode (no external data required).

Plan §5.1: "A query workload generator issuing spatiotemporal range queries."
"""

import random
from datetime import datetime, timedelta, timezone

import pygeohash
import h3

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

# Aircraft category values (mirrors ws1-01 normalizer: 0–19 per ADS-B spec)
_CATEGORIES = list(range(0, 20))

# Sample origin countries for synthetic records — representative European set
_COUNTRIES = [
    "Germany", "France", "United Kingdom", "Netherlands", "Switzerland",
    "Spain", "Italy", "Austria", "Belgium", "Portugal",
]

# Sample callsign prefixes for synthetic aircraft
_CALLSIGN_PREFIXES = ["DLH", "AFR", "BAW", "KLM", "SWR", "IBE", "AZA", "AUA", "BEL", "TAP"]


def generate_synthetic_records(
    n: int = 1_000,
    seed: int = 42,
    space: dict | None = None,
) -> list[dict]:
    """Generate a seeded list of deterministic synthetic silver-format records.

    All spatial coordinates are uniformly distributed within ``space`` (default
    ``DEFAULT_SPACE``).  Timestamps are spread uniformly across the space's
    time range.  geohash7 and h3_r7 are computed from lon/lat so the records
    are valid inputs to all three index backends.

    The generator is fully deterministic: the same ``seed`` + ``n`` always
    produces identical output, enabling reproducible offline benchmarks.

    Args:
        n:     Number of records to generate.
        seed:  Random seed.  Default 42 matches the workload generator default.
        space: Override the default spatial/temporal bounds.  Must contain
               the same keys as ``DEFAULT_SPACE``.

    Returns:
        List of dicts with fields:
            icao24, callsign, event_ts, lon, lat, baro_altitude_m,
            velocity_ms, true_track_deg, vertical_rate_ms, on_ground,
            squawk, origin_country, nearest_airport, geohash7, h3_r7,
            metar_wind_kt, metar_vis_m, metar_ceiling_ft, category
    """
    rng = random.Random(seed)
    sp = space or DEFAULT_SPACE

    lon_min = sp["lon_min"]
    lon_max = sp["lon_max"]
    lat_min = sp["lat_min"]
    lat_max = sp["lat_max"]

    t_start = datetime.fromisoformat(sp["time_start"])
    t_end = datetime.fromisoformat(sp["time_end"])
    time_range_s = (t_end - t_start).total_seconds()

    records = []
    for i in range(n):
        lon = rng.uniform(lon_min, lon_max)
        lat = rng.uniform(lat_min, lat_max)

        # Geospatial index fields — computed deterministically from lon/lat
        geohash7 = pygeohash.encode(lat, lon, precision=7)
        h3_r7 = h3.latlng_to_cell(lat, lon, 7)

        # Spread timestamps uniformly across the space's time range
        offset_s = rng.uniform(0, time_range_s)
        event_ts = (t_start + timedelta(seconds=offset_s)).isoformat()

        # Synthetic aircraft identifier — zero-padded 6-hex to match ICAO24 format
        icao24 = format(i % 0xFFFFFF, "06x")

        prefix = rng.choice(_CALLSIGN_PREFIXES)
        callsign = f"{prefix}{rng.randint(1, 9999):04d}"

        records.append({
            "icao24": icao24,
            "callsign": callsign,
            "event_ts": event_ts,
            "lon": round(lon, 6),
            "lat": round(lat, 6),
            "baro_altitude_m": round(rng.uniform(0, 12_500), 1),
            "velocity_ms": round(rng.uniform(0, 300), 1),
            "true_track_deg": round(rng.uniform(0, 360), 1),
            "vertical_rate_ms": round(rng.uniform(-15, 15), 2),
            "on_ground": False,
            "squawk": f"{rng.randint(0, 7777):04d}",
            "origin_country": rng.choice(_COUNTRIES),
            "nearest_airport": None,
            "geohash7": geohash7,
            "h3_r7": h3_r7,
            "metar_wind_kt": None,
            "metar_vis_m": None,
            "metar_ceiling_ft": None,
            "category": rng.choice(_CATEGORIES),
        })

    return records

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
