#!/usr/bin/env python3
"""
Generate synthetic sample state-vector data in OpenSky raw format.

Produces a JSONL file matching the structure that the OpenSky REST API
returns from /states/all, so downstream ingestion code can consume it
without modification.

The generated data is deterministic (seeded RNG) for reproducibility.
"""

import json
import random
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Seed for reproducibility
SEED = 42

# Realistic sample aircraft — mix of countries and callsign patterns
SAMPLE_AIRCRAFT = [
    {"icao24": "4b1806", "callsign": "SWR162 ", "origin_country": "Switzerland"},
    {"icao24": "48414e", "callsign": "TRA6515", "origin_country": "Netherlands"},
    {"icao24": "3c6751", "callsign": "DLH42N ", "origin_country": "Germany"},
    {"icao24": "400a30", "callsign": "BAW226 ", "origin_country": "United Kingdom"},
    {"icao24": "a0b1c2", "callsign": "UAL455 ", "origin_country": "United States"},
    {"icao24": "39cedc", "callsign": "AFR175 ", "origin_country": "France"},
    {"icao24": "3448a9", "callsign": "VLG8204", "origin_country": "Spain"},
    {"icao24": "30014c", "callsign": None, "origin_country": "Italy"},
    {"icao24": "471f52", "callsign": "NOZ163 ", "origin_country": "Norway"},
    {"icao24": "4ca87b", "callsign": "RYR4125", "origin_country": "Ireland"},
    {"icao24": "c06042", "callsign": "ACA871 ", "origin_country": "Canada"},
    {"icao24": "7c6db5", "callsign": "QFA8   ", "origin_country": "Australia"},
    {"icao24": "a835af", "callsign": "DAL100 ", "origin_country": "United States"},
    {"icao24": "4601f6", "callsign": "SAS937 ", "origin_country": "Sweden"},
    {"icao24": "4081db", "callsign": "EZY83FP", "origin_country": "United Kingdom"},
]

# European airspace bounding box (the primary area for sample data)
BBOX = {
    "lon_min": -5.0,
    "lon_max": 20.0,
    "lat_min": 45.0,
    "lat_max": 55.0,
}

# Emergency squawk probability per state vector
EMERGENCY_SQUAWK_PROB = 0.02


def _random_squawk(rng: random.Random) -> str | None:
    """Generate a realistic squawk code (octal digits 0-7)."""
    if rng.random() < 0.1:  # 10% chance of no squawk
        return None
    if rng.random() < EMERGENCY_SQUAWK_PROB:
        return rng.choice(["7500", "7600", "7700"])
    digits = [str(rng.randint(0, 7)) for _ in range(4)]
    return "".join(digits)


def _generate_trajectory(
    rng: random.Random,
    num_snapshots: int,
    base_time: int,
    interval_s: int = 10,
) -> list[dict]:
    """Generate a simple linear trajectory with realistic parameters."""
    lon = rng.uniform(BBOX["lon_min"], BBOX["lon_max"])
    lat = rng.uniform(BBOX["lat_min"], BBOX["lat_max"])
    altitude = rng.uniform(3000, 12000)  # metres
    velocity = rng.uniform(80, 260)  # m/s
    heading = rng.uniform(0, 360)

    on_ground = rng.random() < 0.05  # 5% chance of being on ground
    if on_ground:
        altitude = 0.0
        velocity = rng.uniform(0, 15)

    points = []
    for i in range(num_snapshots):
        t = base_time + i * interval_s

        # Simple linear movement
        heading_rad = math.radians(heading)
        delta_lon = velocity * math.sin(heading_rad) * interval_s / 111320 / math.cos(math.radians(lat))
        delta_lat = velocity * math.cos(heading_rad) * interval_s / 110540

        lon += delta_lon
        lat += delta_lat

        # Clamp to valid ranges
        lon = max(-180, min(180, lon))
        lat = max(-90, min(90, lat))

        vertical_rate = rng.uniform(-3, 3) if not on_ground else 0.0
        altitude += vertical_rate * interval_s
        altitude = max(0, altitude)

        points.append({
            "time": t,
            "lon": round(lon, 6),
            "lat": round(lat, 6),
            "baro_altitude": round(altitude, 1) if not on_ground else None,
            "geo_altitude": round(altitude + rng.uniform(-20, 20), 1) if not on_ground else None,
            "on_ground": on_ground,
            "velocity": round(velocity + rng.uniform(-5, 5), 1),
            "true_track": round((heading + rng.uniform(-2, 2)) % 360, 1),
            "vertical_rate": round(vertical_rate, 2),
            "squawk": _random_squawk(rng),
        })

    return points


def generate_sample_data(
    output_dir: Path,
    num_aircraft: int = 10,
    num_snapshots: int = 5,
) -> Path:
    """Generate sample state-vector data in OpenSky raw JSONL format.

    Each line is a JSON object representing one API response snapshot,
    matching the structure of OpenSky /states/all.

    Returns the path to the generated file.
    """
    rng = random.Random(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_state_vectors.jsonl"

    # Pick aircraft (may be fewer than available)
    aircraft_pool = SAMPLE_AIRCRAFT[:num_aircraft]

    # Base time: 2024-06-03T12:00:00Z (a Monday, matching curated dataset pattern)
    base_ts = int(datetime(2024, 6, 3, 12, 0, 0, tzinfo=timezone.utc).timestamp())

    # Generate trajectories
    trajectories = {}
    for ac in aircraft_pool:
        trajectories[ac["icao24"]] = _generate_trajectory(
            rng, num_snapshots, base_ts
        )

    # Group by time snapshot and write as OpenSky API response format
    # Each line = one /states/all response at a given time
    time_slots = sorted(set(
        point["time"]
        for traj in trajectories.values()
        for point in traj
    ))

    records_written = 0
    with open(output_path, "w") as f:
        for ts in time_slots:
            states = []
            for ac in aircraft_pool:
                traj = trajectories[ac["icao24"]]
                matching = [p for p in traj if p["time"] == ts]
                if not matching:
                    continue
                point = matching[0]

                # OpenSky state vector array format (17 fields)
                # See: https://openskynetwork.github.io/opensky-api/rest.html
                state = [
                    ac["icao24"],           # 0: icao24
                    ac["callsign"],         # 1: callsign
                    ac["origin_country"],   # 2: origin_country
                    ts,                     # 3: time_position
                    ts,                     # 4: last_contact
                    point["lon"],           # 5: longitude
                    point["lat"],           # 6: latitude
                    point["baro_altitude"], # 7: baro_altitude (m)
                    point["on_ground"],     # 8: on_ground
                    point["velocity"],      # 9: velocity (m/s)
                    point["true_track"],    # 10: true_track (deg)
                    point["vertical_rate"], # 11: vertical_rate (m/s)
                    None,                   # 12: sensors
                    point["geo_altitude"],  # 13: geo_altitude (m)
                    point["squawk"],        # 14: squawk
                    False,                  # 15: spi
                    0,                      # 16: position_source (ADS-B)
                ]
                states.append(state)

            response = {
                "time": ts,
                "states": states,
            }
            f.write(json.dumps(response) + "\n")
            records_written += 1

    print(f"  Generated {records_written} snapshots × {len(aircraft_pool)} aircraft")
    return output_path
