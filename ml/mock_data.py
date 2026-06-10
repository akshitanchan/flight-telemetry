"""
ml/mock_data.py — Offline PRC-shaped fixture generator.
========================================================

Generates a minimal but structurally correct mock of the PRC-2025 dataset
so the offline test suite (test_cv.py, test_obs.py) and the mock-data training
path (make ml-baseline-small) run without real data.

New columns added (ml-03):
  - track        : filled with a mock bearing so avg_track_change is exercisable
  - mach / TAS / CAS : kept as NaN (matches real ADS-B data; fill strategy in
                       extract_features.py handles them)

The flightlist now uses all 4 original mock aircraft types (A320, B738, A359,
B77W), which are all in ml.features.AIRCRAFT_TYPES, so one-hot encoding works
correctly in offline tests.
"""

import os
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_mock_eurocontrol_data(
    out_dir: str, num_flights: int = 5, seed: int = 42
):
    """
    Generates a mock version of the Eurocontrol PRC Data Challenge 2025 dataset
    to unblock the ML pipeline build while awaiting real data access.

    Emits all trajectory columns expected by extract_features.py:
      timestamp, flight_id, typecode, latitude, longitude, altitude,
      groundspeed, track, vertical_rate, mach, TAS, CAS, source
    """
    np.random.seed(seed)  # reproducible mock data (audit m7)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    flight_ids = [f"prc_mock_{i:04d}" for i in range(num_flights)]
    # Use types that are in ml.features.AIRCRAFT_TYPES for valid one-hot encoding
    aircraft_types = ["A320", "B738", "A359", "B77W"]

    # 1. Generate flightlist_train.parquet
    flights = []
    base_time = datetime(2025, 4, 15, 8, 0, 0)

    for fid in flight_ids:
        ac_type = np.random.choice(aircraft_types)
        duration_mins = np.random.randint(45, 180)
        takeoff = base_time + timedelta(minutes=np.random.randint(0, 1000))
        landed = takeoff + timedelta(minutes=duration_mins)

        flights.append({
            "flight_id": fid,
            "flight_date": takeoff.strftime("%Y-%m-%d"),
            "aircraft_type": ac_type,
            "takeoff": takeoff,
            "landed": landed,
            "origin_icao": "EHAM",
            "origin_name": "Amsterdam",
            "destination_icao": "EGLL",
            "destination_name": "London",
        })

    df_flightlist = pd.DataFrame(flights)
    df_flightlist.to_parquet(out_path / "flightlist_train.parquet")

    # 2. Generate trajectory parquets and zip them, and fuel_train.parquet
    fuel_records = []
    zip_path = out_path / "flights_train.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in flights:
            fid = f["flight_id"]
            takeoff = f["takeoff"]
            landed = f["landed"]
            ac_type = f["aircraft_type"]

            # Generate trajectory (1 record every 10 seconds)
            duration_s = int((landed - takeoff).total_seconds())
            timestamps = [
                takeoff + timedelta(seconds=s) for s in range(0, duration_s, 10)
            ]
            n_pts = len(timestamps)

            # Mock track: slowly rotating bearing (270 ± noise) so avg_track_change
            # is non-zero and exercisable in tests.
            track_base = np.linspace(260.0, 280.0, n_pts) + np.random.normal(0, 1, n_pts)
            track_base = track_base % 360.0  # keep in [0, 360)

            # Mock kinematic data — mach/TAS/CAS left as NaN to match real ADS-B.
            df_traj = pd.DataFrame({
                "timestamp": timestamps,
                "flight_id": fid,
                "typecode": ac_type,
                "latitude": np.linspace(52.3, 51.4, n_pts),
                "longitude": np.linspace(4.7, -0.4, n_pts),
                "altitude": np.sin(np.linspace(0, np.pi, n_pts)) * 10000.0,
                "groundspeed": np.random.normal(200, 10, n_pts),
                "track": track_base,
                "vertical_rate": np.random.normal(0, 1, n_pts),
                "mach": np.nan,
                "TAS": np.nan,
                "CAS": np.nan,
                "source": "adsb",
            })

            # Save to temporary parquet and write to zip
            tmp_pq = out_path / f"{fid}.parquet"
            df_traj.to_parquet(tmp_pq)
            zf.write(tmp_pq, arcname=f"{fid}.parquet")
            os.remove(tmp_pq)

            # Generate fuel intervals (every 15 minutes)
            interval_mins = 15
            current_start = takeoff

            while current_start + timedelta(minutes=interval_mins) <= landed:
                current_end = current_start + timedelta(minutes=interval_mins)

                # Mock fuel: ~50kg per minute
                fuel_burned = np.random.normal(50 * interval_mins, 50)

                fuel_records.append({
                    "flight_id": fid,
                    "start": current_start,
                    "end": current_end,
                    "fuel_kg": max(100.0, fuel_burned),
                })
                current_start = current_end

    df_fuel = pd.DataFrame(fuel_records)
    # add index column
    df_fuel.insert(0, "idx", range(len(df_fuel)))
    df_fuel.to_parquet(out_path / "fuel_train.parquet")

    print(f"Mock Eurocontrol data generated at {out_path}/")
    print(f" - flightlist_train.parquet ({len(df_flightlist)} flights)")
    print(f" - fuel_train.parquet ({len(df_fuel)} intervals)")
    print(f" - flights_train.zip (containing {len(df_flightlist)} trajectory files)")


if __name__ == "__main__":
    generate_mock_eurocontrol_data("data/ml/prc_2025_mock")
