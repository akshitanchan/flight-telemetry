import argparse
import zipfile
import io
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO  [%(name)s] %(message)s")
logger = logging.getLogger("ml.extract")

def extract_features(data_dir: str, split: str = "train"):
    """
    Extracts features for all fuel intervals by iterating through the zip file exactly once.
    Saves a flat features_{split}.parquet file to speed up PyTorch training.
    """
    data_path = Path(data_dir)
    fuel_path = data_path / f"fuel_{split}.parquet"
    zip_path = data_path / f"flights_{split}.zip"
    
    if not fuel_path.exists() or not zip_path.exists():
        raise FileNotFoundError(f"Missing required datasets in {data_path}")
        
    logger.info(f"Loading fuel intervals from {fuel_path}")
    df_fuel = pd.read_parquet(fuel_path)
    
    # Group intervals by flight_id so we only process each trajectory once
    flight_intervals = df_fuel.groupby('flight_id')
    
    extracted_features = []
    
    logger.info(f"Extracting features from {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get list of all parquets in the zip
        zip_files = set(zf.namelist())
        
        # We iterate over the distinct flights that have fuel labels
        for flight_id, group in tqdm(flight_intervals, desc="Processing flights"):
            parquet_filename = f"{flight_id}.parquet"
            
            if parquet_filename not in zip_files:
                # Flight missing from zip, skip or yield zeros
                for _, row in group.iterrows():
                    extracted_features.append({
                        "idx": row["idx"],
                        "flight_id": flight_id,
                        "duration_s": 0.0,
                        "alt_change": 0.0,
                        "avg_speed": 0.0,
                        "max_vrate": 0.0,
                        "fuel_kg": row["fuel_kg"]
                    })
                continue
                
            # Read trajectory once
            with zf.open(parquet_filename) as f:
                df_traj = pd.read_parquet(io.BytesIO(f.read()))
                
            df_traj['timestamp'] = pd.to_datetime(df_traj['timestamp'])
            
            # Extract features for all intervals of this flight
            for _, row in group.iterrows():
                start_ts = pd.to_datetime(row['start'])
                end_ts = pd.to_datetime(row['end'])
                
                mask = (df_traj['timestamp'] >= start_ts) & (df_traj['timestamp'] <= end_ts)
                df_interval = df_traj[mask]
                
                duration_s = (end_ts - start_ts).total_seconds()
                
                if len(df_interval) < 2:
                    alt_change = 0.0
                    avg_speed = 0.0
                    max_vrate = 0.0
                else:
                    alt_change = float(df_interval['altitude'].iloc[-1] - df_interval['altitude'].iloc[0])
                    avg_speed = float(df_interval['groundspeed'].mean())
                    max_vrate = float(df_interval['vertical_rate'].abs().max())
                    
                    if np.isnan(avg_speed): avg_speed = 0.0
                    if np.isnan(max_vrate): max_vrate = 0.0
                    
                extracted_features.append({
                    "idx": row["idx"],
                    "flight_id": flight_id,
                    "duration_s": duration_s,
                    "alt_change": alt_change,
                    "avg_speed": avg_speed,
                    "max_vrate": max_vrate,
                    "fuel_kg": row["fuel_kg"]
                })
                
    df_features = pd.DataFrame(extracted_features)
    out_path = data_path / f"features_{split}.parquet"
    df_features.to_parquet(out_path)
    logger.info(f"Successfully extracted features for {len(df_features)} intervals.")
    logger.info(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    if args.data_dir == "data/ml/prc_2025_mock" and not Path(args.data_dir).exists():
        # Import the mock generator lazily so importing this module elsewhere does
        # not pull it in for every caller (audit M6).
        from ml.mock_data import generate_mock_eurocontrol_data
        logger.info(f"Generating mock dataset in {args.data_dir}...")
        generate_mock_eurocontrol_data(args.data_dir, num_flights=20)

    extract_features(args.data_dir, args.split)
