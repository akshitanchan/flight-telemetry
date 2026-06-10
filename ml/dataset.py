import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path

from ml.features import FEATURE_COLUMNS


class FuelBurnDataset(Dataset):
    """
    PyTorch Dataset for Eurocontrol PRC Fuel Burn Estimation Challenge.
    Loads pre-extracted features from memory to eliminate IO bottlenecks.

    The feature tensor is assembled from FEATURE_COLUMNS (imported from
    ml/features.py).  This is the single source of truth — changing the
    feature schema requires only editing ml/features.py.
    """
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split

        features_path = self.data_dir / f"features_{split}.parquet"

        if not features_path.exists():
            raise FileNotFoundError(f"Features file missing at {features_path}. Run extract_features.py first.")

        # Load the entire feature table into memory
        self.df_features = pd.read_parquet(features_path)

        # Validate that all expected columns are present; missing columns get
        # filled with 0.0 so old feature files degrade gracefully rather than
        # crashing (useful during incremental schema rollout).
        missing = [c for c in FEATURE_COLUMNS if c not in self.df_features.columns]
        if missing:
            import warnings
            warnings.warn(
                f"Features parquet is missing {len(missing)} column(s): {missing}. "
                "Filling with 0.0 — re-run extract_features.py to regenerate.",
                stacklevel=2,
            )
            for col in missing:
                self.df_features[col] = 0.0

    # Expose flight_id per row so callers can perform group-aware splits.
    # Returns a numpy array of shape (N,) aligned with __getitem__ indices.
    @property
    def flight_ids(self) -> np.ndarray:
        return self.df_features["flight_id"].to_numpy()

    def __len__(self):
        return len(self.df_features)

    def __getitem__(self, idx):
        row = self.df_features.iloc[idx]

        target_fuel = torch.tensor(row['fuel_kg'], dtype=torch.float32)

        # Build feature vector from the shared ordered column list.
        features = torch.tensor(
            [float(row[col]) for col in FEATURE_COLUMNS],
            dtype=torch.float32,
        )

        return features, target_fuel
