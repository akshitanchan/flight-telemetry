import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

class FuelBurnDataset(Dataset):
    """
    PyTorch Dataset for Eurocontrol PRC Fuel Burn Estimation Challenge.
    Loads pre-extracted features from memory to eliminate IO bottlenecks.
    """
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        
        features_path = self.data_dir / f"features_{split}.parquet"
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file missing at {features_path}. Run extract_features.py first.")
            
        # Load the entire feature table into memory
        self.df_features = pd.read_parquet(features_path)
        
    def __len__(self):
        return len(self.df_features)
        
    def __getitem__(self, idx):
        row = self.df_features.iloc[idx]
        
        target_fuel = torch.tensor(row['fuel_kg'], dtype=torch.float32)
        
        features = torch.tensor([
            row['duration_s'], 
            row['alt_change'], 
            row['avg_speed'], 
            row['max_vrate']
        ], dtype=torch.float32)
        
        return features, target_fuel
        
    def close(self):
        pass

