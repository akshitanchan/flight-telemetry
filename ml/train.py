import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import math
import copy
import logging
from pathlib import Path

from ml.dataset import FuelBurnDataset
from ml.model import FuelBurnMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO  [%(name)s] %(message)s")
logger = logging.getLogger("ml.train")

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for features, target in dataloader:
        features, target = features.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(features)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * features.size(0)
        
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for features, target in dataloader:
            features, target = features.to(device), target.to(device)
            output = model(features)
            loss = criterion(output, target)
            total_loss += loss.item() * features.size(0)
            
    return total_loss / len(dataloader.dataset)

def main():
    parser = argparse.ArgumentParser(description="Eurocontrol Fuel Burn Baseline Training")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to Eurocontrol dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    if args.data_dir == "data/ml/prc_2025_mock" and not Path(args.data_dir).exists():
        raise FileNotFoundError("Mock data missing. Run generate_mock in extract script.")
    device = torch.device("cpu") # Keep simple for local scaffold
    
    logger.info(f"Loading dataset from {args.data_dir}")
    dataset = FuelBurnDataset(data_dir=args.data_dir, split="train")
    
    # Simple split for scaffold
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    model = FuelBurnMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize MLflow tracking
    mlflow.set_experiment("FuelBurn_Baseline")
    
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "model_type": "MLP_Baseline"
        })
        
        logger.info(f"Starting training for {args.epochs} epochs...")
        best_val_rmse = float('inf')
        best_state = None

        for epoch in range(1, args.epochs + 1):
            train_mse = train_epoch(model, train_loader, criterion, optimizer, device)
            val_mse = evaluate(model, val_loader, criterion, device)

            train_rmse = math.sqrt(train_mse)
            val_rmse = math.sqrt(val_mse)

            mlflow.log_metrics({
                "train_rmse": train_rmse,
                "val_rmse": val_rmse
            }, step=epoch)

            logger.info(f"Epoch {epoch:03d} | Train RMSE: {train_rmse:.2f} | Val RMSE: {val_rmse:.2f}")

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = copy.deepcopy(model.state_dict())

        logger.info(f"Training complete. Best Val RMSE: {best_val_rmse:.2f}")
        mlflow.log_metric("best_val_rmse", best_val_rmse)

        # Restore the best-performing weights so we register the best model, not
        # the (possibly overfit) final-epoch model (audit M4).
        if best_state is not None:
            model.load_state_dict(best_state)

        # Log the PyTorch model
        mlflow.pytorch.log_model(model, "model")
        
    dataset.close()

if __name__ == "__main__":
    main()
