import torch
import torch.nn as nn

class FuelBurnMLP(nn.Module):
    """
    Baseline Multi-Layer Perceptron for Fuel Burn Estimation.
    Takes 4 simple trajectory features and outputs a single scalar (fuel_kg).
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # Accept an unbatched [input_dim] tensor by adding a batch dim, so
        # squeeze(-1) is well-defined for both batched and single inputs (audit M9).
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x).squeeze(-1)
