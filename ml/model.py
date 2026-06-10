import torch
import torch.nn as nn

from ml.features import INPUT_DIM


class FuelBurnMLP(nn.Module):
    """
    Baseline Multi-Layer Perceptron for Fuel Burn Estimation.

    Takes INPUT_DIM features (trajectory statistics + aircraft-type one-hot)
    and outputs a single scalar (fuel_kg).

    input_dim defaults to the shared constant INPUT_DIM from ml/features.py so
    the model architecture is always consistent with the feature pipeline.
    Hard-coding a literal 4 here was the previous bug; ml-03 fixes it.
    """
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 64):
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
