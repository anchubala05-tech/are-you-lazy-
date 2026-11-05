"""
model.py
--------
A small feed-forward network for binary classification:
    P(will_procrastinate = 1 | features)

Nothing exotic on purpose — this is a tabular, 10-feature problem, so a deep
or wide net would just overfit. Three hidden layers with dropout is plenty.
"""

import torch
import torch.nn as nn


class ProcrastinationNet(nn.Module):
    def __init__(self, n_features: int = 10, hidden_sizes=(32, 16, 8), dropout: float = 0.2):
        super().__init__()

        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # raw logit; use BCEWithLogitsLoss

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))
