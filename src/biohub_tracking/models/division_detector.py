"""
Neural network models for cell division detection.
"""

import torch
import torch.nn as nn

class DivisionDetectionModel(nn.Module):
    """A neural network that predicts if a cell is dividing."""
    def __init__(self, in_features=256):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))
