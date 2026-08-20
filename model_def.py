"""
model_def.py — the shared network definition (CLASS ONLY, no training code).

Both train.py and watch_wall.py import the class from here. This file must be
importable WITHOUT side effects, so there is NO env/model/model.learn code in it.
Training lives in train.py.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

GRID = 24  # must match GRID in sim_server.ts


class OpenFrontCNN(BaseFeaturesExtractor):
    """Turns the {grid, scalars} observation into one feature vector."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.scalar_mlp = nn.Sequential(nn.Linear(6, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(64 * GRID * GRID + 32, features_dim), nn.ReLU()
        )
        self.register_buffer(
            "scalar_scale",
            torch.tensor([5000.0, 50000.0, 200000.0, 5.0, 20000.0, 5000.0]),
        )

    def forward(self, obs):
        grid = torch.nn.functional.one_hot(obs["grid"].long(), 4)
        grid = grid.permute(0, 3, 1, 2).float()
        scalars = obs["scalars"] / self.scalar_scale
        return self.head(torch.cat([self.conv(grid), self.scalar_mlp(scalars)], dim=1))