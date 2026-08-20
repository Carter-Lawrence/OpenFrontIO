import torch, torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from openfront_env import OpenFrontEnv
from stable_baselines3.common.callbacks import CheckpointCallback

"""
model_def.py — the shared network definition.

Put the CNN here (not inside your training script) so that BOTH your training
script and watch_wall.py can import the exact same class. That matters because
when Stable-Baselines3 saves a model, it records *where* the network class lives
and re-imports it on load. If the class only existed inside your training file,
the wall couldn't reload the saved checkpoints.

Use it in your training script like this:

    from model_def import OpenFrontCNN
    model = PPO("MultiInputPolicy", env,
                policy_kwargs=dict(features_extractor_class=OpenFrontCNN),
                ...)
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

GRID = 24  # must match GRID in sim_server.ts


class OpenFrontCNN(BaseFeaturesExtractor):
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
        # The scalars have wildly different sizes (gold ~100k, tiles ~5k), which
        # destabilizes training. Divide by rough typical values so they're ~0-1.
        self.register_buffer(
            "scalar_scale",
            torch.tensor([5000.0, 50000.0, 200000.0, 5.0, 20000.0, 5000.0]),
        )

    def forward(self, obs):
        # grid: (B,24,24) ints 0..3  ->  one-hot to (B,4,24,24) float
        grid = torch.nn.functional.one_hot(obs["grid"].long(), 4)
        grid = grid.permute(0, 3, 1, 2).float()
        scalars = obs["scalars"] / self.scalar_scale
        return self.head(torch.cat([self.conv(grid), self.scalar_mlp(scalars)], dim=1))
    
env = OpenFrontEnv()
model = PPO("MultiInputPolicy", env,
            policy_kwargs=dict(features_extractor_class=OpenFrontCNN),
            n_steps=2048, batch_size=256, verbose=1)
ckpt = CheckpointCallback(save_freq=50_000, save_path="./checkpoints/", name_prefix="ppo_of")
model.learn(total_timesteps=1_000_000, callback=ckpt)
model.save("ppo_openfront_final")     # -> ppo_openfront_final.zip