"""Neural network modules used by DQN agents."""

from __future__ import annotations

import torch
from torch import nn


class MLPQNetwork(nn.Module):
    """Small feed-forward Q-network for flat maze observations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (64, 64),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
