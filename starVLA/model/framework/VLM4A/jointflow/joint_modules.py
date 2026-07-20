"""PR2: Joint token projection modules for QwenJointFlow."""

from __future__ import annotations

import torch
from torch import nn


######### // code // ##########
class DinoProjector(nn.Module):
    def __init__(self, d_dino: int = 384, hidden_size: int = 896, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_dino),
            nn.Linear(d_dino, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ActionContextEncoder(nn.Module):
    def __init__(self, action_dim: int = 7, hidden_size: int = 896):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.position = nn.Embedding(512, hidden_size)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(actions.shape[1], device=actions.device)
        return self.net(actions) + self.position(pos).unsqueeze(0)


class ActionQueryTokenBank(nn.Module):
    def __init__(self, action_horizon: int = 8, hidden_size: int = 896):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_query = nn.Embedding(self.action_horizon, hidden_size)
        nn.init.normal_(self.action_query.weight, mean=0.0, std=0.02)

    def forward(self, batch_size: int, device=None) -> torch.Tensor:
        ids = torch.arange(self.action_horizon, device=device or self.action_query.weight.device)
        return self.action_query(ids).unsqueeze(0).expand(batch_size, -1, -1)


class FutureDinoQueryTokenBank(nn.Module):
    def __init__(self, max_queries: int = 196, hidden_size: int = 896):
        super().__init__()
        self.max_queries = int(max_queries)
        self.future_dino_query = nn.Embedding(self.max_queries, hidden_size)
        nn.init.normal_(self.future_dino_query.weight, mean=0.0, std=0.02)

    def forward(self, batch_size: int, n_query: int, device=None) -> torch.Tensor:
        if n_query > self.max_queries:
            raise ValueError(f"n_query={n_query} exceeds max_queries={self.max_queries}")
        ids = torch.arange(n_query, device=device or self.future_dino_query.weight.device)
        return self.future_dino_query(ids).unsqueeze(0).expand(batch_size, -1, -1)
######### // code // ##########
