"""PR2: Joint token projection modules for QwenJointFlow."""

from __future__ import annotations

import torch
from torch import nn


######### // code // ##########
# 中文注释：DINO patch feature 投影到 Qwen hidden 维度。
# 输入 z [B,N,384]，输出 [B,N,D_qwen]。这里保留单一路径，SigLIP 分支只留 config 接口。
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


# 中文注释：把 clean action chunk 编成 Qwen token，用于 FDM 的 action context block。
# 输入 action [B,H_a,action_dim]，输出 [B,H_a,D_qwen]。
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


# 中文注释：只取当前 t 的 proprio state 注入 LLM，避免 fdm/passive 泄漏未来 state。
# 输入 state [B,T,state_dim] 或 [B,state_dim]，输出 [B,n_state_tokens,D_qwen]。
class StateEncoder(nn.Module):
    def __init__(self, state_dim: int = 8, hidden_size: int = 896, n_state_tokens: int = 1):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.n_state_tokens = int(n_state_tokens)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * self.n_state_tokens),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, state: torch.Tensor | None, batch_size: int | None = None, device=None, dtype=None) -> torch.Tensor:
        if state is None:
            if batch_size is None:
                raise ValueError("batch_size is required when state is None")
            state = torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype or torch.float32)
        elif state.ndim == 3:
            state = state[:, 0, :]
        elif state.ndim != 2:
            raise ValueError(f"Expected state [B,T,D] or [B,D], got {tuple(state.shape)}")

        state = state[..., : self.state_dim]
        if state.shape[-1] < self.state_dim:
            pad = torch.zeros(*state.shape[:-1], self.state_dim - state.shape[-1], device=state.device, dtype=state.dtype)
            state = torch.cat([state, pad], dim=-1)

        out = self.net(state).view(state.shape[0], self.n_state_tokens, self.hidden_size)
        return self.norm(out)


# 中文注释：管理 action query 和 image query 两组 learned tokens。
# action_query 输出 [B,H_a,D]；image_query 输出 [B,N_q,D]。
class QueryTokenBank(nn.Module):
    def __init__(self, action_horizon: int = 8, max_image_queries: int = 196, hidden_size: int = 896):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.max_image_queries = int(max_image_queries)
        self.action_query = nn.Embedding(self.action_horizon, hidden_size)
        self.image_query = nn.Embedding(self.max_image_queries, hidden_size)
        nn.init.normal_(self.action_query.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.image_query.weight, mean=0.0, std=0.02)

    def get_action_queries(self, batch_size: int, device=None) -> torch.Tensor:
        ids = torch.arange(self.action_horizon, device=device or self.action_query.weight.device)
        return self.action_query(ids).unsqueeze(0).expand(batch_size, -1, -1)

    def get_image_queries(self, batch_size: int, n_query: int, device=None) -> torch.Tensor:
        if n_query > self.max_image_queries:
            raise ValueError(f"n_query={n_query} exceeds max_image_queries={self.max_image_queries}")
        ids = torch.arange(n_query, device=device or self.image_query.weight.device)
        return self.image_query(ids).unsqueeze(0).expand(batch_size, -1, -1)
######### // code // ##########

