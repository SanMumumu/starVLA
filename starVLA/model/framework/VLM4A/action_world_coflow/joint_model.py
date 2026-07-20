"""Single-bridge shared-transformer Action--World Co-Flow model.

Each closed-loop policy call models exactly one physical bridge:
``[context + Z0] -> [A1:16, Z(t+16)]``.  Action and future-world tokens share
self-attention; modality-specific norms and FFNs provide asymmetric capacity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from starVLA.model.modules.action_model.GR00T_ActionHeader import action_prediction_to_velocity

from .attention_mask import (
    TOKEN_ACTION,
    TOKEN_CONTEXT,
    TOKEN_WORLD,
    build_block_causal_attention_mask,
)
from .bridge import QantaraWorldBridge
from .noise_plane import NoisePlaneSampler


def _cfg_get(config, key: str, default=None):
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, hidden_size: int, frequency_dim: int = 256, scale: float = 1000.0) -> None:
        super().__init__()
        if frequency_dim % 2:
            raise ValueError("time_frequency_dim must be even")
        half = frequency_dim // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.scale = float(scale)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        args = tau.float().unsqueeze(-1) * self.scale * self.frequencies
        features = torch.cat([args.cos(), args.sin()], dim=-1)
        # Trigonometric features are intentionally evaluated in float32, but
        # inference does not run under the trainer's autocast context.  Match
        # the MLP parameter dtype explicitly so a bf16 checkpoint remains
        # callable from eval/deployment without relying on ambient autocast.
        first_weight = self.mlp[0].weight
        features = features.to(device=first_weight.device, dtype=first_weight.dtype)
        return self.mlp(features)


class SharedSelfAttention(nn.Module):
    """One shared attention operation for all context/action/world tokens."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(out.transpose(1, 2).reshape(batch, length, hidden))


class ModalityExpertBlock(nn.Module):
    """Shared attention plus context/action/world-specific norm and FFN.

    Legacy checkpoints use one common ``mlp_ratio`` and dense expert
    evaluation.  New asymmetric-MoT checkpoints may opt into distinct expert
    widths and token-sparse routing.  The legacy construction and state-dict
    layout are unchanged when those options are absent.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        *,
        expert_mlp_ratios: tuple[float, float, float] | None = None,
        sparse_expert_routing: bool = False,
    ) -> None:
        super().__init__()
        ratios = (
            tuple(float(value) for value in expert_mlp_ratios)
            if expert_mlp_ratios is not None
            else (float(mlp_ratio),) * 3
        )
        if len(ratios) != 3 or any(value <= 0 for value in ratios):
            raise ValueError(
                "expert_mlp_ratios must contain three positive values in "
                "(context, action, world) order"
            )
        self.expert_mlp_ratios = ratios
        self.sparse_expert_routing = bool(sparse_expert_routing)
        self.attention_norms = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(3)])
        self.ffn_norms = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(3)])
        self.attention = SharedSelfAttention(hidden_size, num_heads, dropout)
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, int(round(hidden_size * ratio))),
                    nn.GELU(approximate="tanh"),
                    nn.Dropout(dropout),
                    nn.Linear(int(round(hidden_size * ratio)), hidden_size),
                )
                for ratio in ratios
            ]
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _route(x: torch.Tensor, token_type_ids: torch.Tensor, experts: nn.ModuleList) -> torch.Tensor:
        result = torch.zeros_like(x)
        for type_id, expert in enumerate(experts):
            selector = (token_type_ids == type_id).view(1, -1, 1)
            result = result + expert(x) * selector.to(dtype=x.dtype)
        return result

    @staticmethod
    def _route_sparse(
        x: torch.Tensor,
        token_type_ids: torch.Tensor,
        experts: nn.ModuleList,
    ) -> torch.Tensor:
        """Evaluate each expert only on tokens assigned to that modality."""

        if token_type_ids.ndim != 1 or token_type_ids.shape[0] != x.shape[1]:
            raise ValueError(
                "sparse expert routing requires one token-type vector shared by the batch; "
                f"got x={tuple(x.shape)}, token_type_ids={tuple(token_type_ids.shape)}"
            )
        result = torch.zeros_like(x)
        for type_id, expert in enumerate(experts):
            indices = torch.nonzero(token_type_ids == type_id, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            selected = x.index_select(1, indices)
            result = result.index_copy(1, indices, expert(selected))
        return result

    def _route_experts(
        self,
        x: torch.Tensor,
        token_type_ids: torch.Tensor,
        experts: nn.ModuleList,
    ) -> torch.Tensor:
        if self.sparse_expert_routing:
            return self._route_sparse(x, token_type_ids, experts)
        return self._route(x, token_type_ids, experts)

    def forward(
        self,
        x: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        normed = self._route_experts(x, token_type_ids, self.attention_norms)
        x = x + self.dropout(self.attention(normed, attention_mask))
        ffn_input = self._route_experts(x, token_type_ids, self.ffn_norms)
        x = x + self.dropout(self._route_experts(ffn_input, token_type_ids, self.ffns))
        return x


class ActionWorldJointTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        gradient_checkpointing: bool,
        *,
        expert_mlp_ratios: tuple[float, float, float] | None = None,
        sparse_expert_routing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.layers = nn.ModuleList(
            [
                ModalityExpertBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    expert_mlp_ratios=expert_mlp_ratios,
                    sparse_expert_routing=sparse_expert_routing,
                )
                for _ in range(int(depth))
            ]
        )
        self.sparse_expert_routing = bool(sparse_expert_routing)
        self.output_norms = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(3)])

    def forward(
        self,
        x: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(layer, x, token_type_ids, attention_mask, use_reentrant=False)
            else:
                x = layer(x, token_type_ids, attention_mask)
        router = (
            ModalityExpertBlock._route_sparse
            if self.sparse_expert_routing
            else ModalityExpertBlock._route
        )
        return router(x, token_type_ids, self.output_norms)


@dataclass
class SequenceResult:
    hidden: torch.Tensor
    slices: dict[str, slice]
    attention_mask: torch.Tensor


class ActionWorldCoFlowModel(nn.Module):
    """One H16 action/future bridge with a shared transformer."""

    def __init__(
        self,
        *,
        context_dim: int,
        world_dim: int,
        action_config,
        coflow_config,
    ) -> None:
        super().__init__()
        self.action_horizon = int(_cfg_get(action_config, "action_horizon", 16))
        self.action_dim = int(_cfg_get(action_config, "action_dim", 14))
        self.state_dim = int(_cfg_get(action_config, "state_dim", 0) or 0)
        if self.action_horizon != 16:
            raise ValueError(
                "Action--World Co-Flow is a closed-loop H16 single-bridge model; "
                f"got action_horizon={self.action_horizon}"
            )

        token_grid = tuple(int(value) for value in _cfg_get(coflow_config, "world_token_grid", [4, 8]))
        self.world_grid = token_grid
        self.num_world_tokens = int(_cfg_get(coflow_config, "num_world_tokens", token_grid[0] * token_grid[1]))
        if token_grid[0] * token_grid[1] != self.num_world_tokens:
            raise ValueError(
                f"num_world_tokens={self.num_world_tokens} disagrees with world_token_grid={token_grid}"
            )
        self.world_dim = int(world_dim)
        self.hidden_size = int(_cfg_get(coflow_config, "hidden_size", 1024))

        self.action_prediction_type = str(_cfg_get(action_config, "prediction_type", "velocity")).lower()
        if self.action_prediction_type not in {"velocity", "jit_x"}:
            raise ValueError(f"Unsupported action prediction_type={self.action_prediction_type!r}")
        self.jit_t_eps = float(_cfg_get(action_config, "jit_t_eps", 5.0e-2))
        self.action_inference_steps = int(_cfg_get(coflow_config, "action_inference_steps", 10))
        self.world_inference_steps = int(_cfg_get(coflow_config, "world_inference_steps", 10))
        self.default_inference_mode = str(
            _cfg_get(coflow_config, "default_inference_mode", "policy")
        ).lower()
        if min(self.action_inference_steps, self.world_inference_steps) <= 0:
            raise ValueError("action_inference_steps and world_inference_steps must be positive")
        if self.default_inference_mode not in {"policy", "diagonal"}:
            raise ValueError(
                "default_inference_mode must be 'policy' or 'diagonal', got "
                f"{self.default_inference_mode!r}"
            )

        bridge_type = str(_cfg_get(coflow_config, "world_bridge_type", "qantara_brownian_bridge")).lower()
        if bridge_type not in {"qantara_brownian_bridge", "qantara_linear_bridge"}:
            raise ValueError(f"Unsupported world_bridge_type={bridge_type!r}")
        bridge_noise = float(_cfg_get(coflow_config, "bridge_noise_scale", 1.0))
        if bridge_type == "qantara_linear_bridge" and bridge_noise != 0.0:
            raise ValueError("qantara_linear_bridge requires bridge_noise_scale=0")
        self.world_bridge = QantaraWorldBridge(
            bridge_noise,
            str(_cfg_get(coflow_config, "world_prediction_type", "qantara_x_delta")),
        )

        self.action_loss_weight = float(_cfg_get(coflow_config, "action_loss_weight", 1.0))
        self.world_loss_weight = float(_cfg_get(coflow_config, "world_loss_weight", 0.1))
        if min(self.action_loss_weight, self.world_loss_weight) < 0:
            raise ValueError("action_loss_weight and world_loss_weight must be non-negative")

        world_loss_schedule = _cfg_get(coflow_config, "world_loss_schedule", {}) or {}
        self.world_loss_schedule_enabled = bool(
            _cfg_get(world_loss_schedule, "enabled", False)
        )
        self.world_loss_schedule_transition_step = int(
            _cfg_get(world_loss_schedule, "transition_step", 0)
        )
        self.world_loss_schedule_before = float(
            _cfg_get(world_loss_schedule, "before_weight", self.world_loss_weight)
        )
        self.world_loss_schedule_after = float(
            _cfg_get(world_loss_schedule, "after_weight", self.world_loss_weight)
        )
        if self.world_loss_schedule_enabled:
            if self.world_loss_schedule_transition_step <= 0:
                raise ValueError(
                    "world_loss_schedule.transition_step must be positive when enabled"
                )
            if min(self.world_loss_schedule_before, self.world_loss_schedule_after) < 0:
                raise ValueError("world-loss schedule weights must be non-negative")
            if abs(self.world_loss_schedule_before - self.world_loss_weight) > 1.0e-12:
                raise ValueError(
                    "world_loss_weight must equal world_loss_schedule.before_weight so the "
                    "saved config has one unambiguous initial weight"
                )

        self.log_attention_statistics = bool(_cfg_get(coflow_config, "log_attention_statistics", True))
        mot_config = _cfg_get(coflow_config, "mot", {}) or {}
        self.mot_enabled = bool(_cfg_get(mot_config, "enabled", False))
        if self.mot_enabled:
            self.mot_expert_mlp_ratios = (
                float(_cfg_get(mot_config, "context_mlp_ratio", 2.0)),
                float(_cfg_get(mot_config, "action_mlp_ratio", 10.5)),
                float(_cfg_get(mot_config, "world_mlp_ratio", 3.5)),
            )
            if any(value <= 0 for value in self.mot_expert_mlp_ratios):
                raise ValueError("all mot expert MLP ratios must be positive")
            self.mot_sparse_routing = bool(
                _cfg_get(mot_config, "sparse_routing", True)
            )
        else:
            self.mot_expert_mlp_ratios = None
            self.mot_sparse_routing = False
        self.action_input_projection = nn.Linear(self.action_dim, self.hidden_size)
        self.world_input_projection = nn.Linear(self.world_dim, self.hidden_size)
        self.context_projection = nn.Linear(int(context_dim), self.hidden_size)
        self.state_projection = (
            nn.Sequential(
                nn.Linear(self.state_dim, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            if self.state_dim > 0
            else None
        )
        self.token_type_embedding = nn.Embedding(3, self.hidden_size)
        self.block_embedding = nn.Embedding(2, self.hidden_size)
        self.action_position_embedding = nn.Embedding(self.action_horizon, self.hidden_size)
        self.world_row_embedding = nn.Embedding(token_grid[0], self.hidden_size)
        self.world_col_embedding = nn.Embedding(token_grid[1], self.hidden_size)
        self.time_embedding = ContinuousTimeEmbedding(
            self.hidden_size,
            int(_cfg_get(coflow_config, "time_frequency_dim", 256)),
        )
        dropout = float(_cfg_get(coflow_config, "dropout", 0.1))
        self.input_dropout = nn.Dropout(dropout)
        self.transformer = ActionWorldJointTransformer(
            hidden_size=self.hidden_size,
            depth=int(_cfg_get(coflow_config, "num_layers", 12)),
            num_heads=int(_cfg_get(coflow_config, "num_attention_heads", 16)),
            mlp_ratio=float(_cfg_get(coflow_config, "mlp_ratio", 4.0)),
            dropout=dropout,
            gradient_checkpointing=bool(_cfg_get(coflow_config, "enable_gradient_checkpointing", True)),
            expert_mlp_ratios=self.mot_expert_mlp_ratios,
            sparse_expert_routing=self.mot_sparse_routing,
        )
        self.action_output_norm = nn.RMSNorm(self.hidden_size)
        self.world_output_norm = nn.RMSNorm(self.hidden_size)
        self.action_output_head = nn.Linear(self.hidden_size, self.action_dim)
        self.world_output_head = nn.Linear(self.hidden_size, self.world_dim)
        if self.world_bridge.prediction_type == "qantara_x_delta":
            # Qantara's delta head starts at the identity bridge prior.  This is
            # an output parameterization, not a scalar gate between modalities.
            nn.init.zeros_(self.world_output_head.weight)
            nn.init.zeros_(self.world_output_head.bias)

        self.capacity_report = self._build_capacity_report()
        minimum_action_path_capacity = int(
            _cfg_get(mot_config, "minimum_action_path_parameters", 0)
        )
        minimum_action_expert_capacity = int(
            _cfg_get(mot_config, "minimum_action_expert_parameters", 0)
        )
        if self.mot_enabled and minimum_action_path_capacity > 0:
            actual_action_path_capacity = self.capacity_report["action_path_parameters"]
            if actual_action_path_capacity < minimum_action_path_capacity:
                raise ValueError(
                    "asymmetric MoT action capacity is below the configured baseline: "
                    f"actual={actual_action_path_capacity:,}, "
                    f"minimum={minimum_action_path_capacity:,}"
                )
        if self.mot_enabled and minimum_action_expert_capacity > 0:
            actual_action_expert_capacity = self.capacity_report["action_expert_parameters"]
            if actual_action_expert_capacity < minimum_action_expert_capacity:
                raise ValueError(
                    "asymmetric MoT action-only expert capacity is below the configured "
                    "baseline: "
                    f"actual={actual_action_expert_capacity:,}, "
                    f"minimum={minimum_action_expert_capacity:,}"
                )
        target_world_ratio = _cfg_get(
            mot_config, "target_world_to_action_expert_ratio", None
        )
        if self.mot_enabled and target_world_ratio is not None:
            target_world_ratio = float(target_world_ratio)
            if target_world_ratio <= 0:
                raise ValueError(
                    "mot.target_world_to_action_expert_ratio must be positive"
                )
            actual_world_ratio = self.capacity_report[
                "world_to_action_expert_parameter_ratio"
            ]
            if abs(actual_world_ratio - target_world_ratio) > 1.0e-3:
                raise ValueError(
                    "asymmetric MoT world/action expert capacity ratio disagrees with the "
                    "configured target: "
                    f"actual={actual_world_ratio:.6f}, target={target_world_ratio:.6f}"
                )

        ratios = _cfg_get(coflow_config, "noise_plane_sampling", {})
        self.noise_plane = NoisePlaneSampler(
            ratios,
            action_timestep_sampling=str(
                _cfg_get(coflow_config, "action_timestep_sampling", "starvla_gr00t")
            ),
            world_timestep_sampling=str(
                _cfg_get(coflow_config, "world_timestep_sampling", "qantara_monotone")
            ),
            action_beta_alpha=float(_cfg_get(action_config, "noise_beta_alpha", 1.5)),
            action_beta_beta=float(_cfg_get(action_config, "noise_beta_beta", 1.0)),
            action_noise_s=float(_cfg_get(action_config, "noise_s", 0.999)),
        )

        self.use_correlated_noise = bool(_cfg_get(action_config, "use_correlated_noise", False))
        flat_dim = self.action_horizon * self.action_dim
        self.register_buffer("_action_corr_chol", torch.zeros(flat_dim, flat_dim), persistent=False)
        self._action_corr_loaded = False

    @staticmethod
    def _parameter_count(module: nn.Module | None) -> int:
        return 0 if module is None else sum(parameter.numel() for parameter in module.parameters())

    def _build_capacity_report(self) -> dict[str, int | float]:
        """Report unique parameter budgets for the asymmetric expert paths."""

        shared_attention = sum(
            self._parameter_count(layer.attention) for layer in self.transformer.layers
        )
        context_expert = sum(
            self._parameter_count(layer.ffns[TOKEN_CONTEXT])
            for layer in self.transformer.layers
        )
        action_expert = sum(
            self._parameter_count(layer.ffns[TOKEN_ACTION])
            for layer in self.transformer.layers
        )
        world_expert = sum(
            self._parameter_count(layer.ffns[TOKEN_WORLD])
            for layer in self.transformer.layers
        )
        action_norms = sum(
            self._parameter_count(layer.attention_norms[TOKEN_ACTION])
            + self._parameter_count(layer.ffn_norms[TOKEN_ACTION])
            for layer in self.transformer.layers
        ) + self._parameter_count(self.transformer.output_norms[TOKEN_ACTION])
        action_conditioning = sum(
            self._parameter_count(module)
            for module in (
                self.action_input_projection,
                self.action_output_norm,
                self.action_output_head,
                self.state_projection,
                self.action_position_embedding,
                self.time_embedding,
                self.token_type_embedding,
                self.block_embedding,
            )
        )
        return {
            "shared_attention_parameters": shared_attention,
            "context_expert_parameters": context_expert,
            "action_expert_parameters": action_expert,
            "world_expert_parameters": world_expert,
            "world_to_action_expert_parameter_ratio": (
                world_expert / action_expert if action_expert else 0.0
            ),
            "action_path_parameters": (
                shared_attention + action_expert + action_norms + action_conditioning
            ),
            "total_parameters": self._parameter_count(self),
        }

    def world_loss_weight_at_step(self, global_step: int) -> float:
        """Return the checkpoint-configured world weight for one optimizer step."""

        if not self.world_loss_schedule_enabled:
            return self.world_loss_weight
        return (
            self.world_loss_schedule_before
            if int(global_step) < self.world_loss_schedule_transition_step
            else self.world_loss_schedule_after
        )

    def set_action_correlation(self, chol: torch.Tensor) -> None:
        chol = torch.as_tensor(chol, dtype=torch.float32, device="cpu")
        if tuple(chol.shape) != tuple(self._action_corr_chol.shape):
            raise ValueError(
                f"Action-correlation Cholesky shape={tuple(chol.shape)}, "
                f"expected={tuple(self._action_corr_chol.shape)}"
            )
        if not bool(torch.isfinite(chol).all()) or not torch.allclose(
            chol, torch.tril(chol), rtol=0.0, atol=1.0e-6
        ):
            raise ValueError("invalid action-correlation Cholesky factor")
        if not bool((torch.diagonal(chol) > 0).all()):
            raise ValueError("action-correlation Cholesky diagonal must be positive")
        self._action_corr_chol.copy_(chol.to(self._action_corr_chol.device))
        self._action_corr_loaded = True

    def _sample_action_noise(
        self,
        batch_size: int,
        device,
        dtype,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not self.use_correlated_noise:
            return torch.randn(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        if not self._action_corr_loaded:
            raise RuntimeError(
                "use_correlated_noise=true but no Cholesky factor was injected before Action--World Co-Flow"
            )
        base = torch.randn(
            batch_size,
            self.action_horizon * self.action_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        chol = self._action_corr_chol.to(device=device, dtype=dtype)
        return (base @ chol.T).reshape(batch_size, self.action_horizon, self.action_dim)

    def _world_grid_position(self, device, dtype) -> torch.Tensor:
        rows = torch.arange(self.world_grid[0], device=device).repeat_interleave(self.world_grid[1])
        cols = torch.arange(self.world_grid[1], device=device).repeat(self.world_grid[0])
        return (self.world_row_embedding(rows) + self.world_col_embedding(cols)).to(dtype=dtype)

    def _prepare_state(self, state: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
        if self.state_projection is None:
            return None
        if state is None:
            raise ValueError("Action--World Co-Flow was configured with state_dim>0 but the batch has no state")
        if state.shape[0] != batch_size or state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state must have batch={batch_size}, dim={self.state_dim}; got {tuple(state.shape)}"
            )
        return state.reshape(batch_size, -1, self.state_dim)[:, -1]

    def _assemble_sequence(
        self,
        *,
        context: torch.Tensor,
        context_valid: torch.Tensor,
        state: torch.Tensor | None,
        z0: torch.Tensor,
        action: torch.Tensor,
        future: torch.Tensor,
        tau_action: torch.Tensor,
        tau_world: torch.Tensor,
        action_valid: torch.Tensor | None = None,
    ) -> SequenceResult:
        batch, context_length, _ = context.shape
        if context_valid.shape != (batch, context_length):
            raise ValueError(
                f"context_valid must be {(batch, context_length)}, got {tuple(context_valid.shape)}"
            )
        if z0.shape != (batch, self.num_world_tokens, self.world_dim):
            raise ValueError(f"z0 has shape {tuple(z0.shape)}, expected [B,{self.num_world_tokens},{self.world_dim}]")
        state = self._prepare_state(state, batch)
        device, dtype = context.device, context.dtype
        world_pos = self._world_grid_position(device, dtype).unsqueeze(0)

        pieces: list[torch.Tensor] = []
        type_pieces: list[torch.Tensor] = []
        block_pieces: list[torch.Tensor] = []
        time_pieces: list[torch.Tensor] = []
        valid_pieces: list[torch.Tensor] = []
        slices: dict[str, slice] = {}
        cursor = 0

        def append(name: str, value: torch.Tensor, token_type: int, block_id: int, tau, valid=None):
            nonlocal cursor
            length = value.shape[1]
            pieces.append(value)
            type_pieces.append(torch.full((length,), token_type, device=device, dtype=torch.long))
            block_pieces.append(torch.full((length,), block_id, device=device, dtype=torch.long))
            tau_tensor = torch.as_tensor(tau, device=device, dtype=dtype)
            if tau_tensor.ndim == 0:
                tau_tensor = tau_tensor.expand(batch)
            time_pieces.append(tau_tensor.reshape(batch, 1).expand(batch, length))
            if valid is None:
                valid = torch.ones(batch, length, device=device, dtype=torch.bool)
            else:
                valid = torch.as_tensor(valid, device=device, dtype=torch.bool)
                if tuple(valid.shape) != (batch, length):
                    raise ValueError(
                        f"valid-key mask for {name} must be {(batch, length)}, got {tuple(valid.shape)}"
                    )
            valid_pieces.append(valid)
            slices[name] = slice(cursor, cursor + length)
            cursor += length

        context_value = self.context_projection(context)
        append("context", context_value, TOKEN_CONTEXT, 0, 1.0, context_valid.to(torch.bool))
        if state is not None:
            append("state", self.state_projection(state).unsqueeze(1), TOKEN_CONTEXT, 0, 1.0)
        append("z0", self.world_input_projection(z0) + world_pos, TOKEN_WORLD, 0, 1.0)

        action_position = self.action_position_embedding(
            torch.arange(self.action_horizon, device=device)
        ).unsqueeze(0)
        append(
            "action",
            self.action_input_projection(action) + action_position,
            TOKEN_ACTION,
            1,
            tau_action,
            valid=action_valid,
        )
        append(
            "future",
            self.world_input_projection(future) + world_pos,
            TOKEN_WORLD,
            1,
            tau_world,
        )

        token_type_ids = torch.cat(type_pieces, dim=0)
        block_ids = torch.cat(block_pieces, dim=0)
        key_valid = torch.cat(valid_pieces, dim=1)
        tau_tokens = torch.cat(time_pieces, dim=1)
        token_type = self.token_type_embedding(token_type_ids).unsqueeze(0)
        block_type = self.block_embedding(block_ids).unsqueeze(0)
        x = torch.cat(pieces, dim=1) + token_type + block_type + self.time_embedding(tau_tokens).to(dtype)
        x = self.input_dropout(x)
        attention_mask = build_block_causal_attention_mask(block_ids, key_valid_mask=key_valid)
        hidden = self.transformer(x, token_type_ids, attention_mask)
        return SequenceResult(hidden=hidden, slices=slices, attention_mask=attention_mask)

    def _action_outputs(
        self,
        hidden: torch.Tensor,
        noisy_action: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.action_output_head(self.action_output_norm(hidden))
        tau_b = tau[:, None, None].to(device=noisy_action.device, dtype=noisy_action.dtype)
        velocity = action_prediction_to_velocity(
            raw,
            noisy_action,
            tau_b,
            prediction_type=self.action_prediction_type,
            t_eps=self.jit_t_eps,
        )
        if self.action_prediction_type == "jit_x":
            clean = raw
        else:
            clean = noisy_action + (1.0 - tau_b) * velocity
        return raw, velocity, clean

    def _world_outputs(self, hidden: torch.Tensor, start: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.world_output_head(self.world_output_norm(hidden))
        return raw, self.world_bridge.clean_from_prediction(raw, start)

    @staticmethod
    def _masked_action_loss(error: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        per_step = error.float().mean(dim=-1)
        valid_f = valid.to(device=per_step.device, dtype=per_step.dtype)
        valid_count = valid_f.sum(dim=1)
        per_sample = (per_step * valid_f).sum(dim=1) / valid_count.clamp_min(1.0)
        # Forward-edge rows deliberately have tau_a=1 and therefore no action
        # supervision.  Exclude those zero-valid rows from the action-loss
        # denominator (Qantara's mask-normalized convention) instead of
        # silently shrinking the policy objective by forward_ratio.
        active = (valid_count > 0).to(per_sample.dtype)
        return (per_sample * active).sum() / active.sum().clamp_min(1.0)

    @staticmethod
    def _masked_world_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        per_sample = (prediction.float() - target.float()).pow(2).mean(dim=(1, 2))
        valid_f = valid.to(device=per_sample.device, dtype=per_sample.dtype)
        return (per_sample * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    def forward_train(
        self,
        *,
        context: torch.Tensor,
        context_valid: torch.Tensor,
        z0: torch.Tensor,
        z16_target: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor | None,
        action_is_pad: torch.Tensor | None,
        future_valid_16: torch.Tensor,
        global_step: int,
    ) -> dict[str, torch.Tensor]:
        batch = actions.shape[0]
        expected_actions = (batch, self.action_horizon, self.action_dim)
        if tuple(actions.shape) != expected_actions:
            raise ValueError(f"actions must have shape {expected_actions}, got {tuple(actions.shape)}")
        if action_is_pad is None:
            action_is_pad = torch.zeros(batch, self.action_horizon, device=actions.device, dtype=torch.bool)
        else:
            action_is_pad = action_is_pad.to(device=actions.device, dtype=torch.bool)
            if tuple(action_is_pad.shape) != (batch, self.action_horizon):
                raise ValueError(
                    f"action_is_pad must be [B,{self.action_horizon}], got {tuple(action_is_pad.shape)}"
                )

        future_valid_16 = future_valid_16.reshape(batch).to(
            device=actions.device, dtype=torch.bool
        )
        plane = self.noise_plane.sample(batch, actions.device, actions.dtype)
        action_noise = self._sample_action_noise(batch, actions.device, actions.dtype)
        world_noise = torch.randn_like(z16_target)
        tau_action = plane.tau_action
        tau_world = torch.where(
            future_valid_16,
            plane.tau_world,
            torch.zeros_like(plane.tau_world),
        )
        noisy_action = (
            (1.0 - tau_action[:, None, None]) * action_noise
            + tau_action[:, None, None] * actions
        )
        noisy_future = self.world_bridge.interpolate(
            z0, z16_target, tau_world, world_noise
        )
        bridge = self._assemble_sequence(
            context=context,
            context_valid=context_valid,
            state=state,
            z0=z0,
            action=noisy_action,
            future=noisy_future,
            tau_action=tau_action,
            tau_world=tau_world,
            action_valid=~action_is_pad,
        )
        _, action_velocity, _ = self._action_outputs(
            bridge.hidden[:, bridge.slices["action"]], noisy_action, tau_action
        )
        _, z16_prediction = self._world_outputs(
            bridge.hidden[:, bridge.slices["future"]], z0
        )

        if self.action_prediction_type == "velocity":
            target_velocity = actions - action_noise
        else:
            # This exactly mirrors FlowmatchingActionHead's JiT clean-x
            # target conversion, including its near-endpoint denominator cap.
            target_velocity = action_prediction_to_velocity(
                actions,
                noisy_action,
                tau_action[:, None, None],
                prediction_type="jit_x",
                t_eps=self.jit_t_eps,
            )

        action_error = (action_velocity - target_velocity).pow(2)
        action_time_valid = (tau_action < 1.0 - 1.0e-6)[:, None].expand(
            -1, self.action_horizon
        )
        action_valid = (~action_is_pad) & action_time_valid
        action_loss = self._masked_action_loss(action_error, action_valid)
        z16_valid = future_valid_16 & (tau_world < 1.0 - 1.0e-6)
        z16_loss = self._masked_world_loss(z16_prediction, z16_target, z16_valid)
        world_loss = z16_loss
        weighted_action_loss = self.action_loss_weight * action_loss
        active_world_loss_weight = self.world_loss_weight_at_step(global_step)
        weighted_world_loss = active_world_loss_weight * world_loss
        total = weighted_action_loss + weighted_world_loss

        metrics: dict[str, torch.Tensor] = {
            "action_loss": total,
            "coflow_action_loss_raw": action_loss.detach(),
            "coflow_action_loss_weighted": weighted_action_loss.detach(),
            "coflow_world_loss_raw": world_loss.detach(),
            "coflow_world_loss_weighted": weighted_world_loss.detach(),
            "coflow_world_to_action_weighted_ratio": (
                weighted_world_loss.detach().abs() / weighted_action_loss.detach().abs().clamp_min(1.0e-12)
            ),
            "coflow_z16_loss_raw": z16_loss.detach(),
            "coflow_future_valid_16_ratio": future_valid_16.float().mean().detach(),
            "coflow_z16_valid_ratio": z16_valid.float().mean().detach(),
            "coflow_action_nonpad_ratio": (~action_is_pad).float().mean().detach(),
            "coflow_action_valid_ratio": action_valid.float().mean().detach(),
        }
        if self.world_loss_schedule_enabled:
            metrics["coflow_world_loss_weight"] = total.new_tensor(
                active_world_loss_weight
            )
        for name, fraction in self.noise_plane.fractions(plane.mode_ids).items():
            metrics[f"coflow_mode_{name}_ratio"] = fraction.detach()
        if self.log_attention_statistics:
            metrics["coflow_attention_allowed_fraction"] = (
                bridge.attention_mask.float().mean().detach()
            )
        return metrics

    def _sample_bridge(
        self,
        *,
        context: torch.Tensor,
        context_valid: torch.Tensor,
        state: torch.Tensor | None,
        z0: torch.Tensor,
        action_noise: torch.Tensor,
        inference_mode: str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one physical 16-step bridge.

        ``policy`` follows the policy edge of the training time plane:
        :math:`tau_a:0\rightarrow1` while :math:`tau_z=0`.  The world head is
        evaluated but its state is never fed back into action integration.

        ``diagonal`` follows the tied path :math:`tau_a=tau_z`; action and
        future state are advanced together.
        """

        inference_mode = str(inference_mode).lower()
        if inference_mode not in {"policy", "diagonal"}:
            raise ValueError(
                f"inference_mode must be 'policy' or 'diagonal', got {inference_mode!r}"
            )
        batch = context.shape[0]
        action = action_noise
        world = z0
        predicted_world = z0

        if inference_mode == "diagonal" and self.action_inference_steps != self.world_inference_steps:
            raise ValueError(
                "diagonal inference requires equal action/world integration steps so tau_a=tau_z "
                f"exactly, got action={self.action_inference_steps}, world={self.world_inference_steps}"
            )
        steps = self.action_inference_steps
        for index in range(steps):
            action_tau = index / float(steps)
            world_tau = action_tau if inference_mode == "diagonal" else 0.0
            tau_a = torch.full((batch,), action_tau, device=context.device, dtype=context.dtype)
            tau_z = torch.full((batch,), world_tau, device=context.device, dtype=context.dtype)
            result = self._assemble_sequence(
                context=context,
                context_valid=context_valid,
                state=state,
                z0=z0,
                action=action,
                future=world,
                tau_action=tau_a,
                tau_world=tau_z,
            )
            _, velocity, _ = self._action_outputs(
                result.hidden[:, result.slices["action"]], action, tau_a
            )
            _, clean_world = self._world_outputs(
                result.hidden[:, result.slices["future"]], z0
            )
            predicted_world = clean_world
            next_tau = (index + 1) / float(steps)
            action = action + (next_tau - action_tau) * velocity
            if inference_mode == "diagonal":
                marginal_noise = (
                    torch.randn(
                        z0.shape,
                        device=z0.device,
                        dtype=z0.dtype,
                        generator=generator,
                    )
                    if generator is not None and next_tau < 1.0 - 1.0e-12
                    else None
                )
                world = self.world_bridge.reproject(
                    z0,
                    clean_world,
                    next_tau,
                    add_marginal_noise=next_tau < 1.0 - 1.0e-12,
                    noise=marginal_noise,
                )
        return action, world if inference_mode == "diagonal" else predicted_world

    def sample_actions(
        self,
        *,
        context: torch.Tensor,
        context_valid: torch.Tensor,
        z0: torch.Tensor,
        state: torch.Tensor | None,
        inference_mode: str | None = None,
        output_horizon: int | None = None,
        inference_seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer one H16 action/future bridge from the current observation."""

        mode = self.default_inference_mode if inference_mode is None else str(inference_mode).lower()
        horizon = self.action_horizon if output_horizon is None else int(output_horizon)
        if mode not in {"policy", "diagonal"}:
            raise ValueError(f"Unsupported Co-Flow inference_mode={mode!r}")
        if horizon != self.action_horizon:
            raise ValueError(f"single-bridge Co-Flow output_horizon must be 16, got {horizon}")

        generator = None
        if inference_seed is not None:
            if int(inference_seed) < 0:
                raise ValueError(f"inference_seed must be non-negative, got {inference_seed}")
            generator = torch.Generator(device=context.device)
            generator.manual_seed(int(inference_seed))
        noise = self._sample_action_noise(
            context.shape[0], context.device, context.dtype, generator=generator
        )
        actions, z16_prediction = self._sample_bridge(
            context=context,
            context_valid=context_valid,
            state=state,
            z0=z0,
            action_noise=noise,
            inference_mode=mode,
            generator=generator,
        )
        return actions, z16_prediction
