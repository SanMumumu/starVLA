"""Shared-transformer Action--World Co-Flow model.

The model performs two physically aligned calls with shared weights:

1. ``[context+Z0] [A1:16, Z16]`` predicts the first action/world block.
2. ``[context+Z0] [A1:16_hat, Z16_source] [A17:32, Z32]`` predicts the
   second block.  At evaluation ``Z16_source`` is always the model prediction.

Within every block action and world tokens use the same self-attention.  FFNs
and normalization remain modality-specific (an MM-DiT style compromise).
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
from .noise_plane import MODE_TO_ID, NoisePlaneSampler


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
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))


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
    """Shared attention plus context/action/world-specific norm and FFN."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        intermediate = int(round(hidden_size * float(mlp_ratio)))
        self.attention_norms = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(3)])
        self.ffn_norms = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(3)])
        self.attention = SharedSelfAttention(hidden_size, num_heads, dropout)
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, intermediate),
                    nn.GELU(approximate="tanh"),
                    nn.Dropout(dropout),
                    nn.Linear(intermediate, hidden_size),
                )
                for _ in range(3)
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

    def forward(
        self,
        x: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        normed = self._route(x, token_type_ids, self.attention_norms)
        x = x + self.dropout(self.attention(normed, attention_mask))
        ffn_input = self._route(x, token_type_ids, self.ffn_norms)
        x = x + self.dropout(self._route(ffn_input, token_type_ids, self.ffns))
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
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.layers = nn.ModuleList(
            [ModalityExpertBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(int(depth))]
        )
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
        return ModalityExpertBlock._route(x, token_type_ids, self.output_norms)


@dataclass
class SequenceResult:
    hidden: torch.Tensor
    slices: dict[str, slice]
    attention_mask: torch.Tensor


class ActionWorldCoFlowModel(nn.Module):
    """Two-block action/world flow model with a single shared transformer."""

    def __init__(
        self,
        *,
        context_dim: int,
        world_dim: int,
        action_config,
        coflow_config,
    ) -> None:
        super().__init__()
        self.action_horizon = int(_cfg_get(action_config, "action_horizon", 32))
        self.action_dim = int(_cfg_get(action_config, "action_dim", 14))
        self.state_dim = int(_cfg_get(action_config, "state_dim", 0) or 0)
        self.segment_boundaries = tuple(int(value) for value in _cfg_get(coflow_config, "segment_boundaries", [16, 32]))
        if self.action_horizon != 32 or self.segment_boundaries != (16, 32):
            raise ValueError(
                "RoboTwin Action--World Co-Flow currently requires action_horizon=32 and "
                f"segment_boundaries=[16,32], got H={self.action_horizon}, segments={self.segment_boundaries}"
            )
        self.segment_size = 16

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
        if min(self.action_inference_steps, self.world_inference_steps) <= 0:
            raise ValueError("action_inference_steps and world_inference_steps must be positive")

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
        self.z16_loss_weight = float(_cfg_get(coflow_config, "z16_loss_weight", 1.0))
        self.z32_loss_weight = float(_cfg_get(coflow_config, "z32_loss_weight", 1.0))
        if min(self.action_loss_weight, self.world_loss_weight, self.z16_loss_weight, self.z32_loss_weight) < 0:
            raise ValueError("all action/world loss weights must be non-negative")

        self.intermediate_state_source = str(
            _cfg_get(coflow_config, "intermediate_state_source", "scheduled")
        ).lower()
        valid_sources = {"ground_truth", "predicted_detach", "predicted_e2e", "scheduled"}
        if self.intermediate_state_source not in valid_sources:
            raise ValueError(
                f"Unsupported intermediate_state_source={self.intermediate_state_source!r}; "
                f"expected one of {sorted(valid_sources)}"
            )
        self.predicted_z16_detach = bool(_cfg_get(coflow_config, "predicted_z16_detach", True))
        self.predicted_action_prefix_detach = bool(
            _cfg_get(coflow_config, "predicted_action_prefix_detach", True)
        )
        self.z16_teacher_ratio_start = float(_cfg_get(coflow_config, "z16_teacher_ratio_start", 1.0))
        self.z16_teacher_ratio_end = float(_cfg_get(coflow_config, "z16_teacher_ratio_end", 0.0))
        self.z16_teacher_decay_steps = int(_cfg_get(coflow_config, "z16_teacher_decay_steps", 30000))
        if not 0 <= self.z16_teacher_ratio_start <= 1 or not 0 <= self.z16_teacher_ratio_end <= 1:
            raise ValueError("z16 teacher ratios must be in [0,1]")
        if self.z16_teacher_decay_steps <= 0:
            raise ValueError("z16_teacher_decay_steps must be positive")

        self.log_attention_statistics = bool(_cfg_get(coflow_config, "log_attention_statistics", True))
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
        self.block_embedding = nn.Embedding(3, self.hidden_size)
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

    def _sample_action_noise(self, batch_size: int, device, dtype) -> torch.Tensor:
        if not self.use_correlated_noise:
            return torch.randn(batch_size, self.action_horizon, self.action_dim, device=device, dtype=dtype)
        if not self._action_corr_loaded:
            raise RuntimeError(
                "use_correlated_noise=true but no Cholesky factor was injected before Action--World Co-Flow"
            )
        base = torch.randn(batch_size, self.action_horizon * self.action_dim, device=device, dtype=dtype)
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
        a1: torch.Tensor,
        z16: torch.Tensor,
        tau_a1: torch.Tensor,
        tau_z16: torch.Tensor,
        a2: torch.Tensor | None = None,
        z32: torch.Tensor | None = None,
        tau_a2: torch.Tensor | None = None,
        tau_z32: torch.Tensor | None = None,
        block1_clean_prefix: bool = False,
        a1_valid: torch.Tensor | None = None,
        a2_valid: torch.Tensor | None = None,
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

        action_pos1 = self.action_position_embedding(torch.arange(0, 16, device=device)).unsqueeze(0)
        append(
            "a1",
            self.action_input_projection(a1) + action_pos1,
            TOKEN_ACTION,
            1,
            1.0 if block1_clean_prefix else tau_a1,
            valid=a1_valid,
        )
        append(
            "z16",
            self.world_input_projection(z16) + world_pos,
            TOKEN_WORLD,
            1,
            1.0 if block1_clean_prefix else tau_z16,
        )

        has_block2 = any(value is not None for value in (a2, z32, tau_a2, tau_z32))
        if has_block2:
            if any(value is None for value in (a2, z32, tau_a2, tau_z32)):
                raise ValueError("block 2 requires a2, z32, tau_a2, and tau_z32 together")
            action_pos2 = self.action_position_embedding(torch.arange(16, 32, device=device)).unsqueeze(0)
            append(
                "a2",
                self.action_input_projection(a2) + action_pos2,
                TOKEN_ACTION,
                2,
                tau_a2,
                valid=a2_valid,
            )
            append("z32", self.world_input_projection(z32) + world_pos, TOKEN_WORLD, 2, tau_z32)

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

    def teacher_ratio(self, global_step: int) -> float:
        progress = min(max(int(global_step), 0) / float(self.z16_teacher_decay_steps), 1.0)
        return self.z16_teacher_ratio_start + progress * (
            self.z16_teacher_ratio_end - self.z16_teacher_ratio_start
        )

    def _choose_intermediate_source(
        self,
        z16_target: torch.Tensor,
        z16_prediction: torch.Tensor,
        future_valid_16: torch.Tensor,
        global_step: int,
        teacher_eligible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        batch = z16_target.shape[0]
        future_valid_16 = future_valid_16.reshape(batch).to(device=z16_target.device, dtype=torch.bool)
        if teacher_eligible is None:
            teacher_eligible = future_valid_16
        else:
            teacher_eligible = teacher_eligible.reshape(batch).to(
                device=z16_target.device, dtype=torch.bool
            ) & future_valid_16
        source = self.intermediate_state_source
        if source == "ground_truth":
            # An episode-tail clamp isn't a real teacher target.  Fall back to
            # a detached model prediction for those samples so padded future
            # images can neither leak into valid action tokens nor create an
            # accidental Stage-2 -> Stage-1 gradient path in this ablation.
            teacher_mask = future_valid_16
            mixed = torch.where(
                teacher_mask[:, None, None],
                z16_target,
                z16_prediction.detach(),
            )
            return mixed, teacher_mask, 1.0
        if source == "predicted_e2e":
            teacher_mask = torch.zeros(batch, device=z16_target.device, dtype=torch.bool)
            return z16_prediction, teacher_mask, 0.0
        prediction = z16_prediction.detach() if (source == "predicted_detach" or self.predicted_z16_detach) else z16_prediction
        if source == "predicted_detach":
            teacher_mask = torch.zeros(batch, device=z16_target.device, dtype=torch.bool)
            return prediction, teacher_mask, 0.0

        configured_ratio = self.teacher_ratio(global_step)
        teacher_mask = (torch.rand(batch, device=z16_target.device) < configured_ratio) & teacher_eligible
        mixed = torch.where(teacher_mask[:, None, None], z16_target, prediction)
        return mixed, teacher_mask, configured_ratio

    def forward_train(
        self,
        *,
        context: torch.Tensor,
        context_valid: torch.Tensor,
        z0: torch.Tensor,
        z16_target: torch.Tensor,
        z32_target: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor | None,
        action_is_pad: torch.Tensor | None,
        future_valid_16: torch.Tensor,
        future_valid_32: torch.Tensor,
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
                raise ValueError(f"action_is_pad must be [B,32], got {tuple(action_is_pad.shape)}")

        future_valid_16 = future_valid_16.reshape(batch).to(device=actions.device, dtype=torch.bool)
        future_valid_32 = future_valid_32.reshape(batch).to(device=actions.device, dtype=torch.bool)
        if bool((future_valid_32 & ~future_valid_16).any()):
            raise ValueError("future_valid_32=true requires future_valid_16=true for an ordered episode")

        plane = self.noise_plane.sample(batch, actions.device, actions.dtype)
        action_noise = self._sample_action_noise(batch, actions.device, actions.dtype)
        world_noise16 = torch.randn_like(z16_target)
        world_noise32 = torch.randn_like(z32_target)
        a1_target, a2_target = actions[:, :16], actions[:, 16:]
        eps1, eps2 = action_noise[:, :16], action_noise[:, 16:]
        tau_a1, tau_a2 = plane.tau_action[:, 0], plane.tau_action[:, 1]
        tau_z16 = torch.where(future_valid_16, plane.tau_world[:, 0], torch.zeros_like(plane.tau_world[:, 0]))
        tau_z32 = torch.where(future_valid_32, plane.tau_world[:, 1], torch.zeros_like(plane.tau_world[:, 1]))

        a1_noisy = (1.0 - tau_a1[:, None, None]) * eps1 + tau_a1[:, None, None] * a1_target
        z16_noisy = self.world_bridge.interpolate(z0, z16_target, tau_z16, world_noise16)
        stage1 = self._assemble_sequence(
            context=context,
            context_valid=context_valid,
            state=state,
            z0=z0,
            a1=a1_noisy,
            z16=z16_noisy,
            tau_a1=tau_a1,
            tau_z16=tau_z16,
            a1_valid=~action_is_pad[:, :16],
        )
        _, a1_velocity, a1_prediction = self._action_outputs(
            stage1.hidden[:, stage1.slices["a1"]], a1_noisy, tau_a1
        )
        _, z16_prediction = self._world_outputs(stage1.hidden[:, stage1.slices["z16"]], z0)

        forward_edge = plane.mode_ids == MODE_TO_ID["forward"]
        inverse_edge = plane.mode_ids == MODE_TO_ID["inverse"]
        policy_edge = plane.mode_ids == MODE_TO_ID["policy"]
        # The policy locus must generate the *entire* 32-step action without
        # clean future information, including Block 2.  It therefore uses
        # predicted Z16 from step 0 even while scheduled teacher forcing is
        # active for other eligible modes.  Inverse has its own explicit clean
        # transition semantics and is applied immediately below.
        teacher_eligible = future_valid_16 & ~policy_edge & ~inverse_edge
        z16_source, teacher_mask, configured_teacher_ratio = self._choose_intermediate_source(
            z16_target,
            z16_prediction,
            future_valid_16,
            global_step,
            teacher_eligible=teacher_eligible,
        )
        # Preserve the exact semantics of the two clean conditioning edges.
        # Forward mode supplies clean actions, so Block 2 must see clean A1
        # rather than an unsupervised endpoint-head estimate at tau_a=1.
        # Inverse mode supplies the clean transition, so its valid Z16 prefix
        # is GT by definition.  All other modes retain the configured
        # scheduled/predicted sequential exposure used at policy inference.
        predicted_a1_prefix = a1_prediction.detach() if self.predicted_action_prefix_detach else a1_prediction
        a1_prefix = torch.where(forward_edge[:, None, None], a1_target, predicted_a1_prefix)
        inverse_clean_z16 = inverse_edge & future_valid_16
        z16_source = torch.where(inverse_clean_z16[:, None, None], z16_target, z16_source)
        a2_noisy = (1.0 - tau_a2[:, None, None]) * eps2 + tau_a2[:, None, None] * a2_target
        z32_noisy = self.world_bridge.interpolate(z16_source, z32_target, tau_z32, world_noise32)
        stage2 = self._assemble_sequence(
            context=context,
            context_valid=context_valid,
            state=state,
            z0=z0,
            a1=a1_prefix,
            z16=z16_source,
            tau_a1=torch.ones_like(tau_a1),
            tau_z16=torch.ones_like(tau_z16),
            a2=a2_noisy,
            z32=z32_noisy,
            tau_a2=tau_a2,
            tau_z32=tau_z32,
            block1_clean_prefix=True,
            a1_valid=~action_is_pad[:, :16],
            a2_valid=~action_is_pad[:, 16:],
        )
        _, a2_velocity, _ = self._action_outputs(stage2.hidden[:, stage2.slices["a2"]], a2_noisy, tau_a2)
        _, z32_prediction = self._world_outputs(stage2.hidden[:, stage2.slices["z32"]], z16_source)

        if self.action_prediction_type == "velocity":
            target_velocity1 = a1_target - eps1
            target_velocity2 = a2_target - eps2
        else:
            # This exactly mirrors FlowmatchingActionHead's JiT clean-x
            # target conversion, including its near-endpoint denominator cap.
            target_velocity1 = action_prediction_to_velocity(
                a1_target,
                a1_noisy,
                tau_a1[:, None, None],
                prediction_type="jit_x",
                t_eps=self.jit_t_eps,
            )
            target_velocity2 = action_prediction_to_velocity(
                a2_target,
                a2_noisy,
                tau_a2[:, None, None],
                prediction_type="jit_x",
                t_eps=self.jit_t_eps,
            )
        action_error = torch.cat(
            [(a1_velocity - target_velocity1).pow(2), (a2_velocity - target_velocity2).pow(2)], dim=1
        )
        tau_action_valid = torch.cat(
            [
                (tau_a1 < 1.0 - 1.0e-6)[:, None].expand(-1, 16),
                (tau_a2 < 1.0 - 1.0e-6)[:, None].expand(-1, 16),
            ],
            dim=1,
        )
        action_valid = (~action_is_pad) & tau_action_valid
        action_loss = self._masked_action_loss(action_error, action_valid)

        z16_valid = future_valid_16 & (tau_z16 < 1.0 - 1.0e-6)
        z32_valid = future_valid_32 & (tau_z32 < 1.0 - 1.0e-6)
        z16_loss = self._masked_world_loss(z16_prediction, z16_target, z16_valid)
        z32_loss = self._masked_world_loss(z32_prediction, z32_target, z32_valid)
        world_loss = self.z16_loss_weight * z16_loss + self.z32_loss_weight * z32_loss
        weighted_action_loss = self.action_loss_weight * action_loss
        weighted_world_loss = self.world_loss_weight * world_loss
        total = weighted_action_loss + weighted_world_loss

        schedule_eligible = future_valid_16 & ~inverse_edge
        schedule_count = schedule_eligible.float().sum().clamp_min(1.0)
        teacher_realized = (teacher_mask & schedule_eligible).float().sum() / schedule_count
        predicted_realized = ((~teacher_mask) & schedule_eligible).float().sum() / schedule_count
        actual_gt_prefix = teacher_mask | inverse_clean_z16
        valid16_count = future_valid_16.float().sum().clamp_min(1.0)
        policy_valid = policy_edge & future_valid_16
        policy_valid_count = policy_valid.float().sum().clamp_min(1.0)
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
            "coflow_z32_loss_raw": z32_loss.detach(),
            "coflow_teacher_ratio_configured": total.new_tensor(configured_teacher_ratio),
            "coflow_teacher_ratio_realized": teacher_realized.detach(),
            "coflow_predicted_z16_ratio": predicted_realized.detach(),
            "coflow_actual_gt_z16_prefix_ratio": (
                (actual_gt_prefix & future_valid_16).float().sum() / valid16_count
            ).detach(),
            "coflow_policy_predicted_z16_prefix_ratio": (
                ((~actual_gt_prefix) & policy_valid).float().sum() / policy_valid_count
            ).detach(),
            "coflow_forward_clean_action_prefix_ratio": forward_edge.float().mean().detach(),
            "coflow_inverse_clean_z16_prefix_ratio": inverse_clean_z16.float().mean().detach(),
            "coflow_future_valid_16_ratio": future_valid_16.float().mean().detach(),
            "coflow_future_valid_32_ratio": future_valid_32.float().mean().detach(),
            "coflow_z16_valid_ratio": z16_valid.float().mean().detach(),
            "coflow_z32_valid_ratio": z32_valid.float().mean().detach(),
            "coflow_action_nonpad_ratio": (~action_is_pad).float().mean().detach(),
            "coflow_action_valid_ratio": action_valid.float().mean().detach(),
        }
        for name, fraction in self.noise_plane.fractions(plane.mode_ids).items():
            metrics[f"coflow_mode_{name}_ratio"] = fraction.detach()
        if self.log_attention_statistics:
            metrics["coflow_attention_allowed_fraction"] = stage2.attention_mask.float().mean().detach()
        return metrics

    @staticmethod
    def _next_event(index: int, steps: int) -> float:
        return (index + 1) / float(steps) if index < steps else math.inf

    def _sample_block(
        self,
        *,
        context: torch.Tensor,
        context_valid: torch.Tensor,
        state: torch.Tensor | None,
        z0: torch.Tensor,
        world_start: torch.Tensor,
        action_noise: torch.Tensor,
        prefix_action: torch.Tensor | None,
        prefix_world: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context.shape[0]
        action = action_noise
        world = world_start
        action_tau = 0.0
        world_tau = 0.0
        action_index = 0
        world_index = 0

        while action_index < self.action_inference_steps or world_index < self.world_inference_steps:
            tau_a = torch.full((batch,), action_tau, device=context.device, dtype=context.dtype)
            tau_z = torch.full((batch,), world_tau, device=context.device, dtype=context.dtype)
            if prefix_action is None:
                result = self._assemble_sequence(
                    context=context,
                    context_valid=context_valid,
                    state=state,
                    z0=z0,
                    a1=action,
                    z16=world,
                    tau_a1=tau_a,
                    tau_z16=tau_z,
                )
                action_slice, world_slice = result.slices["a1"], result.slices["z16"]
            else:
                result = self._assemble_sequence(
                    context=context,
                    context_valid=context_valid,
                    state=state,
                    z0=z0,
                    a1=prefix_action,
                    z16=prefix_world,
                    tau_a1=torch.ones_like(tau_a),
                    tau_z16=torch.ones_like(tau_z),
                    a2=action,
                    z32=world,
                    tau_a2=tau_a,
                    tau_z32=tau_z,
                    block1_clean_prefix=True,
                )
                action_slice, world_slice = result.slices["a2"], result.slices["z32"]

            _, velocity, _ = self._action_outputs(result.hidden[:, action_slice], action, tau_a)
            _, clean_world = self._world_outputs(result.hidden[:, world_slice], world_start)
            next_action = self._next_event(action_index, self.action_inference_steps)
            next_world = self._next_event(world_index, self.world_inference_steps)
            event = min(next_action, next_world)
            if abs(next_action - event) < 1.0e-12:
                action = action + (next_action - action_tau) * velocity
                action_tau = next_action
                action_index += 1
            if abs(next_world - event) < 1.0e-12:
                world = self.world_bridge.reproject(
                    world_start,
                    clean_world,
                    next_world,
                    add_marginal_noise=next_world < 1.0 - 1.0e-12,
                )
                world_tau = next_world
                world_index += 1
        return action, world

    def sample_actions(
        self,
        *,
        context: torch.Tensor,
        context_valid: torch.Tensor,
        z0: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference from current observation only; no future target arguments exist."""

        noise = self._sample_action_noise(context.shape[0], context.device, context.dtype)
        action1, z16_prediction = self._sample_block(
            context=context,
            context_valid=context_valid,
            state=state,
            z0=z0,
            world_start=z0,
            action_noise=noise[:, :16],
            prefix_action=None,
            prefix_world=None,
        )
        action2, z32_prediction = self._sample_block(
            context=context,
            context_valid=context_valid,
            state=state,
            z0=z0,
            world_start=z16_prediction,
            action_noise=noise[:, 16:],
            prefix_action=action1,
            prefix_world=z16_prediction,
        )
        return torch.cat([action1, action2], dim=1), z16_prediction, z32_prediction
