"""Short-horizon conditional World--Action Mixture-of-Transformers.

FastWAM-style condition-cross layers alternate with joint physical self-
attention layers.  Only noisy action and noisy future-DINO tokens enter the
MoT stream.  State/ACTION-PLAN and WORLD-PLAN/current-DINO remain read-only
K/V memories.  Action and world experts own independent attention projections,
normalization, feed-forward parameters, and output heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from torch.utils.checkpoint import checkpoint


def _cfg_get(config, key: str, default=None):
    if config is None:
        return default
    getter = getattr(config, "get", None)
    return getter(key, default) if callable(getter) else getattr(config, key, default)


class ContinuousTimeEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_dim: int = 256,
        time_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if frequency_dim <= 0 or frequency_dim % 2:
            raise ValueError("time_frequency_dim must be a positive even integer")
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        self.frequency_dim = int(frequency_dim)
        self.time_scale = float(time_scale)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phase = (
            time.float().reshape(-1, 1)
            * self.time_scale
            * frequencies.reshape(1, -1)
        )
        embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
        return self.mlp(embedding.to(dtype=next(self.mlp.parameters()).dtype))


class AdaptiveRMSNorm(nn.Module):
    """RMSNorm modulated by one stream's continuous flow-time embedding."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(
        self, hidden: torch.Tensor, time_condition: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.modulation(time_condition).chunk(2, dim=-1)
        normalized = self.norm(hidden)
        return normalized * (1.0 + scale[:, None]) + shift[:, None]


class ActionInputEncoder(nn.Module):
    """FastWAM-style multi-layer action encoder with continuous time input."""

    def __init__(self, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.action_projection = nn.Linear(action_dim, hidden_size)
        self.action_time_fusion = nn.Linear(2 * hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, action: torch.Tensor, time_condition: torch.Tensor
    ) -> torch.Tensor:
        action_hidden = self.action_projection(action)
        time_hidden = time_condition[:, None].expand(-1, action.shape[1], -1)
        hidden = torch.cat([action_hidden, time_hidden], dim=-1)
        return self.output_projection(F.silu(self.action_time_fusion(hidden)))


class StateEncoder(nn.Module):
    """Encode the normalized current proprioception as Action-expert tokens."""

    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(state_dim, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.output_projection(F.relu(self.input_projection(state)))


def _masked_action_mse(
    squared_error: torch.Tensor, action_is_pad: torch.Tensor | None
) -> torch.Tensor:
    """Average valid steps per sample before averaging the batch.

    This matches the successful FastWAM action objective: short padded samples
    receive the same sample weight as full-horizon samples.
    """

    if action_is_pad is None:
        return squared_error.mean()
    if tuple(action_is_pad.shape) != tuple(squared_error.shape[:2]):
        raise ValueError(
            "action_is_pad must match squared_error's batch and horizon dimensions, "
            f"got mask={tuple(action_is_pad.shape)} and error={tuple(squared_error.shape)}"
        )
    per_step = squared_error.mean(dim=-1)
    valid = (~action_is_pad).to(device=squared_error.device, dtype=squared_error.dtype)
    per_sample = (per_step * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return per_sample.mean()


class StreamAttention(nn.Module):
    """One modality's private Q/K/V/O projections."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // self.num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def qkv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = hidden.shape

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        return (
            split_heads(self.q_proj(hidden)),
            split_heads(self.k_proj(hidden)),
            split_heads(self.v_proj(hidden)),
        )

    def output(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, heads, length, head_dim = hidden.shape
        return self.o_proj(hidden.transpose(1, 2).reshape(batch, length, heads * head_dim))


class StreamCrossAttention(nn.Module):
    """One modality's private Q and condition-memory K/V/O projections."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // self.num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_dropout = float(dropout)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self, hidden: torch.Tensor, condition_memory: torch.Tensor
    ) -> torch.Tensor:
        query = self._split_heads(self.q_proj(hidden))
        key = self._split_heads(self.k_proj(condition_memory))
        value = self._split_heads(self.v_proj(condition_memory))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        batch, heads, length, head_dim = attended.shape
        attended = attended.transpose(1, 2).reshape(
            batch, length, heads * head_dim
        )
        return self.o_proj(attended)


class MoTExpertBlock(nn.Module):
    """Joint full self-attention over only the two noisy physical streams."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        action_mlp_ratio: float,
        world_mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.action_attn_norm = AdaptiveRMSNorm(hidden_size)
        self.world_attn_norm = AdaptiveRMSNorm(hidden_size)
        self.action_attention = StreamAttention(hidden_size, num_heads)
        self.world_attention = StreamAttention(hidden_size, num_heads)
        # Match the proven FastWAM DiT pattern: time modulates attention at
        # every layer, while the following FFN keeps a plain private norm.
        self.action_ffn_norm = nn.RMSNorm(hidden_size)
        self.world_ffn_norm = nn.RMSNorm(hidden_size)
        self.action_ffn = self._make_ffn(hidden_size, action_mlp_ratio, dropout)
        self.world_ffn = self._make_ffn(hidden_size, world_mlp_ratio, dropout)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = float(dropout)

    @staticmethod
    def _make_ffn(hidden_size: int, ratio: float, dropout: float) -> nn.Module:
        width = int(round(hidden_size * float(ratio)))
        if width <= 0:
            raise ValueError("expert MLP ratio must be positive")
        return nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(width, hidden_size),
        )

    def forward(
        self,
        action_hidden: torch.Tensor,
        world_hidden: torch.Tensor,
        _action_memory: torch.Tensor,
        _world_memory: torch.Tensor,
        action_time_condition: torch.Tensor,
        world_time_condition: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_normed = self.action_attn_norm(action_hidden, action_time_condition)
        world_normed = self.world_attn_norm(world_hidden, world_time_condition)
        aq, ak, av = self.action_attention.qkv(action_normed)
        wq, wk, wv = self.world_attention.qkv(world_normed)

        # Q/K/V parameters are private; only the projected representations are
        # concatenated for global cross-modal attention.
        queries = torch.cat([aq, wq], dim=2)
        keys = torch.cat([ak, wk], dim=2)
        values = torch.cat([av, wv], dim=2)
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_bias.to(dtype=queries.dtype),
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        action_length = action_hidden.shape[1]
        action_delta = self.action_attention.output(attended[:, :, :action_length])
        world_delta = self.world_attention.output(attended[:, :, action_length:])
        action_hidden = action_hidden + self.dropout(action_delta)
        world_hidden = world_hidden + self.dropout(world_delta)
        action_hidden = action_hidden + self.dropout(
            self.action_ffn(self.action_ffn_norm(action_hidden))
        )
        world_hidden = world_hidden + self.dropout(
            self.world_ffn(self.world_ffn_norm(world_hidden))
        )
        return action_hidden, world_hidden


class ConditionCrossAttentionBlock(nn.Module):
    """FastWAM-style private cross-attention into read-only condition KV."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        action_mlp_ratio: float,
        world_mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.action_attn_norm = AdaptiveRMSNorm(hidden_size)
        self.world_attn_norm = AdaptiveRMSNorm(hidden_size)
        self.action_attention = StreamCrossAttention(
            hidden_size, num_heads, dropout
        )
        self.world_attention = StreamCrossAttention(
            hidden_size, num_heads, dropout
        )
        self.action_ffn_norm = nn.RMSNorm(hidden_size)
        self.world_ffn_norm = nn.RMSNorm(hidden_size)
        self.action_ffn = MoTExpertBlock._make_ffn(
            hidden_size, action_mlp_ratio, dropout
        )
        self.world_ffn = MoTExpertBlock._make_ffn(
            hidden_size, world_mlp_ratio, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        action_hidden: torch.Tensor,
        world_hidden: torch.Tensor,
        action_memory: torch.Tensor,
        world_memory: torch.Tensor,
        action_time_condition: torch.Tensor,
        world_time_condition: torch.Tensor,
        _attention_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_query = self.action_attn_norm(
            action_hidden, action_time_condition
        )
        world_query = self.world_attn_norm(
            world_hidden, world_time_condition
        )
        action_hidden = action_hidden + self.dropout(
            self.action_attention(action_query, action_memory)
        )
        world_hidden = world_hidden + self.dropout(
            self.world_attention(world_query, world_memory)
        )
        action_hidden = action_hidden + self.dropout(
            self.action_ffn(self.action_ffn_norm(action_hidden))
        )
        world_hidden = world_hidden + self.dropout(
            self.world_ffn(self.world_ffn_norm(world_hidden))
        )
        return action_hidden, world_hidden


class WorldActionMoT(nn.Module):
    """Joint flow model for an action chunk and a future DINO token field."""

    def __init__(
        self,
        *,
        planner_dim: int,
        world_dim: int,
        action_config,
        mot_config,
    ) -> None:
        super().__init__()
        self.action_horizon = int(_cfg_get(action_config, "action_horizon", 16))
        self.action_dim = int(_cfg_get(action_config, "action_dim", 14))
        self.state_dim = int(_cfg_get(action_config, "state_dim", 0) or 0)
        self.world_dim = int(world_dim)
        self.hidden_size = int(_cfg_get(mot_config, "hidden_size", 1024))
        self.num_layers = int(_cfg_get(mot_config, "num_layers", 12))
        self.attention_pattern = str(
            _cfg_get(
                mot_config,
                "attention_pattern",
                "alternating_condition_joint",
            )
        ).lower()
        self.inference_steps = int(_cfg_get(mot_config, "num_inference_timesteps", 10))
        self.action_loss_weight = float(_cfg_get(mot_config, "action_loss_weight", 1.0))
        self.world_loss_weight = float(_cfg_get(mot_config, "world_loss_weight", 0.1))
        self.action_prediction_type = str(
            _cfg_get(mot_config, "action_prediction_type", "velocity")
        ).lower()
        self.jit_t_eps = float(_cfg_get(mot_config, "jit_t_eps", 0.05))
        self.flow_time_sampling = str(
            _cfg_get(mot_config, "flow_time_sampling", "uniform")
        ).lower()
        self.noise_beta_alpha = float(
            _cfg_get(mot_config, "noise_beta_alpha", 1.5)
        )
        self.noise_beta_beta = float(
            _cfg_get(mot_config, "noise_beta_beta", 1.0)
        )
        self.noise_s = float(_cfg_get(mot_config, "noise_s", 0.999))
        self.num_timestep_buckets = int(
            _cfg_get(mot_config, "num_timestep_buckets", 1000)
        )
        self.repeated_diffusion_steps = int(
            _cfg_get(mot_config, "repeated_diffusion_steps", 1)
        )
        self.gradient_checkpointing = bool(
            _cfg_get(mot_config, "enable_gradient_checkpointing", True)
        )
        if self.action_horizon <= 0 or self.action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive")
        if self.state_dim < 0:
            raise ValueError("state_dim must be non-negative")
        if self.num_layers <= 0 or self.num_layers % 2:
            raise ValueError(
                "num_layers must be a positive even number for alternating "
                "condition-cross and joint-self attention"
            )
        if self.attention_pattern != "alternating_condition_joint":
            raise ValueError(
                "attention_pattern must be 'alternating_condition_joint'"
            )
        if self.inference_steps <= 0:
            raise ValueError("num_inference_timesteps must be positive")
        if min(self.action_loss_weight, self.world_loss_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.action_prediction_type not in {"velocity", "jit_x"}:
            raise ValueError(
                "action_prediction_type must be 'velocity' or 'jit_x', got "
                f"{self.action_prediction_type!r}"
            )
        if self.flow_time_sampling not in {"uniform", "gr00t"}:
            raise ValueError(
                "flow_time_sampling must be 'uniform' or 'gr00t', got "
                f"{self.flow_time_sampling!r}"
            )
        if self.jit_t_eps <= 0:
            raise ValueError("jit_t_eps must be positive")
        if self.noise_beta_alpha <= 0 or self.noise_beta_beta <= 0:
            raise ValueError("noise Beta distribution parameters must be positive")
        if not 0 < self.noise_s <= 1:
            raise ValueError("noise_s must be in (0, 1]")
        if self.num_timestep_buckets <= 0:
            raise ValueError("num_timestep_buckets must be positive")
        if self.repeated_diffusion_steps <= 0:
            raise ValueError("repeated_diffusion_steps must be positive")

        self.action_plan_projection = nn.Linear(int(planner_dim), self.hidden_size)
        self.world_plan_projection = nn.Linear(int(planner_dim), self.hidden_size)
        self.action_input_encoder = ActionInputEncoder(
            self.action_dim, self.hidden_size
        )
        self.state_encoder = (
            StateEncoder(self.state_dim, self.hidden_size)
            if self.state_dim > 0
            else None
        )
        self.world_input_projection = nn.Linear(self.world_dim, self.hidden_size)
        self.current_world_projection = nn.Linear(self.world_dim, self.hidden_size)
        self.action_position = nn.Embedding(self.action_horizon, self.hidden_size)
        self.world_position = nn.Parameter(
            torch.randn(1, int(_cfg_get(mot_config, "max_world_tokens", 512)), self.hidden_size)
            * 0.01
        )
        self.action_time = ContinuousTimeEmbedding(
            self.hidden_size,
            int(_cfg_get(mot_config, "time_frequency_dim", 256)),
            time_scale=self.num_timestep_buckets,
        )
        self.world_time = ContinuousTimeEmbedding(
            self.hidden_size,
            int(_cfg_get(mot_config, "time_frequency_dim", 256)),
            time_scale=self.num_timestep_buckets,
        )
        dropout = float(_cfg_get(mot_config, "dropout", 0.1))
        self.input_dropout = nn.Dropout(dropout)
        num_heads = int(_cfg_get(mot_config, "num_attention_heads", 16))
        action_mlp_ratio = float(
            _cfg_get(mot_config, "action_mlp_ratio", 4.0)
        )
        world_mlp_ratio = float(
            _cfg_get(mot_config, "world_mlp_ratio", 4.0)
        )
        layers = []
        for layer_index in range(self.num_layers):
            block_type = (
                ConditionCrossAttentionBlock
                if layer_index % 2 == 0
                else MoTExpertBlock
            )
            layers.append(
                block_type(
                    self.hidden_size,
                    num_heads,
                    action_mlp_ratio,
                    world_mlp_ratio,
                    dropout,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.action_output_norm = AdaptiveRMSNorm(self.hidden_size)
        self.world_output_norm = AdaptiveRMSNorm(self.hidden_size)
        self.action_output = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.action_dim),
        )
        self.world_output = nn.Linear(self.hidden_size, self.world_dim)

    def _sample_flow_time(
        self, batch_size: int, *, device: torch.device
    ) -> torch.Tensor:
        if self.flow_time_sampling == "gr00t":
            alpha = torch.tensor(
                self.noise_beta_alpha, device=device, dtype=torch.float32
            )
            beta = torch.tensor(
                self.noise_beta_beta, device=device, dtype=torch.float32
            )
            sample = Beta(alpha, beta).sample((int(batch_size),))
            return (1.0 - sample) * self.noise_s
        return torch.rand(int(batch_size), device=device, dtype=torch.float32) * self.noise_s

    def _action_to_velocity(
        self,
        prediction: torch.Tensor,
        noisy_action: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if self.action_prediction_type == "velocity":
            return prediction
        # Flow time is intentionally sampled in fp32, while mixed-precision
        # model states are commonly bf16.  Keep the JiT-x conversion in the
        # prediction dtype; otherwise fp32 `time` promotes the Euler state to
        # fp32 after the first step and the next bf16 Linear fails.
        noisy_action = noisy_action.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        time = time.to(device=prediction.device, dtype=prediction.dtype)
        denominator = (1.0 - time[:, None, None]).clamp_min(self.jit_t_eps)
        return (prediction - noisy_action) / denominator

    @staticmethod
    def _repeat_batch(tensor: torch.Tensor | None, repeats: int):
        if tensor is None or repeats == 1:
            return tensor
        return tensor.repeat(int(repeats), *([1] * (tensor.ndim - 1)))

    def _physical_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        weight = self.action_input_encoder.action_projection.weight
        return weight.device, weight.dtype

    def _prepare_physical_conditions(
        self,
        *,
        action_plan: torch.Tensor,
        world_plan: torch.Tensor,
        current_world: torch.Tensor,
        current_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        device, dtype = self._physical_device_dtype()
        action_plan = action_plan.to(device=device, dtype=dtype)
        world_plan = world_plan.to(device=device, dtype=dtype)
        current_world = current_world.to(device=device, dtype=dtype)
        if current_state is not None:
            current_state = current_state.to(device=device, dtype=dtype)
        return action_plan, world_plan, current_world, current_state

    def _tokens(
        self,
        *,
        action_plan: torch.Tensor,
        world_plan: torch.Tensor,
        current_world: torch.Tensor,
        noisy_action: torch.Tensor,
        noisy_world: torch.Tensor,
        action_time: torch.Tensor,
        world_time: torch.Tensor,
        action_is_pad: torch.Tensor | None,
        current_state: torch.Tensor | None = None,
    ):
        batch, world_tokens, _ = noisy_world.shape
        if world_tokens > self.world_position.shape[1]:
            raise ValueError(
                f"world token count {world_tokens} exceeds max_world_tokens="
                f"{self.world_position.shape[1]}"
            )
        action_time_condition = self.action_time(action_time)
        world_time_condition = self.world_time(world_time)
        action_data = self.action_input_encoder(noisy_action, action_time_condition)
        action_data = action_data + self.action_position.weight[None, : noisy_action.shape[1]]
        world_data = self.world_input_projection(noisy_world)
        world_data = world_data + self.world_position[:, :world_tokens]
        world_data = world_data + world_time_condition[:, None]

        # Read-only condition memories. They are never concatenated with the
        # noisy streams, never receive flow-time embeddings, and are never
        # updated by the physical Transformer.
        action_memory_parts = []
        if self.state_encoder is not None:
            if current_state is None:
                raise ValueError(
                    "current_state is required when action_model.state_dim > 0"
                )
            if current_state.ndim == 2:
                current_state = current_state[:, None, :]
            if current_state.ndim != 3 or current_state.shape[-1] != self.state_dim:
                raise ValueError(
                    "current_state must have shape [B,N,state_dim], got "
                    f"{tuple(current_state.shape)} for state_dim={self.state_dim}"
                )
            action_memory_parts.append(self.state_encoder(current_state))
        elif current_state is not None:
            raise ValueError(
                "current_state was provided but action_model.state_dim is zero"
            )
        action_memory_parts.append(self.action_plan_projection(action_plan))
        action_memory = torch.cat(action_memory_parts, dim=1)
        world_memory = torch.cat(
            [
                self.world_plan_projection(world_plan),
                self.current_world_projection(current_world),
            ],
            dim=1,
        )
        action_hidden = self.input_dropout(action_data)
        world_hidden = self.input_dropout(world_data)
        action_valid = torch.ones(
            (batch, action_hidden.shape[1]), device=action_hidden.device, dtype=torch.bool
        )
        if action_is_pad is not None:
            action_valid[:] = ~action_is_pad.to(
                device=action_hidden.device, dtype=torch.bool
            )
        world_valid = torch.ones(
            (batch, world_hidden.shape[1]), device=world_hidden.device, dtype=torch.bool
        )
        # Full bidirectional physical attention.  Action and future-world
        # variables are jointly denoised, so every valid token can read every
        # other valid token.  The only restriction is padded action keys.
        key_valid = torch.cat([action_valid, world_valid], dim=1)
        allowed = key_valid[:, None, None, :]
        attention_bias = torch.zeros(
            allowed.shape, device=action_hidden.device, dtype=action_hidden.dtype
        )
        attention_bias.masked_fill_(~allowed, float("-inf"))
        return (
            action_hidden,
            world_hidden,
            action_memory,
            world_memory,
            attention_bias,
            action_time_condition,
            world_time_condition,
        )

    def _predict(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        (
            action_hidden,
            world_hidden,
            action_memory,
            world_memory,
            attention_bias,
            action_time_condition,
            world_time_condition,
        ) = self._tokens(**kwargs)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                action_hidden, world_hidden = checkpoint(
                    layer,
                    action_hidden,
                    world_hidden,
                    action_memory,
                    world_memory,
                    action_time_condition,
                    world_time_condition,
                    attention_bias,
                    use_reentrant=False,
                )
            else:
                action_hidden, world_hidden = layer(
                    action_hidden,
                    world_hidden,
                    action_memory,
                    world_memory,
                    action_time_condition,
                    world_time_condition,
                    attention_bias,
                )
        action_prediction = self.action_output(
            self.action_output_norm(action_hidden, action_time_condition)
        )
        world_velocity = self.world_output(
            self.world_output_norm(world_hidden, world_time_condition)
        )
        return action_prediction, world_velocity

    def forward_train(
        self,
        *,
        action_plan: torch.Tensor,
        world_plan: torch.Tensor,
        current_world: torch.Tensor,
        target_action: torch.Tensor,
        target_world: torch.Tensor,
        action_is_pad: torch.Tensor | None,
        future_valid: torch.Tensor,
        current_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        (
            action_plan,
            world_plan,
            current_world,
            current_state,
        ) = self._prepare_physical_conditions(
            action_plan=action_plan,
            world_plan=world_plan,
            current_world=current_world,
            current_state=current_state,
        )
        physical_device, physical_dtype = self._physical_device_dtype()
        target_action = target_action.to(
            device=physical_device,
            dtype=physical_dtype,
        )
        target_world = target_world.to(
            device=physical_device,
            dtype=physical_dtype,
        )
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=physical_device,
                dtype=torch.bool,
            )
        future_valid = future_valid.to(device=physical_device)

        repeats = self.repeated_diffusion_steps
        action_plan = self._repeat_batch(action_plan, repeats)
        world_plan = self._repeat_batch(world_plan, repeats)
        current_world = self._repeat_batch(current_world, repeats)
        target_action = self._repeat_batch(target_action, repeats)
        target_world = self._repeat_batch(target_world, repeats)
        current_state = self._repeat_batch(current_state, repeats)
        action_is_pad = self._repeat_batch(action_is_pad, repeats)
        future_valid = self._repeat_batch(future_valid, repeats)
        batch = target_action.shape[0]
        action_noise = torch.randn_like(target_action)
        world_noise = torch.randn_like(target_world)
        # A shared physical clock keeps the joint training path aligned with
        # the synchronized action/world Euler sampler used at inference.
        flow_time = self._sample_flow_time(batch, device=target_action.device)
        # Keep the accurately sampled clock in fp32 for time embeddings, but
        # perform the physical interpolation in the exact latent/model dtype
        # used by inference.  Otherwise fp32 time silently promotes bf16
        # training latents and makes training depend on an outer autocast.
        latent_time = flow_time.to(dtype=physical_dtype)
        noisy_action = (
            (1.0 - latent_time[:, None, None]) * action_noise
            + latent_time[:, None, None] * target_action
        )
        noisy_world = (
            (1.0 - latent_time[:, None, None]) * world_noise
            + latent_time[:, None, None] * target_world
        )
        predicted_action, predicted_world = self._predict(
            action_plan=action_plan,
            world_plan=world_plan,
            current_world=current_world,
            noisy_action=noisy_action,
            noisy_world=noisy_world,
            action_time=flow_time,
            world_time=flow_time,
            action_is_pad=action_is_pad,
            current_state=current_state,
        )
        predicted_action_velocity = self._action_to_velocity(
            predicted_action, noisy_action, flow_time
        )
        target_action_velocity = (
            self._action_to_velocity(target_action, noisy_action, flow_time)
            if self.action_prediction_type == "jit_x"
            else target_action - action_noise
        )
        action_error = (
            predicted_action_velocity.float() - target_action_velocity.float()
        ).square()
        action_loss = _masked_action_mse(action_error, action_is_pad)

        per_sample_world = (
            predicted_world.float() - (target_world - world_noise).float()
        ).square().mean(dim=(1, 2))
        valid = future_valid.to(device=per_sample_world.device).reshape(batch, -1)[:, 0] > 0.5
        world_loss = (
            per_sample_world[valid].mean()
            if bool(valid.any())
            else per_sample_world.sum() * 0.0
        )
        total = self.action_loss_weight * action_loss + self.world_loss_weight * world_loss
        return {
            "loss": total,
            "action_loss_raw": action_loss.detach(),
            "world_loss_raw": world_loss.detach(),
        }

    @torch.inference_mode()
    def sample(
        self,
        *,
        action_plan: torch.Tensor,
        world_plan: torch.Tensor,
        current_world: torch.Tensor,
        current_state: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            action_plan,
            world_plan,
            current_world,
            current_state,
        ) = self._prepare_physical_conditions(
            action_plan=action_plan,
            world_plan=world_plan,
            current_world=current_world,
            current_state=current_state,
        )
        sample_device, sample_dtype = self._physical_device_dtype()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=sample_device)
            generator.manual_seed(int(seed))
        batch = action_plan.shape[0]
        action = torch.randn(
            (batch, self.action_horizon, self.action_dim),
            device=sample_device,
            dtype=sample_dtype,
            generator=generator,
        )
        world = torch.randn(
            current_world.shape,
            device=sample_device,
            dtype=sample_dtype,
            generator=generator,
        )
        step_size = 1.0 / self.inference_steps
        for step in range(self.inference_steps):
            time = torch.full(
                (batch,),
                step / self.inference_steps,
                device=action.device,
                dtype=torch.float32,
            )
            action_prediction, world_velocity = self._predict(
                action_plan=action_plan,
                world_plan=world_plan,
                current_world=current_world,
                noisy_action=action,
                noisy_world=world,
                action_time=time,
                world_time=time,
                action_is_pad=None,
                current_state=current_state,
            )
            action_velocity = self._action_to_velocity(
                action_prediction, action, time
            )
            # Preserve the model dtype across every Euler step.  This is also
            # defensive against a future velocity parameterization doing
            # intermediate arithmetic in fp32.
            action = (action + step_size * action_velocity).to(sample_dtype)
            world = (world + step_size * world_velocity).to(sample_dtype)
        return action, world
