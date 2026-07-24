"""Causal current/future-DINO and action Mixture-of-Transformers.

The physical stream is ordered as ``[z0, z_future_t, action_t]``.  ``z0`` is
the clean current-observation DINO grid and is clamped throughout sampling;
only future DINO and action tokens are noised and denoised.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def _cfg_get(config, key: str, default=None):
    if config is None:
        return default
    getter = getattr(config, "get", None)
    return getter(key, default) if callable(getter) else getattr(config, key, default)


def _sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim <= 0 or dim % 2:
        raise ValueError("time_frequency_dim must be a positive even integer")
    sinusoid = torch.outer(
        position.to(torch.float64),
        torch.pow(
            10_000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device)
            / (dim // 2),
        ),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(
        position.dtype
    )


def _precompute_freqs_cis(
    dim: int,
    end: int = 1024,
    theta: float = 10_000.0,
) -> torch.Tensor:
    if dim <= 0 or dim % 2:
        raise ValueError("RoPE dimensions must be positive even integers")
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.float64)[: dim // 2]
            / float(dim)
        )
    )
    phase = torch.outer(torch.arange(end), frequencies)
    return torch.polar(torch.ones_like(phase), phase)


def _precompute_freqs_cis_2d(
    dim: int,
    end: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    height_dim = dim // 2
    width_dim = dim - height_dim
    if min(height_dim, width_dim) <= 0 or height_dim % 2 or width_dim % 2:
        raise ValueError(
            "attention_head_dim must split into even height/width RoPE dimensions"
        )
    return (
        _precompute_freqs_cis(height_dim, end=end),
        _precompute_freqs_cis(width_dim, end=end),
    )


def _rope_apply(
    hidden: torch.Tensor,
    frequencies: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    batch, length, inner_dim = hidden.shape
    if inner_dim % num_heads:
        raise ValueError(
            f"attention inner dim {inner_dim} is not divisible by heads={num_heads}"
        )
    head_dim = inner_dim // num_heads
    values = hidden.reshape(batch, length, num_heads, head_dim)
    complex_values = torch.view_as_complex(
        values.to(torch.float64).reshape(batch, length, num_heads, -1, 2)
    )
    frequencies = frequencies.to(device=hidden.device)
    if hidden.device.type == "npu":
        frequencies = frequencies.to(torch.complex64)
    complex_values = complex_values * frequencies
    return torch.view_as_real(complex_values).flatten(2).to(hidden.dtype)


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch, query_length, inner_dim = query.shape
    head_dim = inner_dim // num_heads

    def split(value_tensor: torch.Tensor) -> torch.Tensor:
        return value_tensor.view(
            batch,
            value_tensor.shape[1],
            num_heads,
            head_dim,
        ).transpose(1, 2)

    attended = F.scaled_dot_product_attention(
        split(query),
        split(key),
        split(value),
        attn_mask=attention_mask,
    )
    return attended.transpose(1, 2).reshape(batch, query_length, inner_dim)


class InnerRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype) * self.weight


class ExpertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_inner_dim: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_size, attention_inner_dim)
        self.k = nn.Linear(hidden_size, attention_inner_dim)
        self.v = nn.Linear(hidden_size, attention_inner_dim)
        self.o = nn.Linear(attention_inner_dim, hidden_size)
        self.norm_q = InnerRMSNorm(attention_inner_dim, eps)
        self.norm_k = InnerRMSNorm(attention_inner_dim, eps)


class ExpertCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_inner_dim: int,
        num_heads: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.q = nn.Linear(hidden_size, attention_inner_dim)
        self.k = nn.Linear(hidden_size, attention_inner_dim)
        self.v = nn.Linear(hidden_size, attention_inner_dim)
        self.o = nn.Linear(attention_inner_dim, hidden_size)
        self.norm_q = InnerRMSNorm(attention_inner_dim, eps)
        self.norm_k = InnerRMSNorm(attention_inner_dim, eps)

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        query = self.norm_q(self.q(hidden))
        key = self.norm_k(self.k(context))
        value = self.v(context)
        return self.o(
            _attention(
                query,
                key,
                value,
                num_heads=self.num_heads,
                attention_mask=None,
            )
        )


class PhysicalExpertBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        ffn_dim: int,
        attention_inner_dim: int,
        num_heads: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.self_attn = ExpertSelfAttention(
            self.hidden_size,
            attention_inner_dim,
            eps,
        )
        self.cross_attn = ExpertCrossAttention(
            self.hidden_size,
            attention_inner_dim,
            self.num_heads,
            eps,
        )
        self.norm1 = nn.LayerNorm(
            self.hidden_size,
            eps=eps,
            elementwise_affine=False,
        )
        self.norm2 = nn.LayerNorm(
            self.hidden_size,
            eps=eps,
            elementwise_affine=False,
        )
        self.norm3 = nn.LayerNorm(self.hidden_size, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, int(ffn_dim)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(ffn_dim), self.hidden_size),
        )
        self.modulation = nn.Parameter(
            torch.randn(1, 6, self.hidden_size) / self.hidden_size**0.5
        )

    def attention_io(
        self,
        hidden: torch.Tensor,
        time_modulation: torch.Tensor,
        frequencies: torch.Tensor,
    ):
        has_token_time = time_modulation.ndim == 4
        chunk_dim = 2 if has_token_time else 1
        base = self.modulation.to(
            device=time_modulation.device,
            dtype=time_modulation.dtype,
        )
        modulation = base + time_modulation
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation.chunk(6, dim=chunk_dim)
        if has_token_time:
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            shift_mlp = shift_mlp.squeeze(2)
            scale_mlp = scale_mlp.squeeze(2)
            gate_mlp = gate_mlp.squeeze(2)

        attention_input = self.norm1(hidden) * (1.0 + scale_msa) + shift_msa
        query = self.self_attn.norm_q(self.self_attn.q(attention_input))
        key = self.self_attn.norm_k(self.self_attn.k(attention_input))
        value = self.self_attn.v(attention_input)
        query = _rope_apply(query, frequencies, self.num_heads)
        key = _rope_apply(key, frequencies, self.num_heads)
        return (
            query,
            key,
            value,
            hidden,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def post_attention(
        self,
        *,
        residual_hidden: torch.Tensor,
        mixed_attention: torch.Tensor,
        context: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
    ) -> torch.Tensor:
        hidden = residual_hidden + gate_msa * self.self_attn.o(mixed_attention)
        hidden = hidden + self.cross_attn(self.norm3(hidden), context)
        ffn_input = self.norm2(hidden) * (1.0 + scale_mlp) + shift_mlp
        return hidden + gate_mlp * self.ffn(ffn_input)


class CausalDINOActionLayer(nn.Module):
    def __init__(
        self,
        *,
        world_hidden_size: int,
        action_hidden_size: int,
        world_ffn_dim: int,
        action_ffn_dim: int,
        attention_inner_dim: int,
        num_heads: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.world = PhysicalExpertBlock(
            hidden_size=world_hidden_size,
            ffn_dim=world_ffn_dim,
            attention_inner_dim=attention_inner_dim,
            num_heads=num_heads,
            eps=eps,
        )
        self.action = PhysicalExpertBlock(
            hidden_size=action_hidden_size,
            ffn_dim=action_ffn_dim,
            attention_inner_dim=attention_inner_dim,
            num_heads=num_heads,
            eps=eps,
        )

    def forward(
        self,
        action_hidden: torch.Tensor,
        world_hidden: torch.Tensor,
        action_context: torch.Tensor,
        world_context: torch.Tensor,
        action_time_modulation: torch.Tensor,
        world_time_modulation: torch.Tensor,
        action_frequencies: torch.Tensor,
        world_frequencies: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        world_io = self.world.attention_io(
            world_hidden,
            world_time_modulation,
            world_frequencies,
        )
        action_io = self.action.attention_io(
            action_hidden,
            action_time_modulation,
            action_frequencies,
        )
        world_query, world_key, world_value = world_io[:3]
        action_query, action_key, action_value = action_io[:3]
        world_length = world_hidden.shape[1]

        mixed = _attention(
            torch.cat([world_query, action_query], dim=1),
            torch.cat([world_key, action_key], dim=1),
            torch.cat([world_value, action_value], dim=1),
            num_heads=self.num_heads,
            attention_mask=attention_mask,
        )
        world_hidden = self.world.post_attention(
            residual_hidden=world_io[3],
            mixed_attention=mixed[:, :world_length],
            context=world_context,
            gate_msa=world_io[4],
            shift_mlp=world_io[5],
            scale_mlp=world_io[6],
            gate_mlp=world_io[7],
        )
        action_hidden = self.action.post_attention(
            residual_hidden=action_io[3],
            mixed_attention=mixed[:, world_length:],
            context=action_context,
            gate_msa=action_io[4],
            shift_mlp=action_io[5],
            scale_mlp=action_io[6],
            gate_mlp=action_io[7],
        )
        return action_hidden, world_hidden


class ShiftedFlowScheduler:
    def __init__(
        self,
        *,
        num_train_timesteps: int,
        shift: float,
        eps: float = 1.0e-10,
    ) -> None:
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if shift <= 0:
            raise ValueError("flow shift must be positive")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._weight_statistics()

    @staticmethod
    def _phi(value: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * value / (1.0 + (shift - 1.0) * value)

    def _weight_statistics(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        timestep = self._phi(grid, self.shift) * float(steps)
        values = torch.exp(
            -2.0 * ((timestep - (steps / 2.0)) / float(steps)).square()
        )
        minimum = float(values.min())
        normalization = float((values - minimum).mean())
        return minimum, normalization

    def sample_training_t(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        uniform = torch.rand(batch_size, device=device, dtype=torch.float32)
        sigma = self._phi(uniform, self.shift)
        return (sigma * float(self.num_train_timesteps)).to(dtype=dtype)

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            device=clean.device,
            dtype=clean.dtype,
        )
        sigma = sigma.view(-1, *([1] * (clean.ndim - 1)))
        return (1.0 - sigma) * clean + sigma * noise

    @staticmethod
    def training_target(
        clean: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return noise - clean

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        value = timestep.float()
        steps = float(self.num_train_timesteps)
        score = torch.exp(-2.0 * ((value - steps / 2.0) / steps).square())
        return (score - self._y_min) / (self._weight_norm_const + self.eps)

    def inference_schedule(
        self,
        num_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_steps <= 0:
            raise ValueError("num_inference_timesteps must be positive")
        uniform = torch.linspace(
            1.0,
            0.0,
            num_steps + 1,
            device=device,
            dtype=torch.float32,
        )
        sigma = self._phi(uniform, self.shift)
        timesteps = sigma[:-1] * float(self.num_train_timesteps)
        deltas = sigma[1:] - sigma[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(
        prediction: torch.Tensor,
        delta: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        delta = delta.to(device=sample.device, dtype=sample.dtype)
        return sample + prediction * delta


class DINOFlowHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(
            hidden_size,
            eps=eps,
            elementwise_affine=False,
        )
        self.output = nn.Linear(hidden_size, output_size)
        self.modulation = nn.Parameter(
            torch.randn(1, 2, hidden_size) / hidden_size**0.5
        )

    def forward(
        self,
        hidden: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = (
            self.modulation.unsqueeze(0).to(
                device=time_embedding.device,
                dtype=time_embedding.dtype,
            )
            + time_embedding.unsqueeze(2)
        ).chunk(2, dim=2)
        return self.output(
            self.norm(hidden) * (1.0 + scale.squeeze(2)) + shift.squeeze(2)
        )


class CausalDINOActionMoT(nn.Module):
    """Per-layer mixed-attention MoT over ``[z0, z_future_t, action_t]``."""

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
        self.interaction_mode = str(
            _cfg_get(mot_config, "interaction_mode", "base")
        ).lower()
        if self.interaction_mode not in {"base", "joint"}:
            raise ValueError("interaction_mode must be 'base' or 'joint'")
        self.world_attention_mask_mode = str(
            _cfg_get(
                mot_config,
                "world_attention_mask_mode",
                "first_frame_causal",
            )
        ).lower()
        if self.world_attention_mask_mode != "first_frame_causal":
            raise ValueError(
                "CausalDINOActionMoT requires "
                "world_attention_mask_mode='first_frame_causal'"
            )

        self.world_hidden_size = int(
            _cfg_get(mot_config, "world_hidden_size", 3072)
        )
        self.action_hidden_size = int(
            _cfg_get(mot_config, "action_hidden_size", 1024)
        )
        self.world_ffn_dim = int(_cfg_get(mot_config, "world_ffn_dim", 14336))
        self.action_ffn_dim = int(_cfg_get(mot_config, "action_ffn_dim", 4096))
        self.num_heads = int(_cfg_get(mot_config, "num_attention_heads", 24))
        self.head_dim = int(_cfg_get(mot_config, "attention_head_dim", 128))
        self.attention_inner_dim = self.num_heads * self.head_dim
        self.num_layers = int(_cfg_get(mot_config, "num_layers", 30))
        self.norm_eps = float(_cfg_get(mot_config, "norm_eps", 1.0e-6))
        self.time_frequency_dim = int(
            _cfg_get(mot_config, "time_frequency_dim", 256)
        )
        self.grid_height = int(_cfg_get(mot_config, "world_grid_height", 12))
        self.grid_width = int(_cfg_get(mot_config, "world_grid_width", 10))
        self.world_tokens = self.grid_height * self.grid_width
        self.inference_steps = int(
            _cfg_get(mot_config, "num_inference_timesteps", 20)
        )
        self.gradient_checkpointing = bool(
            _cfg_get(mot_config, "enable_gradient_checkpointing", True)
        )
        self.action_loss_weight = float(
            _cfg_get(mot_config, "action_loss_weight", 1.0)
        )
        self.world_loss_weight = float(
            _cfg_get(mot_config, "world_loss_weight", 1.0)
        )
        if min(
            self.action_horizon,
            self.action_dim,
            self.world_hidden_size,
            self.action_hidden_size,
            self.world_ffn_dim,
            self.action_ffn_dim,
            self.num_heads,
            self.head_dim,
            self.num_layers,
            self.grid_height,
            self.grid_width,
            self.inference_steps,
        ) <= 0:
            raise ValueError("all architecture sizes must be positive")
        if self.head_dim % 2:
            raise ValueError("attention_head_dim must be even for RoPE")

        self.world_input = nn.Linear(self.world_dim, self.world_hidden_size)
        self.action_input = nn.Linear(self.action_dim, self.action_hidden_size)
        self.world_context = nn.Sequential(
            nn.Linear(int(planner_dim), self.world_hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.world_hidden_size, self.world_hidden_size),
        )
        self.action_context = nn.Sequential(
            nn.Linear(int(planner_dim), self.action_hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.action_hidden_size, self.action_hidden_size),
        )
        self.state_to_planner = (
            nn.Linear(self.state_dim, int(planner_dim))
            if self.state_dim > 0
            else None
        )
        self.world_time_embedding = nn.Sequential(
            nn.Linear(self.time_frequency_dim, self.world_hidden_size),
            nn.SiLU(),
            nn.Linear(self.world_hidden_size, self.world_hidden_size),
        )
        self.world_time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.world_hidden_size, 6 * self.world_hidden_size),
        )
        self.action_time_embedding = nn.Sequential(
            nn.Linear(self.time_frequency_dim, self.action_hidden_size),
            nn.SiLU(),
            nn.Linear(self.action_hidden_size, self.action_hidden_size),
        )
        self.action_time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.action_hidden_size, 6 * self.action_hidden_size),
        )
        self.layers = nn.ModuleList(
            [
                CausalDINOActionLayer(
                    world_hidden_size=self.world_hidden_size,
                    action_hidden_size=self.action_hidden_size,
                    world_ffn_dim=self.world_ffn_dim,
                    action_ffn_dim=self.action_ffn_dim,
                    attention_inner_dim=self.attention_inner_dim,
                    num_heads=self.num_heads,
                    eps=self.norm_eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.action_output = nn.Linear(self.action_hidden_size, self.action_dim)
        self.world_output = DINOFlowHead(
            self.world_hidden_size,
            self.world_dim,
            self.norm_eps,
        )

        action_train_shift = float(
            _cfg_get(mot_config, "action_train_shift", 5.0)
        )
        action_infer_shift = float(
            _cfg_get(mot_config, "action_infer_shift", 5.0)
        )
        world_train_shift = float(
            _cfg_get(mot_config, "world_train_shift", 5.0)
        )
        world_infer_shift = float(
            _cfg_get(mot_config, "world_infer_shift", 5.0)
        )
        action_train_steps = int(
            _cfg_get(mot_config, "action_num_train_timesteps", 1000)
        )
        world_train_steps = int(
            _cfg_get(mot_config, "world_num_train_timesteps", 1000)
        )
        self.train_action_scheduler = ShiftedFlowScheduler(
            num_train_timesteps=action_train_steps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = ShiftedFlowScheduler(
            num_train_timesteps=action_train_steps,
            shift=action_infer_shift,
        )
        self.train_world_scheduler = ShiftedFlowScheduler(
            num_train_timesteps=world_train_steps,
            shift=world_train_shift,
        )
        self.infer_world_scheduler = ShiftedFlowScheduler(
            num_train_timesteps=world_train_steps,
            shift=world_infer_shift,
        )

        action_frequencies = _precompute_freqs_cis(
            self.head_dim,
            end=max(1024, self.action_horizon),
        )
        rope_cache_end = max(1024, self.grid_height, self.grid_width)
        height, width = _precompute_freqs_cis_2d(
            self.head_dim,
            end=rope_cache_end,
        )
        spatial_frequencies = torch.cat(
            [
                height[: self.grid_height]
                .view(self.grid_height, 1, -1)
                .expand(self.grid_height, self.grid_width, -1),
                width[: self.grid_width]
                .view(1, self.grid_width, -1)
                .expand(self.grid_height, self.grid_width, -1),
            ],
            dim=-1,
        ).reshape(self.world_tokens, 1, -1)
        world_frequencies = spatial_frequencies.repeat(2, 1, 1)
        self.world_frame_embedding = nn.Parameter(
            torch.empty(1, 2, self.world_hidden_size)
        )
        nn.init.normal_(self.world_frame_embedding, std=0.02)
        # Keep complex RoPE caches as plain CPU tensors, as in the source
        # architecture.  Module-wide BF16 conversion must never discard their
        # imaginary component; _rope_apply moves them to the active device.
        self.action_frequencies = action_frequencies[: self.action_horizon].view(
            self.action_horizon,
            1,
            -1,
        )
        self.world_frequencies = world_frequencies
        self.register_buffer(
            "physical_attention_mask",
            self._build_attention_mask(),
            persistent=False,
        )

    def _build_attention_mask(self) -> torch.Tensor:
        current = self.world_tokens
        future_end = 2 * self.world_tokens
        total = future_end + self.action_horizon
        mask = torch.zeros(total, total, dtype=torch.bool)
        # z0/zf video-style first-frame-causal block.
        mask[:future_end, :future_end] = True
        mask[:current, current:future_end] = False
        # Action tokens always mix with each other.
        mask[future_end:, future_end:] = True
        # Base mode reads z0 only; joint mode reads z0 and zf.
        world_key_end = current if self.interaction_mode == "base" else future_end
        mask[future_end:, :world_key_end] = True
        return mask

    def _device_dtype(self) -> tuple[torch.device, torch.dtype]:
        weight = self.action_input.weight
        return weight.device, weight.dtype

    def _prepare_conditions(
        self,
        *,
        action_plan: torch.Tensor,
        world_plan: torch.Tensor,
        current_world: torch.Tensor,
        current_state: torch.Tensor | None,
    ):
        device, dtype = self._device_dtype()
        action_plan = action_plan.to(device=device, dtype=dtype)
        world_plan = world_plan.to(device=device, dtype=dtype)
        current_world = current_world.to(device=device, dtype=dtype)
        if current_world.ndim != 3 or current_world.shape[1] != self.world_tokens:
            raise ValueError(
                "current DINO must have shape "
                f"[B,{self.world_tokens},{self.world_dim}], got {tuple(current_world.shape)}"
            )
        if self.state_to_planner is not None:
            if current_state is None:
                raise ValueError("current_state is required when state_dim > 0")
            if current_state.ndim == 2:
                current_state = current_state[:, None]
            current_state = current_state.to(device=device, dtype=dtype)
            state_token = self.state_to_planner(current_state[:, :1])
            action_plan = torch.cat([action_plan, state_token], dim=1)
            world_plan = torch.cat([world_plan, state_token], dim=1)
        elif current_state is not None:
            raise ValueError("current_state was provided but state_dim is zero")
        return action_plan, world_plan, current_world

    def _time_features(
        self,
        action_time: torch.Tensor,
        world_time: torch.Tensor,
    ):
        batch = action_time.shape[0]
        action_time_embedding = self.action_time_embedding(
            _sinusoidal_embedding_1d(
                self.time_frequency_dim,
                action_time,
            )
        )
        action_time_modulation = self.action_time_projection(
            action_time_embedding
        ).unflatten(1, (6, self.action_hidden_size))

        token_times = torch.cat(
            [
                torch.zeros(
                    batch,
                    self.world_tokens,
                    device=world_time.device,
                    dtype=world_time.dtype,
                ),
                world_time[:, None].expand(-1, self.world_tokens),
            ],
            dim=1,
        )
        world_time_embedding = self.world_time_embedding(
            _sinusoidal_embedding_1d(
                self.time_frequency_dim,
                token_times.reshape(-1),
            )
        ).reshape(batch, 2 * self.world_tokens, self.world_hidden_size)
        world_time_modulation = self.world_time_projection(
            world_time_embedding
        ).unflatten(2, (6, self.world_hidden_size))
        return (
            action_time_embedding,
            action_time_modulation,
            world_time_embedding,
            world_time_modulation,
        )

    def _predict(
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del action_is_pad
        action_plan, world_plan, current_world = self._prepare_conditions(
            action_plan=action_plan,
            world_plan=world_plan,
            current_world=current_world,
            current_state=current_state,
        )
        device, dtype = self._device_dtype()
        noisy_action = noisy_action.to(device=device, dtype=dtype)
        noisy_world = noisy_world.to(device=device, dtype=dtype)
        action_time = action_time.to(device=device, dtype=dtype)
        world_time = world_time.to(device=device, dtype=dtype)
        if noisy_world.shape != current_world.shape:
            raise ValueError(
                "noisy future DINO and current DINO must have identical shapes"
            )

        (
            _action_time_embedding,
            action_time_modulation,
            world_time_embedding,
            world_time_modulation,
        ) = self._time_features(action_time, world_time)
        action_hidden = self.action_input(noisy_action)
        world_hidden = self.world_input(
            torch.cat([current_world, noisy_world], dim=1)
        )
        frame_embedding = (
            self.world_frame_embedding[:, :, None, :]
            .expand(-1, -1, self.world_tokens, -1)
            .reshape(1, 2 * self.world_tokens, self.world_hidden_size)
        )
        world_hidden = world_hidden + frame_embedding
        action_context = self.action_context(action_plan)
        world_context = self.world_context(world_plan)
        attention_mask = self.physical_attention_mask.to(device=device)
        action_frequencies = self.action_frequencies.to(device=device)
        world_frequencies = self.world_frequencies.to(device=device)

        for layer in self.layers:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                action_hidden, world_hidden = checkpoint(
                    layer,
                    action_hidden,
                    world_hidden,
                    action_context,
                    world_context,
                    action_time_modulation,
                    world_time_modulation,
                    action_frequencies,
                    world_frequencies,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                action_hidden, world_hidden = layer(
                    action_hidden,
                    world_hidden,
                    action_context,
                    world_context,
                    action_time_modulation,
                    world_time_modulation,
                    action_frequencies,
                    world_frequencies,
                    attention_mask,
                )
        action_prediction = self.action_output(action_hidden)
        future_hidden = world_hidden[:, self.world_tokens :]
        future_time_embedding = world_time_embedding[:, self.world_tokens :]
        world_prediction = self.world_output(
            future_hidden,
            future_time_embedding,
        )
        return action_prediction, world_prediction

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
        device, dtype = self._device_dtype()
        target_action = target_action.to(device=device, dtype=dtype)
        target_world = target_world.to(device=device, dtype=dtype)
        if target_world.ndim != 3 or target_world.shape[1] != self.world_tokens:
            raise ValueError(
                "target future DINO must have shape "
                f"[B,{self.world_tokens},{self.world_dim}], got {tuple(target_world.shape)}"
            )
        batch = target_action.shape[0]
        action_noise = torch.randn_like(target_action)
        world_noise = torch.randn_like(target_world)
        action_time = self.train_action_scheduler.sample_training_t(
            batch,
            device=device,
            dtype=dtype,
        )
        world_time = self.train_world_scheduler.sample_training_t(
            batch,
            device=device,
            dtype=dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            target_action,
            action_noise,
            action_time,
        )
        noisy_world = self.train_world_scheduler.add_noise(
            target_world,
            world_noise,
            world_time,
        )
        predicted_action, predicted_world = self._predict(
            action_plan=action_plan,
            world_plan=world_plan,
            current_world=current_world,
            noisy_action=noisy_action,
            noisy_world=noisy_world,
            action_time=action_time,
            world_time=world_time,
            action_is_pad=action_is_pad,
            current_state=current_state,
        )

        action_per_step = (
            predicted_action.float()
            - self.train_action_scheduler.training_target(
                target_action,
                action_noise,
            ).float()
        ).square().mean(dim=-1)
        if action_is_pad is not None:
            valid_action = (~action_is_pad.to(device=device, dtype=torch.bool)).to(
                action_per_step.dtype
            )
            action_per_sample = (
                action_per_step * valid_action
            ).sum(dim=1) / valid_action.sum(dim=1).clamp_min(1.0)
        else:
            action_per_sample = action_per_step.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(
            action_time
        ).to(device=device, dtype=action_per_sample.dtype)
        action_loss = (action_per_sample * action_weight).mean()

        world_per_sample = (
            predicted_world.float()
            - self.train_world_scheduler.training_target(
                target_world,
                world_noise,
            ).float()
        ).square().mean(dim=(1, 2))
        world_weight = self.train_world_scheduler.training_weight(
            world_time
        ).to(device=device, dtype=world_per_sample.dtype)
        world_per_sample = world_per_sample * world_weight
        valid_world = (
            future_valid.to(device=device).reshape(batch, -1)[:, 0] > 0.5
        )
        world_loss = (
            world_per_sample[valid_world].mean()
            if bool(valid_world.any())
            else world_per_sample.sum() * 0.0
        )
        total = (
            self.action_loss_weight * action_loss
            + self.world_loss_weight * world_loss
        )
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
        device, dtype = self._device_dtype()
        current_world = current_world.to(device=device, dtype=dtype)
        if current_world.ndim != 3 or current_world.shape[1:] != (
            self.world_tokens,
            self.world_dim,
        ):
            raise ValueError(
                "current DINO must have shape "
                f"[B,{self.world_tokens},{self.world_dim}], "
                f"got {tuple(current_world.shape)}"
            )
        action_generator = None
        world_generator = None
        if seed is not None:
            action_generator = torch.Generator(device="cpu")
            world_generator = torch.Generator(device="cpu")
            action_generator.manual_seed(int(seed))
            world_generator.manual_seed(int(seed))
        batch = current_world.shape[0]
        action = torch.randn(
            batch,
            self.action_horizon,
            self.action_dim,
            device="cpu",
            dtype=torch.float32,
            generator=action_generator,
        ).to(device=device, dtype=dtype)
        future_world = torch.randn(
            current_world.shape,
            device="cpu",
            dtype=torch.float32,
            generator=world_generator,
        ).to(device=device, dtype=dtype)
        action_times, action_deltas = self.infer_action_scheduler.inference_schedule(
            self.inference_steps,
            device=device,
            dtype=dtype,
        )
        world_times, world_deltas = self.infer_world_scheduler.inference_schedule(
            self.inference_steps,
            device=device,
            dtype=dtype,
        )
        for action_time, action_delta, world_time, world_delta in zip(
            action_times,
            action_deltas,
            world_times,
            world_deltas,
        ):
            action_prediction, world_prediction = self._predict(
                action_plan=action_plan,
                world_plan=world_plan,
                current_world=current_world,
                noisy_action=action,
                noisy_world=future_world,
                action_time=action_time.expand(batch),
                world_time=world_time.expand(batch),
                action_is_pad=None,
                current_state=current_state,
            )
            action = self.infer_action_scheduler.step(
                action_prediction,
                action_delta,
                action,
            ).to(dtype)
            future_world = self.infer_world_scheduler.step(
                world_prediction,
                world_delta,
                future_world,
            ).to(dtype)
        return action, future_world
