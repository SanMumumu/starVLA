"""Two-dimensional action/world time-plane sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Beta

MODE_NAMES = ("policy", "forward", "inverse", "joint", "diagonal")
MODE_TO_ID = {name: index for index, name in enumerate(MODE_NAMES)}


@dataclass
class NoisePlaneBatch:
    tau_action: torch.Tensor  # [B]
    tau_world: torch.Tensor  # [B]
    mode_ids: torch.Tensor  # [B]


class NoisePlaneSampler:
    """Sample edge, interior, and diagonal loci of ``(tau_action,tau_world)``.

    Semantics follow Qantara's inference-role naming:

    - policy: action varies, world is pinned to bridge source (tau_z=0)
    - forward: clean action (tau_a=1), world varies
    - inverse: action varies, clean future (tau_z=1)
    - joint: independent action/world interior samples (Qantara ``square``)
    - diagonal: tied action/world time (Qantara ``joint``)

    StarVLA samples one action time and one world time for the single H16
    action--future bridge.  There is no second temporal block or chained world
    time in this model.
    """

    def __init__(
        self,
        ratios: dict,
        *,
        action_timestep_sampling: str = "starvla_gr00t",
        world_timestep_sampling: str = "qantara_monotone",
        action_beta_alpha: float = 1.5,
        action_beta_beta: float = 1.0,
        action_noise_s: float = 0.999,
    ) -> None:
        ratio_values = []
        for name in MODE_NAMES:
            value = float(ratios.get(f"{name}_ratio", 0.0))
            if value < 0:
                raise ValueError(f"noise_plane_sampling.{name}_ratio must be non-negative, got {value}")
            ratio_values.append(value)
        total = sum(ratio_values)
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError(f"noise_plane_sampling ratios must sum to 1.0, got {total:.9f}")
        self.ratios = torch.tensor(ratio_values, dtype=torch.float64)

        self.action_timestep_sampling = str(action_timestep_sampling).lower()
        if self.action_timestep_sampling not in {"starvla_gr00t", "starvla_legacy", "uniform"}:
            raise ValueError(f"Unsupported action_timestep_sampling={action_timestep_sampling!r}")
        self.world_timestep_sampling = str(world_timestep_sampling).lower()
        if self.world_timestep_sampling not in {"qantara_monotone", "independent_uniform", "uniform"}:
            raise ValueError(f"Unsupported world_timestep_sampling={world_timestep_sampling!r}")
        self.action_beta_alpha = float(action_beta_alpha)
        self.action_beta_beta = float(action_beta_beta)
        self.action_noise_s = float(action_noise_s)
        if self.action_beta_alpha <= 0 or self.action_beta_beta <= 0:
            raise ValueError("action Beta parameters must be positive")
        if not 0 < self.action_noise_s <= 1:
            raise ValueError("action_noise_s must be in (0,1]")

    def _sample_action_variable(self, batch_size: int, device, dtype) -> torch.Tensor:
        if self.action_timestep_sampling == "uniform":
            scalar = torch.rand(batch_size, device=device, dtype=dtype)
        else:
            alpha = torch.tensor(self.action_beta_alpha, device=device, dtype=torch.float32)
            beta = torch.tensor(self.action_beta_beta, device=device, dtype=torch.float32)
            sample = Beta(alpha, beta).sample((batch_size,))
            if self.action_timestep_sampling == "starvla_gr00t":
                scalar = (1.0 - sample) * self.action_noise_s
            else:
                scalar = (self.action_noise_s - sample.clamp(max=self.action_noise_s)) / self.action_noise_s
            scalar = scalar.to(dtype=dtype)
        return scalar

    def _sample_world_variable(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.rand(batch_size, device=device, dtype=dtype)

    def sample(self, batch_size: int, device, dtype=torch.float32) -> NoisePlaneBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        ratios = self.ratios.to(device=device, dtype=torch.float32)
        mode_ids = torch.multinomial(ratios, batch_size, replacement=True)
        # Treat ``dtype`` as the public output contract even if an autocast
        # eligible sampling op internally promotes its result to float32.
        action_var = self._sample_action_variable(batch_size, device, dtype).to(
            device=device, dtype=dtype
        )
        world_var = self._sample_world_variable(batch_size, device, dtype).to(
            device=device, dtype=dtype
        )
        tau_action = action_var.clone()
        tau_world = world_var.clone()

        policy = mode_ids == MODE_TO_ID["policy"]
        forward = mode_ids == MODE_TO_ID["forward"]
        inverse = mode_ids == MODE_TO_ID["inverse"]
        diagonal = mode_ids == MODE_TO_ID["diagonal"]

        tau_world[policy] = 0.0
        tau_action[forward] = 1.0
        tau_world[inverse] = 1.0
        # The diagonal inherits StarVLA's one-time-per-action-chunk draw.
        tau_world[diagonal] = action_var[diagonal]
        return NoisePlaneBatch(tau_action=tau_action, tau_world=tau_world, mode_ids=mode_ids)

    @staticmethod
    def fractions(mode_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            name: (mode_ids == mode_id).float().mean()
            for name, mode_id in MODE_TO_ID.items()
        }
