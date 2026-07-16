"""Qantara-aligned current-state bridge for frozen visual latents."""

from __future__ import annotations

import torch
from torch import nn


class QantaraWorldBridge(nn.Module):
    """Brownian bridge with Qantara's clean-x recursion.

    Training input:

    ``z_tau = (1-tau) z_start + tau z_target + gamma sqrt(tau(1-tau)) eps``.

    Qantara predicts the clean endpoint (by default as a residual from
    ``z_start``), not an Euler velocity on the state axis.  Sampling therefore
    predicts ``x_hat`` and re-projects it onto the bridge at the next tau.
    """

    _PREDICTION_TYPES = {"qantara_x_delta", "qantara_x"}

    def __init__(self, noise_scale: float = 1.0, prediction_type: str = "qantara_x_delta") -> None:
        super().__init__()
        self.noise_scale = float(noise_scale)
        if self.noise_scale < 0:
            raise ValueError(f"bridge_noise_scale must be >= 0, got {self.noise_scale}")
        self.prediction_type = str(prediction_type).lower()
        if self.prediction_type not in self._PREDICTION_TYPES:
            raise ValueError(
                f"Unsupported world_prediction_type={self.prediction_type!r}; "
                f"expected one of {sorted(self._PREDICTION_TYPES)}"
            )

    @staticmethod
    def _broadcast_tau(tau: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        tau = torch.as_tensor(tau, device=reference.device, dtype=reference.dtype)
        while tau.ndim < reference.ndim:
            tau = tau.unsqueeze(-1)
        return tau

    def standard_deviation(self, tau: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        tau = self._broadcast_tau(tau, reference)
        return self.noise_scale * torch.sqrt((tau * (1.0 - tau)).clamp_min(0.0))

    def interpolate(
        self,
        start: torch.Tensor,
        target: torch.Tensor,
        tau: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if start.shape != target.shape:
            raise ValueError(f"world bridge endpoints must match, got {start.shape} and {target.shape}")
        tau_b = self._broadcast_tau(tau, start)
        value = (1.0 - tau_b) * start + tau_b * target
        if self.noise_scale > 0:
            noise = torch.randn_like(start) if noise is None else noise
            if noise.shape != start.shape:
                raise ValueError(f"bridge noise must have shape {start.shape}, got {noise.shape}")
            value = value + self.standard_deviation(tau, start) * noise
        return value

    def prediction_target(self, start: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.prediction_type == "qantara_x_delta":
            return target - start
        return target

    def clean_from_prediction(self, prediction: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        if self.prediction_type == "qantara_x_delta":
            return start + prediction
        return prediction

    def reproject(
        self,
        start: torch.Tensor,
        clean_prediction: torch.Tensor,
        tau_next: torch.Tensor | float,
        *,
        add_marginal_noise: bool,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Qantara x-hat recursion step at ``tau_next``.

        Intermediate stochastic steps draw from the *bridge marginal* using
        ``gamma*sqrt(tau(1-tau))``.  This is intentionally not an
        Euler--Maruyama ``sqrt(delta_tau)`` increment.
        """

        tau_b = self._broadcast_tau(torch.as_tensor(tau_next), start)
        value = (1.0 - tau_b) * start + tau_b * clean_prediction
        if add_marginal_noise and self.noise_scale > 0:
            noise = torch.randn_like(start) if noise is None else noise
            value = value + self.standard_deviation(torch.as_tensor(tau_next), start) * noise
        return value
