"""PR2: ELF flow-matching 数学 (PyTorch 移植).

复用 (逻辑移植, 不 import JAX 代码):
- Third_github/ELF/src/utils/sampling_utils.py: add_noise / sample_timesteps /
  net_out_to_v_x / _ode_step / _sde_step
- Third_github/ELF/src/train_step.py: 速度目标与解码分支 λ 采样

约定 (ELF 原版):
- t=1 为干净, t=0 为纯噪声;  z_t = t·x0 + (1-t)·ε·noise_scale
- 网络预测 x0;  v = (x0_pred - z) / max(1-t, t_eps);  训练目标 v* 同式代入真 x0
- 训练 t ~ logit-normal: sigmoid(N(P_mean, P_std))
- 采样: Euler ODE  z ← z + (t_next - t)·v;  SDE 先回噪 α=1-γh 再走到 t_next
"""

from __future__ import annotations

import torch


######### // code // ##########
# 中文注释：训练 timestep 采样。返回 [B] ∈ (0,1)，logit-normal 偏向中等噪声段。
def sample_t_logit_normal(
    batch_size: int,
    p_mean: float,
    p_std: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    z = torch.randn(batch_size, device=device, generator=generator) * p_std + p_mean
    return torch.sigmoid(z)


# 中文注释：前向加噪。x0/noise [B,n,D]，t [B]。z = t·x0 + (1-t)·ε·scale。
def add_noise(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, noise_scale: float) -> torch.Tensor:
    t_exp = t.reshape(-1, 1, 1).to(x0.dtype)
    return t_exp * x0 + (1.0 - t_exp) * noise * float(noise_scale)


# 中文注释：由 x0（真值或网络预测）与当前 z, t 计算速度。分母 clamp 到 t_eps 防 t→1 爆炸。
def velocity_from_x0(x0: torch.Tensor, z: torch.Tensor, t: torch.Tensor, t_eps: float = 0.05) -> torch.Tensor:
    denom = (1.0 - t.reshape(-1, 1, 1).to(x0.dtype)).clamp_min(float(t_eps))
    return (x0 - z) / denom


# 中文注释：解码分支的 per-token λ（ELF decoder branch）：sigmoid(N(p_mean,p_std)) [B,S,1]。
# 解码输入 z = λ·x0 + (1-λ)·ε·decoder_noise_scale，t 喂 1.0。
def sample_decoder_lambda(
    batch_size: int,
    seq_len: int,
    p_mean: float,
    p_std: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    z = torch.randn(batch_size, seq_len, 1, device=device, generator=generator) * p_std + p_mean
    return torch.sigmoid(z)


# 中文注释：采样时间格点（uniform 0→1，n_steps 段 / n_steps+1 个端点）。
def uniform_t_grid(n_steps: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, int(n_steps) + 1, device=device)


# 中文注释：Euler ODE 一步：z ← z + (t_next - t)·v。
def ode_step(z: torch.Tensor, v: torch.Tensor, t: float, t_next: float) -> torch.Tensor:
    return z + (float(t_next) - float(t)) * v


# 中文注释：SDE 回噪（ELF hybrid 版）：h=t_next-t, α=clip(1-γh,0,1), t_back=α·t,
# z_back = α·z + (1-α)·ε·noise_scale；调用方在 t_back 处 forward 后走 z_back + (t_next-t_back)·v。
def sde_renoise(
    z: torch.Tensor,
    t: float,
    t_next: float,
    gamma: float,
    noise_scale: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, float]:
    h = float(t_next) - float(t)
    alpha = max(min(1.0 - float(gamma) * h, 1.0), 0.0)
    t_back = alpha * float(t)
    eps = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator) * float(noise_scale)
    z_back = alpha * z + (1.0 - alpha) * eps
    return z_back, t_back


# 中文注释：带掩码的逐 token 平均（ELF reduce_token_loss）。
# per_token [B,n] (已对维度取 mean)，mask [B,n] (1=计入)。返回标量。
def masked_token_mean(per_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(per_token.dtype)
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)
######### // code // ##########
