"""PR5: Visual DINO flow-matching head.

复用:
- starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit.DiT
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.distributions import Beta

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


######### // code // ##########
# 中文注释：DINO feature 的 continuous flow-matching 头。
# forward 输入 cond [B,N_q,D_qwen] 和 z_gt [B,N_v,384]，输出 velocity MSE。
# predict 从高斯噪声 Euler 积分，输出 z_hat [B,N_v,384]，仍在 norm 后的 DINO feature 空间。
class VisualFlowMatchingHead(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        cfg = full_config.framework.visual_model
        self.d_dino = int(cfg.get("d_dino", 384))
        self.hidden_size = int(cfg.get("hidden_size", 768))
        self.num_timestep_buckets = int(cfg.get("num_timestep_buckets", 1000))
        self.noise_s = float(cfg.get("noise_s", 0.999))
        self.num_inference_timesteps = int(cfg.get("num_inference_timesteps", 4))
        self.add_pos_embed = bool(cfg.get("add_pos_embed", True))

        self.x_embed = nn.Linear(self.d_dino, self.hidden_size)
        self.x_decode = nn.Linear(self.hidden_size, self.d_dino)
        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(int(cfg.get("max_seq_len", 1024)), self.hidden_size)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        dit_cfg = dict(cfg.get("diffusion_model_cfg", {}))
        dit_cfg.setdefault("num_attention_heads", int(cfg.get("num_attention_heads", 12)))
        dit_cfg.setdefault("attention_head_dim", int(cfg.get("attention_head_dim", self.hidden_size // dit_cfg["num_attention_heads"])))
        dit_cfg.setdefault("num_layers", int(cfg.get("num_layers", 8)))
        dit_cfg.setdefault("output_dim", self.hidden_size)
        dit_cfg.setdefault("dropout", float(cfg.get("dropout", 0.1)))
        dit_cfg.setdefault("final_dropout", True)
        dit_cfg.setdefault("interleave_self_attention", True)
        dit_cfg.setdefault("norm_type", "ada_norm")
        dit_cfg.setdefault("positional_embeddings", None)
        dit_cfg["cross_attention_dim"] = int(cfg.get("cross_attention_dim", full_config.framework.qwenvl.get("vl_hidden_dim", 896)))
        self.model = DiT(**dit_cfg)

        self.beta_dist = Beta(float(cfg.get("noise_beta_alpha", 1.5)), float(cfg.get("noise_beta_beta", 1.0)))

    #######
    # 中文注释：DeepSpeed bf16 会把 visual head 参数转成 bf16；禁用 autocast 后，
    # Linear/DiT 输入必须显式跟随参数 dtype，loss 和日志仍在 fp32 中计算。
    @staticmethod
    def _module_dtype(module: nn.Module, fallback: torch.dtype = torch.float32) -> torch.dtype:
        for param in module.parameters(recurse=True):
            return param.dtype
        return fallback
    #######

    def sample_time(self, batch_size: int, device, dtype) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype).clamp(max=self.noise_s)
        return (self.noise_s - sample) / self.noise_s

    def _embed_noisy(self, z: torch.Tensor) -> torch.Tensor:
        x = self.x_embed(z)
        if self.add_pos_embed:
            pos = torch.arange(z.shape[1], device=z.device)
            x = x + self.position_embedding(pos).unsqueeze(0)
        return x

    ######### // code // ##########
    # 中文注释（CED C1）：forward 扩展三个可选入参，全部缺省时与旧实现逐位等价：
    #   noise [B,N,D]：外部共享噪声（CED 的 Δ 路径要求 fdm⁺/fdm⁰ 两次调用共享同一 ε）；
    #   t：外部时间（标量或 [B]）。t=0 即纯噪声端（noisy=ε），此时 v̂=Ê[z_H|cond]−ε，
    #      共享 ε 的配对相减恰好消去 ε，得到条件均值差 Δ；
    #   weights [B,N]：逐 patch 权重（change-based 加权）。None 时 mean 与旧标量数值一致
    #      （先对 D 取 mean 再对 B,N 取 mean == 对全部元素取 mean）。
    # return_pred=True 时返回 (loss, pred_velocity, per_patch_loss[B,N])，否则只返回 loss（旧调用点零改动）；
    # per_patch_loss 供 fdm 静/动态 patch 拆分日志直接使用，免额外前向。
    # 关闭 autocast 后输入 dtype 跟随模块参数，避免 DeepSpeed bf16 Linear mismatch；loss/Δ 日志用 fp32。
    def forward(
        self,
        cond: torch.Tensor,
        z_gt: torch.Tensor,
        noise: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
        weights: torch.Tensor | None = None,
        return_pred: bool = False,
    ):
        device_type = z_gt.device.type
        autocast_ctx = torch.autocast(device_type=device_type, enabled=False) if device_type in {"cuda", "cpu"} else nullcontext()
        with autocast_ctx:
            #######
            # 中文注释：原生 trainer 的 bf16/DeepSpeed 会改变参数 dtype；这里让 z、cond、noise
            # 与 visual head 参数一致，避免禁用 autocast 后矩阵乘法 dtype 不匹配。
            compute_dtype = self._module_dtype(self)
            z_gt = z_gt.to(dtype=compute_dtype)
            cond = cond.to(dtype=compute_dtype)
            noise = torch.randn_like(z_gt) if noise is None else noise.to(dtype=compute_dtype, device=z_gt.device)
            #######
            if t is None:
                t = self.sample_time(z_gt.shape[0], z_gt.device, z_gt.dtype)[:, None, None]
            else:
                if not torch.is_tensor(t):
                    t = torch.full((z_gt.shape[0],), float(t), device=z_gt.device, dtype=z_gt.dtype)
                t = t.to(device=z_gt.device, dtype=z_gt.dtype).reshape(-1)[:, None, None]
            noisy = (1 - t) * noise + t * z_gt
            velocity = z_gt - noise
            t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

            hidden = self._embed_noisy(noisy)
            out = self.model(
                hidden_states=hidden,
                encoder_hidden_states=cond,
                timestep=t_discretized,
                return_all_hidden_states=False,
            )
            pred_velocity = self.x_decode(out)
            per_patch = ((pred_velocity.float() - velocity.float()) ** 2).mean(dim=-1)  # [B,N]
            if weights is None:
                loss = per_patch.mean()
            else:
                loss = (per_patch * weights.to(dtype=per_patch.dtype, device=per_patch.device)).mean()
            if return_pred:
                return loss, pred_velocity, per_patch
            return loss
    ######### // code // ##########

    @torch.inference_mode()
    def predict(self, cond: torch.Tensor, n: int) -> torch.Tensor:
        batch_size = cond.shape[0]
        #######
        # 中文注释：推理阶段同样让 latent 和条件 token 跟随 head 参数 dtype，兼容 bf16 checkpoint。
        compute_dtype = self._module_dtype(self, fallback=cond.dtype)
        cond = cond.to(dtype=compute_dtype)
        z = torch.randn(batch_size, n, self.d_dino, device=cond.device, dtype=compute_dtype)
        #######
        dt = 1.0 / float(self.num_inference_timesteps)

        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timestep = torch.full((batch_size,), t_discretized, device=cond.device, dtype=torch.long)
            hidden = self._embed_noisy(z)
            out = self.model(hidden_states=hidden, encoder_hidden_states=cond, timestep=timestep)
            pred_velocity = self.x_decode(out)
            z = z + dt * pred_velocity
        return z
######### // code // ##########
