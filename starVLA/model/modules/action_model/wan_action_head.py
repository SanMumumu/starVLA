"""Wan-initialized action head (flow-matching) for QwenGR00T WAM.

动机：把 Wan2.2-TI2V-5B 视频 DiT 的骨干（transformer blocks + text/time 嵌入）当作动作专家的初始化
（参考 Third_github/FastWAM 的 ActionDiT + preprocess_action_dit_backbone.py）。

要点：
- 本头的 transformer 内核 **就是 Wan 的 DiTBlock**（RMSNorm 的 q/k + RoPE 自注意 + cross-attn 到 context
  + 每块 6 路 modulation 调制 + ffn），所以预处理脚本能把 Wan 权重按 key 截断(前 N 层)+插值装进来。
  内部模块/参数名与 FastWAM `ActionDiT` 一致（action_encoder / text_embedding / time_embedding /
  time_projection / blocks / head），骨干 = 除 `action_encoder.`/`head.` 外的全部键。
- 对外接口与 `FlowmatchingActionHead` 一致：`forward(vl_embs, actions, state, encoder_attention_mask)->loss`、
  `predict_action(...)`、`set_action_correlation(...)`，因此 QwenGR00T 的 wam 路径**不需要改调用**。
- 维度由 config 给（默认对齐 DiT-B：hidden=768/16 层；注意力保持 Wan 的 24×128 以便干净迁移；
  text_dim=2048 直接条件在 Qwen hidden 上）。时间走 Wan 式 t_mod 调制（非旧头的 concat-time），
  以便 Wan 的 time_embedding/time_projection 权重可装。

Wan 内核函数/类移植自 FastWAM `wan_video_dit.py`（无 FastWAM 运行时依赖）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Beta
from torch.utils.checkpoint import checkpoint as _ckpt


# ============================ Wan 内核（移植自 FastWAM）============================
def flash_attention(q, k, v, num_heads, ctx_mask=None):
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, attn_head_dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.attn_hidden_dim = num_heads * attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x, freqs, self_attn_mask=None):
        q = rope_apply(self.norm_q(self.q(x)), freqs, self.num_heads)
        k = rope_apply(self.norm_k(self.k(x)), freqs, self.num_heads)
        v = self.v(x)
        return self.o(flash_attention(q, k, v, self.num_heads, ctx_mask=self_attn_mask))


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, attn_head_dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.attn_hidden_dim = num_heads * attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x, ctx, ctx_mask=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        return self.o(flash_attention(q, k, v, self.num_heads, ctx_mask=ctx_mask))


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, attn_head_dim, num_heads, ffn_dim, eps=1e-6):
        super().__init__()
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask=None):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)  # (B,1,T,L) broadcast over heads
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask)
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


# ============================ 骨干（key 名对齐 FastWAM ActionDiT）============================
class WanActionBackbone(nn.Module):
    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")

    def __init__(self, action_dim, hidden_dim, ffn_dim, text_dim, freq_dim, eps,
                 num_heads, attn_head_dim, num_layers, max_seq_len=1024, use_grad_ckpt=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.use_grad_ckpt = bool(use_grad_ckpt)
        if attn_head_dim % 2 != 0:
            raise ValueError(f"attn_head_dim must be even for RoPE, got {attn_head_dim}")

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.GELU(approximate="tanh"), nn.Linear(hidden_dim, hidden_dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("freqs", precompute_freqs_cis(attn_head_dim, end=max_seq_len), persistent=False)

    @classmethod
    def backbone_key_set(cls, keys):
        return {k for k in keys if not any(k.startswith(p) for p in cls.ACTION_BACKBONE_SKIP_PREFIXES)}

    def forward(self, action_tokens, timestep, context, context_mask=None):
        """action_tokens [B,T,action_dim]; timestep [B] 离散桶; context [B,L,text_dim]; context_mask [B,L] bool(keep)。
        返回 velocity [B,T,action_dim]。"""
        B, T, _ = action_tokens.shape
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(action_tokens.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))           # [B,6,hidden]
        x = self.action_encoder(action_tokens)
        ctx = self.text_embedding(context.to(x.dtype))
        freqs = self.freqs[:T].view(T, 1, -1).to(x.device)
        cm = None
        if context_mask is not None:
            cm = context_mask
            if cm.dim() == 2:
                cm = cm.unsqueeze(1).expand(-1, T, -1)                               # [B,T,L]
            cm = cm.to(torch.bool)
        for blk in self.blocks:
            if self.use_grad_ckpt and self.training:
                x = _ckpt(blk, x, ctx, t_mod, freqs, cm, use_reentrant=False)
            else:
                x = blk(x, ctx, t_mod, freqs, context_mask=cm)
        return self.head(x)

    def load_backbone_payload(self, payload: dict, strict_meta: bool = True) -> None:
        """装载 preprocess 产出的骨干 payload（{backbone_state_dict, meta, policy}）。只覆盖骨干键。"""
        bsd = payload["backbone_state_dict"]
        state = self.state_dict()
        expected = self.backbone_key_set(state.keys())
        provided = set(bsd.keys())
        missing, unexpected = sorted(expected - provided), sorted(provided - expected)
        if missing or unexpected:
            raise ValueError(f"[wan-init] 骨干键不匹配 missing={missing[:6]} unexpected={unexpected[:6]}")
        for k in expected:
            v = bsd[k]
            if tuple(v.shape) != tuple(state[k].shape):
                raise ValueError(f"[wan-init] `{k}` 形状不符 expect {tuple(state[k].shape)} got {tuple(v.shape)}")
            state[k] = v.to(state[k].dtype)
        self.load_state_dict(state, strict=True)


# ============================ flow-matching 包装（接口同 FlowmatchingActionHead）============================
class WanFlowMatchingActionHead(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        cfg = full_config.framework.action_model
        wan = cfg.get("wan", {}) or {}
        self.config = cfg
        self.action_dim = int(cfg.action_dim)
        self.action_horizon = int(cfg.action_horizon)
        self.hidden_dim = int(wan.get("hidden_dim", 768))
        text_dim = int(wan.get("text_dim", full_config.framework.qwenvl.get("vl_hidden_dim", 2048)))
        self.backbone = WanActionBackbone(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            ffn_dim=int(wan.get("ffn_dim", 3072)),
            text_dim=text_dim,
            freq_dim=int(wan.get("freq_dim", 256)),
            eps=float(wan.get("eps", 1e-6)),
            # 默认对齐我的 DiT-B 动作头(12×64)→新头与旧头同尺寸(~150M)；Wan 原生 24×128 迁移更干净但~2.5×大,
            # 想要后者就在 config 里设 num_heads=24/attn_head_dim=128。
            num_heads=int(wan.get("num_heads", 12)),
            attn_head_dim=int(wan.get("attn_head_dim", 64)),
            num_layers=int(wan.get("num_layers", 16)),
            max_seq_len=int(cfg.get("max_seq_len", 1024)),
            use_grad_ckpt=bool(full_config.framework.qwenvl.get("enable_gradient_checkpointing", False)),
        )

        # flow-matching 噪声调度（与 FlowmatchingActionHead 同口径）
        self.num_timestep_buckets = int(cfg.get("num_timestep_buckets", 1000))
        self.num_inference_timesteps = int(cfg.get("num_inference_timesteps", 10) or 10)
        self.noise_s = float(cfg.get("noise_s", 0.999))
        self.beta_dist = Beta(float(cfg.get("noise_beta_alpha", 1.5)), float(cfg.get("noise_beta_beta", 1.0)))
        # correlated noise / multi-step FM（与旧头一致的 trick 兼容）
        self.use_correlated_noise = bool(cfg.get("use_correlated_noise", False))
        self.flow_matching_steps = int(cfg.get("flow_matching_steps", 1))
        self.register_buffer(
            "_action_corr_chol",
            torch.zeros(self.action_horizon * self.action_dim, self.action_horizon * self.action_dim),
            persistent=False,
        )
        self._action_corr_loaded = False

        init_path = wan.get("init_path", None)
        if init_path:
            import os
            if os.path.isfile(str(init_path)):
                payload = torch.load(str(init_path), map_location="cpu")
                self.backbone.load_backbone_payload(payload)
                print(f"[wan-init] loaded Wan backbone <- {init_path}", flush=True)
            else:
                print(f"[wan-init] init_path 不存在，随机初始化: {init_path}", flush=True)

    # ----- flow-matching 工具（copy 自 FlowmatchingActionHead） -----
    def sample_time(self, batch_size, device, dtype):
        s = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.noise_s)
        return (self.noise_s - s) / self.noise_s

    def set_action_correlation(self, chol):
        chol = torch.as_tensor(chol, dtype=torch.float32, device="cpu")
        expected = tuple(self._action_corr_chol.shape)
        if tuple(chol.shape) != expected:
            raise ValueError(f"Action-correlation Cholesky shape={tuple(chol.shape)}, expected={expected}.")
        if not bool(torch.isfinite(chol).all()):
            raise ValueError("Action-correlation Cholesky contains NaN or infinite values.")
        if not torch.allclose(chol, torch.tril(chol), rtol=0.0, atol=1.0e-6):
            raise ValueError("Action-correlation Cholesky must be lower triangular.")
        if not bool((torch.diagonal(chol) > 0).all()):
            raise ValueError("Action-correlation Cholesky must have a strictly positive diagonal.")
        self._action_corr_chol.copy_(chol.to(self._action_corr_chol.device))
        self._action_corr_loaded = True

    def _sample_initial_noise(self, bsz, device, dtype):
        if self.use_correlated_noise:
            if not self._action_corr_loaded:
                raise RuntimeError(
                    "use_correlated_noise=true but no action-correlation Cholesky factor was injected. "
                    "Call set_action_correlation before training or inference."
                )
            z = torch.randn(bsz, self.action_horizon * self.action_dim, device=device, dtype=dtype)
            L = self._action_corr_chol.to(device=device, dtype=dtype)
            return (z @ L.T).reshape(bsz, self.action_horizon, self.action_dim)
        return torch.randn(bsz, self.action_horizon, self.action_dim, device=device, dtype=dtype)

    def forward(self, vl_embs, actions, state=None, encoder_attention_mask=None):
        n_fm = int(self.flow_matching_steps)
        if n_fm > 1:
            vl_embs = vl_embs.repeat(n_fm, 1, 1)
            actions = actions.repeat(n_fm, 1, 1)
            if encoder_attention_mask is not None and torch.is_tensor(encoder_attention_mask):
                encoder_attention_mask = encoder_attention_mask.repeat(n_fm, *([1] * (encoder_attention_mask.ndim - 1)))
        noise = self._sample_initial_noise(actions.shape[0], actions.device, actions.dtype)
        t = self.sample_time(actions.shape[0], actions.device, actions.dtype)[:, None, None]
        noisy = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_disc = (t[:, 0, 0] * self.num_timestep_buckets).long()
        pred = self.backbone(noisy, t_disc, context=vl_embs, context_mask=encoder_attention_mask)
        return ((pred - velocity) ** 2).mean()

    @torch.no_grad()
    def predict_action(self, vl_embs, state=None, encoder_attention_mask=None):
        bsz = vl_embs.shape[0]
        actions = self._sample_initial_noise(bsz, vl_embs.device, vl_embs.dtype)
        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            t_disc = int((step / self.num_inference_timesteps) * self.num_timestep_buckets)
            ts = torch.full((bsz,), t_disc, device=vl_embs.device, dtype=torch.long)
            pred_v = self.backbone(actions, ts, context=vl_embs, context_mask=encoder_attention_mask)
            actions = actions + dt * pred_v
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
