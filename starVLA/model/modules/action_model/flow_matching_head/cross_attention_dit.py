# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn
from torch.utils.checkpoint import checkpoint


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        #######
        # 中文注释：World→Action guidance（M5/M6+）——可选的 world cross-attention（gated）。
        #   world_cross_attention=True 时本 block 额外做一路对 world memory 的 cross-attn，
        #   X' = X + CA_action(X, M_a) + tanh(gate)·CA_world(X, M_w)；gate 初值=world_gate_init（默认 0 → 等价 baseline）。
        world_cross_attention: bool = False,
        world_cross_attention_dim: Optional[int] = None,
        world_gate_init: float = 0.0,
        #######
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type
        self.world_cross_attention = world_cross_attention

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        #######
        # 中文注释：World cross-attention（M5/M6+）。读与 attn1 相同的 block 输入 X（pre-attn），gated 残差相加。
        if world_cross_attention:
            self.world_norm = (
                AdaLayerNorm(dim) if norm_type == "ada_norm"
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            )
            self.world_attn = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=world_cross_attention_dim,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )
            self.world_gate = nn.Parameter(torch.full((1,), float(world_gate_init)))
        else:
            self.world_norm = None
            self.world_attn = None
            self.world_gate = None
        #######

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        #######
        # 中文注释：World→Action guidance（M5/M6+）的 world memory + mask；为 None 时本 block 行为与 baseline 完全一致。
        world_hidden_states: Optional[torch.Tensor] = None,
        world_attention_mask: Optional[torch.Tensor] = None,
        #######
    ) -> torch.Tensor:

        # 0. Self-Attention
        x_in = hidden_states  # 中文注释：保存 block 输入 X（world cross-attn 与 attn1 读同一 X）
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,  # @JinhuiYE original attention_mask=attention_mask
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        #######
        # 中文注释：M5/M6+ gated world cross-attention——X' += tanh(gate)·CA_world(X, M_w)。
        # 读 block 输入 X（pre-attn1），与 action cross-attn 并联；gate 初值 0 → 平滑从 baseline 起步。
        if self.world_attn is not None and world_hidden_states is not None:
            if self.norm_type == "ada_norm":
                norm_world = self.world_norm(x_in, temb)
            else:
                norm_world = self.world_norm(x_in)
            world_out = self.world_attn(
                norm_world,
                encoder_hidden_states=world_hidden_states,
                attention_mask=world_attention_mask,
            )
            if world_out.ndim == 4:
                world_out = world_out.squeeze(1)
            hidden_states = hidden_states + torch.tanh(self.world_gate) * world_out
        #######

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    # register_to_config auto-registers constructor params into config, enabling access via self.config.xxx instead of self.xxx
    @register_to_config  # Registers passed params to config. TODO: replace with our singleton pattern, implement a mergeable @merge_param_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
        #######
        # 中文注释：World→Action guidance（M4/M5/M6/M6+）的 DiT 级开关，默认全 False → 与原 DiT 完全一致。
        #   world_cross_attention：在每个 cross block 加 gated world cross-attn（M5/M6+）。
        #   world_adaln：加 world_to_temb（zero-init），把 pooled world 向量加到 timestep embedding 驱动 AdaLN（M6/M6+）。
        #   world_cross_attention_dim/world_global_dim：world memory / world 全局向量维度（默认=cross_attention_dim）。
        world_cross_attention: bool = False,
        world_adaln: bool = False,
        world_cross_attention_dim: Optional[int] = None,
        world_global_dim: Optional[int] = None,
        world_gate_init: float = 0.0,
        #######
        **kwargs,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        # Timestep encoder
        #  self.config.compute_dtype may not exist, handle it in advance
        compute_dtype = getattr(self.config, "compute_dtype", torch.float32)
        self.timestep_encoder = (
            TimestepEncoder(  # TODO BUG: self.config.compute_dtype doesn't error during training but fails at eval
                embedding_dim=self.inner_dim, compute_dtype=compute_dtype
            )
        )

        all_blocks = []
        for idx in range(self.config.num_layers):

            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None
            # world cross-attn 只加在 cross block（self block 不读 memory）。
            block_world_xattn = bool(world_cross_attention) and (not use_self_attn)

            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                    world_cross_attention=block_world_xattn,
                    world_cross_attention_dim=(world_cross_attention_dim or cross_attention_dim),
                    world_gate_init=world_gate_init,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        #######
        # 中文注释：M6/M6+ world AdaLN——zero-init 的 Linear 把 pooled world 向量映到 inner_dim 加到 temb；
        # zero-init → 训练起点 temb 不变（等价 baseline），让 world 调制平滑生效。
        if world_adaln:
            wgd = int(world_global_dim or cross_attention_dim or self.inner_dim)
            self.world_to_temb = nn.Linear(wgd, self.inner_dim)
            nn.init.zeros_(self.world_to_temb.weight)
            nn.init.zeros_(self.world_to_temb.bias)
        else:
            self.world_to_temb = None
        #######

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask=None,
        #######
        # 中文注释：World→Action guidance（M4/M5/M6+）。默认全 None/"none" → 与原 DiT 完全一致。
        #   world_mode="alternate"(M4)：cross block 交替 attend action / world memory（复用 attn1，无新参数）。
        #   world_mode="dual"(M5/M6+)：每个 cross block 并联 gated world cross-attn（用 block.world_attn）。
        #   world_global(M6/M6+)：pooled world 向量经 world_to_temb 加到 temb 驱动 AdaLN。
        world_hidden_states=None,
        world_attention_mask=None,
        world_global=None,
        world_mode: str = "none",
        #######
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        #######
        # 中文注释：M6/M6+ world AdaLN 调制（zero-init → 起点等价 baseline）。
        if getattr(self, "world_to_temb", None) is not None and world_global is not None:
            temb = temb + self.world_to_temb(world_global.to(temb.dtype))
        #######

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        cross_count = 0  # 中文注释：cross block 计数（M4 交替路由用）
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                block_kwargs = dict(
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                enc, enc_mask = encoder_hidden_states, encoder_attention_mask
                w_hs, w_mask = None, None
                if world_mode == "alternate" and world_hidden_states is not None:
                    # M4：奇数个 cross block → 读 world memory，偶数 → action memory
                    if cross_count % 2 == 1:
                        enc, enc_mask = world_hidden_states, world_attention_mask
                elif world_mode == "dual" and world_hidden_states is not None:
                    # M5/M6+：并联 gated world cross-attn
                    w_hs, w_mask = world_hidden_states, world_attention_mask
                block_kwargs = dict(
                    attention_mask=None,
                    encoder_hidden_states=enc,
                    encoder_attention_mask=enc_mask,
                    temb=temb,
                    world_hidden_states=w_hs,
                    world_attention_mask=w_mask,
                )
                cross_count += 1
            if self.training and self.gradient_checkpointing and not return_all_hidden_states:
                hidden_states = checkpoint(block, hidden_states, use_reentrant=False, **block_kwargs)
            else:
                hidden_states = block(hidden_states, **block_kwargs)
            all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        print(
            "Total number of SelfAttentionTransformer parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        return_all_hidden_states: bool = False,
    ):
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states
