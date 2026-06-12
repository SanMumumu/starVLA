"""PR2: TriFlow 干净/噪声块注意力 mask.

复用:
- starVLA.jointflow.modules.attention_mask.visualize_mask (mask 出图自检, 只 import)

规则 (区别于 v1 的 block-causal):
- 噪声块 (目标) 的 query 看所有块 —— 条件信息全可见, 多个噪声块互见 (joint 联合生成需要);
- 干净块 (条件) 的 query 只看干净块 —— 条件表征永不被噪声污染;
- 文本块的 padding 列对所有 query 屏蔽 (pad 行不屏蔽: 其输出被 loss mask 掉, 无人 attend)。
"""

from __future__ import annotations

import torch


######### // code // ##########
# 中文注释：构造 [B,1,T,T] additive float mask。可见=0，屏蔽=neg_value(-1e4)。
# 输入：
#   block_sizes  [n] 每块 token 数（按序列顺序）
#   noisy_flags  [n] 每块是否为加噪目标块
#   batch_size   B
#   text_block_idx / text_valid_lens [B]：文本块位置与有效长度（含 EOS），用于屏蔽 pad 列
def build_clean_noisy_mask(
    block_sizes: list[int],
    noisy_flags: list[bool],
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    neg_value: float = -1.0e4,
    text_block_idx: int | None = None,
    text_valid_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    assert len(block_sizes) == len(noisy_flags), "block_sizes 与 noisy_flags 长度必须一致"
    n_blocks = len(block_sizes)
    total = int(sum(block_sizes))

    # 块级可见性：vis[i,j] = noisy[i] or (not noisy[j])
    noisy = torch.tensor(noisy_flags, dtype=torch.bool, device=device)
    vis_block = noisy.reshape(-1, 1) | (~noisy.reshape(1, -1))
    assert vis_block.shape == (n_blocks, n_blocks)

    # 展开到 token 级 [T,T]
    token_block_id = torch.repeat_interleave(
        torch.arange(n_blocks, device=device), torch.tensor(block_sizes, device=device)
    )
    vis_tok = vis_block[token_block_id][:, token_block_id]

    mask = torch.where(
        vis_tok,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), float(neg_value), dtype=dtype, device=device),
    )
    mask = mask.reshape(1, 1, total, total).repeat(batch_size, 1, 1, 1)

    # 文本 pad 列屏蔽（对所有 query）
    if text_block_idx is not None and text_valid_lens is not None:
        text_start = int(sum(block_sizes[:text_block_idx]))
        text_len = int(block_sizes[text_block_idx])
        positions = torch.arange(text_len, device=device).reshape(1, -1)
        pad_cols = positions >= text_valid_lens.to(device).reshape(-1, 1)  # [B, text_len]
        pad_cols_4d = pad_cols.reshape(batch_size, 1, 1, text_len)
        col_slice = slice(text_start, text_start + text_len)
        mask[:, :, :, col_slice] = torch.where(
            pad_cols_4d,
            torch.full((), float(neg_value), dtype=dtype, device=device),
            mask[:, :, :, col_slice],
        )
    return mask
######### // code // ##########
