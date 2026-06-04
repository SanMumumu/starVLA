"""PR3: Hybrid block-causal mask builder."""

from __future__ import annotations

from pathlib import Path

import torch


######### // code // ##########
# 中文注释：构造 block-causal 4D additive mask。
# 输入 block_sizes 为同一 batch 的块长度列表，text_valid_lens [B] 只作用于第 0 个 text block。
# 输出 mask [B,1,T,T]：可见位置为 0，不可见位置为 finfo.min。
def build_block_causal_mask(
    block_sizes: list[int],
    text_valid_lens: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    hybrid: bool = True,
) -> torch.Tensor:
    if not block_sizes or block_sizes[0] <= 0:
        raise ValueError(f"Invalid block_sizes: {block_sizes}")
    total = int(sum(block_sizes))
    batch_size = int(text_valid_lens.shape[0])
    neg = torch.finfo(dtype if torch.is_floating_point(torch.empty((), dtype=dtype)) else torch.float32).min

    if hybrid:
        visible = torch.zeros(total, total, dtype=torch.bool, device=device)
        q_start = 0
        for i, q_size in enumerate(block_sizes):
            q_end = q_start + q_size
            k_start = 0
            for j, k_size in enumerate(block_sizes):
                k_end = k_start + k_size
                if j <= i:
                    visible[q_start:q_end, k_start:k_end] = True
                k_start = k_end
            q_start = q_end
    else:
        visible = torch.ones(total, total, dtype=torch.bool, device=device).tril()

    mask = torch.full((batch_size, 1, total, total), neg, dtype=dtype, device=device)
    mask[:, :, visible] = 0.0

    text_block = int(block_sizes[0])
    for batch_idx, valid_len in enumerate(text_valid_lens.tolist()):
        valid_len = int(valid_len)
        if valid_len < text_block:
            mask[batch_idx, :, :, valid_len:text_block] = neg
    return mask


# 中文注释：把 mask 渲染成 PNG，用于核对块内双向、块间 causal、text padding key 全暗。
def visualize_mask(mask4d: torch.Tensor, block_names: list[str], save_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mask = mask4d[0, 0].detach().float().cpu()
    visible = mask == 0
    plt.figure(figsize=(6, 6))
    plt.imshow(visible.numpy(), cmap="gray", interpolation="nearest")
    plt.title(" / ".join(block_names))
    plt.xlabel("key")
    plt.ylabel("query")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
######### // code // ##########

