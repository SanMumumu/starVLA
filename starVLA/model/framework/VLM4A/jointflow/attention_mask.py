"""PR3: Hybrid block-causal mask builder."""

from __future__ import annotations

from pathlib import Path

import torch


######### // code // ##########
def build_block_causal_mask(
    block_sizes: list[int],
    text_valid_lens: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    hybrid: bool = True,
    neg_value: float = -1e4,
) -> torch.Tensor:
    if not block_sizes or block_sizes[0] <= 0:
        raise ValueError(f"Invalid block_sizes: {block_sizes}")
    total = int(sum(block_sizes))
    batch_size = int(text_valid_lens.shape[0])
    neg = float(neg_value)

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
