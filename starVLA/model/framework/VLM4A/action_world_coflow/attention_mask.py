"""Explicit block-causal masks for Action--World Co-Flow.

SDPA uses ``True`` to mean "this query may read this key".  The block id is
therefore the only causal coordinate: tokens inside one block are genuinely
bidirectional, while a token can never read a later block.
"""

from __future__ import annotations

import torch

TOKEN_CONTEXT = 0
TOKEN_ACTION = 1
TOKEN_WORLD = 2
TOKEN_TYPE_NAMES = {
    TOKEN_CONTEXT: "context",
    TOKEN_ACTION: "action",
    TOKEN_WORLD: "world",
}


def build_block_causal_attention_mask(
    block_ids: torch.Tensor,
    *,
    key_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a batched SDPA mask from explicit block ids.

    Args:
        block_ids: ``[L]`` or ``[B, L]`` non-negative integer ids.  Context is
            block 0, the first action/world pair is block 1, and so on.
        key_valid_mask: optional ``[B, L]`` mask used for left-padded Qwen
            context.  Invalid *keys* are hidden.  Invalid context query rows
            still read valid context keys so SDPA never receives an all-False
            row; their outputs are ignored and cannot influence later blocks.

    Returns:
        Boolean mask ``[B, 1, L, L]`` with query on the penultimate axis and
        key on the final axis.
    """

    block_ids = torch.as_tensor(block_ids)
    if block_ids.ndim == 1:
        block_ids = block_ids.unsqueeze(0)
    if block_ids.ndim != 2:
        raise ValueError(f"block_ids must have shape [L] or [B,L], got {tuple(block_ids.shape)}")
    if block_ids.numel() == 0:
        raise ValueError("block_ids cannot be empty")
    if block_ids.dtype == torch.bool or block_ids.is_floating_point():
        raise TypeError(f"block_ids must be an integer tensor, got {block_ids.dtype}")
    if bool((block_ids < 0).any()):
        raise ValueError("block_ids must be non-negative")

    # key_block <= query_block: context only sees context; block 1 sees 0/1;
    # block 2 sees 0/1/2.  Equality gives full bidirectionality within a block.
    allowed = block_ids.unsqueeze(-1) >= block_ids.unsqueeze(-2)

    if key_valid_mask is not None:
        key_valid_mask = torch.as_tensor(key_valid_mask, device=block_ids.device, dtype=torch.bool)
        if key_valid_mask.ndim == 1:
            key_valid_mask = key_valid_mask.unsqueeze(0)
        if key_valid_mask.shape[-1] != block_ids.shape[-1]:
            raise ValueError(
                "key_valid_mask length must match block_ids: "
                f"mask={tuple(key_valid_mask.shape)}, blocks={tuple(block_ids.shape)}"
            )
        if block_ids.shape[0] == 1 and key_valid_mask.shape[0] > 1:
            block_ids = block_ids.expand(key_valid_mask.shape[0], -1)
            allowed = block_ids.unsqueeze(-1) >= block_ids.unsqueeze(-2)
        elif key_valid_mask.shape[0] == 1 and block_ids.shape[0] > 1:
            key_valid_mask = key_valid_mask.expand(block_ids.shape[0], -1)
        elif key_valid_mask.shape[0] != block_ids.shape[0]:
            raise ValueError(
                "key_valid_mask batch must be 1 or match block_ids: "
                f"mask={tuple(key_valid_mask.shape)}, blocks={tuple(block_ids.shape)}"
            )
        if not bool(key_valid_mask.any(dim=-1).all()):
            raise ValueError("every example must expose at least one valid key")
        allowed = allowed & key_valid_mask[:, None, :]

    return allowed.unsqueeze(1)


def render_bool_attention_mask(mask: torch.Tensor, true_char: str = "#", false_char: str = ".") -> str:
    """Return a small human-readable query-by-key visualization for tests/docs."""

    mask = torch.as_tensor(mask, dtype=torch.bool)
    while mask.ndim > 2:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"mask must reduce to [L,L], got {tuple(mask.shape)}")
    return "\n".join("".join(true_char if value else false_char for value in row.tolist()) for row in mask)
