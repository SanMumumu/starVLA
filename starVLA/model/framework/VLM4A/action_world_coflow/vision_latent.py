"""Frozen multi-layer Qwen3-VL patch latents with deterministic 2-D pooling."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn


def select_vision_layers(
    depth: int,
    strategy: str,
    num_layers: int,
    explicit_layers: list[int] | tuple[int, ...] | None,
) -> tuple[int, ...]:
    if depth <= 0:
        raise ValueError(f"Qwen vision depth must be positive, got {depth}")
    strategy = str(strategy).lower()
    if strategy == "all_layers":
        return tuple(range(depth))
    if strategy == "explicit":
        if not explicit_layers:
            raise ValueError("world_feature_layers is required for strategy='explicit'")
        resolved = []
        for index in explicit_layers:
            index = int(index)
            index = depth + index if index < 0 else index
            if not 0 <= index < depth:
                raise ValueError(f"vision layer {index} is outside depth={depth}")
            if index not in resolved:
                resolved.append(index)
        return tuple(resolved)
    if strategy != "evenly_spaced":
        raise ValueError(
            f"Unsupported world_feature_layer_strategy={strategy!r}; "
            "expected evenly_spaced, explicit, or all_layers"
        )
    num_layers = int(num_layers)
    if not 1 <= num_layers <= depth:
        raise ValueError(f"world_feature_num_layers must be in [1,{depth}], got {num_layers}")
    if num_layers == 1:
        return (depth - 1,)
    indices = torch.linspace(0, depth - 1, steps=num_layers).round().to(torch.long).tolist()
    if len(set(indices)) != num_layers:
        raise RuntimeError(f"failed to choose {num_layers} distinct layers from depth={depth}: {indices}")
    return tuple(int(index) for index in indices)


class FrozenQwenMultiLayerVisionLatent(nn.Module):
    """Capture raw Qwen vision blocks, fuse layers, and pool to a fixed grid.

    The Qwen vision module is passed into ``capture``/``encode`` rather than
    registered here.  This prevents a second state-dict path to the same frozen
    encoder and keeps old/new checkpoint ownership unambiguous.
    """

    def __init__(
        self,
        *,
        vision_depth: int,
        strategy: str = "evenly_spaced",
        num_layers: int = 4,
        explicit_layers: list[int] | None = None,
        fusion_type: str = "fixed_mean",
        fixed_layer_weights: list[float] | None = None,
        normalization: str = "layernorm",
        token_grid: tuple[int, int] = (4, 8),
        spatial_pool_type: str = "adaptive_avg_pool2d",
    ) -> None:
        super().__init__()
        self.layer_indices = select_vision_layers(vision_depth, strategy, num_layers, explicit_layers)
        self.fusion_type = str(fusion_type).lower()
        if self.fusion_type not in {"fixed_mean", "fixed_weighted_mean", "trainable_weighted_mean"}:
            raise ValueError(f"Unsupported world_layer_fusion_type={fusion_type!r}")
        self.normalization = str(normalization).lower()
        if self.normalization not in {"layernorm", "l2"}:
            raise ValueError(f"Unsupported world_feature_normalization={normalization!r}")
        self.token_grid = tuple(int(value) for value in token_grid)
        if len(self.token_grid) != 2 or min(self.token_grid) <= 0:
            raise ValueError(f"world_token_grid must contain two positive integers, got {token_grid}")
        self.num_world_tokens = self.token_grid[0] * self.token_grid[1]
        self.spatial_pool_type = str(spatial_pool_type).lower()
        if self.spatial_pool_type != "adaptive_avg_pool2d":
            raise ValueError("The stable target extractor currently supports only adaptive_avg_pool2d")

        count = len(self.layer_indices)
        if self.fusion_type == "fixed_mean":
            weights = torch.full((count,), 1.0 / count, dtype=torch.float32)
            self.register_buffer("layer_weights", weights, persistent=True)
        elif self.fusion_type == "fixed_weighted_mean":
            if fixed_layer_weights is None or len(fixed_layer_weights) != count:
                raise ValueError(
                    f"fixed_layer_weights must have {count} entries for selected layers {self.layer_indices}"
                )
            weights = torch.tensor(fixed_layer_weights, dtype=torch.float32)
            if bool((weights < 0).any()) or float(weights.sum()) <= 0:
                raise ValueError("fixed_layer_weights must be non-negative with a positive sum")
            self.register_buffer("layer_weights", weights / weights.sum(), persistent=True)
        else:
            # Optional ablation only.  The default fixed_mean has no trainable
            # target-space parameters.  Future targets are still detached by
            # the framework even when this ablation is selected.
            self.layer_weight_logits = nn.Parameter(torch.zeros(count, dtype=torch.float32))

    def normalized_layer_weights(self, *, device, dtype) -> torch.Tensor:
        if self.fusion_type == "trainable_weighted_mean":
            return self.layer_weight_logits.softmax(dim=0).to(device=device, dtype=dtype)
        return self.layer_weights.to(device=device, dtype=dtype)

    @contextmanager
    def capture(self, visual: nn.Module) -> Iterator[dict[int, torch.Tensor]]:
        blocks = getattr(visual, "blocks", None)
        if blocks is None:
            raise TypeError("Qwen vision module does not expose .blocks; cannot capture multi-layer patch features")
        if len(blocks) <= max(self.layer_indices):
            raise ValueError(
                f"Qwen vision depth changed after construction: depth={len(blocks)}, layers={self.layer_indices}"
            )
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(index: int):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, (tuple, list)) else output
                if not torch.is_tensor(value) or value.ndim != 2:
                    raise RuntimeError(
                        f"Qwen vision block {index} returned unsupported shape/type: "
                        f"{type(value).__name__} {getattr(value, 'shape', None)}"
                    )
                captured[index] = value

            return hook

        try:
            for index in self.layer_indices:
                handles.append(blocks[index].register_forward_hook(make_hook(index)))
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def encode(self, visual: nn.Module, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        with self.capture(visual) as captured:
            visual(pixel_values, grid_thw=grid_thw)
        return self.pool_captured(captured, grid_thw, int(visual.config.spatial_merge_size))

    def _normalize(self, value: torch.Tensor) -> torch.Tensor:
        if self.normalization == "l2":
            return F.normalize(value, dim=-1, eps=1.0e-6)
        return F.layer_norm(value, (value.shape[-1],), weight=None, bias=None, eps=1.0e-6)

    @staticmethod
    def _restore_spatial_grid(tokens: torch.Tensor, t: int, h: int, w: int, merge: int) -> torch.Tensor:
        if h % merge or w % merge:
            raise ValueError(f"Qwen grid {(t, h, w)} is not divisible by spatial_merge_size={merge}")
        expected = t * h * w
        if tokens.shape[0] != expected:
            raise ValueError(f"raw vision token count={tokens.shape[0]} does not match grid product={expected}")
        # Qwen3-VL permutes (h//m, w//m, m, m) before flattening.  Undo that
        # order so adaptive pooling operates on the physical row/column grid.
        return (
            tokens.view(t, h // merge, w // merge, merge, merge, tokens.shape[-1])
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(t, h, w, tokens.shape[-1])
        )

    def pool_captured(
        self,
        captured: dict[int, torch.Tensor],
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
    ) -> torch.Tensor:
        missing = [index for index in self.layer_indices if index not in captured]
        if missing:
            raise RuntimeError(f"Qwen vision hooks did not capture selected layers: {missing}")
        layers = [captured[index] for index in self.layer_indices]
        total_tokens = sum(int(row.prod().item()) for row in grid_thw)
        if any(layer.ndim != 2 or layer.shape[0] != total_tokens for layer in layers):
            raise ValueError(
                "captured Qwen layers must all be [sum(t*h*w), hidden]; "
                f"grid_total={total_tokens}, shapes={[tuple(layer.shape) for layer in layers]}"
            )
        hidden_dims = {int(layer.shape[-1]) for layer in layers}
        if len(hidden_dims) != 1:
            raise ValueError(f"selected Qwen layers have different hidden dims: {sorted(hidden_dims)}")

        weights = self.normalized_layer_weights(device=layers[0].device, dtype=layers[0].dtype)
        fused = sum(weight * self._normalize(layer) for weight, layer in zip(weights, layers, strict=True))
        sizes = [int(row.prod().item()) for row in grid_thw]
        per_image = fused.split(sizes, dim=0)
        pooled = []
        for tokens, row in zip(per_image, grid_thw.tolist(), strict=True):
            t, h, w = (int(value) for value in row)
            grid = self._restore_spatial_grid(tokens, t, h, w, spatial_merge_size)
            grid = grid.mean(dim=0).permute(2, 0, 1).unsqueeze(0)
            grid = F.adaptive_avg_pool2d(grid, self.token_grid)
            pooled.append(grid.squeeze(0).permute(1, 2, 0).reshape(self.num_world_tokens, -1))
        return torch.stack(pooled, dim=0)
