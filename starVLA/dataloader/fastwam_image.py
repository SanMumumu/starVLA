"""Lightweight, shared FastWAM RoboTwin image composition.

Kept separate from the LeRobot dataset module so the RoboTwin evaluation
environment does not need training-only dependencies such as PyTorch3D.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf


FASTWAM_COMPOSITE_LAYOUT = "fastwam_composite"
FASTWAM_COMPOSITE_VIEW_KEY = "video.fastwam_composite"
# PIL/config order is (width, height); numpy/tensor order is (height, width).
FASTWAM_COMPOSITE_SIZE = (320, 384)


def _to_chw_float(image: Image.Image | np.ndarray) -> torch.Tensor:
    array = np.array(image.convert("RGB") if isinstance(image, Image.Image) else image, copy=True)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected an HWC image with 1/3/4 channels, got {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    tensor = torch.from_numpy(array[..., :3]).permute(2, 0, 1).to(torch.float32)
    if tensor.numel() and float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp_(0.0, 1.0)


def build_robotwin_composite(images: Sequence[Image.Image | np.ndarray]) -> Image.Image:
    """Reproduce FastWAM's two-stage three-camera layout exactly."""

    if len(images) != 3:
        raise ValueError(f"FastWAM composite requires [head, left_wrist, right_wrist], got {len(images)} views")
    frames = [
        tvf.resize(
            _to_chw_float(image),
            [240, 320],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        for image in images
    ]
    head = tvf.resize(frames[0], [256, 320], interpolation=InterpolationMode.BILINEAR, antialias=True)
    left = tvf.resize(frames[1], [128, 160], interpolation=InterpolationMode.BILINEAR, antialias=True)
    right = tvf.resize(frames[2], [128, 160], interpolation=InterpolationMode.BILINEAR, antialias=True)
    composite = torch.cat([head, torch.cat([left, right], dim=-1)], dim=-2)
    array = composite.mul(255.0).round().clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


__all__ = [
    "FASTWAM_COMPOSITE_LAYOUT",
    "FASTWAM_COMPOSITE_SIZE",
    "FASTWAM_COMPOSITE_VIEW_KEY",
    "build_robotwin_composite",
]
