"""Deployment-safe two-camera composition for LIBERO.

LIBERO supplies two square RGB observations.  The canonical model input keeps
both views undistorted and joins them horizontally as::

    [ third-person 256x256 | wrist 256x256 ] -> 512x256

This module is shared by training and simulator-side evaluation so camera
order, interpolation, and output geometry cannot drift between environments.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image


LIBERO_COMPOSITE_LAYOUT = "libero_dual_view_composite"
LIBERO_COMPOSITE_VIEW_KEY = "video.libero_dual_view_composite"
# PIL/config order is (width, height); NumPy/tensor order is (height, width).
LIBERO_COMPOSITE_SIZE = (512, 256)
LIBERO_COMPOSITE_SOURCE_VIEW_KEYS = (
    "video.primary_image",
    "video.wrist_image",
)
LIBERO_VIEW_SIZE = (256, 256)


def _to_rgb(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "LIBERO composite expects HWC images with 1/3/4 channels, "
            f"got {array.shape}"
        )
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if array.size and float(np.nanmax(array)) <= 1.5 else 1.0
        array = np.nan_to_num(array * scale).round().clip(0, 255)
    return Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")


def build_libero_composite(
    images: Sequence[Image.Image | np.ndarray],
) -> Image.Image:
    """Join ``[third_person, wrist]`` into the canonical 512x256 image."""

    if len(images) != 2:
        raise ValueError(
            "LIBERO composite requires [third_person, wrist], "
            f"got {len(images)} views"
        )
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    views = [
        image
        if image.size == LIBERO_VIEW_SIZE
        else image.resize(LIBERO_VIEW_SIZE, resampling)
        for image in (_to_rgb(value) for value in images)
    ]
    composite = Image.new("RGB", LIBERO_COMPOSITE_SIZE)
    composite.paste(views[0], (0, 0))
    composite.paste(views[1], (LIBERO_VIEW_SIZE[0], 0))
    return composite


__all__ = [
    "LIBERO_COMPOSITE_LAYOUT",
    "LIBERO_COMPOSITE_SIZE",
    "LIBERO_COMPOSITE_SOURCE_VIEW_KEYS",
    "LIBERO_COMPOSITE_VIEW_KEY",
    "LIBERO_VIEW_SIZE",
    "build_libero_composite",
]
