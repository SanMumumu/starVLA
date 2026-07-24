"""Backward-compatible training import for the FastWAM compositor.

The canonical implementation is deployment-safe so training and evaluation
share identical pixels without making simulator environments import the
training dataloader package.
"""

from deployment.fastwam_image import (
    FASTWAM_COMPOSITE_LAYOUT,
    FASTWAM_COMPOSITE_SIZE,
    FASTWAM_COMPOSITE_VIEW_KEY,
    TRI_VIEW_COMPOSITE_LAYOUT,
    TRI_VIEW_COMPOSITE_VIEW_KEY,
    build_robotwin_composite,
)

__all__ = [
    "FASTWAM_COMPOSITE_LAYOUT",
    "FASTWAM_COMPOSITE_SIZE",
    "FASTWAM_COMPOSITE_VIEW_KEY",
    "TRI_VIEW_COMPOSITE_LAYOUT",
    "TRI_VIEW_COMPOSITE_VIEW_KEY",
    "build_robotwin_composite",
]
