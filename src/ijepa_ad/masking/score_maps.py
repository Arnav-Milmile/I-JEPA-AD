"""Conversion of patch scores to image-sized anomaly heatmaps."""

from __future__ import annotations

import torch
from torch import Tensor

from .coarse_masker import PatchGrid


def patch_scores_to_heatmap(
    patch_scores: Tensor,
    grid: PatchGrid | None = None,
    *,
    output_size: int | tuple[int, int] = 224,
) -> Tensor:
    """Expand `[196]` or `[N,196]` patch scores to normalized image heatmaps."""
    grid = grid or PatchGrid()
    if patch_scores.shape[-1] != grid.tokens:
        raise ValueError("patch_scores must end with the number of grid tokens")
    if patch_scores.ndim == 1:
        patch_scores = patch_scores.unsqueeze(0)
    if patch_scores.ndim != 2:
        raise ValueError("patch_scores must be one- or two-dimensional")
    maps = patch_scores.reshape(-1, 1, grid.size, grid.size).float()
    size = (output_size, output_size) if isinstance(output_size, int) else output_size
    maps = torch.nn.functional.interpolate(maps, size=size, mode="nearest")
    minimum = maps.amin(dim=(-2, -1), keepdim=True)
    maximum = maps.amax(dim=(-2, -1), keepdim=True)
    return (maps - minimum) / (maximum - minimum).clamp_min(torch.finfo(maps.dtype).eps)


anomaly_heatmap = patch_scores_to_heatmap
build_heatmap = patch_scores_to_heatmap
