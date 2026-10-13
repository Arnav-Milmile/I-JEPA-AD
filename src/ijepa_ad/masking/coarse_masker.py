"""Coarse masking and patch-grid coordinate utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PatchGrid:
    """Mapping between image pixels and I-JEPA patch-token coordinates."""

    image_size: int = 224
    patch_size: int = 16

    @property
    def size(self) -> int:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        return self.image_size // self.patch_size

    @property
    def tokens(self) -> int:
        return self.size * self.size

    def pixel_to_patch(self, x: int, y: int) -> tuple[int, int]:
        """Return `(row, column)` for a pixel coordinate."""
        if not (0 <= x < self.image_size and 0 <= y < self.image_size):
            raise ValueError("pixel coordinate is outside the image")
        return y // self.patch_size, x // self.patch_size

    def patch_to_index(self, row: int, column: int) -> int:
        if not (0 <= row < self.size and 0 <= column < self.size):
            raise ValueError("patch coordinate is outside the grid")
        return row * self.size + column

    def index_to_patch(self, index: int) -> tuple[int, int]:
        if not (0 <= index < self.tokens):
            raise ValueError("patch index is outside the grid")
        return index // self.size, index % self.size


@dataclass(frozen=True)
class MaskWindow:
    """A rectangular token mask and its corresponding image-space box."""

    token_indices: tuple[int, ...]
    row: int
    column: int
    height: int
    width: int


class CoarseMasker:
    """Generate non-overlapping coarse windows over a patch grid."""

    def __init__(self, grid: PatchGrid | None = None, window_size: int = 4) -> None:
        self.grid = grid or PatchGrid()
        if window_size < 1 or window_size > self.grid.size:
            raise ValueError("window_size must fit inside the patch grid")
        self.window_size = window_size

    def windows(self) -> tuple[MaskWindow, ...]:
        step = self.window_size
        windows: list[MaskWindow] = []
        for row in range(0, self.grid.size, step):
            for column in range(0, self.grid.size, step):
                height = min(step, self.grid.size - row)
                width = min(step, self.grid.size - column)
                indices = tuple(
                    self.grid.patch_to_index(r, c)
                    for r in range(row, row + height)
                    for c in range(column, column + width)
                )
                windows.append(MaskWindow(indices, row, column, height, width))
        return tuple(windows)

    def masks(self, *, device: torch.device | None = None) -> Tensor:
        """Return boolean masks shaped `[num_windows, num_tokens]`."""
        result = torch.zeros(
            (len(self.windows()), self.grid.tokens), dtype=torch.bool, device=device
        )
        for index, window in enumerate(self.windows()):
            result[index, list(window.token_indices)] = True
        return result

    def score(self, token_scores: Tensor) -> Tensor:
        """Average token scores in every coarse window."""
        if token_scores.shape[-1] != self.grid.tokens:
            raise ValueError("token_scores must end with the patch-token dimension")
        masks = self.masks(device=token_scores.device).to(token_scores.dtype)
        return token_scores @ masks.T / masks.sum(dim=1).clamp_min(1)


def coarse_anomaly_score(token_scores: Tensor, window_size: int = 4) -> Tensor:
    """Compute coarse-window scores directly from per-patch scores."""
    return CoarseMasker(window_size=window_size).score(token_scores)


def latent_anomaly_scores(actual: Tensor, predicted: Tensor) -> Tensor:
    """Convert actual/predicted latent tokens into one MSE score per patch."""
    if actual.shape != predicted.shape or actual.ndim < 2:
        raise ValueError(
            "actual and predicted latents must have matching [.., tokens, dim] shapes"
        )
    return (actual - predicted).square().mean(dim=-1)
