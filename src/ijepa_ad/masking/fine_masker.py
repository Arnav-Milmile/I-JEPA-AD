"""Suspicious-region selection and fine masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .coarse_masker import MaskWindow, PatchGrid


@dataclass(frozen=True)
class FineSelection:
    windows: tuple[MaskWindow, ...]
    threshold: float

    @property
    def count(self) -> int:
        return len(self.windows)


class FineMasker:
    """Select only coarse regions whose score crosses a suspiciousness gate."""

    def __init__(self, grid: PatchGrid | None = None, fine_size: int = 1) -> None:
        self.grid = grid or PatchGrid()
        if fine_size < 1 or fine_size > self.grid.size:
            raise ValueError("fine_size must fit inside the patch grid")
        self.fine_size = fine_size

    def candidates(self) -> tuple[MaskWindow, ...]:
        result: list[MaskWindow] = []
        for row in range(0, self.grid.size, self.fine_size):
            for column in range(0, self.grid.size, self.fine_size):
                height = min(self.fine_size, self.grid.size - row)
                width = min(self.fine_size, self.grid.size - column)
                indices = tuple(
                    self.grid.patch_to_index(r, c)
                    for r in range(row, row + height)
                    for c in range(column, column + width)
                )
                result.append(MaskWindow(indices, row, column, height, width))
        return tuple(result)

    def select(self, coarse_scores: Tensor, threshold: float) -> FineSelection:
        """Select candidates whose score is at least `threshold`.

        Candidate scores may be token-shaped `[H, W]`, or one score per candidate.
        A scalar threshold produces no masks for a normal all-zero score map.
        """
        if coarse_scores.ndim == 2 and coarse_scores.shape == (
            self.grid.size,
            self.grid.size,
        ):
            values = torch.stack(
                [
                    coarse_scores[
                        w.row : w.row + w.height, w.column : w.column + w.width
                    ].mean()
                    for w in self.candidates()
                ]
            )
        elif coarse_scores.ndim == 1 and coarse_scores.numel() == len(
            self.candidates()
        ):
            values = coarse_scores
        else:
            raise ValueError(
                "coarse_scores must be [grid, grid] or one score per candidate"
            )
        selected = tuple(
            w
            for w, value in zip(self.candidates(), values, strict=True)
            if float(value) >= threshold
        )
        return FineSelection(selected, threshold)

    def masks(
        self, selection: FineSelection, *, device: torch.device | None = None
    ) -> Tensor:
        result = torch.zeros(
            (selection.count, self.grid.tokens), dtype=torch.bool, device=device
        )
        for index, window in enumerate(selection.windows):
            result[index, list(window.token_indices)] = True
        return result


def select_suspicious_regions(
    coarse_scores: Tensor, threshold: float, fine_size: int = 1
) -> FineSelection:
    """Select fine windows from a 14×14 coarse score map."""
    return FineMasker(fine_size=fine_size).select(coarse_scores, threshold)
