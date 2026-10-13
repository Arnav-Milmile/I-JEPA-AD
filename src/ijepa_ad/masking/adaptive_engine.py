"""Coarse-to-fine masking orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .coarse_masker import CoarseMasker, PatchGrid
from .fine_masker import FineMasker
from .fusion import fuse_scores
from .score_maps import patch_scores_to_heatmap


@dataclass(frozen=True)
class MaskingResult:
    coarse_score: Tensor
    fine_score: Tensor
    final_score: Tensor
    heatmap: Tensor
    evaluated_regions: int
    total_regions: int


class AdaptiveMaskingEngine:
    """Run fixed, dense, or gated coarse-to-fine region evaluation."""

    def __init__(
        self,
        *,
        coarse_size: int = 4,
        fine_size: int = 1,
        threshold: float = 0.5,
        lam: float = 1.0,
    ) -> None:
        grid = PatchGrid()
        self.coarse = CoarseMasker(grid, coarse_size)
        self.fine = FineMasker(grid, fine_size)
        self.threshold = threshold
        self.lam = lam

    def evaluate(
        self, patch_scores: Tensor, *, mode: str = "adaptive"
    ) -> MaskingResult:
        """Evaluate a patch score map; score tensors are `[196]` or `[N,196]`."""
        if patch_scores.ndim == 1:
            patch_scores = patch_scores.unsqueeze(0)
        if patch_scores.ndim != 2 or patch_scores.shape[-1] != 196:
            raise ValueError("patch_scores must have shape [N, 196]")
        coarse = self.coarse.score(patch_scores)
        candidates = self.fine.candidates()
        if mode == "dense":
            fine = patch_scores.mean(dim=-1)
            evaluated = len(candidates)
        elif mode == "fixed":
            fine = coarse.mean(dim=-1)
            evaluated = len(self.coarse.windows())
        elif mode == "adaptive":
            selected = [
                self.fine.select(row.reshape(14, 14), self.threshold)
                for row in patch_scores
            ]
            fine_values = []
            for row, selection in zip(patch_scores, selected, strict=True):
                values = [
                    row[list(window.token_indices)].mean()
                    for window in selection.windows
                ]
                fine_values.append(
                    torch.stack(values).mean() if values else row.new_zeros(())
                )
            fine = torch.stack(fine_values)
            evaluated = sum(selection.count for selection in selected)
        else:
            raise ValueError("mode must be adaptive, dense, or fixed")
        final_score = fuse_scores(coarse.mean(dim=-1), fine, self.lam)
        return MaskingResult(
            coarse.mean(dim=-1),
            fine,
            final_score,
            patch_scores_to_heatmap(patch_scores),
            evaluated,
            len(candidates),
        )
