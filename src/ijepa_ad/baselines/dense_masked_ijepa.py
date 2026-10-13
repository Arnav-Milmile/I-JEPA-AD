"""Dense-mask I-JEPA comparison baseline."""

from __future__ import annotations

from torch import Tensor

from ijepa_ad.masking import AdaptiveMaskingEngine


class DenseMaskedIJEPA:
    """Evaluate every fine patch region independently."""

    def __init__(self, *, coarse_size: int = 4, fine_size: int = 1) -> None:
        self.engine = AdaptiveMaskingEngine(
            coarse_size=coarse_size, fine_size=fine_size
        )

    def score(self, patch_scores: Tensor) -> Tensor:
        return self.engine.evaluate(patch_scores, mode="dense").final_score

    def __call__(self, patch_scores: Tensor) -> Tensor:
        return self.score(patch_scores)
