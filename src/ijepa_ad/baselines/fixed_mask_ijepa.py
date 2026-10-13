"""Fixed-mask I-JEPA comparison baseline."""

from __future__ import annotations

from torch import Tensor

from ijepa_ad.masking import AdaptiveMaskingEngine


class FixedMaskIJEPA:
    """Score each image with a fixed coarse mask schedule."""

    def __init__(self, *, coarse_size: int = 4) -> None:
        self.engine = AdaptiveMaskingEngine(coarse_size=coarse_size)

    def score(self, patch_scores: Tensor) -> Tensor:
        return self.engine.evaluate(patch_scores, mode="fixed").final_score

    def __call__(self, patch_scores: Tensor) -> Tensor:
        return self.score(patch_scores)
