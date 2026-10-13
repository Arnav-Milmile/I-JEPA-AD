"""MAE reconstruction comparison baseline with an injected reconstruction model."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor


class MAEReconstructionBaseline:
    """Compute per-image mean squared reconstruction error."""

    def __init__(self, reconstruct: Callable[[Tensor], Tensor] | None = None) -> None:
        self.reconstruct = reconstruct

    def score(self, images: Tensor, reconstruction: Tensor | None = None) -> Tensor:
        predicted = reconstruction
        if predicted is None:
            if self.reconstruct is None:
                raise RuntimeError(
                    "provide reconstruction or a reconstruction callable"
                )
            predicted = self.reconstruct(images)
        if predicted.shape != images.shape:
            raise ValueError("reconstruction must have the same shape as images")
        return (images - predicted).square().flatten(start_dim=1).mean(dim=1)

    def __call__(self, images: Tensor, reconstruction: Tensor | None = None) -> Tensor:
        return self.score(images, reconstruction)


MAEReconstruction = MAEReconstructionBaseline
