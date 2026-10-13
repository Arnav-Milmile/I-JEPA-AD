"""Small dependency-free PatchCore-style nearest-neighbour baseline."""

from __future__ import annotations

import torch
from torch import Tensor


class PatchCoreBaseline:
    """Store nominal patch embeddings and score queries by nearest distance."""

    def __init__(self, coreset_fraction: float = 1.0) -> None:
        if not 0 < coreset_fraction <= 1:
            raise ValueError("coreset_fraction must be in (0, 1]")
        self.coreset_fraction = coreset_fraction
        self.memory_bank: Tensor | None = None

    def fit(self, features: Tensor) -> PatchCoreBaseline:
        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must be a non-empty [N, D] or [N, P, D] tensor")
        count = max(1, int(features.shape[0] * self.coreset_fraction))
        self.memory_bank = features.detach()[:count].clone()
        return self

    def score(self, features: Tensor) -> Tensor:
        if self.memory_bank is None:
            raise RuntimeError("fit must be called before score")
        original_shape = features.shape[:-1]
        flattened = features.reshape(-1, features.shape[-1])
        distances = torch.cdist(
            flattened,
            self.memory_bank.to(device=flattened.device, dtype=flattened.dtype),
        )
        return (
            distances.min(dim=1).values.reshape(original_shape).amax(dim=-1)
            if len(original_shape) == 2
            else distances.min(dim=1).values.reshape(original_shape)
        )

    def __call__(self, features: Tensor) -> Tensor:
        return self.score(features)


PatchCore = PatchCoreBaseline
