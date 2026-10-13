"""Coarse/fine anomaly-score fusion."""

from torch import Tensor


def fuse_scores(
    coarse_score: Tensor | float, fine_score: Tensor | float, lam: float
) -> Tensor:
    """Compute `coarse_score + lam * fine_score` with tensor broadcasting."""
    if lam < 0:
        raise ValueError("lam must be non-negative")
    return coarse_score + lam * fine_score


score_fusion = fuse_scores
fuse_anomaly_scores = fuse_scores
