"""Deterministic nominal-only train/validation split for predictor training.

The official MVTec ``test`` split is never touched here: validation samples are
held out from ``train/good`` only, so model selection never sees defect labels
(DECISIONS.md, 2026-08-20 "Keep official MVTec test data evaluation-only").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from ijepa_ad.utils.seeding import seeded_rng

from .validators import (
    DatasetValidationError,
    SampleMetadata,
    validate_training_records,
)

SampleT = TypeVar("SampleT", bound=SampleMetadata)


class InsufficientNominalSamplesError(DatasetValidationError):
    """Raised when a category has too few nominal images to split."""


@dataclass(frozen=True)
class NominalSplit(Generic[SampleT]):
    """Disjoint nominal-only training and validation samples."""

    train: tuple[SampleT, ...]
    val: tuple[SampleT, ...]
    seed: int
    val_fraction: float


def split_nominal_train_val(
    samples: Sequence[SampleT],
    *,
    val_fraction: float,
    seed: int,
    min_train: int = 1,
    min_val: int = 1,
) -> NominalSplit[SampleT]:
    """Hold out ``round(len(samples) * val_fraction)`` nominal samples.

    ``samples`` must be official training records (``train/good`` only); any
    defect, masked, or non-train record is rejected. Samples are ordered by
    content hash before shuffling, so the split depends only on image content
    and ``seed`` — not on filesystem order or absolute paths. Both returned
    tuples are sorted by image path for readability.
    """
    if isinstance(val_fraction, bool) or not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if min_train < 1 or min_val < 1:
        raise ValueError("min_train and min_val must be at least 1")

    records = tuple(samples)
    validate_training_records(records)

    n_val = round(len(records) * val_fraction)
    n_train = len(records) - n_val
    if n_val < min_val or n_train < min_train:
        raise InsufficientNominalSamplesError(
            f"cannot split {len(records)} nominal images with "
            f"val_fraction={val_fraction}: would give {n_train} train / {n_val} "
            f"validation, need at least {min_train} / {min_val}"
        )

    ordered = sorted(records, key=lambda record: record.image_id)
    seeded_rng(seed).shuffle(ordered)
    val = ordered[:n_val]
    train = ordered[n_val:]

    overlap = {r.image_id for r in train} & {r.image_id for r in val}
    if overlap:
        raise DatasetValidationError(
            f"train/validation leakage detected for image IDs: {sorted(overlap)}"
        )

    return NominalSplit(
        train=tuple(sorted(train, key=lambda record: record.image_path)),
        val=tuple(sorted(val, key=lambda record: record.image_path)),
        seed=seed,
        val_fraction=val_fraction,
    )
