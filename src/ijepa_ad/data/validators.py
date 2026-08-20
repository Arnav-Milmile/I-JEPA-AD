"""Validation rules for MVTec AD samples and split boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


class DatasetValidationError(ValueError):
    """Raised when an MVTec sample or split violates the dataset contract."""


@dataclass(frozen=True)
class SampleMetadata:
    """Metadata required to validate one image in an MVTec split."""

    image_path: Path
    image_id: str
    split: str
    label: str
    mask_path: Path | None = None

    @property
    def is_anomalous(self) -> bool:
        """Whether this sample is a defect sample rather than ``good``."""
        return self.label != "good"


def validate_training_records(records: Iterable[SampleMetadata]) -> None:
    """Require a training split containing only nominal ``good`` images."""
    records = tuple(records)
    if not records:
        raise DatasetValidationError("training split is empty")
    _validate_unique_image_ids(records, split="train")
    for record in records:
        if record.split != "train":
            raise DatasetValidationError(
                f"training record has split {record.split!r}: {record.image_path}"
            )
        if record.label != "good" or record.is_anomalous:
            raise DatasetValidationError(
                f"training split contains defect image: {record.image_path}"
            )
        if record.mask_path is not None:
            raise DatasetValidationError(
                f"nominal training image must not have a mask: {record.image_path}"
            )


def validate_test_records(records: Iterable[SampleMetadata]) -> None:
    """Require official test labels and masks to agree for every sample."""
    records = tuple(records)
    if not records:
        raise DatasetValidationError("test split is empty")
    _validate_unique_image_ids(records, split="test")
    for record in records:
        if record.split != "test":
            raise DatasetValidationError(
                f"test record has split {record.split!r}: {record.image_path}"
            )
        if record.label == "good":
            if record.mask_path is not None:
                raise DatasetValidationError(
                    f"good test image must not have a ground-truth mask: "
                    f"{record.image_path}"
                )
            continue

        if record.mask_path is None:
            raise DatasetValidationError(
                f"defect test image is missing its ground-truth mask: "
                f"{record.image_path}"
            )
        if not record.mask_path.is_file():
            raise DatasetValidationError(
                f"ground-truth mask does not exist: {record.mask_path}"
            )


def validate_no_split_leakage(
    train_records: Iterable[SampleMetadata],
    test_records: Iterable[SampleMetadata],
) -> None:
    """Reject an image ID occurring in both training and test splits."""
    train_ids = {record.image_id for record in train_records}
    test_ids = {record.image_id for record in test_records}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        joined = ", ".join(overlap[:3])
        suffix = "..." if len(overlap) > 3 else ""
        raise DatasetValidationError(
            f"train/test image leakage detected for image IDs: {joined}{suffix}"
        )


def validate_splits(
    train_records: Iterable[SampleMetadata],
    test_records: Iterable[SampleMetadata],
) -> None:
    """Apply all split-level invariants."""
    train_records = tuple(train_records)
    test_records = tuple(test_records)
    validate_training_records(train_records)
    validate_test_records(test_records)
    validate_no_split_leakage(train_records, test_records)


def _validate_unique_image_ids(records: Iterable[SampleMetadata], split: str) -> None:
    seen: set[str] = set()
    for record in records:
        if record.image_id in seen:
            raise DatasetValidationError(
                f"duplicate image ID in {split} split: {record.image_id}"
            )
        seen.add(record.image_id)
