"""MVTec AD directory discovery with strict split validation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

from .validators import (
    DatasetValidationError,
    SampleMetadata,
    validate_splits,
    validate_test_records,
    validate_training_records,
)

Split = Literal["train", "test"]
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})


@dataclass(frozen=True)
class MVTecSample(SampleMetadata):
    """A discovered MVTec image and its official split metadata."""

    category: str = ""


class MVTecADDataset(Sequence[MVTecSample]):
    """Indexable metadata dataset for one MVTec AD category and split.

    The class deliberately returns validated path metadata rather than loading
    images. Image decoding/transforms belong at the model boundary and can be
    added later without weakening split validation.
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: Split,
        *,
        validate: bool = True,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.root = Path(root)
        self.category = category
        self.split = split
        self.category_root = self.root / category
        self._samples = tuple(self._discover())

        if validate:
            if split == "train":
                validate_training_records(self._samples)
            else:
                validate_test_records(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    @overload
    def __getitem__(self, index: int) -> MVTecSample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[MVTecSample]: ...

    def __getitem__(self, index: int | slice) -> MVTecSample | Sequence[MVTecSample]:
        return self._samples[index]

    def __iter__(self) -> Iterator[MVTecSample]:
        return iter(self._samples)

    @property
    def samples(self) -> tuple[MVTecSample, ...]:
        """Return the immutable discovered sample list."""
        return self._samples

    def _discover(self) -> Iterator[MVTecSample]:
        if self.split == "train":
            image_root = self.category_root / "train" / "good"
            if not image_root.is_dir():
                raise DatasetValidationError(
                    f"MVTec training directory does not exist: {image_root}"
                )
            unexpected_dirs = sorted(
                path.name
                for path in image_root.parent.iterdir()
                if path.is_dir() and path.name != "good"
            )
            if unexpected_dirs:
                raise DatasetValidationError(
                    "MVTec training split contains non-nominal directories: "
                    f"{unexpected_dirs}"
                )
            for image_path in _image_paths(image_root):
                yield self._sample(image_path, label="good", mask_path=None)
            return

        image_root = self.category_root / "test"
        if not image_root.is_dir():
            raise DatasetValidationError(
                f"MVTec test directory does not exist: {image_root}"
            )
        for label_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
            label = label_dir.name
            mask_root = self.category_root / "ground_truth" / label
            for image_path in _image_paths(label_dir):
                mask_path = None
                if label != "good":
                    mask_path = _find_mask(mask_root, image_path.stem)
                yield self._sample(
                    image_path,
                    label=label,
                    mask_path=mask_path,
                )

    def _sample(
        self,
        image_path: Path,
        *,
        label: str,
        mask_path: Path | None,
    ) -> MVTecSample:
        return MVTecSample(
            image_path=image_path,
            image_id=_content_id(image_path),
            split=self.split,
            label=label,
            mask_path=mask_path,
            category=self.category,
        )


# Short alias for callers that prefer the dataset name without the suffix.
MVTecDataset = MVTecADDataset


def validate_category_splits(
    root: str | Path, category: str
) -> tuple[MVTecADDataset, MVTecADDataset]:
    """Discover and validate both official splits for one category."""
    train = MVTecADDataset(root, category, "train")
    test = MVTecADDataset(root, category, "test")
    validate_splits(train, test)
    return train, test


def _image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _find_mask(mask_root: Path, stem: str) -> Path | None:
    if not mask_root.is_dir():
        return None
    matches = sorted(
        path
        for path in mask_root.glob(f"{stem}_mask.*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(matches) > 1:
        raise DatasetValidationError(
            f"multiple ground-truth masks found for {stem!r}: {matches}"
        )
    return matches[0] if matches else None


def _content_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
