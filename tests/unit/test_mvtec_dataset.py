"""Synthetic MVTec AD split and leakage tests."""

from collections import Counter
from pathlib import Path

import pytest

from ijepa_ad.data.mvtec_dataset import validate_category_splits
from ijepa_ad.data.validators import DatasetValidationError


def _write(root: Path, relative_path: str, content: bytes) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_fixture(root: Path) -> None:
    for index in range(4):
        _write(root, f"bottle/train/good/{index:03d}.png", f"train-{index}".encode())
    for index in range(2):
        _write(root, f"bottle/test/good/{index:03d}.png", f"good-{index}".encode())
        _write(
            root,
            f"bottle/test/scratch/{index:03d}.png",
            f"scratch-{index}".encode(),
        )
        _write(
            root,
            f"bottle/ground_truth/scratch/{index:03d}_mask.png",
            f"mask-{index}".encode(),
        )


def test_official_splits_have_expected_nominal_defect_counts(tmp_path: Path) -> None:
    _make_fixture(tmp_path)

    train, test = validate_category_splits(tmp_path, "bottle")

    assert len(train) == 4
    assert len(test) == 4
    assert Counter(sample.label for sample in train) == Counter({"good": 4})
    assert Counter(sample.label for sample in test) == Counter(
        {"good": 2, "scratch": 2}
    )
    assert sum(sample.is_anomalous for sample in test) / len(test) == 0.5
    assert all(sample.mask_path is None for sample in train)
    assert all(sample.mask_path is not None for sample in test if sample.is_anomalous)


def test_identical_image_content_in_train_and_test_is_rejected(
    tmp_path: Path,
) -> None:
    _make_fixture(tmp_path)
    shared_content = b"same-image-content"
    _write(tmp_path, "bottle/train/good/shared.png", shared_content)
    _write(tmp_path, "bottle/test/good/shared.png", shared_content)

    with pytest.raises(DatasetValidationError, match="leakage"):
        validate_category_splits(tmp_path, "bottle")


def test_training_defect_directory_is_rejected(tmp_path: Path) -> None:
    _make_fixture(tmp_path)
    _write(tmp_path, "bottle/train/scratch/000.png", b"defect")

    with pytest.raises(DatasetValidationError, match="non-nominal"):
        validate_category_splits(tmp_path, "bottle")


def test_defect_without_official_mask_is_rejected(tmp_path: Path) -> None:
    _make_fixture(tmp_path)
    (tmp_path / "bottle/ground_truth/scratch/001_mask.png").unlink()

    with pytest.raises(DatasetValidationError, match="missing"):
        validate_category_splits(tmp_path, "bottle")
