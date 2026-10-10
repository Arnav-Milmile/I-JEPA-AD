"""Tests for the deterministic nominal-only train/validation split."""

import random
from dataclasses import replace
from pathlib import Path

import pytest

from ijepa_ad.data.mvtec_dataset import MVTecADDataset, MVTecSample
from ijepa_ad.data.splits import (
    InsufficientNominalSamplesError,
    split_nominal_train_val,
)
from ijepa_ad.data.validators import DatasetValidationError


def _write(root: Path, relative_path: str, content: bytes) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _train_samples(root: Path, count: int) -> tuple[MVTecSample, ...]:
    for index in range(count):
        _write(root, f"bottle/train/good/{index:03d}.png", f"train-{index}".encode())
    return MVTecADDataset(root, "bottle", "train").samples


def _test_samples(root: Path) -> tuple[MVTecSample, ...]:
    _write(root, "bottle/test/good/000.png", b"test-good")
    _write(root, "bottle/test/crack/000.png", b"test-crack")
    _write(root, "bottle/ground_truth/crack/000_mask.png", b"mask")
    return MVTecADDataset(root, "bottle", "test").samples


def test_split_is_nominal_only_disjoint_and_complete(tmp_path: Path) -> None:
    samples = _train_samples(tmp_path, 20)

    split = split_nominal_train_val(samples, val_fraction=0.25, seed=0)

    assert len(split.train) == 15
    assert len(split.val) == 5
    train_ids = {sample.image_id for sample in split.train}
    val_ids = {sample.image_id for sample in split.val}
    assert not train_ids & val_ids
    assert train_ids | val_ids == {sample.image_id for sample in samples}
    assert all(s.label == "good" and not s.is_anomalous for s in split.train)
    assert all(s.label == "good" and not s.is_anomalous for s in split.val)
    assert all(s.mask_path is None for s in split.train + split.val)


def test_split_is_reproducible_for_a_fixed_seed(tmp_path: Path) -> None:
    samples = _train_samples(tmp_path, 20)

    first = split_nominal_train_val(samples, val_fraction=0.2, seed=42)
    second = split_nominal_train_val(samples, val_fraction=0.2, seed=42)

    assert first == second


def test_split_ignores_input_order(tmp_path: Path) -> None:
    samples = _train_samples(tmp_path, 20)

    forward = split_nominal_train_val(samples, val_fraction=0.2, seed=3)
    backward = split_nominal_train_val(samples[::-1], val_fraction=0.2, seed=3)

    assert forward == backward


def test_different_seeds_change_the_validation_set(tmp_path: Path) -> None:
    samples = _train_samples(tmp_path, 20)

    val_sets = {
        frozenset(
            s.image_id
            for s in split_nominal_train_val(samples, val_fraction=0.2, seed=seed).val
        )
        for seed in range(5)
    }

    assert len(val_sets) > 1


def test_split_does_not_touch_global_random_state(tmp_path: Path) -> None:
    samples = _train_samples(tmp_path, 10)
    random.seed(99)
    state = random.getstate()

    split_nominal_train_val(samples, val_fraction=0.2, seed=1)

    assert random.getstate() == state


def test_anomalous_test_records_are_rejected(tmp_path: Path) -> None:
    samples = _test_samples(tmp_path)

    with pytest.raises(DatasetValidationError):
        split_nominal_train_val(samples, val_fraction=0.5, seed=0)


def test_defect_label_in_training_records_is_rejected(tmp_path: Path) -> None:
    samples = list(_train_samples(tmp_path, 4))
    samples[0] = replace(samples[0], label="scratch")

    with pytest.raises(DatasetValidationError, match="defect"):
        split_nominal_train_val(samples, val_fraction=0.25, seed=0)


def test_masked_training_record_is_rejected(tmp_path: Path) -> None:
    samples = list(_train_samples(tmp_path, 4))
    samples[1] = replace(samples[1], mask_path=tmp_path / "mask.png")

    with pytest.raises(DatasetValidationError, match="mask"):
        split_nominal_train_val(samples, val_fraction=0.25, seed=0)


def test_too_few_nominal_images_raises_clear_error(tmp_path: Path) -> None:
    samples = _train_samples(tmp_path, 2)

    with pytest.raises(InsufficientNominalSamplesError, match="cannot split 2"):
        split_nominal_train_val(samples, val_fraction=0.1, seed=0)
    with pytest.raises(InsufficientNominalSamplesError, match="need at least 2 / 1"):
        split_nominal_train_val(samples, val_fraction=0.5, seed=0, min_train=2)


def test_empty_training_records_are_rejected() -> None:
    with pytest.raises(DatasetValidationError, match="empty"):
        split_nominal_train_val([], val_fraction=0.2, seed=0)


def test_invalid_val_fraction_is_rejected(tmp_path: Path) -> None:
    samples = _train_samples(tmp_path, 10)

    for val_fraction in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="val_fraction"):
            split_nominal_train_val(samples, val_fraction=val_fraction, seed=0)


def test_invalid_seed_is_rejected(tmp_path: Path) -> None:
    samples = _train_samples(tmp_path, 10)

    with pytest.raises(ValueError, match="non-negative"):
        split_nominal_train_val(samples, val_fraction=0.2, seed=-1)
