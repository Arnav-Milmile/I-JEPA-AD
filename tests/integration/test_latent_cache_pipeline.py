"""MVTec → nominal split → preprocessing → mock encoder → resumable latent cache."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from ijepa_ad.caching.builder import (
    CacheBuildError,
    build_latent_cache,
    build_nominal_latent_caches,
    parts_dir,
)
from ijepa_ad.caching.cache_schema import CacheExpectations, PreprocessingSpec
from ijepa_ad.caching.latent_cache import (
    CacheCorruptedError,
    latent_cache_path,
    load_latent_cache,
    try_load_latent_cache,
)
from ijepa_ad.data.mvtec_dataset import MVTecADDataset, validate_category_splits
from ijepa_ad.data.splits import split_nominal_train_val
from ijepa_ad.data.transforms import PreprocessConfig, preprocess_image
from ijepa_ad.data.validators import DatasetValidationError

CONFIG = PreprocessConfig(image_size=32)
PATCH = 16
DIM = 8
SHA = "c" * 64
COMMON: dict[str, Any] = {
    "category": "bottle",
    "encoder_name": "mock_encoder",
    "checkpoint_sha256": SHA,
    "patch_size": PATCH,
    "embedding_dim": DIM,
    "config": CONFIG,
}


class MockEncoder:
    """Deterministic, content-dependent N×3×32×32 → N×4×8 encoder; records calls."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        generator = torch.Generator().manual_seed(1)
        self.projection = nn.Linear(3 * PATCH * PATCH, DIM)
        with torch.no_grad():
            for parameter in self.projection.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
        self.batch_sizes: list[int] = []
        self.fail_on_call = fail_on_call

    def encode(self, image_batch: Tensor) -> Tensor:
        if self.fail_on_call == len(self.batch_sizes):
            raise RuntimeError("simulated encoder crash")
        self.batch_sizes.append(image_batch.shape[0])
        patches = image_batch.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
        patches = patches.permute(0, 2, 3, 1, 4, 5).flatten(3)
        with torch.no_grad():
            out = self.projection(patches.flatten(0, 2))
        return out.view(image_batch.shape[0], 4, DIM)


def _make_mvtec(root: Path, n_train: int = 10) -> None:
    for index in range(n_train):
        path = root / f"bottle/train/good/{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 40), (index * 20, 40, 90 + index)).save(path)
    for label, shade in (("good", 200), ("broken_large", 250)):
        path = root / f"bottle/test/{label}/000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 40), (shade, shade, 7)).save(path)
    mask = root / "bottle/ground_truth/broken_large/000_mask.png"
    mask.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (40, 40), 255).save(mask)


def _data(tmp_path: Path) -> tuple[MVTecADDataset, MVTecADDataset]:
    _make_mvtec(tmp_path / "mvtec")
    return validate_category_splits(tmp_path / "mvtec", "bottle")


def _expected(
    split: str, ids: tuple[str, ...], **overrides: object
) -> CacheExpectations:
    arguments: dict[str, object] = {
        "category": "bottle",
        "split": split,
        "encoder_name": "mock_encoder",
        "checkpoint_sha256": SHA,
        "patch_size": PATCH,
        "embedding_dim": DIM,
        "preprocessing": PreprocessingSpec.from_config(CONFIG),
        "sample_ids": ids,
    }
    arguments.update(overrides)
    return CacheExpectations(**arguments)  # type: ignore[arg-type]


def test_build_associates_each_latent_with_its_sample(tmp_path: Path) -> None:
    train, _ = _data(tmp_path)
    root = tmp_path / "cache"
    encoder = MockEncoder()

    report = build_latent_cache(
        train.samples, encoder, cache_root=root, split="train", batch_size=4, **COMMON
    )

    assert (report.total, report.encoded, report.reused) == (10, 10, 0)
    assert encoder.batch_sizes == [4, 4, 2]  # batched, never all at once
    entry = load_latent_cache(report.path)
    assert entry.metadata.sample_ids == tuple(s.image_id for s in train.samples)
    reference = MockEncoder()
    for sample in train.samples:
        image = preprocess_image(sample.image_path, CONFIG).unsqueeze(0)
        # Batch composition changes float summation order, so not bit-identical.
        expected = reference.encode(image)[0]
        assert torch.allclose(entry.get(sample.image_id), expected, atol=1e-4)
    assert not parts_dir(root, "bottle", "train").exists()


def test_completed_cache_is_reused_without_encoding(tmp_path: Path) -> None:
    train, _ = _data(tmp_path)
    build_latent_cache(
        train.samples, MockEncoder(), cache_root=tmp_path, split="train", **COMMON
    )
    encoder = MockEncoder()

    report = build_latent_cache(
        train.samples, encoder, cache_root=tmp_path, split="train", **COMMON
    )

    assert report.already_complete and report.encoded == 0
    assert encoder.batch_sizes == []


def test_interrupted_run_resumes_without_recomputing_valid_entries(
    tmp_path: Path,
) -> None:
    train, _ = _data(tmp_path)
    with pytest.raises(RuntimeError, match="simulated"):
        build_latent_cache(
            train.samples,
            MockEncoder(fail_on_call=2),
            cache_root=tmp_path,
            split="train",
            batch_size=4,
            **COMMON,
        )
    final = latent_cache_path(tmp_path, "bottle", "train")
    assert not final.exists()  # never written from partial results
    assert len(list(parts_dir(tmp_path, "bottle", "train").glob("*.pt"))) == 8

    resumed = MockEncoder()
    report = build_latent_cache(
        train.samples,
        resumed,
        cache_root=tmp_path,
        split="train",
        batch_size=4,
        **COMMON,
    )

    assert (report.encoded, report.reused) == (2, 8)
    assert resumed.batch_sizes == [2]
    clean = tmp_path / "clean"
    build_latent_cache(
        train.samples, MockEncoder(), cache_root=clean, split="train", **COMMON
    )
    resumed_latents = load_latent_cache(final).latents
    clean_latents = load_latent_cache(latent_cache_path(clean, "bottle", "train"))
    assert torch.allclose(resumed_latents, clean_latents.latents, atol=1e-4)


def test_corrupt_shards_are_recomputed_and_leftover_temp_files_ignored(
    tmp_path: Path,
) -> None:
    train, _ = _data(tmp_path)
    with pytest.raises(RuntimeError):
        build_latent_cache(
            train.samples,
            MockEncoder(fail_on_call=1),
            cache_root=tmp_path,
            split="train",
            batch_size=5,
            **COMMON,
        )
    shard_dir = parts_dir(tmp_path, "bottle", "train")
    shards = sorted(shard_dir.glob("*.pt"))
    assert len(shards) == 5
    shards[0].write_bytes(b"garbage")  # corrupted
    (shard_dir / ".x.pt.tmp").write_bytes(b"leftover partial write")

    report = build_latent_cache(
        train.samples, MockEncoder(), cache_root=tmp_path, split="train", **COMMON
    )

    assert report.encoded == 6  # 5 never done + 1 corrupt
    assert report.reused == 4


def test_other_checkpoint_or_preprocessing_never_reuses_cache(tmp_path: Path) -> None:
    train, _ = _data(tmp_path)
    build_latent_cache(
        train.samples, MockEncoder(), cache_root=tmp_path, split="train", **COMMON
    )
    path = latent_cache_path(tmp_path, "bottle", "train")
    ids = tuple(s.image_id for s in train.samples)
    other_spec = replace(PreprocessingSpec.from_config(CONFIG), mean=(0.0, 0.0, 0.0))

    assert try_load_latent_cache(path, _expected("train", ids)) is not None
    other_sha = _expected("train", ids, checkpoint_sha256="d" * 64)
    assert try_load_latent_cache(path, other_sha) is None
    other_pre = _expected("train", ids, preprocessing=other_spec)
    assert try_load_latent_cache(path, other_pre) is None

    changed = dict(COMMON, checkpoint_sha256="d" * 64)
    report = build_latent_cache(
        train.samples, MockEncoder(), cache_root=tmp_path, split="train", **changed
    )
    assert report.encoded == 10 and not report.already_complete
    assert load_latent_cache(path).metadata.checkpoint_sha256 == "d" * 64


def test_corrupted_final_entry_is_not_silently_overwritten(tmp_path: Path) -> None:
    train, _ = _data(tmp_path)
    build_latent_cache(
        train.samples, MockEncoder(), cache_root=tmp_path, split="train", **COMMON
    )
    latent_cache_path(tmp_path, "bottle", "train").write_bytes(b"garbage")

    with pytest.raises(CacheCorruptedError):
        build_latent_cache(
            train.samples, MockEncoder(), cache_root=tmp_path, split="train", **COMMON
        )


def test_unreadable_sample_fails_cleanly_and_keeps_good_shards(tmp_path: Path) -> None:
    train, _ = _data(tmp_path)
    bad = train.samples[3]
    bad.image_path.write_bytes(b"not an image")
    encoder = MockEncoder()

    with pytest.raises(CacheBuildError) as raised:
        build_latent_cache(
            train.samples, encoder, cache_root=tmp_path, split="train", **COMMON
        )

    assert set(raised.value.failed) == {bad.image_id}
    assert sum(encoder.batch_sizes) == 9
    assert not latent_cache_path(tmp_path, "bottle", "train").exists()
    assert len(list(parts_dir(tmp_path, "bottle", "train").glob("*.pt"))) == 9


def test_encoder_with_wrong_batch_size_is_rejected(tmp_path: Path) -> None:
    train, _ = _data(tmp_path)

    class Dropping(MockEncoder):
        def encode(self, image_batch: Tensor) -> Tensor:
            return super().encode(image_batch)[:-1]

    with pytest.raises(CacheBuildError, match="refusing"):
        build_latent_cache(
            train.samples, Dropping(), cache_root=tmp_path, split="train", **COMMON
        )
    assert not parts_dir(tmp_path, "bottle", "train").exists()


def test_nominal_caches_cover_train_and_val_only(tmp_path: Path) -> None:
    train, test = _data(tmp_path)

    reports = build_nominal_latent_caches(
        train.samples,
        MockEncoder(),
        val_fraction=0.2,
        seed=0,
        cache_root=tmp_path / "cache",
        **COMMON,
    )

    split = split_nominal_train_val(train.samples, val_fraction=0.2, seed=0)
    train_entry = load_latent_cache(reports["train"].path)
    val_entry = load_latent_cache(reports["val"].path)
    assert train_entry.metadata.sample_ids == tuple(s.image_id for s in split.train)
    assert val_entry.metadata.sample_ids == tuple(s.image_id for s in split.val)
    cached = set(train_entry.metadata.sample_ids) | set(val_entry.metadata.sample_ids)
    assert not cached & {s.image_id for s in test.samples}
    assert not (tmp_path / "cache" / "bottle" / "test.pt").exists()


def test_test_and_defect_samples_cannot_enter_nominal_cache(tmp_path: Path) -> None:
    _, test = _data(tmp_path)

    for split in ("train", "val"):
        with pytest.raises(DatasetValidationError):
            build_latent_cache(
                test.samples,
                MockEncoder(),
                cache_root=tmp_path,
                split=split,
                **COMMON,
            )
    with pytest.raises(DatasetValidationError):
        build_nominal_latent_caches(
            test.samples,
            MockEncoder(),
            val_fraction=0.5,
            seed=0,
            cache_root=tmp_path,
            **COMMON,
        )
    assert not (tmp_path / "bottle").exists()


def test_wrong_category_and_bad_arguments_are_rejected(tmp_path: Path) -> None:
    train, _ = _data(tmp_path)
    wrong = dict(COMMON, category="capsule")

    with pytest.raises(DatasetValidationError, match="category"):
        build_latent_cache(
            train.samples, MockEncoder(), cache_root=tmp_path, split="train", **wrong
        )
    with pytest.raises(ValueError, match="split"):
        build_latent_cache(
            train.samples, MockEncoder(), cache_root=tmp_path, split="dev", **COMMON
        )
    with pytest.raises(ValueError, match="batch_size"):
        build_latent_cache(
            train.samples,
            MockEncoder(),
            cache_root=tmp_path,
            split="train",
            batch_size=0,
            **COMMON,
        )


def test_report_counts_recomputed_shards(tmp_path: Path) -> None:
    train, _ = _data(tmp_path)
    with pytest.raises(RuntimeError):
        build_latent_cache(
            train.samples,
            MockEncoder(fail_on_call=1),
            cache_root=tmp_path,
            split="train",
            batch_size=5,
            **COMMON,
        )
    shards = sorted(parts_dir(tmp_path, "bottle", "train").glob("*.pt"))
    shards[0].write_bytes(b"garbage")

    report = build_latent_cache(
        train.samples, MockEncoder(), cache_root=tmp_path, split="train", **COMMON
    )

    assert (report.encoded, report.recomputed, report.reused) == (6, 1, 4)
    assert report.failed == {}


def test_failed_sample_report_is_attached_to_the_error(tmp_path: Path) -> None:
    train, _ = _data(tmp_path)
    bad = train.samples[0]
    bad.image_path.write_bytes(b"not an image")

    with pytest.raises(CacheBuildError) as raised:
        build_latent_cache(
            train.samples, MockEncoder(), cache_root=tmp_path, split="train", **COMMON
        )

    report = raised.value.report
    assert report is not None
    assert (report.total, report.encoded, report.reused) == (10, 9, 0)
    assert set(report.failed) == {bad.image_id}


def test_wrong_token_count_or_embedding_dim_is_rejected_before_saving(
    tmp_path: Path,
) -> None:
    train, _ = _data(tmp_path)

    with pytest.raises(CacheBuildError, match="expected"):
        build_latent_cache(
            train.samples,
            MockEncoder(),
            cache_root=tmp_path,
            split="train",
            num_tokens=196,  # mock yields 4 tokens
            **COMMON,
        )
    wrong_dim = dict(COMMON, embedding_dim=DIM + 1)
    with pytest.raises(CacheBuildError, match="expected"):
        build_latent_cache(
            train.samples,
            MockEncoder(),
            cache_root=tmp_path,
            split="train",
            **wrong_dim,
        )
    assert not (tmp_path / "bottle").exists()  # nothing persisted

    report = build_latent_cache(
        train.samples,
        MockEncoder(),
        cache_root=tmp_path,
        split="train",
        num_tokens=4,
        **COMMON,
    )
    assert report.encoded == 10


def test_cache_metadata_is_consistent_with_the_run(tmp_path: Path) -> None:
    train, _ = _data(tmp_path)
    report = build_latent_cache(
        train.samples, MockEncoder(), cache_root=tmp_path, split="train", **COMMON
    )

    metadata = load_latent_cache(report.path).metadata
    assert metadata.category == "bottle" and metadata.split == "train"
    assert metadata.encoder_name == "mock_encoder"
    assert metadata.checkpoint_sha256 == SHA
    assert (metadata.num_tokens, metadata.embedding_dim) == (4, DIM)
    assert metadata.sample_ids == tuple(s.image_id for s in train.samples)


def test_test_split_cache_is_built_only_from_official_test_records(
    tmp_path: Path,
) -> None:
    train, test = _data(tmp_path)

    report = build_latent_cache(
        test.samples, MockEncoder(), cache_root=tmp_path, split="test", **COMMON
    )

    ids = load_latent_cache(report.path).metadata.sample_ids
    assert ids == tuple(s.image_id for s in test.samples)
    with pytest.raises(DatasetValidationError):
        build_latent_cache(
            train.samples, MockEncoder(), cache_root=tmp_path, split="test", **COMMON
        )
