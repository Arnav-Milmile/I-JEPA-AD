"""Phase 5: dataset → split → preprocessing → encoder → cache → reload, as one system.

Synthetic MVTec-style trees, the real loader/splitter/preprocessing/cache code, and a
deterministic content-dependent mock encoder. CPU only; nothing here runs a real
checkpoint or touches a real dataset.
"""

import hashlib
import json
import os
import random
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor

from ijepa_ad.caching.builder import CacheBuildError, build_latent_cache, parts_dir
from ijepa_ad.caching.cache_schema import CacheExpectations, PreprocessingSpec
from ijepa_ad.caching.cli import (
    MOCK_CACHE_SUBDIR,
    MOCK_ENCODER_NAME,
    CacheRunConfig,
    MockLatentEncoder,
    mock_checkpoint_sha256,
    run_cache_generation,
)
from ijepa_ad.caching.latent_cache import (
    CacheCorruptedError,
    CacheIncompatibleError,
    latent_cache_path,
    load_latent_cache,
    save_latent_cache,
)
from ijepa_ad.data.mvtec_dataset import MVTecADDataset, validate_category_splits
from ijepa_ad.data.splits import split_nominal_train_val
from ijepa_ad.data.transforms import PreprocessConfig, preprocess_batch
from ijepa_ad.data.validators import DatasetValidationError

SIZE = 32
DIM = 8
TOKENS = (SIZE // 16) ** 2
N_TRAIN = 12
CONFIG = PreprocessConfig(image_size=SIZE)


class SpyEncoder(MockLatentEncoder):
    """The CLI's deterministic mock encoder, recording every batch it receives."""

    def __init__(self) -> None:
        super().__init__(DIM, seed=0)
        self.batches: list[Tensor] = []

    def encode(self, image_batch: Tensor) -> Tensor:
        self.batches.append(image_batch.clone())
        return super().encode(image_batch)

    @property
    def rows_seen(self) -> int:
        return sum(batch.shape[0] for batch in self.batches)


def _save(path: Path, mode: str, size: tuple[int, int], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color: int | tuple[int, int, int] = (
        value if mode == "L" else (value, (value * 7) % 256, 255 - value)
    )
    Image.new(mode, size, color).save(path)


def make_tree(root: Path, *, name: str = "{:03d}.png", reverse: bool = False) -> None:
    """12 nominal train images (RGB + grayscale, mixed sizes), 5 official test images.

    ``reverse`` assigns filenames in the opposite order to the pixel content, so the
    same image set sits at different paths / enumeration positions.
    """
    for index in range(N_TRAIN):
        file_index = N_TRAIN - 1 - index if reverse else index
        mode = "L" if index % 4 == 3 else "RGB"
        size = (48, 40) if index % 2 else (40, 40)
        _save(
            root / "bottle/train/good" / name.format(file_index), mode, size, index * 19
        )
    for label, count, base in (("good", 2, 200), ("crack", 2, 90), ("scratch", 1, 30)):
        for index in range(count):
            file = f"{index:03d}.png"
            _save(root / f"bottle/test/{label}/{file}", "RGB", (40, 40), base + index)
            if label != "good":
                mask = root / f"bottle/ground_truth/{label}/{index:03d}_mask.png"
                _save(mask, "L", (40, 40), 255)


def make_config(tmp_path: Path, **overrides: object) -> CacheRunConfig:
    config = CacheRunConfig(
        dataset_root=tmp_path / "mvtec",
        cache_root=tmp_path / "cache",
        categories=("bottle",),
        batch_size=5,
        image_size=SIZE,
        mock_encoder=True,
        mock_embedding_dim=DIM,
        val_fraction=0.25,
    )
    return replace(config, **overrides)  # type: ignore[arg-type]


def mock_root(config: CacheRunConfig) -> Path:
    return config.cache_root / MOCK_CACHE_SUBDIR


def entry_path(config: CacheRunConfig, split: str, category: str = "bottle") -> Path:
    return latent_cache_path(mock_root(config), category, split)


def expectations(
    ids: tuple[str, ...], split: str, **overrides: object
) -> CacheExpectations:
    arguments: dict[str, object] = {
        "category": "bottle",
        "split": split,
        "encoder_name": MOCK_ENCODER_NAME,
        "checkpoint_sha256": mock_checkpoint_sha256(0),
        "patch_size": 16,
        "embedding_dim": DIM,
        "preprocessing": PreprocessingSpec.from_config(CONFIG),
        "sample_ids": ids,
    }
    arguments.update(overrides)
    return CacheExpectations(**arguments)  # type: ignore[arg-type]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- 1. the whole pipeline, once ------------------------------------------------------


def test_full_pipeline_discovers_splits_encodes_caches_and_reloads(
    tmp_path: Path,
) -> None:
    make_tree(tmp_path / "mvtec")
    config = make_config(tmp_path)
    train, test = validate_category_splits(config.dataset_root, "bottle")

    # 1. discovery
    assert len(train) == N_TRAIN and {s.label for s in train} == {"good"}
    assert sorted(s.label for s in test) == [
        "crack",
        "crack",
        "good",
        "good",
        "scratch",
    ]
    assert all(s.mask_path is not None for s in test if s.label != "good")

    # 2-3. nominal split excludes every test/defect sample and is repeatable
    split = split_nominal_train_val(train, val_fraction=0.25, seed=0)
    again = split_nominal_train_val(train, val_fraction=0.25, seed=0)
    assert split == again
    train_ids, val_ids = [s.image_id for s in split.train], [
        s.image_id for s in split.val
    ]
    assert (len(train_ids), len(val_ids)) == (9, 3)
    assert not set(train_ids) & set(val_ids)
    assert set(train_ids) | set(val_ids) == {s.image_id for s in train}
    assert not (set(train_ids) | set(val_ids)) & {s.image_id for s in test}

    # 4. preprocessed batches have the encoder input shape (RGB and grayscale alike)
    assert preprocess_batch([s.image_path for s in split.train], CONFIG).shape == (
        9,
        3,
        SIZE,
        SIZE,
    )

    # 5-6, 10. generation through the real orchestrator
    spy = SpyEncoder()
    summary = run_cache_generation(config, spy)
    assert summary.ok
    totals = summary.to_dict()["totals"]
    assert totals["samples_discovered"] == 12 and totals["encoded_new"] == 12
    assert spy.rows_seen == 12  # nothing lost, nothing encoded twice, no test image

    # 7. reload and validate; every latent belongs to its own sample
    reference = SpyEncoder()
    for name, records in (("train", split.train), ("val", split.val)):
        ids = tuple(s.image_id for s in records)
        entry = load_latent_cache(entry_path(config, name), expectations(ids, name))
        assert entry.metadata.sample_ids == ids and len(set(ids)) == len(ids)
        assert entry.latents.shape == (len(ids), TOKENS, DIM)
        for sample in records:
            own = preprocess_batch([sample.image_path], CONFIG)
            assert torch.allclose(
                entry.get(sample.image_id), reference.encode(own)[0], atol=1e-5
            ), f"latent of {sample.image_path.name} does not match its own image"
            other = preprocess_batch([records[0].image_path], CONFIG)
            if sample is not records[0]:
                assert not torch.allclose(
                    entry.get(sample.image_id), reference.encode(other)[0], atol=1e-5
                )
    assert not entry_path(config, "test").exists()

    # 8. a second run reuses everything without calling the encoder
    spy_again = SpyEncoder()
    second = run_cache_generation(config, spy_again).to_dict()["totals"]
    assert spy_again.batches == []
    assert (second["encoded_new"], second["skipped_compatible"]) == (0, 12)


# --- 2. leakage and invalid sample usage ----------------------------------------------


def test_split_membership_depends_on_content_not_filenames_or_order(
    tmp_path: Path,
) -> None:
    make_tree(tmp_path / "a")
    make_tree(tmp_path / "b", reverse=True)  # same pixels, opposite path order
    a = validate_category_splits(tmp_path / "a", "bottle")[0]
    b = validate_category_splits(tmp_path / "b", "bottle")[0]
    name_of_a = {s.image_id: s.image_path.name for s in a}
    name_of_b = {s.image_id: s.image_path.name for s in b}
    assert set(name_of_a) == set(name_of_b)
    assert name_of_a != name_of_b  # same content really sits under different names

    split_a = split_nominal_train_val(a, val_fraction=0.25, seed=3)
    split_b = split_nominal_train_val(b, val_fraction=0.25, seed=3)

    assert {s.image_id for s in split_a.val} == {s.image_id for s in split_b.val}
    assert {s.image_id for s in split_a.train} == {s.image_id for s in split_b.train}
    shuffled = list(a)
    random.Random(99).shuffle(shuffled)
    split_shuffled = split_nominal_train_val(shuffled, val_fraction=0.25, seed=3)
    assert {s.image_id for s in split_shuffled.val} == {s.image_id for s in split_a.val}


def test_duplicate_nominal_image_is_rejected_not_silently_deduplicated(
    tmp_path: Path,
) -> None:
    make_tree(tmp_path / "mvtec")
    good = tmp_path / "mvtec/bottle/train/good"
    shutil.copyfile(good / "000.png", good / "copy_of_000.png")

    with pytest.raises(DatasetValidationError, match="duplicate image ID"):
        MVTecADDataset(tmp_path / "mvtec", "bottle", "train")
    run_and_expect_no_encoding(make_config(tmp_path))


def run_and_expect_no_encoding(config: CacheRunConfig) -> None:
    spy = SpyEncoder()
    with pytest.raises(DatasetValidationError):
        run_cache_generation(config, spy)
    assert spy.batches == []


def test_test_image_copied_into_training_is_rejected(tmp_path: Path) -> None:
    make_tree(tmp_path / "mvtec")
    leaked = tmp_path / "mvtec/bottle/test/crack/000.png"
    shutil.copyfile(leaked, tmp_path / "mvtec/bottle/train/good/leaked.png")

    with pytest.raises(DatasetValidationError, match="leakage"):
        validate_category_splits(tmp_path / "mvtec", "bottle")
    run_and_expect_no_encoding(make_config(tmp_path))


def test_existing_cache_does_not_bypass_dataset_validation(tmp_path: Path) -> None:
    make_tree(tmp_path / "mvtec")
    config = make_config(tmp_path)
    run_cache_generation(config, SpyEncoder())
    train_before = entry_path(config, "train").read_bytes()
    # The dataset becomes invalid after a complete, valid cache already exists.
    _save(tmp_path / "mvtec/bottle/train/crack/000.png", "RGB", (40, 40), 1)

    run_and_expect_no_encoding(config)

    assert entry_path(config, "train").read_bytes() == train_before


def test_builder_revalidates_records_even_when_a_complete_cache_exists(
    tmp_path: Path,
) -> None:
    make_tree(tmp_path / "mvtec")
    config = make_config(tmp_path)
    train, test = validate_category_splits(config.dataset_root, "bottle")
    arguments = {
        "cache_root": mock_root(config),
        "category": "bottle",
        "encoder_name": MOCK_ENCODER_NAME,
        "checkpoint_sha256": mock_checkpoint_sha256(0),
        "patch_size": 16,
        "embedding_dim": DIM,
        "config": CONFIG,
    }
    build_latent_cache(train.samples, SpyEncoder(), split="train", **arguments)  # type: ignore[arg-type]

    with pytest.raises(DatasetValidationError):  # defect records, valid train entry
        build_latent_cache(test.samples, SpyEncoder(), split="train", **arguments)  # type: ignore[arg-type]


def test_full_run_keeps_global_rng_and_source_files_untouched(tmp_path: Path) -> None:
    make_tree(tmp_path / "mvtec")
    config = make_config(tmp_path)
    sources = sorted(p for p in config.dataset_root.rglob("*") if p.is_file())
    before = {
        p: (file_digest(p), p.stat().st_mtime_ns, p.stat().st_size) for p in sources
    }
    python_state, torch_state = random.getstate(), torch.get_rng_state()

    run_cache_generation(config, SpyEncoder())
    run_cache_generation(replace(config, purpose="test"), SpyEncoder())

    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert sorted(p for p in config.dataset_root.rglob("*") if p.is_file()) == sources
    after = {
        p: (file_digest(p), p.stat().st_mtime_ns, p.stat().st_size) for p in sources
    }
    assert after == before


def test_cache_file_moved_to_another_category_or_split_is_never_reused(
    tmp_path: Path,
) -> None:
    make_tree(tmp_path / "mvtec")
    config = make_config(tmp_path)
    run_cache_generation(config, SpyEncoder())
    train_ids = load_latent_cache(entry_path(config, "train")).metadata.sample_ids
    val_ids = load_latent_cache(entry_path(config, "val")).metadata.sample_ids

    # train.pt masquerading as val.pt: same file, wrong split and wrong sample ids
    shutil.copyfile(entry_path(config, "train"), entry_path(config, "val"))
    with pytest.raises(CacheIncompatibleError, match="split"):
        load_latent_cache(entry_path(config, "val"), expectations(val_ids, "val"))
    # train.pt placed under another category
    wrong = entry_path(config, "train", category="capsule")
    wrong.parent.mkdir(parents=True)
    shutil.copyfile(entry_path(config, "train"), wrong)
    with pytest.raises(CacheIncompatibleError, match="category"):
        load_latent_cache(wrong, expectations(train_ids, "train", category="capsule"))

    # The orchestrator rebuilds the impostor val entry instead of trusting it.
    spy = SpyEncoder()
    run_cache_generation(config, spy)
    assert spy.rows_seen == len(val_ids)
    rebuilt = load_latent_cache(entry_path(config, "val"), expectations(val_ids, "val"))
    assert rebuilt.metadata.split == "val"


# --- 3. cache reliability and recovery -----------------------------------------------


def test_shard_copied_under_another_samples_name_is_recomputed(tmp_path: Path) -> None:
    make_tree(tmp_path / "mvtec")
    config = make_config(tmp_path)

    class Crashing(SpyEncoder):
        def encode(self, image_batch: Tensor) -> Tensor:
            if len(self.batches) == 1:
                raise KeyboardInterrupt("simulated crash")
            return super().encode(image_batch)

    with pytest.raises(KeyboardInterrupt):
        run_cache_generation(config, Crashing())
    shards = sorted(parts_dir(mock_root(config), "bottle", "train").glob("*.pt"))
    assert len(shards) == 5
    victim, donor = shards[0], shards[1]
    victim.write_bytes(donor.read_bytes())  # right bytes, wrong sample

    spy = SpyEncoder()
    summary = run_cache_generation(config, spy).to_dict()

    assert summary["totals"]["recomputed"] == 1
    train = load_latent_cache(entry_path(config, "train"))
    reference = SpyEncoder()
    paths = {
        s.image_id: s.image_path
        for s in MVTecADDataset(config.dataset_root, "bottle", "train")
    }
    for sample_id in train.metadata.sample_ids:
        own = preprocess_batch([paths[sample_id]], CONFIG)
        assert torch.allclose(train.get(sample_id), reference.encode(own)[0], atol=1e-5)


def test_failed_atomic_replace_leaves_no_entry_and_resume_is_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_tree(tmp_path / "mvtec")
    config = make_config(tmp_path)
    real_replace, calls = os.replace, {"n": 0}

    def flaky_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        calls["n"] += 1
        if calls["n"] == 4:
            raise OSError("simulated disk failure")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(OSError, match="simulated disk failure"):
        run_cache_generation(config, SpyEncoder())
    monkeypatch.undo()

    shard_dir = parts_dir(mock_root(config), "bottle", "train")
    assert not entry_path(config, "train").exists()
    assert len(list(shard_dir.glob("*.pt"))) == 3  # only fully replaced shards
    assert list(shard_dir.glob("*.tmp")) == []  # temp file cleaned up
    for shard in shard_dir.glob("*.pt"):
        load_latent_cache(shard)  # every visible shard is intact

    spy = SpyEncoder()
    assert run_cache_generation(config, spy).ok
    assert spy.rows_seen == 12 - 3
    assert load_latent_cache(entry_path(config, "train")).latents.shape[0] == 9


def test_truncated_or_tampered_final_entry_is_rejected_and_left_in_place(
    tmp_path: Path,
) -> None:
    make_tree(tmp_path / "mvtec")
    config = make_config(tmp_path)
    run_cache_generation(config, SpyEncoder())
    path = entry_path(config, "train")
    good_bytes = path.read_bytes()

    path.write_bytes(good_bytes[: len(good_bytes) // 2])  # truncated tensor file
    truncated = path.read_bytes()
    spy = SpyEncoder()
    summary = run_cache_generation(config, spy)
    assert not summary.ok and spy.batches == []
    assert path.read_bytes() == truncated  # never silently overwritten
    assert "<entry>" in summary.to_dict()["failed_samples"]["bottle/train"]
    assert entry_path(config, "val").is_file()  # the other entry is unaffected

    # Valid torch file, tensor changed but metadata (with its checksum) untouched.
    entry = load_latent_cache(entry_path(config, "val"))
    payload = torch.load(entry_path(config, "val"), weights_only=True)
    payload["latents"] = payload["latents"] + 1.0
    torch.save(payload, entry_path(config, "val"))
    with pytest.raises(CacheCorruptedError, match="checksum"):
        load_latent_cache(entry_path(config, "val"))
    assert entry.latents.shape[0] == 3


def test_reordered_sample_ids_in_a_valid_looking_entry_are_not_trusted(
    tmp_path: Path,
) -> None:
    make_tree(tmp_path / "mvtec")
    config = make_config(tmp_path)
    run_cache_generation(config, SpyEncoder())
    path = entry_path(config, "train")
    entry = load_latent_cache(path)
    swapped_ids = (entry.metadata.sample_ids[1], entry.metadata.sample_ids[0]) + (
        entry.metadata.sample_ids[2:]
    )
    # Internally consistent file (checksum matches), but ids no longer match rows.
    save_latent_cache(
        path, entry.latents, replace(entry.metadata, sample_ids=swapped_ids)
    )

    spy = SpyEncoder()
    run_cache_generation(config, spy)

    assert spy.rows_seen == 9  # rebuilt from scratch, not reused
    rebuilt = load_latent_cache(path)
    assert rebuilt.metadata.sample_ids == entry.metadata.sample_ids
    assert torch.allclose(rebuilt.latents, entry.latents, atol=1e-5)


def test_non_finite_encoder_output_is_a_reported_failure_with_no_partial_batch(
    tmp_path: Path,
) -> None:
    make_tree(tmp_path / "mvtec")
    config = make_config(tmp_path)

    class NanEncoder(SpyEncoder):
        def encode(self, image_batch: Tensor) -> Tensor:
            out = super().encode(image_batch)
            out[1] = float("nan")
            return out

    summary = run_cache_generation(config, NanEncoder())  # must not raise

    assert not summary.ok
    assert not entry_path(config, "train").exists()
    shard_dir = parts_dir(mock_root(config), "bottle", "train")
    assert not shard_dir.exists() or list(shard_dir.glob("*.pt")) == []


def test_changed_embedding_dim_rebuilds_instead_of_reusing(tmp_path: Path) -> None:
    make_tree(tmp_path / "mvtec")
    run_cache_generation(make_config(tmp_path), SpyEncoder())
    wider = make_config(tmp_path, mock_embedding_dim=DIM * 2)

    summary = run_cache_generation(wider, MockLatentEncoder(DIM * 2, 0))

    assert summary.ok and summary.to_dict()["totals"]["encoded_new"] == 12
    assert (
        load_latent_cache(entry_path(wider, "train")).metadata.embedding_dim == DIM * 2
    )


def test_changed_split_configuration_is_not_served_from_the_old_entry(
    tmp_path: Path,
) -> None:
    make_tree(tmp_path / "mvtec")
    run_cache_generation(make_config(tmp_path), SpyEncoder())
    old_val = load_latent_cache(entry_path(make_config(tmp_path), "val")).metadata
    reseeded = make_config(tmp_path, seed=1)
    train, _ = validate_category_splits(reseeded.dataset_root, "bottle")
    new_split = split_nominal_train_val(train, val_fraction=0.25, seed=1)
    assert {s.image_id for s in new_split.val} != set(old_val.sample_ids)

    run_cache_generation(reseeded, SpyEncoder())

    for name, records in (("train", new_split.train), ("val", new_split.val)):
        ids = tuple(s.image_id for s in records)
        assert load_latent_cache(entry_path(reseeded, name)).metadata.sample_ids == ids


# --- 4. reproducibility ---------------------------------------------------------------


def test_repeated_runs_with_the_same_inputs_produce_identical_caches(
    tmp_path: Path,
) -> None:
    """Deterministic guarantee: mock encoder, CPU, same batch size → identical bytes."""
    make_tree(tmp_path / "mvtec")
    first = make_config(tmp_path, cache_root=tmp_path / "run1")
    second = make_config(tmp_path, cache_root=tmp_path / "run2")
    run_cache_generation(first, SpyEncoder())
    run_cache_generation(second, SpyEncoder())

    for name in ("train", "val"):
        one = load_latent_cache(entry_path(first, name))
        two = load_latent_cache(entry_path(second, name))
        assert torch.equal(one.latents, two.latents)
        assert one.metadata.latents_sha256 == two.metadata.latents_sha256
        stable = {"created_at_utc"}  # the only run-dependent field here
        left, right = one.metadata.to_dict(), two.metadata.to_dict()
        assert {k: v for k, v in left.items() if k not in stable} == {
            k: v for k, v in right.items() if k not in stable
        }
        assert json.loads(one.metadata.to_json()) == left


def test_batch_size_changes_latents_only_within_float_tolerance(tmp_path: Path) -> None:
    """Documented limit: not bit-identical across batch compositions, only close."""
    make_tree(tmp_path / "mvtec")
    small = make_config(tmp_path, cache_root=tmp_path / "small", batch_size=1)
    large = make_config(tmp_path, cache_root=tmp_path / "large", batch_size=12)
    run_cache_generation(small, SpyEncoder())
    run_cache_generation(large, SpyEncoder())

    one = load_latent_cache(entry_path(small, "train"))
    two = load_latent_cache(entry_path(large, "train"))
    assert one.metadata.sample_ids == two.metadata.sample_ids
    assert torch.allclose(one.latents, two.latents, atol=1e-4)


def test_failed_build_error_is_catchable_as_cache_build_error(tmp_path: Path) -> None:
    """Builder-level contract behind the CLI's NaN handling."""
    make_tree(tmp_path / "mvtec")
    train, _ = validate_category_splits(tmp_path / "mvtec", "bottle")

    class NanEncoder(SpyEncoder):
        def encode(self, image_batch: Tensor) -> Tensor:
            return super().encode(image_batch) * float("nan")

    with pytest.raises(CacheBuildError, match="non-finite"):
        build_latent_cache(
            train.samples,
            NanEncoder(),
            cache_root=tmp_path / "cache",
            category="bottle",
            split="train",
            encoder_name=MOCK_ENCODER_NAME,
            checkpoint_sha256=mock_checkpoint_sha256(0),
            patch_size=16,
            embedding_dim=DIM,
            config=CONFIG,
        )
