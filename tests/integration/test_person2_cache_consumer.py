"""Downstream (Person 2) consumption of cached latents through the public cache API.

Person 2's predictor does not exist in this repository, so the "consumer" here is a
stand-in that uses only what ``FLOW.md`` §2 documents as the consumption surface:
``latent_cache_path``, ``CacheExpectations``, ``PreprocessingSpec``,
``load_latent_cache`` / ``try_load_latent_cache`` / ``cache_exists`` and the
``LatentCache`` accessors. It never opens a ``.pt`` file itself and never touches the
``.<split>.parts`` staging directory. This verifies the documented contract; it does
not verify any real Person 2 code. Synthetic images, the CLI's deterministic mock
encoder, CPU only.
"""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from ijepa_ad.caching.cache_schema import CacheExpectations, PreprocessingSpec
from ijepa_ad.caching.cli import (
    MOCK_CHECKPOINT_SHA256,
    MOCK_ENCODER_NAME,
    CacheRunConfig,
    MockLatentEncoder,
    run_cache_generation,
)
from ijepa_ad.caching.latent_cache import (
    CacheCorruptedError,
    CacheIncompatibleError,
    CacheNotFoundError,
    LatentCache,
    cache_exists,
    latent_cache_path,
    load_latent_cache,
    try_load_latent_cache,
)
from ijepa_ad.data.mvtec_dataset import validate_category_splits
from ijepa_ad.data.splits import split_nominal_train_val
from ijepa_ad.data.transforms import PreprocessConfig, preprocess_image

CATEGORY = "bottle"
SIZE = 32
DIM = 8
TOKENS = (SIZE // 16) ** 2
VAL_FRACTION = 0.25
PREPROCESS = PreprocessConfig(image_size=SIZE)


def _make_mvtec(root: Path, n_train: int = 8) -> None:
    for index in range(n_train):
        path = root / f"{CATEGORY}/train/good/{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 40), (index * 20, 40, 90 + index)).save(path)
    for label, shade in (("good", 200), ("broken_large", 250)):
        path = root / f"{CATEGORY}/test/{label}/000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 40), (shade, shade, 7)).save(path)
    mask = root / f"{CATEGORY}/ground_truth/broken_large/000_mask.png"
    mask.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (40, 40), 255).save(mask)


def _config(
    tmp_path: Path, purpose: str = "nominal", **overrides: object
) -> CacheRunConfig:
    base = CacheRunConfig(
        dataset_root=tmp_path / "mvtec",
        cache_root=tmp_path / "cache",
        categories=(CATEGORY,),
        purpose=purpose,
        batch_size=3,
        val_fraction=VAL_FRACTION,
        image_size=SIZE,
        mock_encoder=True,
        mock_embedding_dim=DIM,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def _expected(split: str, **overrides: object) -> CacheExpectations:
    """What a consumer must know about the run that produced the cache."""
    base = CacheExpectations(
        category=CATEGORY,
        split=split,
        encoder_name=MOCK_ENCODER_NAME,
        checkpoint_sha256=MOCK_CHECKPOINT_SHA256,
        patch_size=16,
        embedding_dim=DIM,
        preprocessing=PreprocessingSpec.from_config(PREPROCESS),
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def _produce(tmp_path: Path) -> tuple[Path, dict[str, tuple[str, ...]]]:
    """Run the producer (nominal then test); return (cache root, ids per split)."""
    _make_mvtec(tmp_path / "mvtec")
    nominal = run_cache_generation(_config(tmp_path))
    run_cache_generation(_config(tmp_path, purpose="test"))
    assert nominal.ok
    train, test = validate_category_splits(tmp_path / "mvtec", CATEGORY)
    split = split_nominal_train_val(
        train.samples, val_fraction=VAL_FRACTION, seed=0, min_train=1, min_val=1
    )
    ids = {
        "train": tuple(s.image_id for s in split.train),
        "val": tuple(s.image_id for s in split.val),
        "test": tuple(s.image_id for s in test.samples),
    }
    return Path(nominal.cache_root), ids


def _load(root: Path, split: str) -> LatentCache:
    return load_latent_cache(latent_cache_path(root, CATEGORY, split), _expected(split))


def test_consumer_loads_entry_with_expected_shape_dtype_and_metadata(
    tmp_path: Path,
) -> None:
    root, ids = _produce(tmp_path)
    cache = _load(root, "train")

    assert cache.latents.shape == (len(ids["train"]), TOKENS, DIM)
    assert cache.latents.dtype == torch.float32
    assert cache.latents.device.type == "cpu"
    assert cache.metadata.latent_shape == tuple(cache.latents.shape)
    assert cache.metadata.latent_dtype == "float32"
    assert (cache.metadata.category, cache.metadata.split) == (CATEGORY, "train")
    assert (cache.metadata.num_tokens, cache.metadata.embedding_dim) == (TOKENS, DIM)
    assert cache.metadata.sample_ids == ids["train"]


def test_sample_ids_stay_associated_with_their_own_latents(
    tmp_path: Path,
) -> None:
    root, ids = _produce(tmp_path)
    train, _ = validate_category_splits(tmp_path / "mvtec", CATEGORY)
    path_by_id = {sample.image_id: sample.image_path for sample in train.samples}
    encoder = MockLatentEncoder(DIM, seed=0)
    cache = _load(root, "train")

    for sample_id in ids["train"]:
        image = preprocess_image(path_by_id[sample_id], PREPROCESS).unsqueeze(0)
        torch.testing.assert_close(
            cache.get(sample_id), encoder.encode(image)[0], atol=1e-5, rtol=0
        )
    # Distinct images must not share latents (guards against a constant mock).
    first, second = ids["train"][:2]
    assert not torch.equal(cache.get(first), cache.get(second))


def test_get_is_one_sample_select_is_a_batch_in_requested_order(
    tmp_path: Path,
) -> None:
    root, ids = _produce(tmp_path)
    cache = _load(root, "train")
    first, second, third = ids["train"][:3]

    assert cache.get(first).shape == (TOKENS, DIM)  # one sample: no batch dim
    batch = cache.select([third, first])
    assert batch.shape == (2, TOKENS, DIM)  # batch: leading N
    assert torch.equal(batch[0], cache.get(third))
    assert torch.equal(batch[1], cache.get(first))
    assert cache.select([]).shape == (0, TOKENS, DIM)
    assert cache.index_of(second) == 1
    assert second in cache
    assert "not-a-sample-id" not in cache


def test_in_memory_edits_do_not_change_the_stored_entry(
    tmp_path: Path,
) -> None:
    root, ids = _produce(tmp_path)
    cache = _load(root, "train")
    original = cache.get(ids["train"][0]).clone()
    cache.select(list(ids["train"])).zero_()  # select returns a copy
    assert torch.equal(cache.get(ids["train"][0]), original)

    cache.latents.zero_()  # an in-memory edit of the loaded tensor
    assert torch.equal(_load(root, "train").get(ids["train"][0]), original)


def test_train_and_val_are_distinguishable_and_disjoint(
    tmp_path: Path,
) -> None:
    root, ids = _produce(tmp_path)
    train, val = _load(root, "train"), _load(root, "val")

    assert (train.metadata.split, val.metadata.split) == ("train", "val")
    assert set(train.metadata.sample_ids).isdisjoint(val.metadata.sample_ids)
    train_dataset, _ = validate_category_splits(tmp_path / "mvtec", CATEGORY)
    all_nominal = {sample.image_id for sample in train_dataset.samples}
    assert set(train.metadata.sample_ids) | set(val.metadata.sample_ids) == all_nominal
    assert val.metadata.sample_ids == ids["val"]
    first_val = ids["val"][0]
    assert first_val in val and first_val not in train
    with pytest.raises(KeyError):
        train.get(first_val)

    # A val file presented as train (or vice versa) is rejected, not silently used.
    val_path = latent_cache_path(root, CATEGORY, "val")
    with pytest.raises(CacheIncompatibleError, match="split"):
        load_latent_cache(val_path, _expected("train"))


def test_test_split_is_a_separate_entry_with_no_labels_or_nominal_overlap(
    tmp_path: Path,
) -> None:
    root, ids = _produce(tmp_path)
    test = _load(root, "test")
    nominal_ids = set(ids["train"]) | set(ids["val"])

    assert test.metadata.split == "test"
    assert test.metadata.sample_ids == ids["test"]
    assert nominal_ids.isdisjoint(test.metadata.sample_ids)
    # The entry stores no defect labels or masks; a consumer joins on image_id.
    assert "label" not in test.metadata.to_dict()
    assert "mask_path" not in test.metadata.to_dict()
    _, test_dataset = validate_category_splits(tmp_path / "mvtec", CATEGORY)
    labels = {sample.image_id: sample.label for sample in test_dataset.samples}
    assert sorted(labels[i] for i in test.metadata.sample_ids) == [
        "broken_large",
        "good",
    ]


def test_missing_entries_have_documented_errors_and_statuses(tmp_path: Path) -> None:
    path = latent_cache_path(tmp_path / "empty", CATEGORY, "train")
    expected = _expected("train")

    with pytest.raises(CacheNotFoundError):
        load_latent_cache(path, expected)
    with pytest.raises(FileNotFoundError):  # CacheNotFoundError is also this
        load_latent_cache(path, expected)
    assert try_load_latent_cache(path, expected) is None
    assert cache_exists(path, expected) is False


def test_missing_sample_id_is_a_key_error(
    tmp_path: Path,
) -> None:
    root, _ = _produce(tmp_path)
    cache = _load(root, "train")
    with pytest.raises(KeyError, match="not in latent cache"):
        cache.get("0" * 64)
    with pytest.raises(KeyError):
        cache.select(["0" * 64])
    with pytest.raises(KeyError):
        cache.index_of("0" * 64)


def test_incompatible_expectations_are_rejected_not_reused(
    tmp_path: Path,
) -> None:
    root, ids = _produce(tmp_path)
    path = latent_cache_path(root, CATEGORY, "train")
    spec = PreprocessingSpec.from_config(PREPROCESS)
    changes: dict[str, CacheExpectations] = {
        "checkpoint": _expected("train", checkpoint_sha256="a" * 64),
        "encoder name": _expected("train", encoder_name="other_encoder"),
        "embedding dim": _expected("train", embedding_dim=DIM + 1),
        "patch size": _expected("train", patch_size=14),
        "category": _expected("train", category="capsule"),
        "schema version": _expected("train", schema_version=1),
        "preprocessing version": _expected(
            "train", preprocessing=replace(spec, version=spec.version + 1)
        ),
        "preprocessing mean": _expected(
            "train", preprocessing=replace(spec, mean=(0.5, 0.5, 0.5))
        ),
        "sample ids reordered": _expected("train", sample_ids=ids["train"][::-1]),
        "sample ids subset": _expected("train", sample_ids=ids["train"][:-1]),
    }
    for name, expected in changes.items():
        with pytest.raises(CacheIncompatibleError):
            load_latent_cache(path, expected)
        assert try_load_latent_cache(path, expected) is None, name
        assert cache_exists(path, expected) is False, name

    # The exact expectations, with the ids pinned, are still accepted.
    assert cache_exists(path, _expected("train", sample_ids=ids["train"]))


def test_corrupted_entry_is_reported_not_treated_as_missing(
    tmp_path: Path,
) -> None:
    root, _ = _produce(tmp_path)
    path = latent_cache_path(root, CATEGORY, "val")
    path.write_bytes(path.read_bytes()[:-64])

    with pytest.raises(CacheCorruptedError):
        load_latent_cache(path, _expected("val"))
    with pytest.raises(CacheCorruptedError):  # never silently None
        try_load_latent_cache(path, _expected("val"))
    with pytest.raises(CacheCorruptedError):
        cache_exists(path, _expected("val"))


def test_cache_path_helper_rejects_unsafe_names(tmp_path: Path) -> None:
    for category, split in (("..", "train"), (CATEGORY, "../train"), ("a/b", "val")):
        with pytest.raises(ValueError, match="plain names"):
            latent_cache_path(tmp_path, category, split)


def test_consumer_needs_only_the_documented_path_helper(
    tmp_path: Path,
) -> None:
    root, _ = _produce(tmp_path)
    # Layout is documented (<root>/<category>/<split>.pt), and staging shards are
    # gone once an entry is complete, so nothing else is visible to a consumer.
    assert latent_cache_path(root, CATEGORY, "train") == root / CATEGORY / "train.pt"
    assert sorted(p.name for p in (root / CATEGORY).iterdir()) == [
        "test.pt",
        "train.pt",
        "val.pt",
    ]


def test_mock_entry_with_real_geometry_is_196_by_1408(tmp_path: Path) -> None:
    """Shape contract of the real encoder, produced by the mock (not real latents)."""
    _make_mvtec(tmp_path / "mvtec", n_train=4)
    summary = run_cache_generation(
        _config(
            tmp_path,
            image_size=224,
            mock_embedding_dim=1408,
            val_fraction=0.25,
            batch_size=2,
        )
    )
    assert summary.ok and summary.encoder_kind == "mock"
    assert summary.real_checkpoint_inference is False

    expected = _expected(
        "train",
        embedding_dim=1408,
        preprocessing=PreprocessingSpec.from_config(PreprocessConfig(image_size=224)),
    )
    cache = load_latent_cache(
        latent_cache_path(summary.cache_root, CATEGORY, "train"), expected
    )
    assert cache.latents.shape[1:] == (196, 1408)
    assert cache.get(cache.metadata.sample_ids[0]).shape == (196, 1408)
    assert cache.select(cache.metadata.sample_ids).shape == (3, 196, 1408)
