from pathlib import Path

import pytest
import torch

from ijepa_ad.caching.cache_schema import CacheExpectations, PreprocessingSpec
from ijepa_ad.caching.latent_cache import (
    build_metadata,
    latent_cache_path,
    save_latent_cache,
)
from ijepa_ad.data.mvtec_dataset import MVTecSample
from ijepa_ad.data.transforms import PreprocessConfig
from ijepa_ad.predictor.cache_dataset import (
    CachedLatentDataset,
    annotations_by_image_id,
    collate_cached_latents,
    load_cached_latent_dataset,
)

CATEGORY = "bottle"
SPLIT = "train"
ENCODER = "mock_encoder"
CHECKPOINT = "a" * 64
PATCH_SIZE = 16
DIM = 4
IMAGE_SIZE = 32
PREPROCESS = PreprocessConfig(image_size=IMAGE_SIZE)


def _expected(split: str = SPLIT) -> CacheExpectations:
    return CacheExpectations(
        category=CATEGORY,
        split=split,
        encoder_name=ENCODER,
        checkpoint_sha256=CHECKPOINT,
        patch_size=PATCH_SIZE,
        embedding_dim=DIM,
        preprocessing=PreprocessingSpec.from_config(PREPROCESS),
    )


def _write_cache(
    root: Path, split: str = SPLIT
) -> tuple[torch.Tensor, tuple[str, ...]]:
    latents = torch.arange(3 * 4 * DIM, dtype=torch.float32).reshape(3, 4, DIM)
    sample_ids = ("id-a", "id-b", "id-c")
    metadata = build_metadata(
        latents,
        category=CATEGORY,
        split=split,
        sample_ids=sample_ids,
        encoder_name=ENCODER,
        checkpoint_sha256=CHECKPOINT,
        patch_size=PATCH_SIZE,
        preprocessing=PreprocessingSpec.from_config(PREPROCESS),
    )
    save_latent_cache(latent_cache_path(root, CATEGORY, split), latents, metadata)
    return latents, sample_ids


def _sample(
    image_id: str,
    *,
    label: str = "good",
    category: str = CATEGORY,
    split: str = "train",
) -> MVTecSample:
    return MVTecSample(
        image_path=Path(f"{image_id}.png"),
        image_id=image_id,
        split=split,
        label=label,
        mask_path=None,
        category=category,
    )


def test_load_cached_latent_dataset_preserves_cache_order_and_shapes(
    tmp_path: Path,
) -> None:
    latents, sample_ids = _write_cache(tmp_path)

    dataset = load_cached_latent_dataset(tmp_path, _expected())

    assert isinstance(dataset, CachedLatentDataset)
    assert len(dataset) == 3
    assert dataset.category == CATEGORY
    assert dataset.split == SPLIT
    assert dataset.sample_ids == sample_ids
    assert torch.equal(dataset.latents, latents)
    assert dataset[0].image_id == "id-a"
    assert dataset[0].latent.shape == (4, DIM)
    assert torch.equal(dataset[0].latent, latents[0])
    assert [sample.image_id for sample in dataset[1:]] == ["id-b", "id-c"]


def test_dataset_selects_latents_in_requested_id_order(tmp_path: Path) -> None:
    latents, _ = _write_cache(tmp_path)
    dataset = load_cached_latent_dataset(tmp_path, _expected())

    selected = dataset.select(["id-c", "id-a"])

    assert selected.shape == (2, 4, DIM)
    assert torch.equal(selected[0], latents[2])
    assert torch.equal(selected[1], latents[0])


def test_dataset_indexing_uses_cache_row_without_id_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    latents, _ = _write_cache(tmp_path)
    dataset = load_cached_latent_dataset(tmp_path, _expected())

    def reject_id_lookup(_cache: object, _sample_id: str) -> torch.Tensor:
        raise AssertionError("dataset indexing should use the known cache row")

    monkeypatch.setattr(type(dataset.cache), "get", reject_id_lookup)

    assert torch.equal(dataset[1].latent, latents[1])


def test_optional_annotations_join_labels_and_masks_by_image_id(
    tmp_path: Path,
) -> None:
    _write_cache(tmp_path, split="test")
    samples = (
        _sample("id-a", label="good", split="test"),
        _sample("id-b", label="broken", split="test"),
        _sample("id-c", label="good", split="test"),
    )

    dataset = load_cached_latent_dataset(tmp_path, _expected("test"), samples=samples)

    assert dataset[1].annotation is not None
    assert dataset[1].annotation.label == "broken"
    assert dataset[1].annotation.image_path == Path("id-b.png")
    assert dataset[1].annotation.category == CATEGORY
    assert dataset[1].annotation.source_split == "test"


def test_annotations_must_match_cache_category_and_source_split(
    tmp_path: Path,
) -> None:
    _write_cache(tmp_path)

    with pytest.raises(ValueError, match="annotation category"):
        load_cached_latent_dataset(
            tmp_path,
            _expected(),
            samples=(
                _sample("id-a", category="capsule"),
                _sample("id-b", category="capsule"),
                _sample("id-c", category="capsule"),
            ),
        )
    with pytest.raises(ValueError, match="annotation source split"):
        load_cached_latent_dataset(
            tmp_path,
            _expected(),
            samples=(
                _sample("id-a", split="test"),
                _sample("id-b", split="test"),
                _sample("id-c", split="test"),
            ),
        )


def test_annotations_must_cover_every_cached_id(tmp_path: Path) -> None:
    _write_cache(tmp_path)

    with pytest.raises(ValueError, match="missing cached sample ids"):
        load_cached_latent_dataset(tmp_path, _expected(), samples=(_sample("id-a"),))


def test_duplicate_annotation_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate sample id"):
        annotations_by_image_id((_sample("id-a"), _sample("id-a")))


def test_collate_cached_latents_stacks_samples_in_order(tmp_path: Path) -> None:
    latents, _ = _write_cache(tmp_path)
    dataset = load_cached_latent_dataset(tmp_path, _expected())

    batch = collate_cached_latents([dataset[2], dataset[0]])

    assert batch.image_ids == ("id-c", "id-a")
    assert batch.category == CATEGORY
    assert batch.split == SPLIT
    assert torch.equal(batch.latents, torch.stack([latents[2], latents[0]], dim=0))
    assert batch.annotations == (None, None)


def test_collate_rejects_empty_or_mixed_split_batches(tmp_path: Path) -> None:
    _write_cache(tmp_path, split="train")
    _write_cache(tmp_path, split="val")
    train = load_cached_latent_dataset(tmp_path, _expected("train"))
    val = load_cached_latent_dataset(tmp_path, _expected("val"))

    with pytest.raises(ValueError, match="empty"):
        collate_cached_latents([])
    with pytest.raises(ValueError, match="category/split"):
        collate_cached_latents([train[0], val[0]])
