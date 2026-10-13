"""Dataset-style access to cached latents for predictor training/evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import torch
from torch import Tensor

from ijepa_ad.caching.cache_schema import CacheExpectations
from ijepa_ad.caching.latent_cache import (
    LatentCache,
    latent_cache_path,
    load_latent_cache,
)
from ijepa_ad.data.mvtec_dataset import MVTecSample


@dataclass(frozen=True)
class SampleAnnotation:
    """Optional non-latent metadata joined by stable image id."""

    image_path: Path
    label: str
    mask_path: Path | None
    category: str
    source_split: str


@dataclass(frozen=True)
class CachedLatentSample:
    """One cached sample consumed by the predictor."""

    image_id: str
    latent: Tensor
    category: str
    split: str
    annotation: SampleAnnotation | None = None


@dataclass(frozen=True)
class CachedLatentBatch:
    """Batch produced by ``collate_cached_latents``."""

    image_ids: tuple[str, ...]
    latents: Tensor
    category: str
    split: str
    annotations: tuple[SampleAnnotation | None, ...]


class CachedLatentDataset(Sequence[CachedLatentSample]):
    """Indexable predictor input backed by one validated latent-cache entry."""

    def __init__(
        self,
        cache: LatentCache,
        *,
        annotations: Mapping[str, SampleAnnotation] | None = None,
    ) -> None:
        self._cache = cache
        self._annotations = dict(annotations or {})
        source_split = _source_split_for_cache_split(self.split)
        missing = [
            image_id
            for image_id in self._cache.metadata.sample_ids
            if image_id not in self._annotations and annotations is not None
        ]
        if missing:
            raise ValueError(
                "annotations are missing cached sample ids: " + ", ".join(missing[:3])
            )
        mismatched = [
            image_id
            for image_id in self._cache.metadata.sample_ids
            if image_id in self._annotations
            and (
                self._annotations[image_id].category != self.category
                or self._annotations[image_id].source_split != source_split
            )
        ]
        if mismatched:
            raise ValueError(
                "annotations do not match cache category/source split for ids: "
                + ", ".join(mismatched[:3])
            )

    def __len__(self) -> int:
        return len(self._cache.metadata.sample_ids)

    @overload
    def __getitem__(self, index: int) -> CachedLatentSample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[CachedLatentSample]: ...

    def __getitem__(
        self, index: int | slice
    ) -> CachedLatentSample | Sequence[CachedLatentSample]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        image_id = self._cache.metadata.sample_ids[index]
        return CachedLatentSample(
            image_id=image_id,
            latent=self._cache.latents[index],
            category=self.category,
            split=self.split,
            annotation=self._annotations.get(image_id),
        )

    @property
    def category(self) -> str:
        return self._cache.metadata.category

    @property
    def split(self) -> str:
        return self._cache.metadata.split

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._cache.metadata.sample_ids

    @property
    def latents(self) -> Tensor:
        """All latents in cache row order."""
        return self._cache.latents

    @property
    def cache(self) -> LatentCache:
        """The validated cache entry, exposed for metadata/provenance reads."""
        return self._cache

    def select(self, image_ids: Sequence[str]) -> Tensor:
        """Return a latent batch in the requested image-id order."""
        return self._cache.select(image_ids)


def load_cached_latent_dataset(
    cache_root: str | Path,
    expected: CacheExpectations,
    *,
    samples: Sequence[MVTecSample] | None = None,
) -> CachedLatentDataset:
    """Load one category/split cache as the predictor's dataset surface."""
    cache = load_latent_cache(
        latent_cache_path(cache_root, expected.category, expected.split), expected
    )
    annotations = (
        None
        if samples is None
        else annotations_by_image_id(
            samples,
            category=cache.metadata.category,
            source_split=_source_split_for_cache_split(cache.metadata.split),
        )
    )
    return CachedLatentDataset(cache, annotations=annotations)


def annotations_by_image_id(
    samples: Sequence[MVTecSample],
    *,
    category: str | None = None,
    source_split: str | None = None,
) -> dict[str, SampleAnnotation]:
    """Build the label/mask join table for cached ids."""
    annotations: dict[str, SampleAnnotation] = {}
    for sample in samples:
        if category is not None and sample.category != category:
            raise ValueError(
                f"annotation category {sample.category!r} does not match {category!r}"
            )
        if source_split is not None and sample.split != source_split:
            raise ValueError(
                f"annotation source split {sample.split!r} does not match "
                f"{source_split!r}"
            )
        if sample.image_id in annotations:
            raise ValueError(f"duplicate sample id in annotations: {sample.image_id}")
        annotations[sample.image_id] = SampleAnnotation(
            image_path=sample.image_path,
            label=sample.label,
            mask_path=sample.mask_path,
            category=sample.category,
            source_split=sample.split,
        )
    return annotations


def _source_split_for_cache_split(cache_split: str) -> str:
    if cache_split in {"train", "val"}:
        return "train"
    if cache_split == "test":
        return "test"
    raise ValueError(f"unsupported cache split: {cache_split!r}")


def collate_cached_latents(
    samples: Sequence[CachedLatentSample],
) -> CachedLatentBatch:
    """Stack cached latent samples into a predictor batch."""
    if not samples:
        raise ValueError("cannot collate an empty latent batch")
    category = samples[0].category
    split = samples[0].split
    if any(sample.category != category or sample.split != split for sample in samples):
        raise ValueError(
            "all samples in a cached latent batch must share category/split"
        )
    return CachedLatentBatch(
        image_ids=tuple(sample.image_id for sample in samples),
        latents=torch.stack([sample.latent for sample in samples], dim=0),
        category=category,
        split=split,
        annotations=tuple(sample.annotation for sample in samples),
    )
