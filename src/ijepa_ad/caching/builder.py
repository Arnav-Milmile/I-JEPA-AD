"""Resumable latent-cache generation: samples → preprocessing → encoder → cache.

Encoding is done in batches and every sample's latents are written to its own
shard (a valid one-sample cache entry) as soon as its batch finishes, so memory
during encoding is bounded by the batch size and an interrupted run resumes by
skipping compatible shards. Only after every sample has a shard are they assembled
into the final ``<root>/<category>/<split>.pt`` entry, which is never written
from partial results. See DECISIONS.md, 2026-10-10 "Resumable cache generation".
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from ijepa_ad.caching.cache_schema import (
    VALID_SPLITS,
    CacheExpectations,
    PreprocessingSpec,
    is_safe_name,
)
from ijepa_ad.caching.latent_cache import (
    CacheCorruptedError,
    CacheError,
    CacheIncompatibleError,
    build_metadata,
    latent_cache_path,
    load_latent_cache,
    save_latent_cache,
    try_load_latent_cache,
)
from ijepa_ad.data.splits import split_nominal_train_val
from ijepa_ad.data.transforms import PreprocessConfig, preprocess_image
from ijepa_ad.data.validators import (
    DatasetValidationError,
    SampleMetadata,
    validate_test_records,
    validate_training_records,
)

_Expectations = Callable[[tuple[str, ...]], CacheExpectations]


class LatentEncoder(Protocol):
    """Anything with the frozen-encoder contract ``N×3×S×S → N×tokens×dim``."""

    def encode(self, image_batch: Tensor) -> Tensor: ...


class CacheBuildError(CacheError):
    """Generation could not complete; finished shards are kept for resuming.

    ``failed`` maps sample id to the reason that sample could not be encoded.
    """

    def __init__(self, message: str, failed: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.failed = failed or {}


@dataclass(frozen=True)
class BuildReport:
    """What one ``build_latent_cache`` call did."""

    path: Path
    total: int
    encoded: int  # samples the encoder ran on in this call
    reused: int  # samples served from existing shards or the final entry
    already_complete: bool  # a compatible final entry existed; nothing was done


def parts_dir(cache_root: str | Path, category: str, split: str) -> Path:
    """Staging directory for per-sample shards of one entry."""
    final = latent_cache_path(cache_root, category, split)
    return final.parent / f".{split}.parts"


def build_latent_cache(
    samples: Sequence[SampleMetadata],
    encoder: LatentEncoder,
    *,
    cache_root: str | Path,
    category: str,
    split: str,
    encoder_name: str,
    checkpoint_sha256: str,
    patch_size: int,
    embedding_dim: int,
    config: PreprocessConfig | None = None,
    batch_size: int = 8,
) -> BuildReport:
    """Create (or resume, or reuse) the cache entry for one category/split.

    ``samples`` define the entry's rows and order. ``split`` ``train``/``val`` accept
    only official nominal training records; ``test`` only official test records.
    Raises ``CacheBuildError`` if any sample cannot be preprocessed (the rest are
    kept as shards, and no final entry is written).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")
    records = tuple(samples)
    _validate_records(records, category=category, split=split)

    config = config or PreprocessConfig()
    spec = PreprocessingSpec.from_config(config)
    ids = tuple(record.image_id for record in records)
    final_path = latent_cache_path(cache_root, category, split)
    shard_root = parts_dir(cache_root, category, split)

    def expectations(sample_ids: tuple[str, ...]) -> CacheExpectations:
        return CacheExpectations(
            category=category,
            split=split,
            encoder_name=encoder_name,
            checkpoint_sha256=checkpoint_sha256,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            preprocessing=spec,
            sample_ids=sample_ids,
        )

    if try_load_latent_cache(final_path, expectations(ids)) is not None:
        return BuildReport(final_path, len(ids), 0, len(ids), True)

    missing = [
        record
        for record in records
        if _load_shard(shard_root, record.image_id, expectations) is None
    ]
    encoded = 0
    failed: dict[str, str] = {}
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        images: list[Tensor] = []
        good: list[SampleMetadata] = []
        for record in batch:
            try:
                images.append(preprocess_image(record.image_path, config))
                good.append(record)
            except (OSError, ValueError) as error:
                failed[record.image_id] = f"{record.image_path.name}: {error}"
        if not good:
            continue
        latents = encoder.encode(torch.stack(images)).detach().to("cpu")
        if latents.ndim != 3 or latents.shape[0] != len(good):
            raise CacheBuildError(
                f"encoder returned shape {tuple(latents.shape)} for {len(good)} "
                "images; refusing to assign latents to samples"
            )
        for row, record in enumerate(good):
            _save_shard(shard_root, record, latents[row : row + 1], expectations)
        encoded += len(good)

    if failed:
        raise CacheBuildError(
            f"{len(failed)} of {len(ids)} samples failed; no final cache written "
            f"for {category}/{split}: {failed}",
            failed,
        )

    _assemble(records, shard_root, final_path, expectations, spec)
    shutil.rmtree(shard_root, ignore_errors=True)
    return BuildReport(final_path, len(ids), encoded, len(ids) - encoded, False)


def build_nominal_latent_caches(
    train_samples: Sequence[SampleMetadata],
    encoder: LatentEncoder,
    *,
    val_fraction: float,
    seed: int,
    min_train: int = 1,
    min_val: int = 1,
    **build_arguments: object,
) -> dict[str, BuildReport]:
    """Split official ``train/good`` records and cache the ``train`` and ``val`` parts.

    Only nominal training records are accepted (``split_nominal_train_val`` rejects
    defect, masked, and test records), so no test or defect image is ever encoded
    here. ``build_arguments`` are those of ``build_latent_cache`` (``cache_root``,
    ``category``, ``encoder_name``, ...).
    """
    split = split_nominal_train_val(
        train_samples,
        val_fraction=val_fraction,
        seed=seed,
        min_train=min_train,
        min_val=min_val,
    )
    return {
        name: build_latent_cache(
            records,
            encoder,
            split=name,
            **build_arguments,  # type: ignore[arg-type]
        )
        for name, records in (("train", split.train), ("val", split.val))
    }


def _validate_records(
    records: tuple[SampleMetadata, ...], *, category: str, split: str
) -> None:
    if split == "test":
        validate_test_records(records)
    else:
        validate_training_records(records)  # nominal train/good only, never test
    for record in records:
        if not is_safe_name(record.image_id):
            raise DatasetValidationError(f"unsafe sample id: {record.image_id!r}")
        record_category = getattr(record, "category", category)
        if record_category not in ("", category):
            raise DatasetValidationError(
                f"sample from category {record_category!r} given for {category!r}: "
                f"{record.image_path}"
            )


def _load_shard(
    shard_root: Path,
    sample_id: str,
    expectations: _Expectations,
) -> Tensor | None:
    """A compatible, intact shard's latents, else ``None`` (shards are derived data)."""
    make = expectations
    try:
        entry = try_load_latent_cache(
            shard_root / f"{sample_id}.pt", make((sample_id,))
        )
    except CacheCorruptedError:
        return None
    return None if entry is None else entry.latents


def _save_shard(
    shard_root: Path,
    record: SampleMetadata,
    latents: Tensor,
    expectations: _Expectations,
) -> None:
    make = expectations
    expected = make((record.image_id,))
    metadata = build_metadata(
        latents,
        category=expected.category,
        split=expected.split,
        sample_ids=[record.image_id],
        encoder_name=expected.encoder_name,
        checkpoint_sha256=expected.checkpoint_sha256,
        patch_size=expected.patch_size,
        preprocessing=expected.preprocessing,
    )
    if metadata.embedding_dim != expected.embedding_dim:
        raise CacheBuildError(
            f"encoder produced embedding_dim {metadata.embedding_dim}, "
            f"expected {expected.embedding_dim}"
        )
    save_latent_cache(shard_root / f"{record.image_id}.pt", latents, metadata)


def _assemble(
    records: tuple[SampleMetadata, ...],
    shard_root: Path,
    final_path: Path,
    expectations: _Expectations,
    spec: PreprocessingSpec,
) -> None:
    make = expectations
    rows: Tensor | None = None
    for index, record in enumerate(records):
        expected = make((record.image_id,))
        shard = load_latent_cache(shard_root / f"{record.image_id}.pt", expected)
        if rows is None:
            rows = torch.empty(
                (len(records), *shard.latents.shape[1:]), dtype=shard.latents.dtype
            )
        elif shard.latents.dtype != rows.dtype or shard.latents.shape[1:] != (
            rows.shape[1:]
        ):
            raise CacheIncompatibleError(
                f"shard for {record.image_id} has dtype/shape "
                f"{shard.latents.dtype}/{tuple(shard.latents.shape[1:])}, others "
                f"{rows.dtype}/{tuple(rows.shape[1:])}; delete {shard_root} and rerun"
            )
        rows[index] = shard.latents[0]
    assert rows is not None
    final_expected = make(tuple(r.image_id for r in records))
    metadata = build_metadata(
        rows,
        category=final_expected.category,
        split=final_expected.split,
        sample_ids=final_expected.sample_ids or (),
        encoder_name=final_expected.encoder_name,
        checkpoint_sha256=final_expected.checkpoint_sha256,
        patch_size=final_expected.patch_size,
        preprocessing=spec,
    )
    save_latent_cache(final_path, rows, metadata)
