"""Save and load I-JEPA latents with validated, versioned metadata.

Each entry is one ``.pt`` file holding only a latent tensor and a canonical JSON
metadata string, so it can be read with ``torch.load(weights_only=True)`` (no
arbitrary object deserialization). Writes go to a temporary file in the same
directory and are moved into place with ``os.replace``, so a crash never leaves a
truncated file at the final path. See DECISIONS.md, 2026-10-10 "Latent cache format
and schema".
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor

from ijepa_ad.caching.cache_schema import (
    SCHEMA_VERSION,
    CacheExpectations,
    CacheMetadata,
    CacheMetadataError,
    PreprocessingSpec,
    UnsupportedSchemaVersionError,
    is_safe_name,
)

_PAYLOAD_KEYS = {"latents", "metadata_json"}
_HASH_CHUNK_BYTES = 1 << 20
_DTYPE_BY_NAME = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class CacheError(Exception):
    """Base class for latent-cache failures."""


class CacheNotFoundError(CacheError, FileNotFoundError):
    """No cache file exists at the requested path."""


class CacheCorruptedError(CacheError):
    """The file is unreadable, truncated, tampered with, or internally inconsistent."""


class CacheIncompatibleError(CacheError):
    """The file is intact but was built for a different configuration or schema."""


@dataclass(frozen=True)
class LatentCache:
    """A loaded cache entry: ``latents`` is ``N×tokens×dim`` on CPU."""

    latents: Tensor
    metadata: CacheMetadata

    def __contains__(self, sample_id: object) -> bool:
        return sample_id in self.metadata.sample_ids

    def index_of(self, sample_id: str) -> int:
        """Row of ``sample_id`` in ``latents``; ``KeyError`` if it is not cached."""
        try:
            return self.metadata.sample_ids.index(sample_id)
        except ValueError:
            raise KeyError(f"sample id not in latent cache: {sample_id!r}") from None

    def get(self, sample_id: str) -> Tensor:
        """Latents ``tokens×dim`` of one sample, looked up by its stable image id."""
        return self.latents[self.index_of(sample_id)]

    def select(self, sample_ids: Sequence[str]) -> Tensor:
        """Latents ``len(sample_ids)×tokens×dim`` in the order requested."""
        rows = [self.index_of(sample_id) for sample_id in sample_ids]
        return self.latents[rows]


def latent_cache_path(cache_root: str | Path, category: str, split: str) -> Path:
    """Path of the entry for one category/split: ``<root>/<category>/<split>.pt``."""
    if not is_safe_name(category) or not is_safe_name(split):
        raise ValueError(
            f"category and split must be plain names, got {category!r}, {split!r}"
        )
    return Path(cache_root) / category / f"{split}.pt"


def compute_file_sha256(path: str | Path) -> str:
    """Stream a file through SHA-256 (use on the checkpoint, once per run)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def compute_tensor_sha256(tensor: Tensor) -> str:
    """SHA-256 of a tensor's raw bytes in C order; dtype and shape are not hashed."""
    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    raw = flat.view(torch.uint8) if flat.numel() else flat.to(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def build_metadata(
    latents: Tensor,
    *,
    category: str,
    split: str,
    sample_ids: tuple[str, ...] | list[str],
    encoder_name: str,
    checkpoint_sha256: str,
    patch_size: int,
    preprocessing: PreprocessingSpec,
) -> CacheMetadata:
    """Describe ``latents`` (``N×tokens×dim``) so they can be saved and reused."""
    if latents.ndim != 3:
        raise CacheMetadataError(
            f"latents must be N×tokens×dim, got shape {tuple(latents.shape)}"
        )
    n, tokens, dim = (int(size) for size in latents.shape)
    size = preprocessing.image_size
    return CacheMetadata(
        schema_version=SCHEMA_VERSION,
        category=category,
        split=split,
        sample_ids=tuple(sample_ids),
        encoder_name=encoder_name,
        checkpoint_sha256=checkpoint_sha256,
        patch_size=patch_size,
        input_shape=(3, size, size),
        preprocessing=preprocessing,
        latent_shape=(n, tokens, dim),
        latent_dtype=str(latents.dtype).removeprefix("torch."),
        num_tokens=tokens,
        embedding_dim=dim,
        latents_sha256=compute_tensor_sha256(latents),
        created_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        torch_version=str(torch.__version__),
    )


def save_latent_cache(
    path: str | Path, latents: Tensor, metadata: CacheMetadata
) -> Path:
    """Validate and atomically write ``latents`` + ``metadata`` to ``path``."""
    path = Path(path)
    tensor = latents.detach().to("cpu").contiguous()
    _check_tensor_matches_metadata(tensor, metadata)
    if not bool(torch.isfinite(tensor).all()):
        raise CacheMetadataError("latents contain NaN or infinite values")
    if compute_tensor_sha256(tensor) != metadata.latents_sha256:
        raise CacheMetadataError(
            "metadata.latents_sha256 does not match the latents being saved; "
            "build the metadata with build_metadata()"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"latents": tensor.clone(), "metadata_json": metadata.to_json()}
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise
    return path


def load_latent_cache(
    path: str | Path, expected: CacheExpectations | None = None
) -> LatentCache:
    """Load and fully validate a cache entry.

    Raises ``CacheNotFoundError`` (missing), ``CacheCorruptedError`` (unreadable or
    inconsistent) or ``CacheIncompatibleError`` (wrong schema version, or differs
    from ``expected``). Pass ``expected`` whenever the latents will be reused.
    """
    path = Path(path)
    if not path.is_file():
        raise CacheNotFoundError(f"latent cache not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CacheCorruptedError(
            f"cannot read latent cache {path}: {error}"
        ) from error

    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
        raise CacheCorruptedError(
            f"latent cache {path} must contain exactly {sorted(_PAYLOAD_KEYS)}"
        )
    latents, metadata_json = payload["latents"], payload["metadata_json"]
    if not isinstance(latents, Tensor) or not isinstance(metadata_json, str):
        raise CacheCorruptedError(f"latent cache {path} has malformed payload types")

    try:
        metadata = CacheMetadata.from_json(metadata_json)
    except UnsupportedSchemaVersionError as error:
        raise CacheIncompatibleError(f"{path}: {error}") from error
    except CacheMetadataError as error:
        raise CacheCorruptedError(f"invalid metadata in {path}: {error}") from error

    try:
        _check_tensor_matches_metadata(latents, metadata)
    except CacheMetadataError as error:
        raise CacheCorruptedError(f"{path}: {error}") from error
    if compute_tensor_sha256(latents) != metadata.latents_sha256:
        raise CacheCorruptedError(f"{path}: latent data does not match its checksum")

    if expected is not None:
        differences = expected.mismatches(metadata)
        if differences:
            raise CacheIncompatibleError(
                f"latent cache {path} is stale or built for a different "
                f"configuration: " + "; ".join(differences)
            )
    return LatentCache(latents=latents, metadata=metadata)


def try_load_latent_cache(
    path: str | Path, expected: CacheExpectations
) -> LatentCache | None:
    """Return the entry if present and compatible, ``None`` if it must be rebuilt.

    Missing and stale entries (including old schema versions) yield ``None``.
    Corruption still raises ``CacheCorruptedError`` so damage is never silent.
    """
    try:
        return load_latent_cache(path, expected)
    except (CacheNotFoundError, CacheIncompatibleError):
        return None


def cache_exists(path: str | Path, expected: CacheExpectations) -> bool:
    """Whether a valid entry compatible with ``expected`` is at ``path``.

    Fully validates (checksum included), so ``True`` means the entry can be loaded.
    Missing and stale entries are ``False``; corruption raises ``CacheCorruptedError``.
    """
    return try_load_latent_cache(path, expected) is not None


def _check_tensor_matches_metadata(tensor: Tensor, metadata: CacheMetadata) -> None:
    if tuple(tensor.shape) != metadata.latent_shape:
        raise CacheMetadataError(
            f"latents have shape {tuple(tensor.shape)} but metadata declares "
            f"{metadata.latent_shape}"
        )
    if tensor.dtype != _DTYPE_BY_NAME[metadata.latent_dtype]:
        raise CacheMetadataError(
            f"latents have dtype {tensor.dtype} but metadata declares "
            f"{metadata.latent_dtype}"
        )
