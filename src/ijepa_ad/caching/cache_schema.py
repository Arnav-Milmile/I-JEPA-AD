"""Versioned, validated metadata schema for cached I-JEPA latents.

Metadata read from disk is never trusted: ``CacheMetadata.from_dict`` re-runs every
check that constructing a new instance runs, and rejects unknown or missing keys.
See DECISIONS.md, 2026-10-10 "Latent cache format and schema".
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ijepa_ad.data.transforms import PREPROCESSING_VERSION, PreprocessConfig

SCHEMA_VERSION = 2
VALID_SPLITS = frozenset({"train", "val", "test"})
SUPPORTED_LATENT_DTYPES = frozenset({"float16", "bfloat16", "float32"})

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CacheMetadataError(ValueError):
    """Raised when cache metadata is missing, malformed, or inconsistent."""


class UnsupportedSchemaVersionError(CacheMetadataError):
    """Raised when metadata declares a schema version this code cannot read."""


def is_safe_name(value: object) -> bool:
    """Whether ``value`` is usable as one path component (no separators, no ``..``)."""
    return (
        isinstance(value, str)
        and _SAFE_NAME.match(value) is not None
        and ".." not in value
    )


@dataclass(frozen=True)
class PreprocessingSpec:
    """Preprocessing parameters that determine the pixels the encoder saw."""

    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    resize: str = "bilinear"
    crop: str = "none"
    version: int = PREPROCESSING_VERSION

    @classmethod
    def from_config(cls, config: PreprocessConfig) -> PreprocessingSpec:
        """Describe ``config``; resize/crop mirror ``data.transforms`` (fixed)."""
        return cls(image_size=config.image_size, mean=config.mean, std=config.std)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "resize": self.resize,
            "crop": self.crop,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: object) -> PreprocessingSpec:
        mapping = _require_mapping(data, "preprocessing")
        _require_keys(
            mapping,
            {"image_size", "mean", "std", "resize", "crop", "version"},
            "preprocessing",
        )
        spec = cls(
            image_size=_as_positive_int(
                mapping["image_size"], "preprocessing.image_size"
            ),
            mean=_as_float_triple(mapping["mean"], "preprocessing.mean"),
            std=_as_float_triple(mapping["std"], "preprocessing.std"),
            resize=_as_str(mapping["resize"], "preprocessing.resize"),
            crop=_as_str(mapping["crop"], "preprocessing.crop"),
            version=_as_positive_int(mapping["version"], "preprocessing.version"),
        )
        if any(value <= 0 for value in spec.std):
            raise CacheMetadataError(
                f"preprocessing.std values must be positive, got {spec.std}"
            )
        return spec


@dataclass(frozen=True)
class CacheMetadata:
    """Everything needed to decide whether a cached latent tensor is reusable."""

    schema_version: int
    category: str
    split: str
    sample_ids: tuple[str, ...]
    encoder_name: str
    checkpoint_sha256: str
    patch_size: int
    input_shape: tuple[int, int, int]
    preprocessing: PreprocessingSpec
    latent_shape: tuple[int, int, int]
    latent_dtype: str
    num_tokens: int
    embedding_dim: int
    latents_sha256: str
    created_at_utc: str
    torch_version: str

    def __post_init__(self) -> None:
        # Runs on direct construction and on from_dict, so both paths are validated.
        if self.schema_version != SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(
                f"unsupported cache schema_version {self.schema_version!r}; this "
                f"code reads version {SCHEMA_VERSION}. Rebuild the cache."
            )
        if not is_safe_name(self.category):
            raise CacheMetadataError(
                f"category must be a plain name (letters, digits, '_', '-', '.'), "
                f"got {self.category!r}"
            )
        if self.split not in VALID_SPLITS:
            raise CacheMetadataError(
                f"split must be one of {sorted(VALID_SPLITS)}, got {self.split!r}"
            )
        if not self.encoder_name:
            raise CacheMetadataError("encoder_name must be a non-empty string")
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _require_sha256(self.latents_sha256, "latents_sha256")
        if self.patch_size <= 0:
            raise CacheMetadataError(f"patch_size must be positive: {self.patch_size}")
        if any(dim <= 0 for dim in self.input_shape):
            raise CacheMetadataError(
                f"input_shape must be positive: {self.input_shape}"
            )
        if self.latent_dtype not in SUPPORTED_LATENT_DTYPES:
            raise CacheMetadataError(
                f"latent_dtype must be one of {sorted(SUPPORTED_LATENT_DTYPES)}, "
                f"got {self.latent_dtype!r}"
            )
        _validate_sample_ids(self.sample_ids)
        _validate_timestamp(self.created_at_utc)
        if not self.torch_version:
            raise CacheMetadataError("torch_version must be a non-empty string")

        channels, height, width = self.input_shape
        if channels != 3 or height != width:
            raise CacheMetadataError(
                f"input_shape must be (3, S, S), got {self.input_shape}"
            )
        if self.preprocessing.image_size != height:
            raise CacheMetadataError(
                f"input_shape {self.input_shape} disagrees with "
                f"preprocessing.image_size {self.preprocessing.image_size}"
            )
        if any(dim <= 0 for dim in self.latent_shape):
            raise CacheMetadataError(
                f"latent_shape must be positive: {self.latent_shape}"
            )
        count, tokens, dim = self.latent_shape
        if count != len(self.sample_ids):
            raise CacheMetadataError(
                f"latent_shape has {count} samples but sample_ids has "
                f"{len(self.sample_ids)}"
            )
        if (tokens, dim) != (self.num_tokens, self.embedding_dim):
            raise CacheMetadataError(
                f"latent_shape {self.latent_shape} disagrees with num_tokens="
                f"{self.num_tokens}, embedding_dim={self.embedding_dim}"
            )
        expected_tokens = (height // self.patch_size) ** 2
        if self.num_tokens != expected_tokens:
            raise CacheMetadataError(
                f"num_tokens {self.num_tokens} does not match a {height}px input "
                f"with {self.patch_size}px patches ({expected_tokens} tokens)"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "category": self.category,
            "split": self.split,
            "sample_ids": list(self.sample_ids),
            "encoder_name": self.encoder_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "patch_size": self.patch_size,
            "input_shape": list(self.input_shape),
            "preprocessing": self.preprocessing.to_dict(),
            "latent_shape": list(self.latent_shape),
            "latent_dtype": self.latent_dtype,
            "num_tokens": self.num_tokens,
            "embedding_dim": self.embedding_dim,
            "latents_sha256": self.latents_sha256,
            "created_at_utc": self.created_at_utc,
            "torch_version": self.torch_version,
        }

    def to_json(self) -> str:
        """Canonical JSON (sorted keys, no whitespace): equal metadata, equal text."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: object) -> CacheMetadata:
        mapping = _require_mapping(data, "metadata")
        # Version first: an old/new schema may legitimately have different keys.
        version = mapping.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise CacheMetadataError(
                f"schema_version must be an integer, got {version!r}"
            )
        if version != SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(
                f"unsupported cache schema_version {version!r}; this code reads "
                f"version {SCHEMA_VERSION}. Rebuild the cache."
            )
        _require_keys(mapping, set(cls.__dataclass_fields__), "metadata")
        sample_ids = mapping["sample_ids"]
        if not isinstance(sample_ids, list):
            raise CacheMetadataError("sample_ids must be a list of strings")
        return cls(
            schema_version=version,
            category=_as_str(mapping["category"], "category"),
            split=_as_str(mapping["split"], "split"),
            sample_ids=tuple(_as_str(item, "sample_ids[]") for item in sample_ids),
            encoder_name=_as_str(mapping["encoder_name"], "encoder_name"),
            checkpoint_sha256=_as_str(
                mapping["checkpoint_sha256"], "checkpoint_sha256"
            ),
            patch_size=_as_positive_int(mapping["patch_size"], "patch_size"),
            input_shape=_as_int_triple(mapping["input_shape"], "input_shape"),
            preprocessing=PreprocessingSpec.from_dict(mapping["preprocessing"]),
            latent_shape=_as_int_triple(mapping["latent_shape"], "latent_shape"),
            latent_dtype=_as_str(mapping["latent_dtype"], "latent_dtype"),
            num_tokens=_as_positive_int(mapping["num_tokens"], "num_tokens"),
            embedding_dim=_as_positive_int(mapping["embedding_dim"], "embedding_dim"),
            latents_sha256=_as_str(mapping["latents_sha256"], "latents_sha256"),
            created_at_utc=_as_str(mapping["created_at_utc"], "created_at_utc"),
            torch_version=_as_str(mapping["torch_version"], "torch_version"),
        )

    @classmethod
    def from_json(cls, text: str) -> CacheMetadata:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise CacheMetadataError(f"metadata is not valid JSON: {error}") from error
        return cls.from_dict(data)


@dataclass(frozen=True)
class CacheExpectations:
    """What the caller needs a cache entry to match in order to reuse it.

    ``sample_ids`` is optional; when given, the entry must contain exactly those
    samples in that order.
    """

    category: str
    split: str
    encoder_name: str
    checkpoint_sha256: str
    patch_size: int
    embedding_dim: int
    preprocessing: PreprocessingSpec
    sample_ids: tuple[str, ...] | None = None
    schema_version: int = SCHEMA_VERSION

    def mismatches(self, metadata: CacheMetadata) -> list[str]:
        """Human-readable differences between ``metadata`` and these expectations."""
        checks: list[tuple[str, object, object]] = [
            ("schema_version", self.schema_version, metadata.schema_version),
            ("category", self.category, metadata.category),
            ("split", self.split, metadata.split),
            ("encoder_name", self.encoder_name, metadata.encoder_name),
            ("checkpoint_sha256", self.checkpoint_sha256, metadata.checkpoint_sha256),
            ("patch_size", self.patch_size, metadata.patch_size),
            ("embedding_dim", self.embedding_dim, metadata.embedding_dim),
            ("preprocessing", self.preprocessing, metadata.preprocessing),
        ]
        if self.sample_ids is not None:
            checks.append(("sample_ids", self.sample_ids, metadata.sample_ids))
        return [
            f"{name}: expected {_brief(expected)}, cache has {_brief(actual)}"
            for name, expected, actual in checks
            if expected != actual
        ]


def _brief(value: object, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _require_mapping(data: object, name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise CacheMetadataError(
            f"{name} must be a JSON object, got {type(data).__name__}"
        )
    return data


def _require_keys(mapping: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing:
        raise CacheMetadataError(f"{name} is missing required fields: {missing}")
    if unknown:
        raise CacheMetadataError(f"{name} has unknown fields: {unknown}")


def _as_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise CacheMetadataError(f"{name} must be a string, got {value!r}")
    return value


def _as_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CacheMetadataError(f"{name} must be a positive integer, got {value!r}")
    return value


def _as_int_triple(value: object, name: str) -> tuple[int, int, int]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise CacheMetadataError(f"{name} must be a list of 3 integers, got {value!r}")
    a, b, c = (_as_positive_int(item, f"{name}[]") for item in value)
    return (a, b, c)


def _as_float_triple(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise CacheMetadataError(f"{name} must be a list of 3 numbers, got {value!r}")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise CacheMetadataError(f"{name} must contain numbers, got {item!r}")
    a, b, c = (float(item) for item in value)
    return (a, b, c)


def _require_sha256(value: str, name: str) -> None:
    if _SHA256.match(value) is None:
        raise CacheMetadataError(
            f"{name} must be 64 lowercase hex characters, got {_brief(value)}"
        )


def _validate_sample_ids(sample_ids: tuple[str, ...]) -> None:
    if not sample_ids:
        raise CacheMetadataError("sample_ids must not be empty")
    if any(not item for item in sample_ids):
        raise CacheMetadataError("sample_ids must not contain empty strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise CacheMetadataError("sample_ids must be unique")


def _validate_timestamp(value: str) -> None:
    try:
        datetime.fromisoformat(value)
    except ValueError as error:
        raise CacheMetadataError(
            f"created_at_utc must be an ISO-8601 timestamp, got {value!r}"
        ) from error
