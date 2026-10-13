"""Tests for the versioned latent cache schema and its save/load behavior."""

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from ijepa_ad.caching.cache_schema import (
    SCHEMA_VERSION,
    CacheExpectations,
    CacheMetadata,
    CacheMetadataError,
    PreprocessingSpec,
    UnsupportedSchemaVersionError,
)
from ijepa_ad.caching.latent_cache import (
    CacheCorruptedError,
    CacheIncompatibleError,
    CacheNotFoundError,
    build_metadata,
    cache_exists,
    compute_file_sha256,
    latent_cache_path,
    load_latent_cache,
    save_latent_cache,
    try_load_latent_cache,
)
from ijepa_ad.data.transforms import PreprocessConfig

CHECKPOINT_SHA = "a" * 64
IMAGE_SIZE = 32
PATCH_SIZE = 16
TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2
DIM = 8
IDS = ("id-0", "id-1", "id-2")


def _spec(**overrides: Any) -> PreprocessingSpec:
    spec = PreprocessingSpec.from_config(PreprocessConfig(image_size=IMAGE_SIZE))
    return replace(spec, **overrides)


def _latents(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(len(IDS), TOKENS, DIM, generator=generator).to(dtype)


def _metadata(latents: torch.Tensor, **overrides: Any) -> CacheMetadata:
    arguments: dict[str, Any] = {
        "category": "bottle",
        "split": "train",
        "sample_ids": IDS,
        "encoder_name": "ijepa_vit_giant_16",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "patch_size": PATCH_SIZE,
        "preprocessing": _spec(),
    }
    arguments.update(overrides)
    return build_metadata(latents, **arguments)


def _expected(**overrides: Any) -> CacheExpectations:
    arguments: dict[str, Any] = {
        "category": "bottle",
        "split": "train",
        "encoder_name": "ijepa_vit_giant_16",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "patch_size": PATCH_SIZE,
        "embedding_dim": DIM,
        "preprocessing": _spec(),
    }
    arguments.update(overrides)
    return CacheExpectations(**arguments)


def _saved(tmp_path: Path, dtype: torch.dtype = torch.float32) -> Path:
    latents = _latents(dtype)
    path = latent_cache_path(tmp_path, "bottle", "train")
    return save_latent_cache(path, latents, _metadata(latents))


def _rewrite_metadata(path: Path, mutate: Any) -> None:
    """Replace the stored metadata JSON, keeping the latents (simulates disk edits)."""
    payload = torch.load(path, weights_only=True)
    data = json.loads(payload["metadata_json"])
    mutate(data)
    payload["metadata_json"] = json.dumps(data)
    torch.save(payload, path)


# --- metadata -----------------------------------------------------------------


def test_metadata_round_trips_through_canonical_json() -> None:
    metadata = _metadata(_latents())

    restored = CacheMetadata.from_json(metadata.to_json())

    assert restored == metadata
    assert restored.to_json() == metadata.to_json()
    assert metadata.schema_version == SCHEMA_VERSION
    assert metadata.latent_shape == (3, TOKENS, DIM)
    assert metadata.latent_dtype == "float32"
    assert metadata.input_shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_metadata_json_is_deterministic_and_sorted() -> None:
    metadata = _metadata(_latents())
    keys = list(json.loads(metadata.to_json()))

    assert keys == sorted(keys)
    assert metadata.to_json() == metadata.to_json()


def test_missing_required_field_is_rejected() -> None:
    full = _metadata(_latents()).to_dict()
    for field in full:
        data = {key: value for key, value in full.items() if key != field}

        with pytest.raises(CacheMetadataError, match=field):
            CacheMetadata.from_dict(data)


def test_unknown_field_is_rejected() -> None:
    data = _metadata(_latents()).to_dict()
    data["surprise"] = 1

    with pytest.raises(CacheMetadataError, match="unknown fields"):
        CacheMetadata.from_dict(data)


def test_unsupported_schema_version_is_rejected() -> None:
    for version in (0, SCHEMA_VERSION + 1, 99):
        data = _metadata(_latents()).to_dict()
        data["schema_version"] = version

        with pytest.raises(UnsupportedSchemaVersionError, match="schema_version"):
            CacheMetadata.from_dict(data)


INVALID_VALUES: list[tuple[str, Any]] = [
    ("schema_version", "1"),
    ("schema_version", True),
    ("category", 3),
    ("category", "../escape"),
    ("category", "a/b"),
    ("split", "holdout"),
    ("sample_ids", "id-0"),
    ("sample_ids", ["a", "a", "b"]),
    ("sample_ids", ["a", "", "b"]),
    ("encoder_name", ""),
    ("checkpoint_sha256", "xyz"),
    ("checkpoint_sha256", "A" * 64),
    ("patch_size", 0),
    ("patch_size", 16.0),
    ("patch_size", True),
    ("input_shape", [3, 32]),
    ("input_shape", [1, 32, 32]),
    ("input_shape", [3, 32, 64]),
    ("latent_shape", [3, TOKENS]),
    ("latent_shape", [5, TOKENS, DIM]),
    ("latent_dtype", "float64"),
    ("num_tokens", 5),
    ("embedding_dim", DIM + 1),
    ("latents_sha256", "short"),
    ("created_at_utc", "yesterday"),
    ("torch_version", ""),
    ("preprocessing", "bilinear"),
]


def test_invalid_metadata_values_are_rejected() -> None:
    for field, value in INVALID_VALUES:
        data = _metadata(_latents()).to_dict()
        data[field] = value

        with pytest.raises(CacheMetadataError):
            CacheMetadata.from_dict(data)


def test_preprocessing_must_match_input_shape() -> None:
    data = _metadata(_latents()).to_dict()
    data["preprocessing"]["image_size"] = 64

    with pytest.raises(CacheMetadataError, match="disagrees"):
        CacheMetadata.from_dict(data)


def test_build_metadata_rejects_non_3d_latents() -> None:
    with pytest.raises(CacheMetadataError, match="N×tokens×dim"):
        _metadata(torch.zeros(TOKENS, DIM))


def test_from_json_rejects_garbage() -> None:
    with pytest.raises(CacheMetadataError, match="not valid JSON"):
        CacheMetadata.from_json("{nope")


# --- round trip ---------------------------------------------------------------


def test_round_trip_preserves_values_and_dtype(tmp_path: Path) -> None:
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        latents = _latents(dtype)
        path = latent_cache_path(tmp_path / str(dtype), "bottle", "train")
        save_latent_cache(path, latents, _metadata(latents))

        loaded = load_latent_cache(path, _expected())

        assert loaded.latents.dtype == dtype
        assert torch.equal(loaded.latents, latents)
        assert CacheMetadata.from_json(loaded.metadata.to_json()) == loaded.metadata
        assert loaded.metadata.sample_ids == IDS


def test_saves_inference_mode_encoder_output(tmp_path: Path) -> None:
    with torch.inference_mode():
        latents = torch.randn(len(IDS), TOKENS, DIM)
    path = latent_cache_path(tmp_path, "bottle", "train")

    save_latent_cache(path, latents, _metadata(latents))

    assert torch.equal(load_latent_cache(path).latents, latents)


def test_accepts_encoder_contract_shape(tmp_path: Path) -> None:
    """The documented N×196×1408 encoder output (random stand-in) is cacheable."""
    latents = torch.randn(2, 196, 1408)
    spec = PreprocessingSpec.from_config(PreprocessConfig())
    metadata = build_metadata(
        latents,
        category="bottle",
        split="val",
        sample_ids=["a", "b"],
        encoder_name="ijepa_vit_giant_16",
        checkpoint_sha256=CHECKPOINT_SHA,
        patch_size=16,
        preprocessing=spec,
    )
    path = latent_cache_path(tmp_path, "bottle", "val")

    save_latent_cache(path, latents, metadata)

    loaded = load_latent_cache(path)
    assert (loaded.metadata.num_tokens, loaded.metadata.embedding_dim) == (196, 1408)
    assert torch.equal(loaded.latents, latents)


def test_cache_path_layout_and_unsafe_names(tmp_path: Path) -> None:
    assert latent_cache_path(tmp_path, "bottle", "train") == (
        tmp_path / "bottle" / "train.pt"
    )
    with pytest.raises(ValueError, match="plain names"):
        latent_cache_path(tmp_path, "../x", "train")


# --- save-side consistency ----------------------------------------------------


def test_save_rejects_shape_mismatch(tmp_path: Path) -> None:
    latents = _latents()
    metadata = _metadata(latents)

    with pytest.raises(CacheMetadataError, match="shape"):
        save_latent_cache(tmp_path / "x.pt", latents[:2], metadata)


def test_save_rejects_dtype_mismatch(tmp_path: Path) -> None:
    latents = _latents()
    metadata = _metadata(latents)

    with pytest.raises(CacheMetadataError, match="dtype"):
        save_latent_cache(tmp_path / "x.pt", latents.half(), metadata)


def test_save_rejects_changed_values(tmp_path: Path) -> None:
    latents = _latents()
    metadata = _metadata(latents)

    with pytest.raises(CacheMetadataError, match="latents_sha256"):
        save_latent_cache(tmp_path / "x.pt", latents + 1.0, metadata)


def test_save_rejects_non_finite_values(tmp_path: Path) -> None:
    latents = _latents()
    latents[0, 0, 0] = float("nan")

    with pytest.raises(CacheMetadataError, match="NaN"):
        save_latent_cache(tmp_path / "x.pt", latents, _metadata(latents))


# --- missing / corrupt --------------------------------------------------------


def test_missing_file_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(CacheNotFoundError):
        load_latent_cache(tmp_path / "bottle" / "train.pt")


def test_truncated_file_is_corrupt(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])

    with pytest.raises(CacheCorruptedError):
        load_latent_cache(path)


def test_garbage_file_is_corrupt(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    path.write_bytes(b"not a torch file")

    with pytest.raises(CacheCorruptedError):
        load_latent_cache(path)


def test_wrong_payload_keys_are_corrupt(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    torch.save({"latents": _latents()}, path)

    with pytest.raises(CacheCorruptedError, match="exactly"):
        load_latent_cache(path)


def test_non_tensor_payload_is_corrupt(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    torch.save({"latents": [1, 2], "metadata_json": "{}"}, path)

    with pytest.raises(CacheCorruptedError, match="payload types"):
        load_latent_cache(path)


def test_tampered_latents_fail_checksum(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload["latents"][0, 0, 0] += 1.0
    torch.save(payload, path)

    with pytest.raises(CacheCorruptedError, match="checksum"):
        load_latent_cache(path)


def test_disk_metadata_is_revalidated(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    _rewrite_metadata(path, lambda data: data.update(latent_dtype="float64"))

    with pytest.raises(CacheCorruptedError, match="invalid metadata"):
        load_latent_cache(path)


def test_disk_metadata_shape_disagreeing_with_tensor_is_corrupt(
    tmp_path: Path,
) -> None:
    path = _saved(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload["latents"] = payload["latents"][:2]
    torch.save(payload, path)

    with pytest.raises(CacheCorruptedError, match="shape"):
        load_latent_cache(path)


def test_pickled_objects_are_not_deserialized(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    torch.save({"latents": _latents(), "metadata_json": Path("x")}, path)

    with pytest.raises(CacheCorruptedError):
        load_latent_cache(path)


# --- invalidation -------------------------------------------------------------


INCOMPATIBLE: list[tuple[dict[str, Any], str]] = [
    ({"checkpoint_sha256": "b" * 64}, "checkpoint_sha256"),
    ({"preprocessing": _spec(image_size=IMAGE_SIZE * 2)}, "preprocessing"),
    ({"preprocessing": _spec(mean=(0.0, 0.0, 0.0))}, "preprocessing"),
    ({"preprocessing": _spec(resize="bicubic")}, "preprocessing"),
    ({"encoder_name": "ijepa_vit_huge_14"}, "encoder_name"),
    ({"patch_size": 14}, "patch_size"),
    ({"embedding_dim": DIM * 2}, "embedding_dim"),
    ({"category": "capsule"}, "category"),
    ({"split": "val"}, "split"),
    ({"sample_ids": ("id-0", "id-1")}, "sample_ids"),
    ({"sample_ids": tuple(reversed(IDS))}, "sample_ids"),
]


def test_incompatible_configuration_is_rejected(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    for override, fragment in INCOMPATIBLE:
        with pytest.raises(CacheIncompatibleError, match=fragment):
            load_latent_cache(path, _expected(**override))
        assert try_load_latent_cache(path, _expected(**override)) is None


def test_matching_sample_ids_are_accepted(tmp_path: Path) -> None:
    path = _saved(tmp_path)

    assert load_latent_cache(path, _expected(sample_ids=IDS)).metadata.sample_ids == IDS


def test_old_schema_version_invalidates_cache(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    _rewrite_metadata(path, lambda data: data.update(schema_version=0))

    with pytest.raises(CacheIncompatibleError, match="schema_version"):
        load_latent_cache(path, _expected())
    assert try_load_latent_cache(path, _expected()) is None


def test_expectation_for_other_schema_version_invalidates_cache(
    tmp_path: Path,
) -> None:
    path = _saved(tmp_path)

    with pytest.raises(CacheIncompatibleError, match="schema_version"):
        load_latent_cache(path, _expected(schema_version=SCHEMA_VERSION + 1))


def test_try_load_returns_entry_when_compatible_and_none_when_missing(
    tmp_path: Path,
) -> None:
    path = _saved(tmp_path)

    assert try_load_latent_cache(path, _expected()) is not None
    assert try_load_latent_cache(tmp_path / "nope.pt", _expected()) is None


def test_try_load_does_not_hide_corruption(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    path.write_bytes(b"junk")

    with pytest.raises(CacheCorruptedError):
        try_load_latent_cache(path, _expected())


# --- atomic writes ------------------------------------------------------------


def test_failed_write_leaves_no_file_and_keeps_previous_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _saved(tmp_path)
    original = path.read_bytes()
    new_latents = _latents() * 2
    new_metadata = _metadata(new_latents)

    def explode(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", explode)
    with pytest.raises(OSError, match="disk full"):
        save_latent_cache(path, new_latents, new_metadata)
    monkeypatch.undo()

    assert path.read_bytes() == original
    assert [p.name for p in path.parent.iterdir()] == [path.name]
    assert load_latent_cache(path, _expected()) is not None


def test_failed_first_write_leaves_nothing_at_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    latents = _latents()
    path = latent_cache_path(tmp_path, "bottle", "train")

    def fail_replace(*_: object, **__: object) -> None:
        raise OSError("interrupted")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        save_latent_cache(path, latents, _metadata(latents))
    monkeypatch.undo()

    assert not path.exists()
    assert list(path.parent.iterdir()) == []
    with pytest.raises(CacheNotFoundError):
        load_latent_cache(path)


def test_overwrite_replaces_entry(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    new_latents = _latents() * 2

    save_latent_cache(path, new_latents, _metadata(new_latents))

    assert torch.equal(load_latent_cache(path).latents, new_latents)


# --- misc ---------------------------------------------------------------------


def test_compute_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    file = tmp_path / "ckpt.bin"
    file.write_bytes(b"checkpoint-bytes" * 100_000)

    assert compute_file_sha256(file) == hashlib.sha256(file.read_bytes()).hexdigest()


def test_everything_runs_on_cpu_without_checkpoint(tmp_path: Path) -> None:
    path = _saved(tmp_path)

    assert load_latent_cache(path).latents.device.type == "cpu"


def test_lookup_by_sample_id_returns_matching_rows(tmp_path: Path) -> None:
    latents = _latents()
    path = _saved(tmp_path)
    entry = load_latent_cache(path, _expected(sample_ids=IDS))

    assert "id-1" in entry and "other" not in entry
    assert entry.index_of("id-2") == 2
    assert torch.equal(entry.get("id-1"), latents[1])
    assert torch.equal(entry.select(["id-2", "id-0"]), latents[[2, 0]])
    with pytest.raises(KeyError, match="other"):
        entry.get("other")


def test_cache_exists_validates_and_distinguishes_states(tmp_path: Path) -> None:
    path = _saved(tmp_path)

    assert cache_exists(path, _expected())
    assert not cache_exists(path, _expected(checkpoint_sha256="b" * 64))
    assert not cache_exists(tmp_path / "missing.pt", _expected())
    path.write_bytes(path.read_bytes()[:50])
    with pytest.raises(CacheCorruptedError):
        cache_exists(path, _expected())


def test_preprocessing_version_invalidates_cache(tmp_path: Path) -> None:
    path = _saved(tmp_path)

    with pytest.raises(CacheIncompatibleError, match="preprocessing"):
        load_latent_cache(path, _expected(preprocessing=_spec(version=2)))


# --- review fixes: spec validation, finite load, list/tuple normalization -------


def _config_with(mean: Any, std: Any) -> PreprocessConfig:
    return PreprocessConfig(image_size=IMAGE_SIZE, mean=mean, std=std)


INVALID_STDS = [
    (0.0, 0.2, 0.2),
    (-0.1, 0.2, 0.2),
    (float("nan"), 0.2, 0.2),
    (float("inf"), 0.2, 0.2),
]
MALFORMED_SPEC_FIELDS: list[tuple[str, Any]] = [
    ("mean", (0.5, 0.5)),
    ("mean", "abc"),
    ("mean", None),
    ("mean", (True, 0.5, 0.5)),
    ("mean", ("a", "b", "c")),
    ("mean", (float("nan"), 0.5, 0.5)),
    ("mean", (float("-inf"), 0.5, 0.5)),
    ("std", (0.2, 0.2)),
    ("std", ("a", "b", "c")),
    ("image_size", 0),
    ("image_size", True),
    ("image_size", 32.0),
    ("version", 0),
    ("resize", 3),
]


def test_direct_construction_rejects_invalid_std() -> None:
    for std in INVALID_STDS:
        with pytest.raises(CacheMetadataError):
            PreprocessingSpec(image_size=IMAGE_SIZE, mean=(0.5, 0.5, 0.5), std=std)
        with pytest.raises(CacheMetadataError):
            replace(_spec(), std=std)
        data = _spec().to_dict()
        data["std"] = list(std)
        with pytest.raises(CacheMetadataError):
            PreprocessingSpec.from_dict(data)


def test_direct_construction_rejects_malformed_fields() -> None:
    for field, value in MALFORMED_SPEC_FIELDS:
        with pytest.raises(CacheMetadataError):
            replace(_spec(), **{field: value})
        data = _spec().to_dict()
        data[field] = value
        with pytest.raises(CacheMetadataError):
            PreprocessingSpec.from_dict(data)


def test_spec_normalizes_mean_and_std_to_float_tuples() -> None:
    mean: Any = [1, 0.5, 0.25]
    std: Any = [1, 2, 3]

    spec = PreprocessingSpec(image_size=IMAGE_SIZE, mean=mean, std=std)

    assert spec.mean == (1.0, 0.5, 0.25) and isinstance(spec.mean, tuple)
    assert spec.std == (1.0, 2.0, 3.0) and isinstance(spec.std, tuple)
    assert all(isinstance(value, float) for value in spec.mean + spec.std)


def test_list_and_tuple_configs_give_equal_specs_and_survive_a_round_trip() -> None:
    as_lists = PreprocessingSpec.from_config(_config_with([0.5] * 3, [0.25] * 3))
    as_tuples = PreprocessingSpec.from_config(_config_with((0.5,) * 3, (0.25,) * 3))

    assert as_lists == as_tuples
    assert PreprocessingSpec.from_dict(as_lists.to_dict()) == as_tuples


def test_cache_saved_with_list_config_matches_tuple_expectations(
    tmp_path: Path,
) -> None:
    lists = PreprocessingSpec.from_config(_config_with([0.5] * 3, [0.25] * 3))
    tuples = PreprocessingSpec.from_config(_config_with((0.5,) * 3, (0.25,) * 3))
    latents = _latents()
    path = latent_cache_path(tmp_path, "bottle", "train")
    save_latent_cache(path, latents, _metadata(latents, preprocessing=lists))

    assert load_latent_cache(path, _expected(preprocessing=lists)).latents.shape
    assert load_latent_cache(path, _expected(preprocessing=tuples)).latents.shape
    different = replace(tuples, mean=(0.5, 0.5, 0.4))
    with pytest.raises(CacheIncompatibleError, match="preprocessing"):
        load_latent_cache(path, _expected(preprocessing=different))


def test_save_and_load_reject_every_non_finite_value(tmp_path: Path) -> None:
    for index, bad in enumerate((float("nan"), float("inf"), float("-inf"))):
        root = tmp_path / str(index)
        latents = _latents()
        latents[1, 0, 2] = bad
        metadata = _metadata(latents)  # build_metadata does not look at values

        with pytest.raises(CacheMetadataError, match="NaN or infinite"):
            save_latent_cache(root / "x.pt", latents, metadata)
        assert not (root / "x.pt").exists()

        # A file whose checksum matches its (non-finite) data, e.g. from another tool.
        path = latent_cache_path(root, "bottle", "train")
        path.parent.mkdir(parents=True)
        torch.save({"latents": latents, "metadata_json": metadata.to_json()}, path)
        with pytest.raises(CacheCorruptedError, match="NaN or infinite"):
            load_latent_cache(path)
        with pytest.raises(CacheCorruptedError):
            try_load_latent_cache(path, _expected())


def test_expectations_check_num_tokens_only_when_given() -> None:
    latents = _latents()
    metadata = _metadata(latents)

    assert _expected().mismatches(metadata) == []
    assert _expected(num_tokens=TOKENS).mismatches(metadata) == []
    problems = _expected(num_tokens=TOKENS + 1).mismatches(metadata)
    assert len(problems) == 1 and "num_tokens" in problems[0]
