"""Run summary and CLI for cache generation: synthetic data, mock encoder, CPU only."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image
from torch import Tensor

from ijepa_ad.caching.builder import parts_dir
from ijepa_ad.caching.cli import (
    MOCK_CACHE_SUBDIR,
    CacheRunConfig,
    CacheRunConfigError,
    MockLatentEncoder,
    main,
    run_cache_generation,
)
from ijepa_ad.caching.latent_cache import latent_cache_path, load_latent_cache
from ijepa_ad.data.mvtec_dataset import MVTecADDataset, validate_category_splits
from ijepa_ad.data.splits import split_nominal_train_val
from ijepa_ad.data.validators import DatasetValidationError

DIM = 8


def _make_category(root: Path, category: str = "bottle", n_train: int = 10) -> None:
    for index in range(n_train):
        path = root / f"{category}/train/good/{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 40), (index * 20, 40, 90 + index)).save(path)
    for label, shade in (("good", 200), ("broken_large", 250)):
        path = root / f"{category}/test/{label}/000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 40), (shade, shade, 7)).save(path)
    mask = root / f"{category}/ground_truth/broken_large/000_mask.png"
    mask.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (40, 40), 255).save(mask)


def _config(tmp_path: Path, **overrides: object) -> CacheRunConfig:
    config = CacheRunConfig(
        dataset_root=tmp_path / "mvtec",
        cache_root=tmp_path / "cache",
        categories=("bottle",),
        batch_size=4,
        image_size=32,
        mock_encoder=True,
        mock_embedding_dim=DIM,
        val_fraction=0.2,
    )
    return replace(config, **overrides)  # type: ignore[arg-type]


def _mock_root(config: CacheRunConfig) -> Path:
    return config.cache_root / MOCK_CACHE_SUBDIR


def test_nominal_run_summary_and_nominal_only_caches(tmp_path: Path) -> None:
    _make_category(tmp_path / "mvtec")
    config = _config(tmp_path)

    summary = run_cache_generation(config)

    data = summary.to_dict()
    assert summary.ok and data["ok"]
    assert data["encoder_kind"] == "mock"
    assert data["real_checkpoint_inference"] is False
    assert data["device"] == "cpu"
    assert data["totals"] == {
        "samples_discovered": 10,
        "skipped_compatible": 0,
        "encoded_new": 10,
        "recomputed": 0,
        "failed": 0,
    }
    train, test = validate_category_splits(config.dataset_root, "bottle")
    split = split_nominal_train_val(train.samples, val_fraction=0.2, seed=0)
    root = _mock_root(config)
    for name, records in (("train", split.train), ("val", split.val)):
        entry = load_latent_cache(latent_cache_path(root, "bottle", name))
        assert entry.metadata.sample_ids == tuple(r.image_id for r in records)
        assert entry.metadata.encoder_name == "mock_encoder"
    cached = {
        i
        for name in ("train", "val")
        for i in load_latent_cache(
            latent_cache_path(root, "bottle", name)
        ).metadata.sample_ids
    }
    assert not cached & {s.image_id for s in test.samples}
    assert not latent_cache_path(root, "bottle", "test").exists()
    assert not config.cache_root.joinpath("bottle").exists()  # real cache untouched


def test_second_run_skips_everything(tmp_path: Path) -> None:
    _make_category(tmp_path / "mvtec")
    config = _config(tmp_path)
    run_cache_generation(config)

    totals = run_cache_generation(config).to_dict()["totals"]

    assert totals["encoded_new"] == 0 and totals["skipped_compatible"] == 10


def test_resume_after_interrupted_run_recomputes_only_the_missing(
    tmp_path: Path,
) -> None:
    _make_category(tmp_path / "mvtec")
    config = _config(tmp_path)

    class Crashing(MockLatentEncoder):
        calls = 0

        def encode(self, image_batch: Tensor) -> Tensor:
            Crashing.calls += 1
            if Crashing.calls == 2:
                raise KeyboardInterrupt("simulated crash")
            return super().encode(image_batch)

    with pytest.raises(KeyboardInterrupt, match="simulated"):
        run_cache_generation(config, Crashing(DIM, config.seed))
    root = _mock_root(config)
    assert not latent_cache_path(root, "bottle", "train").exists()
    assert len(list(parts_dir(root, "bottle", "train").glob("*.pt"))) == 4

    totals = run_cache_generation(config).to_dict()["totals"]

    assert totals["skipped_compatible"] == 4 and totals["encoded_new"] == 6
    assert latent_cache_path(root, "bottle", "train").is_file()


def test_corrupted_shard_is_reported_as_recomputed(tmp_path: Path) -> None:
    _make_category(tmp_path / "mvtec")
    config = _config(tmp_path)

    class Crashing(MockLatentEncoder):
        calls = 0

        def encode(self, image_batch: Tensor) -> Tensor:
            Crashing.calls += 1
            if Crashing.calls == 2:
                raise KeyboardInterrupt("simulated crash")
            return super().encode(image_batch)

    with pytest.raises(KeyboardInterrupt):
        run_cache_generation(config, Crashing(DIM, config.seed))
    shard = sorted(parts_dir(_mock_root(config), "bottle", "train").glob("*.pt"))[0]
    shard.write_bytes(b"garbage")

    totals = run_cache_generation(config).to_dict()["totals"]

    assert totals["recomputed"] == 1
    assert totals["encoded_new"] == 6 and totals["skipped_compatible"] == 3


def test_failed_stale_replacement_never_yields_negative_encoded_new(
    tmp_path: Path,
) -> None:
    _make_category(tmp_path / "mvtec")
    config = _config(tmp_path)

    class Crashing(MockLatentEncoder):
        calls = 0

        def encode(self, image_batch: Tensor) -> Tensor:
            Crashing.calls += 1
            if Crashing.calls == 2:
                raise KeyboardInterrupt("simulated crash")
            return super().encode(image_batch)

    with pytest.raises(KeyboardInterrupt):
        run_cache_generation(config, Crashing(DIM, config.seed))
    shards = sorted(parts_dir(_mock_root(config), "bottle", "train").glob("*.pt"))
    for shard in shards[:2]:
        shard.write_bytes(b"garbage")

    class AlwaysFailing(MockLatentEncoder):
        def encode(self, image_batch: Tensor) -> Tensor:
            raise RuntimeError("out of memory")

    summary = run_cache_generation(config, AlwaysFailing(DIM, config.seed))

    data = summary.to_dict()
    totals = data["totals"]
    assert not summary.ok
    assert totals["recomputed"] == 0  # no replacement succeeded
    assert totals["encoded_new"] == 0
    assert all(r.encoded >= r.recomputed >= 0 for r in summary.results)
    assert totals["failed"] == 8  # every train and val sample failed


def test_unreadable_image_is_reported_and_other_category_still_built(
    tmp_path: Path,
) -> None:
    _make_category(tmp_path / "mvtec", "bottle")
    _make_category(tmp_path / "mvtec", "capsule")
    config = _config(tmp_path, categories=("bottle", "capsule"))
    bad = MVTecADDataset(config.dataset_root, "bottle", "train").samples[3]
    bad.image_path.write_bytes(b"not an image")

    summary = run_cache_generation(config)

    assert not summary.ok
    data = summary.to_dict()
    reasons = [r for ids in data["failed_samples"].values() for r in ids.values()]
    assert len(reasons) == 1 and bad.image_path.name in reasons[0]
    capsule = [r for r in summary.results if r.category == "capsule"]
    assert capsule and all(r.status == "complete" for r in capsule)
    assert latent_cache_path(_mock_root(config), "capsule", "train").is_file()


def test_test_purpose_caches_only_the_official_test_split(tmp_path: Path) -> None:
    _make_category(tmp_path / "mvtec")
    config = _config(tmp_path, purpose="test")

    summary = run_cache_generation(config)

    root = _mock_root(config)
    assert [r.split for r in summary.results] == ["test"]
    assert summary.to_dict()["totals"]["samples_discovered"] == 2
    assert latent_cache_path(root, "bottle", "test").is_file()
    assert not latent_cache_path(root, "bottle", "train").exists()


def test_rebuild_ignores_existing_entries(tmp_path: Path) -> None:
    _make_category(tmp_path / "mvtec")
    config = _config(tmp_path)
    run_cache_generation(config)

    totals = run_cache_generation(replace(config, rebuild=True)).to_dict()["totals"]

    assert totals["encoded_new"] == 10 and totals["skipped_compatible"] == 0


def test_misconfiguration_fails_clearly_before_any_encoding(tmp_path: Path) -> None:
    _make_category(tmp_path / "mvtec")
    config = _config(tmp_path)

    for bad in (
        replace(config, dataset_root=tmp_path / "missing"),
        replace(config, batch_size=0),
        replace(config, categories=()),
        replace(config, purpose="dev"),
        replace(config, device="cuda"),  # mock encoder is CPU-only
        replace(config, device="not-a-device"),
        replace(config, mock_encoder=False, checkpoint=tmp_path / "nope.pth.tar"),
    ):
        with pytest.raises(CacheRunConfigError):
            run_cache_generation(bad)
    with pytest.raises(DatasetValidationError):
        run_cache_generation(replace(config, categories=("nonexistent",)))
    assert not (tmp_path / "cache").exists()


def test_cli_prints_json_summary_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_category(tmp_path / "mvtec")
    summary_file = tmp_path / "summary.json"

    code = main(
        [
            "--dataset-root", str(tmp_path / "mvtec"),
            "--cache-root", str(tmp_path / "cache"),
            "--category", "bottle",
            "--batch-size", "4",
            "--image-size", "32",
            "--mock-encoder",
            "--mock-embedding-dim", str(DIM),
            "--val-fraction", "0.2",
            "--summary-json", str(summary_file),
        ]
    )  # fmt: skip

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] and printed["encoder_kind"] == "mock"
    assert json.loads(summary_file.read_text(encoding="utf-8")) == printed


def test_cli_exit_codes_for_bad_config_and_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_category(tmp_path / "mvtec")
    base = [
        "--dataset-root", str(tmp_path / "mvtec"),
        "--cache-root", str(tmp_path / "cache"),
        "--image-size", "32",
        "--mock-encoder",
        "--mock-embedding-dim", str(DIM),
    ]  # fmt: skip

    assert main([*base, "--category", "nonexistent"]) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["--dataset-root", str(tmp_path / "none"), "--category", "x"]) == 2

    bad = MVTecADDataset(tmp_path / "mvtec", "bottle", "train").samples[0]
    bad.image_path.write_bytes(b"not an image")
    assert main([*base, "--category", "bottle", "--val-fraction", "0.2"]) == 1


def test_cli_requires_a_category(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2
    assert "--category" in capsys.readouterr().err


# --- review fixes: encoder failures, mock identity, val_fraction, recovery ---------


class _FailingOnce(MockLatentEncoder):
    """Raises on its first call only (e.g. one out-of-memory batch)."""

    def __init__(self, embedding_dim: int, seed: int) -> None:
        super().__init__(embedding_dim, seed)
        self.calls = 0

    def encode(self, image_batch: Tensor) -> Tensor:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("CUDA out of memory")
        return super().encode(image_batch)


def test_encoder_failure_is_summarized_and_other_work_continues(
    tmp_path: Path,
) -> None:
    _make_category(tmp_path / "mvtec", "bottle")
    _make_category(tmp_path / "mvtec", "capsule")
    config = _config(tmp_path, categories=("bottle", "capsule"))
    encoder = _FailingOnce(DIM, config.seed)

    summary = run_cache_generation(config, encoder)

    assert not summary.ok
    by_entry = {(r.category, r.split): r for r in summary.results}
    failed = by_entry[("bottle", "train")]
    assert failed.status == "failed" and len(failed.failed) == 4  # first batch only
    assert all("out of memory" in why for why in failed.failed.values())
    assert by_entry[("bottle", "val")].status == "complete"
    assert by_entry[("capsule", "train")].status == "complete"
    assert summary.to_dict()["totals"]["failed"] == 4
    root = _mock_root(config)
    assert not latent_cache_path(root, "bottle", "train").exists()
    assert latent_cache_path(root, "capsule", "train").is_file()

    again = run_cache_generation(config).to_dict()  # resume encodes only the 4 missing
    assert again["ok"] and again["totals"]["encoded_new"] == 4


def test_changing_the_mock_seed_never_reuses_other_projection(tmp_path: Path) -> None:
    _make_category(tmp_path / "mvtec")
    config = _config(tmp_path, purpose="test", seed=0)  # test ids ignore the seed
    path = latent_cache_path(_mock_root(config), "bottle", "test")
    run_cache_generation(config)
    seed0 = load_latent_cache(path)

    other = run_cache_generation(replace(config, seed=1))

    assert other.to_dict()["totals"]["encoded_new"] == 2  # nothing reused
    seed1 = load_latent_cache(path)
    assert seed1.metadata.checkpoint_sha256 != seed0.metadata.checkpoint_sha256
    assert seed1.metadata.encoder_name == seed0.metadata.encoder_name == "mock_encoder"
    assert not (seed1.latents == seed0.latents).all()

    same = run_cache_generation(replace(config, seed=1)).to_dict()["totals"]
    assert same["encoded_new"] == 0 and same["skipped_compatible"] == 2
    assert (load_latent_cache(path).latents == seed1.latents).all()


def test_invalid_val_fraction_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_category(tmp_path / "mvtec")
    base = [
        "--dataset-root", str(tmp_path / "mvtec"),
        "--cache-root", str(tmp_path / "cache"),
        "--category", "bottle",
        "--image-size", "32",
        "--mock-encoder",
        "--mock-embedding-dim", str(DIM),
    ]  # fmt: skip
    for fraction in (0.0, 1.0, -0.1, 1.5, float("nan"), float("inf")):
        config = _config(tmp_path, val_fraction=fraction)
        with pytest.raises(CacheRunConfigError, match="val_fraction"):
            run_cache_generation(config)
        assert main([*base, f"--val-fraction={fraction}"]) == 2
        assert "val_fraction" in capsys.readouterr().err
        assert not (tmp_path / "cache").exists()  # rejected before any encoding


def test_huge_integer_val_fraction_is_a_config_error(tmp_path: Path) -> None:
    _make_category(tmp_path / "mvtec")

    for fraction in (10**400, -(10**400)):
        with pytest.raises(CacheRunConfigError, match="val_fraction"):
            run_cache_generation(_config(tmp_path, val_fraction=fraction))
    assert not (tmp_path / "cache").exists()  # rejected before any encoding


def test_boolean_val_fraction_is_a_config_error(tmp_path: Path) -> None:
    _make_category(tmp_path / "mvtec")

    with pytest.raises(CacheRunConfigError, match="val_fraction"):
        run_cache_generation(_config(tmp_path, val_fraction=True))


def test_valid_val_fractions_are_accepted(tmp_path: Path) -> None:
    _make_category(tmp_path / "mvtec")

    for index, fraction in enumerate((0.1, 0.2, 0.5)):
        config = _config(
            tmp_path, val_fraction=fraction, cache_root=tmp_path / f"cache{index}"
        )
        assert run_cache_generation(config).ok


def test_corrupted_final_entry_is_reported_and_rebuild_recovers(
    tmp_path: Path,
) -> None:
    _make_category(tmp_path / "mvtec")
    config = _config(tmp_path)
    run_cache_generation(config)
    root = _mock_root(config)
    train_path = latent_cache_path(root, "bottle", "train")
    val_path = latent_cache_path(root, "bottle", "val")
    val_bytes = val_path.read_bytes()
    train_path.write_bytes(b"garbage")

    summary = run_cache_generation(config)  # documented: reported, never overwritten

    results = {r.split: r for r in summary.results}
    assert results["train"].status == "failed" and train_path.read_bytes() == b"garbage"
    assert results["val"].status == "complete" and results["val"].already_complete
    assert val_path.read_bytes() == val_bytes  # valid entry stays reusable

    recovered = run_cache_generation(replace(config, rebuild=True))

    assert recovered.ok
    assert load_latent_cache(train_path).latents.shape[0] == 8


def test_failed_rebuild_never_leaves_a_valid_looking_entry(tmp_path: Path) -> None:
    _make_category(tmp_path / "mvtec")
    config = _config(tmp_path)
    run_cache_generation(config)
    train_path = latent_cache_path(_mock_root(config), "bottle", "train")
    train_path.write_bytes(b"garbage")

    summary = run_cache_generation(
        replace(config, rebuild=True), _FailingOnce(DIM, config.seed)
    )

    failed = [r for r in summary.results if r.status == "failed"]
    assert [r.split for r in failed] == ["train"]
    assert not train_path.exists()  # absent, not a stale or partial entry
