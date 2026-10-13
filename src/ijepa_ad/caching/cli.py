"""Command-line entry point and run summary for latent-cache generation.

``python -m ijepa_ad.caching.cli --help``. This only wires existing pieces together
(dataset discovery, nominal split, ``build_latent_cache``); it adds no caching logic.
A mock encoder is available for CPU smoke tests and never touches a checkpoint. See
DECISIONS.md, 2026-10-10 "Cache-generation CLI and run summary".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ijepa_ad.caching.builder import (
    BuildReport,
    CacheBuildError,
    LatentEncoder,
    build_latent_cache,
    parts_dir,
)
from ijepa_ad.caching.latent_cache import (
    CacheError,
    compute_file_sha256,
    latent_cache_path,
)
from ijepa_ad.data.mvtec_dataset import validate_category_splits
from ijepa_ad.data.splits import split_nominal_train_val
from ijepa_ad.data.transforms import PreprocessConfig
from ijepa_ad.data.validators import DatasetValidationError, SampleMetadata
from ijepa_ad.utils.logging_setup import setup_logging

PURPOSES = ("nominal", "test")
PATCH_SIZE = 16
REAL_NUM_TOKENS = 196
REAL_EMBEDDING_DIM = 1408
MOCK_ENCODER_NAME = "mock_encoder"
MOCK_CACHE_SUBDIR = "_mock"
ENTRY_FAILURE_KEY = "<entry>"

logger = logging.getLogger("ijepa_ad.cache")


def mock_checkpoint_sha256(seed: int) -> str:
    """Stand-in checkpoint identity of the mock projection drawn with ``seed``.

    The mock has no checkpoint; this hash makes its identity explicit and
    deterministic, so a different seed (a different projection) never reuses a cache.
    The embedding dimension is part of the cache identity separately.
    """
    text = f"{MOCK_ENCODER_NAME}:random-projection:seed={seed}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CacheRunConfigError(Exception):
    """The requested run is misconfigured (missing files, unusable device, ...)."""


@dataclass(frozen=True)
class CacheRunConfig:
    """Everything one cache-generation run needs; defaults mirror ``configs/``."""

    dataset_root: Path
    cache_root: Path
    categories: tuple[str, ...]
    purpose: str = "nominal"  # "nominal": train+val caches; "test": evaluation cache
    batch_size: int = 8
    device: str = "cpu"
    val_fraction: float = 0.1
    seed: int = 0
    image_size: int = 224
    rebuild: bool = False
    mock_encoder: bool = False
    mock_embedding_dim: int = REAL_EMBEDDING_DIM
    checkpoint: Path = Path("checkpoints/IN22K-vit.g.16-600e.pth.tar")
    official_repo: Path = Path("external/ijepa")
    encoder_name: str = "ijepa_vit_g16_in22k_ep44"


@dataclass
class SplitResult:
    """Outcome for one (category, split) cache entry."""

    category: str
    split: str
    path: str
    total: int
    encoded: int  # encoder ran on these samples in this run (includes recomputed)
    skipped: int  # served from a compatible entry or shard
    recomputed: int  # of ``encoded``: shards that existed but were unusable
    already_complete: bool
    status: str  # "complete" or "failed"
    failed: dict[str, str] = field(default_factory=dict)


@dataclass
class RunSummary:
    """What a run did. ``real_checkpoint_inference`` is False for mock runs."""

    purpose: str
    encoder_kind: str  # "mock" or "real"
    real_checkpoint_inference: bool
    encoder_name: str
    checkpoint_sha256: str
    device: str
    batch_size: int
    cache_root: str
    results: list[SplitResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(result.status == "complete" for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        results = self.results
        data = asdict(self)
        data["totals"] = {
            "samples_discovered": sum(r.total for r in results),
            "skipped_compatible": sum(r.skipped for r in results),
            "encoded_new": sum(r.encoded - r.recomputed for r in results),
            "recomputed": sum(r.recomputed for r in results),
            "failed": sum(len(r.failed) for r in results),
        }
        data["failed_samples"] = {
            f"{r.category}/{r.split}": r.failed for r in results if r.failed
        }
        data["ok"] = self.ok
        return data


class MockLatentEncoder:
    """Deterministic content-dependent stand-in: ``N×3×S×S → N×(S/16)²×dim``."""

    def __init__(self, embedding_dim: int, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.projection = (
            torch.randn(3 * PATCH_SIZE * PATCH_SIZE, embedding_dim, generator=generator)
            / float(3 * PATCH_SIZE * PATCH_SIZE) ** 0.5
        )

    def encode(self, image_batch: Tensor) -> Tensor:
        count, _, height, width = image_batch.shape
        patches = image_batch.unfold(2, PATCH_SIZE, PATCH_SIZE)
        patches = patches.unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(0, 2, 3, 1, 4, 5).flatten(3)
        tokens = (height // PATCH_SIZE) * (width // PATCH_SIZE)
        with torch.no_grad():
            return (patches.flatten(0, 2) @ self.projection).view(count, tokens, -1)


class _DeviceEncoder:
    """Moves each input batch to the encoder's device; the builder expects CPU back."""

    def __init__(self, encoder: LatentEncoder, device: torch.device) -> None:
        self._encoder = encoder
        self._device = device

    def encode(self, image_batch: Tensor) -> Tensor:
        return self._encoder.encode(image_batch.to(self._device))


def run_cache_generation(
    config: CacheRunConfig, encoder: LatentEncoder | None = None
) -> RunSummary:
    """Build the caches described by ``config`` and return a structured summary.

    ``encoder`` overrides construction (tests); identity and mock/real labelling
    still follow ``config.mock_encoder``. Per-sample and per-entry failures are
    recorded in the summary and the run continues; configuration problems raise
    ``CacheRunConfigError`` or ``DatasetValidationError`` before any encoding.
    """
    _check_config(config)
    cache_root = config.cache_root
    if config.mock_encoder:
        cache_root = cache_root / MOCK_CACHE_SUBDIR  # never touch real caches
        name, sha, device_name = (
            MOCK_ENCODER_NAME,
            mock_checkpoint_sha256(config.seed),
            "cpu",
        )
        embedding_dim = config.mock_embedding_dim
        num_tokens = (config.image_size // PATCH_SIZE) ** 2
        if encoder is None:
            encoder = MockLatentEncoder(embedding_dim, config.seed)
    else:
        name, device_name = config.encoder_name, config.device
        embedding_dim, num_tokens = REAL_EMBEDDING_DIM, REAL_NUM_TOKENS
        if encoder is None:
            sha, encoder = _load_real_encoder(config)
        else:
            sha = compute_file_sha256(config.checkpoint)

    summary = RunSummary(
        purpose=config.purpose,
        encoder_kind="mock" if config.mock_encoder else "real",
        real_checkpoint_inference=not config.mock_encoder,
        encoder_name=name,
        checkpoint_sha256=sha,
        device=device_name,
        batch_size=config.batch_size,
        cache_root=str(cache_root),
    )
    preprocess = PreprocessConfig(image_size=config.image_size)
    for category in config.categories:
        # Discovery validates both official splits (incl. cross-split leakage).
        train, test = validate_category_splits(config.dataset_root, category)
        for split, records in _plan(config, train.samples, test.samples):
            if config.rebuild:
                _discard(cache_root, category, split)
            summary.results.append(
                _build_one(
                    records,
                    encoder,
                    cache_root=cache_root,
                    category=category,
                    split=split,
                    encoder_name=name,
                    checkpoint_sha256=sha,
                    patch_size=PATCH_SIZE,
                    embedding_dim=embedding_dim,
                    num_tokens=num_tokens,
                    config=preprocess,
                    batch_size=config.batch_size,
                )
            )
    return summary


def _plan(
    config: CacheRunConfig,
    train: Sequence[SampleMetadata],
    test: Sequence[SampleMetadata],
) -> list[tuple[str, Sequence[SampleMetadata]]]:
    """(split, records) pairs. Nominal runs never see the official test records."""
    if config.purpose == "test":
        return [("test", test)]
    nominal = split_nominal_train_val(
        train, val_fraction=config.val_fraction, seed=config.seed
    )
    return [("train", nominal.train), ("val", nominal.val)]


def _build_one(
    records: Sequence[SampleMetadata], encoder: LatentEncoder, **arguments: Any
) -> SplitResult:
    category, split = arguments["category"], arguments["split"]
    path = latent_cache_path(arguments["cache_root"], category, split)
    try:
        report = build_latent_cache(records, encoder, **arguments)
    except CacheBuildError as error:
        logger.error("cache %s/%s failed: %s", category, split, error)
        partial = error.report
        failed = error.failed or {ENTRY_FAILURE_KEY: str(error)}
        return _result(category, split, path, len(records), partial, failed)
    except CacheError as error:  # e.g. a corrupted final entry: never overwritten
        logger.error("cache %s/%s unusable: %s", category, split, error)
        return _result(
            category, split, path, len(records), None, {ENTRY_FAILURE_KEY: str(error)}
        )
    logger.info(
        "cache %s/%s ok: %d encoded (%d recomputed), %d skipped",
        category,
        split,
        report.encoded,
        report.recomputed,
        report.reused,
    )
    return _result(category, split, path, report.total, report, {})


def _result(
    category: str,
    split: str,
    path: Path,
    total: int,
    report: BuildReport | None,
    failed: dict[str, str],
) -> SplitResult:
    return SplitResult(
        category=category,
        split=split,
        path=str(path),
        total=total,
        encoded=report.encoded if report else 0,
        skipped=report.reused if report else 0,
        recomputed=report.recomputed if report else 0,
        already_complete=report.already_complete if report else False,
        status="failed" if failed else "complete",
        failed=failed,
    )


def _discard(cache_root: Path, category: str, split: str) -> None:
    """``--rebuild``: drop the final entry and its shards so nothing is reused."""
    latent_cache_path(cache_root, category, split).unlink(missing_ok=True)
    shutil.rmtree(parts_dir(cache_root, category, split), ignore_errors=True)


def _check_config(config: CacheRunConfig) -> None:
    if config.purpose not in PURPOSES:
        raise CacheRunConfigError(f"purpose must be one of {PURPOSES}")
    if not config.categories:
        raise CacheRunConfigError("at least one category is required")
    if config.batch_size < 1:
        raise CacheRunConfigError("batch_size must be at least 1")
    if config.purpose == "nominal" and (
        isinstance(config.val_fraction, bool)
        or not isinstance(config.val_fraction, int | float)
        or not 0.0 < config.val_fraction < 1.0
    ):  # same rule as split_nominal_train_val; also rejects NaN/±inf/huge ints
        raise CacheRunConfigError(
            f"val_fraction must be a number in (0, 1), got {config.val_fraction!r}"
        )
    if not config.dataset_root.is_dir():
        raise CacheRunConfigError(f"dataset root not found: {config.dataset_root}")
    try:
        device = torch.device(config.device)
    except (RuntimeError, ValueError) as error:
        raise CacheRunConfigError(f"invalid device {config.device!r}") from error
    if config.mock_encoder:
        if device.type != "cpu":
            raise CacheRunConfigError("the mock encoder runs on CPU only")
        if config.mock_embedding_dim < 1:
            raise CacheRunConfigError("mock_embedding_dim must be at least 1")
    else:
        if config.image_size != 224:
            raise CacheRunConfigError("the real encoder requires image_size 224")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise CacheRunConfigError("CUDA device requested but CUDA is unavailable")
        if not config.checkpoint.is_file():
            raise CacheRunConfigError(f"checkpoint not found: {config.checkpoint}")
        if not config.official_repo.is_dir():
            raise CacheRunConfigError(
                f"official I-JEPA checkout not found: {config.official_repo}"
            )


def _load_real_encoder(config: CacheRunConfig) -> tuple[str, LatentEncoder]:
    from ijepa_ad.encoder.ijepa_wrapper import IJEPAEncoder

    sha = compute_file_sha256(config.checkpoint)  # once per run (multi-GB read)
    logger.info("checkpoint sha256 %s", sha)
    device = torch.device(config.device)
    encoder = IJEPAEncoder(
        config.checkpoint, official_repo_path=config.official_repo, device=device
    )
    return sha, _DeviceEncoder(encoder, device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ijepa_ad.caching.cli",
        description=(
            "Generate resumable I-JEPA latent caches. Reruns skip compatible entries "
            "and shards and recompute missing, corrupted, or incompatible shards; a "
            "corrupted final entry is reported and left untouched (use --rebuild to "
            "replace it). 'nominal' caches "
            "a train/val split of train/good only; 'test' caches the official test "
            "split for evaluation and is never mixed into training caches."
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/latents"))
    parser.add_argument(
        "--category", dest="categories", nargs="+", required=True, metavar="NAME"
    )
    parser.add_argument("--purpose", choices=PURPOSES, default="nominal")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="e.g. cpu, cuda, cuda:0")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--image-size", type=int, default=224, help="must be 224 for the real encoder"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="delete the selected entries and their shards first (no resume)",
    )
    parser.add_argument(
        "--mock-encoder",
        action="store_true",
        help="CPU smoke test with a random-projection stand-in; no checkpoint is "
        f"read and caches go to <cache-root>/{MOCK_CACHE_SUBDIR}",
    )
    parser.add_argument("--mock-embedding-dim", type=int, default=REAL_EMBEDDING_DIM)
    defaults = CacheRunConfig(Path("."), Path("."), ())
    parser.add_argument("--checkpoint", type=Path, default=defaults.checkpoint)
    parser.add_argument("--official-repo", type=Path, default=defaults.official_repo)
    parser.add_argument("--encoder-name", default=defaults.encoder_name)
    parser.add_argument(
        "--summary-json", type=Path, help="also write the run summary to this file"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Exit codes: 0 all entries complete, 1 some failed, 2 bad configuration."""
    args = build_parser().parse_args(argv)
    setup_logging()
    config = CacheRunConfig(
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        categories=tuple(args.categories),
        purpose=args.purpose,
        batch_size=args.batch_size,
        device=args.device,
        val_fraction=args.val_fraction,
        seed=args.seed,
        image_size=args.image_size,
        rebuild=args.rebuild,
        mock_encoder=args.mock_encoder,
        mock_embedding_dim=args.mock_embedding_dim,
        checkpoint=args.checkpoint,
        official_repo=args.official_repo,
        encoder_name=args.encoder_name,
    )
    try:
        summary = run_cache_generation(config)
    except (CacheRunConfigError, DatasetValidationError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text = json.dumps(summary.to_dict(), indent=2, sort_keys=True)
    print(text)
    if args.summary_json is not None:
        args.summary_json.write_text(text + "\n", encoding="utf-8")
    return 0 if summary.ok else 1


if __name__ == "__main__":
    sys.exit(main())
