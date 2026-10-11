from pathlib import Path

import torch

from ijepa_ad.caching.cache_schema import CacheExpectations, PreprocessingSpec
from ijepa_ad.caching.latent_cache import (
    build_metadata,
    latent_cache_path,
    save_latent_cache,
)
from ijepa_ad.data.transforms import PreprocessConfig
from ijepa_ad.predictor.cache_dataset import load_cached_latent_dataset
from ijepa_ad.predictor.training import (
    TrainingConfig,
    build_random_target_mask,
    train_predictor,
)
from ijepa_ad.predictor.transformer_predictor import (
    LatentTransformerPredictor,
    PredictorConfig,
)

CATEGORY = "bottle"
ENCODER = "synthetic_encoder"
CHECKPOINT = "b" * 64
PATCH_SIZE = 16
TOKENS = 4
DIM = 4
PREPROCESS = PreprocessConfig(image_size=32)


def _expected(split: str) -> CacheExpectations:
    return CacheExpectations(
        category=CATEGORY,
        split=split,
        encoder_name=ENCODER,
        checkpoint_sha256=CHECKPOINT,
        patch_size=PATCH_SIZE,
        embedding_dim=DIM,
        preprocessing=PreprocessingSpec.from_config(PREPROCESS),
    )


def _write_cache(root: Path, split: str, count: int) -> None:
    sample_ids = tuple(f"{split}-{index}" for index in range(count))
    latents = _synthetic_latents(count)
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


def _synthetic_latents(count: int) -> torch.Tensor:
    rows = []
    for sample_index in range(count):
        base = float(sample_index + 1)
        tokens = [[base, base + 1.0, base - 1.0, base * 0.5] for _ in range(TOKENS)]
        rows.append(tokens)
    return torch.tensor(rows, dtype=torch.float32)


def test_random_target_mask_keeps_context_and_target_tokens() -> None:
    mask = build_random_target_mask(
        batch_size=5,
        num_tokens=8,
        mask_ratio=0.5,
        generator=torch.Generator().manual_seed(0),
    )

    assert mask.shape == (5, 8)
    assert mask.dtype == torch.bool
    assert torch.equal(mask.sum(dim=1), torch.full((5,), 4))


def test_predictor_overfits_tiny_synthetic_cache_and_saves_checkpoint(
    tmp_path: Path,
) -> None:
    _write_cache(tmp_path, "train", 4)
    _write_cache(tmp_path, "val", 4)
    train = load_cached_latent_dataset(tmp_path, _expected("train"))
    val = load_cached_latent_dataset(tmp_path, _expected("val"))
    model = LatentTransformerPredictor(
        PredictorConfig(
            embedding_dim=DIM,
            num_tokens=TOKENS,
            hidden_dim=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
        )
    )
    checkpoint = tmp_path / "predictor.pt"

    result = train_predictor(
        model,
        train,
        val,
        TrainingConfig(
            batch_size=4,
            max_epochs=120,
            learning_rate=0.02,
            weight_decay=0.0,
            mask_ratio=0.5,
            loss_type="l2",
            patience=0,
            min_delta=0.0,
            seed=0,
            device="cpu",
        ),
        checkpoint_path=checkpoint,
    )

    assert result.history[0].val_loss > result.best_val_loss
    assert result.best_val_loss < result.history[0].val_loss * 0.25
    assert result.best_val_loss < 1.0
    assert result.checkpoint_path == checkpoint
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["best_epoch"] == result.best_epoch
    assert payload["model_config"]["embedding_dim"] == DIM
