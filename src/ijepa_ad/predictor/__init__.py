"""Predictor-side public interfaces."""

from ijepa_ad.predictor.cache_dataset import (
    CachedLatentBatch,
    CachedLatentDataset,
    CachedLatentSample,
    SampleAnnotation,
    annotations_by_image_id,
    collate_cached_latents,
    load_cached_latent_dataset,
)
from ijepa_ad.predictor.losses import (
    cosine_prediction_loss,
    l2_prediction_loss,
    prediction_loss,
)
from ijepa_ad.predictor.training import (
    EpochMetrics,
    TrainingConfig,
    TrainingResult,
    build_random_target_mask,
    train_predictor,
)
from ijepa_ad.predictor.transformer_predictor import (
    LatentTransformerPredictor,
    PredictorConfig,
)

__all__ = [
    "CachedLatentBatch",
    "CachedLatentDataset",
    "CachedLatentSample",
    "EpochMetrics",
    "LatentTransformerPredictor",
    "PredictorConfig",
    "SampleAnnotation",
    "TrainingConfig",
    "TrainingResult",
    "annotations_by_image_id",
    "build_random_target_mask",
    "collate_cached_latents",
    "cosine_prediction_loss",
    "l2_prediction_loss",
    "load_cached_latent_dataset",
    "prediction_loss",
    "train_predictor",
]
