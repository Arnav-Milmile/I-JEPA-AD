import pytest
import torch

from ijepa_ad.predictor.transformer_predictor import (
    LatentTransformerPredictor,
    PredictorConfig,
)


def _config() -> PredictorConfig:
    return PredictorConfig(
        embedding_dim=6,
        num_tokens=4,
        hidden_dim=12,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
    )


def test_predictor_returns_full_sequence_predictions() -> None:
    model = LatentTransformerPredictor(_config())
    latents = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, False, False, True], [False, True, False, False]])

    predicted = model(latents, mask)

    assert predicted.shape == latents.shape


def test_predictor_rejects_bad_config_and_inputs() -> None:
    with pytest.raises(ValueError, match="divisible"):
        PredictorConfig(6, 4, 10, 1, 3, 2.0, 0.0)

    model = LatentTransformerPredictor(_config())
    with pytest.raises(ValueError, match="token/dim"):
        model(torch.randn(2, 5, 6), torch.ones(2, 5, dtype=torch.bool))
    with pytest.raises(ValueError, match="target_mask"):
        model(torch.randn(2, 4, 6), torch.ones(2, 5, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one"):
        model(torch.randn(2, 4, 6), torch.zeros(2, 4, dtype=torch.bool))
