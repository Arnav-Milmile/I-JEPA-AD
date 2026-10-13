import pytest
import torch

from ijepa_ad.predictor.losses import (
    cosine_prediction_loss,
    l2_prediction_loss,
    prediction_loss,
)


def test_l2_prediction_loss_matches_hand_computed_masked_value() -> None:
    predicted = torch.tensor([[[1.0, 3.0], [2.0, 4.0], [5.0, 5.0]]])
    target = torch.tensor([[[0.0, 1.0], [2.0, 6.0], [9.0, 5.0]]])
    mask = torch.tensor([[True, False, True]])

    loss = l2_prediction_loss(predicted, target, mask)

    assert loss.item() == pytest.approx((1.0 + 4.0 + 16.0 + 0.0) / 4.0)


def test_cosine_prediction_loss_matches_hand_computed_value() -> None:
    predicted = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    loss = cosine_prediction_loss(predicted, target)

    expected = (0.0 + (1.0 - 2.0**-0.5)) / 2.0
    assert loss.item() == pytest.approx(expected)


def test_prediction_loss_dispatches_and_rejects_unknown_loss() -> None:
    predicted = torch.zeros(1, 2, 3)
    target = torch.ones(1, 2, 3)
    mask = torch.tensor([[True, False]])

    assert prediction_loss(predicted, target, mask, "l2").item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="loss_type"):
        prediction_loss(predicted, target, mask, "mae")


def test_losses_validate_shapes_and_nonempty_mask() -> None:
    predicted = torch.zeros(1, 2, 3)
    target = torch.zeros(1, 2, 3)

    with pytest.raises(ValueError, match="same shape"):
        l2_prediction_loss(predicted, torch.zeros(1, 2, 4))
    with pytest.raises(ValueError, match="B x tokens"):
        l2_prediction_loss(predicted, target, torch.zeros(1, 2, 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one"):
        l2_prediction_loss(predicted, target, torch.zeros(1, 2, dtype=torch.bool))
