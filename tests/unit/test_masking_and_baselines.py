from __future__ import annotations

import torch

from ijepa_ad.baselines import (
    DenseMaskedIJEPA,
    FixedMaskIJEPA,
    MAEReconstructionBaseline,
    PatchCoreBaseline,
)
from ijepa_ad.masking import (
    AdaptiveMaskingEngine,
    FineMasker,
    PatchGrid,
    fuse_scores,
    latent_anomaly_scores,
    patch_scores_to_heatmap,
)


def test_patch_grid_maps_image_coordinates_to_14_by_14_tokens() -> None:
    grid = PatchGrid()
    assert grid.size == 14
    assert grid.pixel_to_patch(0, 0) == (0, 0)
    assert grid.pixel_to_patch(223, 223) == (13, 13)
    assert grid.patch_to_index(13, 13) == 195
    assert grid.index_to_patch(195) == (13, 13)


def test_normal_images_do_not_generate_fine_masks() -> None:
    selection = FineMasker().select(torch.zeros(14, 14), threshold=0.5)
    assert selection.count == 0


def test_latent_features_produce_one_score_per_patch() -> None:
    actual = torch.zeros(1, 196, 4)
    predicted = torch.ones(1, 196, 4)
    scores = latent_anomaly_scores(actual, predicted)
    assert scores.shape == (1, 196)
    assert torch.allclose(scores, torch.ones_like(scores))


def test_suspicious_region_selects_only_correct_location() -> None:
    scores = torch.zeros(14, 14)
    scores[3, 5] = 1.0
    selection = FineMasker().select(scores, threshold=0.5)
    assert selection.count == 1
    assert selection.windows[0].token_indices == (3 * 14 + 5,)


def test_fixed_and_dense_modes_are_independent() -> None:
    scores = torch.zeros(1, 196)
    assert (
        AdaptiveMaskingEngine(threshold=0.5)
        .evaluate(scores, mode="fixed")
        .evaluated_regions
        == 16
    )
    assert (
        AdaptiveMaskingEngine(threshold=0.5)
        .evaluate(scores, mode="dense")
        .evaluated_regions
        == 196
    )


def test_fusion_matches_formula() -> None:
    assert torch.equal(
        fuse_scores(torch.tensor([2.0]), torch.tensor([3.0]), 0.25),
        torch.tensor([2.75]),
    )


def test_heatmaps_have_expected_shape_and_range() -> None:
    heatmap = patch_scores_to_heatmap(torch.arange(196, dtype=torch.float32))
    assert heatmap.shape == (1, 1, 224, 224)
    assert float(heatmap.min()) == 0.0
    assert float(heatmap.max()) == 1.0


def test_each_baseline_runs_on_synthetic_input() -> None:
    patch_scores = torch.rand(2, 196)
    assert FixedMaskIJEPA().score(patch_scores).shape == (2,)
    assert DenseMaskedIJEPA().score(patch_scores).shape == (2,)
    features = torch.rand(3, 4, 8)
    assert PatchCoreBaseline().fit(features).score(features).shape == (3,)
    images = torch.rand(2, 3, 8, 8)
    assert MAEReconstructionBaseline().score(images, images * 0.9).shape == (2,)


def test_adaptive_evaluates_fewer_regions_than_dense() -> None:
    scores = torch.zeros(1, 196)
    scores[0, 5 * 14 + 6] = 1.0
    engine = AdaptiveMaskingEngine(threshold=0.5)
    adaptive = engine.evaluate(scores, mode="adaptive")
    dense = engine.evaluate(scores, mode="dense")
    assert adaptive.evaluated_regions < dense.evaluated_regions
