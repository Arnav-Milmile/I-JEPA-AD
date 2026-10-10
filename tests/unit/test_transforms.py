"""Tests for deterministic I-JEPA image preprocessing."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from ijepa_ad.data.transforms import (
    IJEPA_MEAN,
    IJEPA_STD,
    ImagePreprocessingError,
    PreprocessConfig,
    preprocess_batch,
    preprocess_image,
)


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IJEPA_MEAN).view(3, 1, 1)
    std = torch.tensor(IJEPA_STD).view(3, 1, 1)
    return tensor * std + mean


def test_rgb_image_produces_normalized_float_tensor() -> None:
    image = Image.new("RGB", (300, 300), (10, 120, 250))

    tensor = preprocess_image(image)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_normalization_matches_official_ijepa_statistics() -> None:
    assert IJEPA_MEAN == (0.485, 0.456, 0.406)
    assert IJEPA_STD == (0.229, 0.224, 0.225)

    white = preprocess_image(Image.new("RGB", (64, 64), (255, 255, 255)))
    black = preprocess_image(Image.new("RGB", (64, 64), (0, 0, 0)))

    for channel in range(3):
        mean, std = IJEPA_MEAN[channel], IJEPA_STD[channel]
        assert torch.allclose(white[channel], torch.tensor((1.0 - mean) / std))
        assert torch.allclose(black[channel], torch.tensor(-mean / std))


def test_grayscale_is_replicated_to_three_equal_channels() -> None:
    image = Image.new("L", (128, 128), 77)

    pixels = _denormalize(preprocess_image(image))

    assert pixels.shape == (3, 224, 224)
    assert torch.allclose(pixels, torch.full_like(pixels, 77 / 255), atol=1e-5)


def test_rgba_transparency_is_composited_onto_black() -> None:
    image = Image.new("RGBA", (64, 64), (255, 0, 0, 0))
    image.paste((0, 255, 0, 255), (0, 0, 32, 64))

    pixels = _denormalize(preprocess_image(image))

    # Left half: opaque green survives. Right half: transparent red becomes black.
    assert torch.allclose(pixels[:, 112, 10], torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)
    assert torch.allclose(pixels[:, 112, 200], torch.zeros(3), atol=1e-5)


def test_palette_with_transparency_is_composited() -> None:
    image = Image.new("P", (32, 32), 0)
    image.putpalette([255, 255, 255] + [0, 0, 0] * 255)
    image.info["transparency"] = 0

    pixels = _denormalize(preprocess_image(image))

    assert torch.allclose(pixels, torch.zeros_like(pixels), atol=1e-5)


def test_full_image_is_resized_without_cropping_borders() -> None:
    image = Image.new("RGB", (1024, 1024), (0, 0, 0))
    image.paste((255, 255, 255), (0, 0, 64, 64))

    pixels = _denormalize(preprocess_image(image))

    # A center crop would discard this corner; a full resize keeps it.
    assert pixels[:, 0, 0].min() > 0.99
    assert pixels[:, -1, -1].max() < 0.01


def test_non_square_image_is_resized_to_square() -> None:
    assert preprocess_image(Image.new("RGB", (400, 250))).shape == (3, 224, 224)


def test_preprocessing_is_deterministic_for_paths_and_images(tmp_path: Path) -> None:
    image = Image.effect_noise((257, 257), 64).convert("RGB")
    path = tmp_path / "noise.png"
    image.save(path)

    first = preprocess_image(path)
    second = preprocess_image(path)
    in_memory = preprocess_image(image)

    assert torch.equal(first, second)
    assert torch.equal(first, in_memory)


def test_batch_stacks_to_encoder_shape(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    Image.new("L", (100, 100), 5).save(path)

    batch = preprocess_batch([path, Image.new("RGB", (50, 80)), str(path)])

    assert batch.shape == (3, 3, 224, 224)
    assert torch.equal(batch[0], batch[2])


def test_custom_image_size_is_respected() -> None:
    tensor = preprocess_image(Image.new("RGB", (64, 64)), PreprocessConfig(32))
    assert tensor.shape == (3, 32, 32)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        preprocess_image(tmp_path / "missing.png")


def test_non_image_file_raises_informative_error(tmp_path: Path) -> None:
    path = tmp_path / "not_an_image.png"
    path.write_bytes(b"definitely not a png")

    with pytest.raises(ImagePreprocessingError, match="not a readable image"):
        preprocess_image(path)


def test_unsupported_input_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match="file path or PIL image"):
        preprocess_image(torch.zeros(3, 224, 224))


def test_high_bit_depth_image_is_rejected() -> None:
    with pytest.raises(ImagePreprocessingError, match="unsupported image mode"):
        preprocess_image(Image.new("I;16", (32, 32)))


def test_batch_rejects_empty_and_single_inputs() -> None:
    with pytest.raises(ImagePreprocessingError, match="empty"):
        preprocess_batch([])
    with pytest.raises(TypeError, match="sequence"):
        preprocess_batch(Image.new("RGB", (8, 8)))


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="image_size"):
        PreprocessConfig(image_size=0)
    with pytest.raises(ValueError, match="std"):
        PreprocessConfig(std=(0.1, 0.0, 0.1))
