"""Deterministic image preprocessing for the frozen I-JEPA ViT-g/16 encoder.

Normalization and value scaling follow Meta's official ``ijepa`` repository
(``src/transforms.py``: ``ToTensor()`` then ``Normalize`` with ImageNet
statistics). The official repository only ships a *training* transform
(``RandomResizedCrop``); for anomaly detection every split uses the same
deterministic full-image resize so no image region (and no defect) is cropped
away. See DECISIONS.md, 2026-10-10 "Deterministic I-JEPA preprocessing".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor

# Official I-JEPA normalization (facebookresearch/ijepa, src/transforms.py).
IJEPA_MEAN = (0.485, 0.456, 0.406)
IJEPA_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
# Bump when preprocess_image changes behavior for the same PreprocessConfig, so
# latent caches built with the old behavior are invalidated.
PREPROCESSING_VERSION = 1

ImageInput: TypeAlias = "str | Path | Image.Image"

# 8-bit modes PIL converts to RGB without changing value semantics.
_DIRECT_RGB_MODES = frozenset({"1", "L", "P", "RGB", "CMYK", "YCbCr"})
_ALPHA_MODES = frozenset({"LA", "PA", "RGBA", "RGBa", "La"})


class ImagePreprocessingError(ValueError):
    """Raised when an input cannot be turned into an encoder-ready tensor."""


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing parameters; defaults match ``configs/data/mvtec.yaml``."""

    image_size: int = IMAGE_SIZE
    mean: tuple[float, float, float] = IJEPA_MEAN
    std: tuple[float, float, float] = IJEPA_STD

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must each have 3 channel values")
        if any(value <= 0 for value in self.std):
            raise ValueError(f"std values must be positive, got {self.std}")


def load_image(path: str | Path) -> Image.Image:
    """Decode an image file fully into memory and return it as RGB."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image file does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            return to_rgb(image)
    except UnidentifiedImageError as error:
        raise ImagePreprocessingError(
            f"file is not a readable image: {path}"
        ) from error


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert an 8-bit PIL image to RGB.

    Grayscale is replicated across three channels. Images with transparency are
    alpha-composited onto black so hidden pixel values never leak into the
    tensor. High-bit-depth and float modes are rejected rather than silently
    clipped to 8 bits.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected a PIL image, got {type(image).__name__}")
    if image.width == 0 or image.height == 0:
        raise ImagePreprocessingError(f"image has zero size: {image.size}")

    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    if image.mode in _ALPHA_MODES:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if image.mode in _DIRECT_RGB_MODES:
        return image.convert("RGB")
    raise ImagePreprocessingError(
        f"unsupported image mode {image.mode!r}; expected an 8-bit image"
    )


def preprocess_image(
    image: ImageInput, config: PreprocessConfig | None = None
) -> Tensor:
    """Return a normalized ``float32`` tensor of shape ``3×S×S``.

    Steps: decode (for paths) → RGB → bilinear resize of the full image to
    ``S×S`` (no crop) → scale to ``[0, 1]`` → per-channel normalization.
    """
    config = config or PreprocessConfig()
    if isinstance(image, str | Path):
        rgb = load_image(image)
    elif isinstance(image, Image.Image):
        rgb = to_rgb(image)
    else:
        raise TypeError(
            f"image must be a file path or PIL image, got {type(image).__name__}"
        )

    size = (config.image_size, config.image_size)
    resized = rgb.resize(size, resample=Image.Resampling.BILINEAR)
    pixels = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
    tensor = pixels.view(config.image_size, config.image_size, 3).permute(2, 0, 1)
    tensor = tensor.to(torch.float32) / 255.0

    mean = torch.tensor(config.mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config.std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def preprocess_batch(
    images: Sequence[ImageInput], config: PreprocessConfig | None = None
) -> Tensor:
    """Preprocess images and stack them into an ``N×3×S×S`` batch."""
    if isinstance(images, str | Path | Image.Image):
        raise TypeError("images must be a sequence of images, not a single image")
    if len(images) == 0:
        raise ImagePreprocessingError("cannot preprocess an empty image batch")
    config = config or PreprocessConfig()
    return torch.stack([preprocess_image(image, config) for image in images])
