"""Visualization component for anomaly heatmaps, overlays, and detected regions."""
from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.cm as cm


def create_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap_name: str = "jet",
) -> Image.Image:
    """Blend a continuous 2D anomaly heatmap with the original image.

    Args:
        image: Original RGB input PIL image.
        heatmap: 2D numpy array with values in [0.0, 1.0].
        alpha: Blending weight (0.0 = only image, 1.0 = only heatmap).
        colormap_name: Matplotlib colormap ('jet', 'inferno', 'viridis', etc.).

    Returns:
        PIL Image with the colorized heatmap overlaid.
    """
    img_rgb = image.convert("RGB")
    orig_w, orig_h = img_rgb.size

    # Ensure heatmap is a 2D float array in [0, 1]
    heat_2d = np.squeeze(heatmap)
    if heat_2d.ndim != 2:
        raise ValueError(f"Expected 2D heatmap array, received shape {heat_2d.shape}")

    heat_norm = np.clip(heat_2d, 0.0, 1.0).astype(np.float32)

    # Resize heatmap to match original image dimensions if needed
    if heat_norm.shape != (orig_h, orig_w):
        heat_pil = Image.fromarray((heat_norm * 255.0).astype(np.uint8), mode="L")
        heat_resized = heat_pil.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)
        heat_norm = np.array(heat_resized, dtype=np.float32) / 255.0

    # Colorize using matplotlib colormaps (supports matplotlib >= 3.8)
    import matplotlib as mpl
    try:
        cmap = mpl.colormaps[colormap_name]
    except Exception:
        cmap = mpl.colormaps["jet"]

    rgba = cmap(heat_norm)
    rgb_overlay = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    overlay_pil = Image.fromarray(rgb_overlay, mode="RGB")

    # Alpha blend
    blended = Image.blend(img_rgb, overlay_pil, alpha=float(np.clip(alpha, 0.0, 1.0)))
    return blended


def draw_detected_regions(
    image: Image.Image,
    detected_regions: list[dict[str, Any]],
    box_color: str = "#FF0033",
) -> Image.Image:
    """Draw bounding boxes and labels around detected anomaly clusters."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for region in detected_regions:
        box = region.get("box")
        conf = region.get("confidence", 0.0)
        label = region.get("label", "Defect")

        if box and len(box) == 4:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            # Text tag
            tag_text = f"{label} ({conf:.2f})"
            draw.rectangle([x1, max(0, y1 - 18), x1 + 140, y1], fill=box_color)
            draw.text((x1 + 4, max(0, y1 - 16)), tag_text, fill="white")

    return annotated
