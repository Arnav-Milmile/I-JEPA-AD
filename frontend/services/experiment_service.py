"""Experiment and training log service for Person 4 visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_training_history(
    category: str = "bottle", epochs: int = 50
) -> tuple[pd.DataFrame, bool]:
    """Load real training history from a checkpoint or generate illustrative curves.

    Inspects `results/predictor/{category}.pt` (the trainer's saved checkpoint).
    If found, extracts recorded `history` (epoch, train_loss, val_loss).

    Returns:
        tuple[pd.DataFrame, bool]: (DataFrame, is_synthetic_flag)
    """
    root = Path(__file__).resolve().parent.parent.parent
    ckpt_path = root / "results" / "predictor" / f"{category}.pt"

    if ckpt_path.is_file():
        try:
            import torch

            checkpoint: dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
            history = checkpoint.get("history", [])
            if history:
                records: list[dict[str, Any]] = []
                for entry in history:
                    records.append(
                        {
                            "Epoch": entry.get("epoch", len(records) + 1),
                            "Train Loss": entry.get("train_loss", 0.0),
                            "Val Loss": entry.get("val_loss", 0.0),
                        }
                    )
                df = pd.DataFrame(records)
                return df, False
        except Exception:
            pass

    # Reproducible illustrative learning curves demonstrating expected convergence
    rng = np.random.default_rng(seed=42 + len(category))
    epoch_arr = np.arange(1, epochs + 1)

    train_loss = (
        0.42 * np.exp(-epoch_arr / 14.0) + 0.08 + rng.normal(0, 0.004, size=epochs)
    )
    val_loss = (
        0.45 * np.exp(-epoch_arr / 14.0) + 0.09 + rng.normal(0, 0.006, size=epochs)
    )

    df = pd.DataFrame(
        {
            "Epoch": epoch_arr,
            "Train Loss": np.clip(train_loss, 0.0, None),
            "Val Loss": np.clip(val_loss, 0.0, None),
        }
    )
    return df, True


def load_benchmark_summary() -> tuple[pd.DataFrame, bool]:
    """Return benchmark comparison table across MVTec categories and methods.

    Returns:
        tuple[pd.DataFrame, bool]: (DataFrame, is_synthetic_flag)
    """
    root = Path(__file__).resolve().parent.parent.parent
    bench_file = root / "results" / "benchmarks" / "summary.csv"

    if bench_file.is_file():
        try:
            df = pd.read_csv(bench_file)
            return df, False
        except Exception:
            pass

    # Target / illustrative baseline comparisons from project proposal
    data = {
        "Method": [
            "Adaptive I-JEPA (Ours)",
            "Dense-mask I-JEPA",
            "Fixed-mask I-JEPA",
            "PatchCore",
            "MAE Reconstruction",
        ],
        "Image AUROC (%)": [98.4, 98.7, 91.5, 97.9, 92.6],
        "Pixel AUROC (%)": [96.8, 97.1, 87.8, 96.2, 88.3],
        "Avg Evaluated Masks": [14, 196, 16, 0, 1],
        "Mask Reduction vs Dense (%)": [92.9, 0.0, 91.8, 100.0, 99.5],
        "Inference Latency (ms)": [42.5, 218.4, 31.0, 115.8, 88.2],
    }
    return pd.DataFrame(data), True


def load_category_auroc_breakdown() -> tuple[pd.DataFrame, bool]:
    """Return category-wise AUROC breakdown for all 15 MVTec objects and textures.

    Returns:
        tuple[pd.DataFrame, bool]: (DataFrame, is_synthetic_flag)
    """
    root = Path(__file__).resolve().parent.parent.parent
    cat_file = root / "results" / "benchmarks" / "category_breakdown.csv"

    if cat_file.is_file():
        try:
            df = pd.read_csv(cat_file)
            return df, False
        except Exception:
            pass

    categories = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    # Target numbers per proposal
    adaptive_scores = [
        99.2,
        98.1,
        98.5,
        97.4,
        97.9,
        98.8,
        99.5,
        98.0,
        97.8,
        96.9,
        98.2,
        97.5,
        98.6,
        99.1,
        98.4,
    ]
    dense_scores = [
        99.4,
        98.3,
        98.7,
        97.6,
        98.1,
        99.0,
        99.6,
        98.2,
        98.0,
        97.1,
        98.4,
        97.7,
        98.8,
        99.3,
        98.6,
    ]
    patchcore_scores = [
        98.5,
        97.2,
        97.8,
        96.5,
        97.0,
        98.1,
        99.0,
        97.4,
        97.0,
        95.8,
        97.6,
        96.8,
        98.0,
        98.5,
        97.8,
    ]

    df = pd.DataFrame(
        {
            "Category": categories,
            "Adaptive I-JEPA": adaptive_scores,
            "Dense I-JEPA": dense_scores,
            "PatchCore": patchcore_scores,
        }
    )
    return df, True
