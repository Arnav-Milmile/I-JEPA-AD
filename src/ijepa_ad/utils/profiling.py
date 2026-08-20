"""GPU memory profiling helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .logging_setup import LOGGER_NAME


@dataclass
class PeakMemory:
    """Peak CUDA memory observed by a profiling context."""

    label: str
    allocated_bytes: int

    @property
    def allocated_mb(self) -> float:
        """Return peak allocated memory in decimal megabytes."""
        return self.allocated_bytes / 1_000_000


@contextmanager
def profile_peak_vram(
    label: str,
    logger: logging.Logger | None = None,
    device: Any | None = None,
) -> Iterator[PeakMemory]:
    """Log peak CUDA memory allocated while executing a code block.

    On CPU-only hosts, the context is a safe no-op that yields zero bytes. This
    keeps unit tests and CPU CI usable while preserving the same call contract on
    GPU hosts.
    """
    target_logger = logger or logging.getLogger(LOGGER_NAME)
    torch = _load_torch()
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    result = PeakMemory(label=label, allocated_bytes=0)
    if cuda is None or not cuda.is_available():
        yield result
        _log_peak(target_logger, result)
        return

    cuda.reset_peak_memory_stats(device)
    try:
        yield result
    finally:
        allocated_bytes = int(cuda.max_memory_allocated(device))
        result.allocated_bytes = allocated_bytes
        _log_peak(target_logger, result)


def _log_peak(logger: logging.Logger, result: PeakMemory) -> None:
    logger.info(
        "peak CUDA memory: %s bytes (%.3f MB)",
        result.allocated_bytes,
        result.allocated_mb,
        extra={"peak_memory": result},
    )


def _load_torch() -> Any | None:
    """Return PyTorch when installed, otherwise ``None``."""
    try:
        return import_module("torch")
    except ImportError:
        return None
