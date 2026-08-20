"""Consistent structured logging for training and evaluation runs."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER_NAME = "ijepa_ad"


class _JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    logger_name: str = LOGGER_NAME,
) -> logging.Logger:
    """Configure and return the project's logger.

    Calling this repeatedly replaces handlers owned by this function, preventing
    duplicate lines in notebooks and test processes.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in logger.handlers[:]:
        if getattr(handler, "_ijepa_ad_handler", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = _JsonFormatter()
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler._ijepa_ad_handler = True  # type: ignore[attr-defined]
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler._ijepa_ad_handler = True  # type: ignore[attr-defined]
        logger.addHandler(file_handler)

    return logger
