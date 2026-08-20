"""Tests for structured logging setup."""

import json
import logging
from pathlib import Path
from typing import Any

from ijepa_ad.utils.logging_setup import setup_logging


def test_setup_logging_emits_json_lines(tmp_path: Path, capsys: Any) -> None:
    logger = setup_logging(level=logging.INFO, log_file=tmp_path / "run.jsonl")
    logger.info("hello")

    stderr_record = json.loads(capsys.readouterr().err)
    file_record = json.loads((tmp_path / "run.jsonl").read_text().strip())
    assert stderr_record["level"] == "INFO"
    assert stderr_record["message"] == "hello"
    assert file_record["logger"] == "ijepa_ad"


def test_setup_logging_is_idempotent() -> None:
    first = setup_logging(logger_name="test-idempotent")
    second = setup_logging(logger_name="test-idempotent")

    assert first is second
    assert len(second.handlers) == 1
