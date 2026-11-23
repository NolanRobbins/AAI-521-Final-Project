"""Logging configuration: timestamped console + file handlers."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_dir: Path, run_name: str, level: int = logging.INFO) -> Path:
    """Configure the root logger to write to the console and a timestamped log file.

    Args:
        log_dir: Directory for log files; created if missing.
        run_name: Prefix for the log file name (e.g. ``"download_data"``).
        level: Root logging level.

    Returns:
        Path to the log file that was created.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}.log"

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT)
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Third-party HTTP libraries are extremely chatty at INFO.
    for noisy in ("urllib3", "PIL", "httpx", "httpx2", "huggingface_hub.file_download"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path
