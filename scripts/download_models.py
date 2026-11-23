"""Download the pretrained checkpoints listed in ``configs/models.yaml`` to the HF cache.

Usage:
    python scripts/download_models.py                    # all models
    python scripts/download_models.py --models colorize super_resolution
"""

from __future__ import annotations

import argparse
import logging
import sys

from restoration.config import load_models_config, logs_dir
from restoration.models.registry import fetch_model
from restoration.utils.logging import setup_logging

logger = logging.getLogger("download_models")


def main(argv: list[str] | None = None) -> int:
    """Download requested models; returns a process exit code."""
    registry = load_models_config().models
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(registry), default=list(registry))
    args = parser.parse_args(argv)
    setup_logging(logs_dir(), "download_models")

    failed: list[str] = []
    for name in args.models:
        try:
            fetch_model(name, registry[name])
        except Exception:  # keep going so one gated/missing repo doesn't block the rest
            logger.exception("Model %s failed", name)
            failed.append(name)
    if failed:
        logger.error("Failed models: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
