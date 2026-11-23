"""Download and locate pretrained checkpoints declared in ``configs/models.yaml``."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from restoration.config import ModelSpec, load_models_config

logger = logging.getLogger(__name__)


def fetch_model(name: str, spec: ModelSpec, local_files_only: bool = False) -> Path:
    """Download (or locate in the HF cache) the files for one model.

    Args:
        name: Registry key (e.g. ``"colorize"``).
        spec: Model spec from the registry.
        local_files_only: If True, never hit the network (fails if not cached).

    Returns:
        Local snapshot directory containing the requested files.

    Raises:
        FileNotFoundError: If the snapshot contains none of the requested files.
    """
    path = Path(
        snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            allow_patterns=spec.allow_patterns,
            local_files_only=local_files_only,
        )
    )
    files = [p for p in path.rglob("*") if p.is_file()]
    if not files:
        raise FileNotFoundError(
            f"{name}: snapshot {path} has no files matching {spec.allow_patterns}"
        )
    size_mb = sum(p.stat().st_size for p in files) / 2**20
    logger.info("%-18s %-55s %4d files %8.1f MB", name, spec.repo_id, len(files), size_mb)
    return path


def model_path(name: str) -> Path:
    """Return the cached snapshot directory for a registry entry (no network access)."""
    spec = load_models_config().models[name]
    return fetch_model(name, spec, local_files_only=True)
