"""Project paths and typed YAML configuration.

All large artifacts live under a single data root taken from ``$AAI521_DATA_ROOT`` so
the repository can sit in a synced folder (OneDrive, Google Drive) without syncing
gigabytes of images, and so Colab runs can point at a mounted Drive path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, HttpUrl

DATA_ROOT_ENV = "AAI521_DATA_ROOT"
CHECKPOINT_ROOT_ENV = "AAI521_CHECKPOINT_ROOT"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_dotenv(path: Path) -> None:
    """Populate ``os.environ`` from a simple ``KEY=VALUE`` file without overriding."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


def data_root() -> Path:
    """Return the data root, creating it if needed.

    Returns:
        ``$AAI521_DATA_ROOT`` if set, else ``<project>/data``.
    """
    root = Path(os.environ.get(DATA_ROOT_ENV, PROJECT_ROOT / "data")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def checkpoint_root() -> Path:
    """Where fine-tuned weights and run logs go.

    Defaults to ``<data_root>/checkpoints``. On Colab, point ``$AAI521_CHECKPOINT_ROOT``
    at Google Drive so checkpoints survive the session, while ``$AAI521_DATA_ROOT``
    stays on the fast local disk.
    """
    default = data_root() / "checkpoints"
    root = Path(os.environ.get(CHECKPOINT_ROOT_ENV, default)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def raw_dir(source: str) -> Path:
    """Directory holding the raw download for a named source."""
    return data_root() / "raw" / source


def processed_dir() -> Path:
    """Directory for curated manifests and derived artifacts."""
    path = data_root() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    """Directory for run logs."""
    return data_root() / "logs"


class _Base(BaseModel):
    model_config = {"extra": "forbid"}


class LocSourceConfig(_Base):
    """Library of Congress collection crawl settings."""

    collection: str
    max_side: int = Field(2048, gt=0)
    limit: int | None = Field(None, gt=0)
    requests_per_second: float = Field(2.0, gt=0)
    workers: int = Field(4, gt=0)


class CommonsSourceConfig(_Base):
    """Wikimedia Commons category crawl settings."""

    category: str
    max_side: int = Field(1920, gt=0)
    limit: int | None = Field(None, gt=0)
    requests_per_second: float = Field(4.0, gt=0)
    workers: int = Field(4, gt=0)


class ArchiveSourceConfig(_Base):
    """Single-archive (zip) download."""

    url: HttpUrl


class SourcesConfig(_Base):
    """All data sources."""

    loc_fsa_color: LocSourceConfig
    loc_fsa_bw: LocSourceConfig
    documerica: CommonsSourceConfig
    nvidia_masks: ArchiveSourceConfig
    div2k_valid: ArchiveSourceConfig


class CurationConfig(_Base):
    """Quality filters applied before splitting (see ``configs/data.yaml``)."""

    train_sources: list[str]
    min_side: int = Field(1024, gt=0)
    min_colorfulness: float = Field(8.0, ge=0)
    sharpness_min_percentile: float = Field(5, ge=0, le=100)
    noise_max_percentile: float = Field(95, ge=0, le=100)
    max_border_fraction: float = Field(0.45, gt=0, le=0.5)


class SplitConfig(_Base):
    """Group-aware train/val/test split settings."""

    val_fraction: float = Field(0.1, gt=0, lt=0.5)
    test_fraction: float = Field(0.1, gt=0, lt=0.5)
    phash_hamming_threshold: int = Field(8, ge=0, le=64)


class PatchConfig(_Base):
    """Training patch settings."""

    size: int = Field(512, gt=0)
    sr_scale: int = Field(4, gt=1)


class DataConfig(_Base):
    """Top-level data config (``configs/data.yaml``)."""

    seed: int = 521
    sources: SourcesConfig
    curation: CurationConfig
    splits: SplitConfig
    patches: PatchConfig


class ModelSpec(_Base):
    """A pretrained checkpoint on the Hugging Face Hub."""

    repo_id: str
    allow_patterns: list[str]
    role: Literal["primary", "comparison"]
    revision: str | None = None
    notes: str = ""


class ModelsConfig(_Base):
    """Top-level model registry (``configs/models.yaml``)."""

    models: dict[str, ModelSpec]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping at the top of {path}, got {type(loaded).__name__}")
    return loaded


def load_data_config(path: Path = CONFIG_DIR / "data.yaml") -> DataConfig:
    """Load and validate the data config."""
    return DataConfig.model_validate(_read_yaml(path))


def load_models_config(path: Path = CONFIG_DIR / "models.yaml") -> ModelsConfig:
    """Load and validate the model registry."""
    return ModelsConfig.model_validate(_read_yaml(path))
