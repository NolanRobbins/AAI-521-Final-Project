"""Integration tests: pair datasets against a synthetic data root on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from restoration.training.datasets import PairDatasetConfig, RestorationPairs


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake data root: 6 colour images, splits.csv and 2 NVIDIA-style masks."""
    monkeypatch.setenv("AAI521_DATA_ROOT", str(tmp_path))
    rng = np.random.default_rng(0)
    img_dir = tmp_path / "raw" / "fake" / "images"
    img_dir.mkdir(parents=True)
    rows = []
    for i in range(6):
        arr = (rng.random((300, 400, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(img_dir / f"img{i}.jpg")
        rows.append(
            {
                "image_id": f"fake_{i}",
                "source": "fake",
                "path": f"images/img{i}.jpg",
                "title": f"Scene {i}",
                "split": "train" if i < 4 else "val",
                "border_top": 0,
                "border_bottom": 0,
                "border_left": 0,
                "border_right": 0,
            }
        )
    (tmp_path / "processed").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "processed" / "splits.csv", index=False)
    mask_dir = tmp_path / "raw" / "nvidia_masks" / "extracted"
    mask_dir.mkdir(parents=True)
    for i in range(2):
        m = np.full((128, 128), 255, dtype=np.uint8)
        m[40:80, 40:80] = 0  # black hole on white background
        Image.fromarray(m).save(mask_dir / f"{i:05d}.png")
    pd.DataFrame({"path": [f"extracted/{i:05d}.png" for i in range(2)]}).to_csv(
        tmp_path / "raw" / "nvidia_masks" / "manifest.csv", index=False
    )
    return tmp_path


def _check(t: torch.Tensor, shape: tuple[int, ...]) -> None:
    assert t.shape == shape and t.dtype == torch.float32
    assert torch.isfinite(t).all() and t.min() >= 0 and t.max() <= 1


@pytest.mark.parametrize("split", ["train", "val"])
def test_denoise_and_sr_shapes(data_root: Path, split: str) -> None:
    ds = RestorationPairs(PairDatasetConfig(task="denoise", split=split, patch_size=128))  # type: ignore[arg-type]
    item = ds[0]
    _check(item["inp"], (3, 128, 128))  # type: ignore[arg-type]
    sr = RestorationPairs(PairDatasetConfig(task="super_resolution", split=split, patch_size=128))  # type: ignore[arg-type]
    item = sr[0]
    _check(item["inp"], (3, 32, 32))  # type: ignore[arg-type]
    _check(item["target"], (3, 128, 128))  # type: ignore[arg-type]


def test_validation_items_are_deterministic(data_root: Path) -> None:
    ds = RestorationPairs(PairDatasetConfig(task="denoise", split="val", patch_size=64))
    a, b = ds[1], ds[1]
    assert torch.equal(a["inp"], b["inp"])  # type: ignore[arg-type]


def test_colorize_item_is_grey_input_with_colour_target(data_root: Path) -> None:
    ds = RestorationPairs(PairDatasetConfig(task="colorize", split="val", patch_size=96))
    item = ds[0]
    inp, target = item["inp"], item["target"]
    assert isinstance(inp, torch.Tensor) and isinstance(target, torch.Tensor)
    _check(inp, (3, 96, 96))
    assert (inp[0] - inp[1]).abs().max() < 0.02  # grey
    assert item["target_ab"].shape == (2, 96, 96)  # type: ignore[union-attr]


def test_inpaint_item_has_mask_and_caption(data_root: Path) -> None:
    ds = RestorationPairs(
        PairDatasetConfig(task="inpaint", split="val", patch_size=128, nvidia_mask_prob=1.0)
    )
    item = ds[0]
    mask = item["mask"]
    assert isinstance(mask, torch.Tensor) and mask.shape == (1, 128, 128)
    assert 0 < mask.mean() < 0.5
    assert item["caption"] == "Scene 4"
