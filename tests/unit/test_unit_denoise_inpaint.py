"""Unit tests for the denoise_inpaint test items and the BOPBtL export/loader round trip."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from restoration.evaluation import denoise_inpaint as dni


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake data root: 3 colour test images and 2 NVIDIA-style masks."""
    monkeypatch.setenv("AAI521_DATA_ROOT", str(tmp_path))
    rng = np.random.default_rng(0)
    img_dir = tmp_path / "raw" / "fake" / "images"
    img_dir.mkdir(parents=True)
    rows = []
    for i in range(3):
        Image.fromarray((rng.random((600, 640, 3)) * 255).astype(np.uint8)).save(
            img_dir / f"img{i}.png"
        )
        rows.append({"image_id": f"fake_{i}", "source": "fake", "path": f"images/img{i}.png"})
    (tmp_path / "processed").mkdir()
    pd.DataFrame(rows).assign(split="test").to_csv(
        tmp_path / "processed" / "splits.csv", index=False
    )
    mask_dir = tmp_path / "raw" / "nvidia_masks" / "extracted"
    mask_dir.mkdir(parents=True)
    for i in range(2):
        m = np.full((128, 128), 255, dtype=np.uint8)
        m[40:80, 40:80] = 0
        Image.fromarray(m).save(mask_dir / f"{i:05d}.png")
    pd.DataFrame({"path": [f"extracted/{i:05d}.png" for i in range(2)]}).to_csv(
        tmp_path / "raw" / "nvidia_masks" / "manifest.csv", index=False
    )
    return tmp_path


def test_build_items_deterministic_and_8bit(data_root: Path) -> None:
    a, b = dni.build_items(3, seed=1), dni.build_items(3, seed=1)
    assert [it.mask_type for it in a] == ["procedural", "nvidia", "procedural"]
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.inp, y.inp)
        np.testing.assert_array_equal(x.mask, y.mask)
        assert x.inp.shape == x.target.shape == (512, 512, 3) and x.inp.dtype == np.float32
        np.testing.assert_array_equal(x.inp * 255, np.round(x.inp * 255))
        assert x.mask is not None and x.mask.any() and not np.array_equal(x.inp, x.target)
    assert not np.array_equal(a[0].inp, dni.build_items(1, seed=2)[0].inp)


def test_export_loader_round_trip(data_root: Path) -> None:
    items = dni.build_items(3, seed=1)
    root = data_root / "outputs" / "bopbtl"
    manifest = dni.export_inputs(items, root, 1, 3, ["a", "b", "c"])
    assert list(pd.read_csv(manifest)["image_id"]) == ["a", "b", "c"]
    shutil.copytree(root / "inputs", root / "outputs")  # an identity "restoration"
    method = dni.bopbtl_method(root, items, seed=1, limit=3)
    assert method is not None
    for it in items:
        out = method({"inp": it.inp, "mask": it.mask, "index": it.index})
        np.testing.assert_array_equal(out, it.inp)


def test_loader_skips_when_outputs_missing(
    data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    items = dni.build_items(2, seed=1)
    root = data_root / "outputs" / "bopbtl"
    assert dni.bopbtl_method(root, items, seed=1, limit=2) is None
    dni.export_inputs(items, root, 1, 2, ["a", "b"])
    (root / "outputs").mkdir()
    shutil.copy(root / "inputs" / "0000.png", root / "outputs")  # one of two
    with caplog.at_level(logging.WARNING):
        assert dni.bopbtl_method(root, items, seed=1, limit=2) is None
    assert "missing for 1/2" in caplog.text


def test_loader_resizes_mismatched_output_and_warns_on_manifest(
    data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    items = dni.build_items(1, seed=1)
    root = data_root / "outputs" / "bopbtl"
    dni.export_inputs(items, root, 1, 1, ["a"])
    (root / "outputs").mkdir()
    Image.fromarray(np.full((496, 528, 3), 128, dtype=np.uint8)).save(root / "outputs/0000.png")
    with caplog.at_level(logging.WARNING):
        method = dni.bopbtl_method(root, items, seed=2, limit=1)
        assert method is not None
        out = method({"inp": items[0].inp, "mask": items[0].mask, "index": 0})
    assert out.shape == (512, 512, 3) and out.dtype == np.float32
    np.testing.assert_allclose(out, 128 / 255, atol=1e-6)
    assert "exported with seed=1" in caplog.text and "resized back" in caplog.text


def test_export_clears_stale_inputs(data_root: Path) -> None:
    root = data_root / "outputs" / "bopbtl"
    dni.export_inputs(dni.build_items(3, seed=1), root, 1, 3, ["a", "b", "c"])
    dni.export_inputs(dni.build_items(1, seed=1), root, 1, 1, ["a"])
    assert sorted(p.name for p in (root / "inputs").iterdir()) == ["0000.png"]
