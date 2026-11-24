"""Unit tests for classical baselines and the evaluation harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from restoration.data import degradations as deg
from restoration.evaluation import baselines, harness


@pytest.fixture
def smooth() -> deg.FloatImage:
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    return np.stack([xx / 63, yy / 63, 0.5 * np.ones_like(xx)], axis=-1).astype(np.float32)


def test_nlm_reduces_noise(smooth: deg.FloatImage) -> None:
    noisy = deg.add_gaussian_noise(smooth, 0.08, np.random.default_rng(0))
    out = baselines.nlm_denoise(noisy)
    assert np.abs(out - smooth).mean() < np.abs(noisy - smooth).mean()


def test_bicubic_scales(smooth: deg.FloatImage) -> None:
    assert baselines.bicubic_upscale(smooth[:16, :20]).shape == (64, 80, 3)


def test_telea_only_changes_hole(smooth: deg.FloatImage) -> None:
    mask = np.zeros(smooth.shape[:2], dtype=bool)
    mask[30:34, 10:50] = True
    damaged = smooth.copy()
    damaged[mask] = 1.0
    out = baselines.telea_inpaint(damaged, mask)
    assert np.abs(out[mask] - smooth[mask]).mean() < 0.05
    np.testing.assert_allclose(out[~mask], smooth[~mask], atol=1 / 255 + 1e-6)


def _items(n: int = 6) -> list[harness.TestItem]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        target = rng.random((32, 32, 3), dtype=np.float32)
        out.append(harness.TestItem(i, deg.add_gaussian_noise(target, 0.1, rng), target))
    return out


def test_evaluate_task_scores_every_method_on_every_image() -> None:
    items = _items()
    targets = {it.index: it.target for it in items}
    methods: dict[str, harness.Method] = {
        "identity": lambda s: s["inp"],
        "oracle": lambda s: targets[s["index"]],
    }
    frame = harness.evaluate_task("denoise", items, methods, torch.device("cpu"))
    assert len(frame) == 12 and set(frame["method"]) == {"identity", "oracle"}
    summary = harness.summarise(frame)
    assert list(summary.index) == ["identity", "oracle"]
    assert summary.loc["oracle", ("psnr", "mean")] > 90
    assert summary.loc["oracle", ("lpips", "mean")] < summary.loc["identity", ("lpips", "mean")]


def test_paired_bootstrap_ci_contains_true_difference() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(30, 2, 200)
    frame = pd.DataFrame(
        {
            "method": ["a"] * 200 + ["b"] * 200,
            "image": list(range(200)) * 2,
            "psnr": np.concatenate([base + 1.0 + rng.normal(0, 0.2, 200), base]),
        }
    )
    mean, lo, hi = harness.paired_bootstrap(frame, "psnr", "a", "b")
    assert lo < 1.0 < hi and mean == pytest.approx(1.0, abs=0.1)
    assert hi - lo < 0.2  # pairing removes the large between-image variance
