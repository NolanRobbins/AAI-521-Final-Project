"""Unit tests for quality features and model-agnostic inference helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from restoration.data import quality
from restoration.models import inference


def test_noise_estimate_tracks_true_sigma() -> None:
    rng = np.random.default_rng(0)
    flat = np.full((256, 256), 0.5, dtype=np.float32)
    for sigma in (0.01, 0.05):
        noisy = flat + rng.normal(0, sigma, flat.shape).astype(np.float32)
        assert quality.estimate_noise_sigma(noisy) == pytest.approx(sigma, rel=0.1)


def test_sharpness_drops_with_blur() -> None:
    rng = np.random.default_rng(0)
    img = rng.random((128, 128), dtype=np.float32)
    import cv2

    assert quality.laplacian_variance(
        cv2.GaussianBlur(img, (0, 0), 2)
    ) < quality.laplacian_variance(img)


def test_colorfulness_zero_for_grey() -> None:
    grey = np.full((32, 32, 3), 0.4, dtype=np.float32)
    assert quality.colorfulness(grey) == pytest.approx(0.0, abs=1e-4)
    red = np.zeros_like(grey)
    red[..., 0] = 1.0
    assert quality.colorfulness(red) > 50


def test_border_extent_detects_slide_mount() -> None:
    rng = np.random.default_rng(0)
    gray = np.zeros((100, 120), dtype=np.float32)
    gray[10:90, 15:105] = rng.random((80, 90), dtype=np.float32)
    assert quality.border_extent(gray) == (10, 10, 15, 15)


def test_compute_features_reports_original_size() -> None:
    img = np.random.default_rng(0).random((300, 2000, 3), dtype=np.float32)
    feats = quality.compute_features(img, analysis_side=500)
    assert (feats.width, feats.height) == (2000, 300)
    assert feats.aspect == pytest.approx(2000 / 300)


@pytest.mark.parametrize("scale", [1, 2])
def test_tiled_apply_matches_untiled(scale: int) -> None:
    x = torch.rand(1, 3, 37, 53)

    def fn(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(t, scale_factor=scale, mode="nearest") * 2

    full = fn(x)
    tiled = inference.tiled_apply(fn, x, tile=16, overlap=4, scale=scale)
    torch.testing.assert_close(tiled, full)


def test_pad_to_multiple_and_crop_back() -> None:
    x = torch.rand(1, 3, 10, 13)
    padded, h, w = inference.pad_to_multiple(x, 8)
    assert padded.shape[-2:] == (16, 16) and (h, w) == (10, 13)
    torch.testing.assert_close(padded[..., :h, :w], x)


def test_composite_leaves_unmasked_pixels_untouched() -> None:
    original = np.zeros((20, 20, 3), dtype=np.float32)
    filled = np.ones_like(original)
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True
    out = inference.composite(original, filled, mask, dilate_px=0, feather_px=0)
    assert out[mask].min() == 1.0 and out[~mask].max() == 0.0


def test_lab_round_trip() -> None:
    img = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32)
    # OpenCV's float Lab path uses a spline-interpolated gamma, so ~2e-3 error near black
    # is inherent; anything below 1/255 is visually lossless.
    np.testing.assert_allclose(inference.lab_to_rgb(inference.rgb_to_lab(img)), img, atol=1 / 255)
