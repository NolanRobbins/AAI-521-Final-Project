"""Unit tests for metrics, Lab conversion and LoRA helpers."""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest
import torch

from restoration.evaluation import metrics
from restoration.training.sd_inpaint import GENERIC_PROMPT, caption_for, latent_mask
from restoration.training.tasks import charbonnier


def test_psnr_known_value() -> None:
    target = torch.zeros(2, 3, 8, 8)
    pred = torch.full_like(target, 0.1)  # MSE = 0.01 -> 20 dB
    torch.testing.assert_close(metrics.psnr(pred, target), torch.tensor([20.0, 20.0]))


def test_masked_psnr_ignores_unmasked_errors() -> None:
    target = torch.zeros(1, 3, 8, 8)
    pred = target.clone()
    pred[..., :4, :] = 1.0  # error only outside the hole
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., 4:, :] = 1
    assert metrics.masked_psnr(pred, target, mask).item() > 90


def test_ssim_identity_and_degradation() -> None:
    img = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(metrics.ssim(img, img), torch.ones(2))
    assert (metrics.ssim(img, img.flip(-1)) < 0.5).all()


def test_colorfulness_matches_numpy_reference() -> None:
    from restoration.data import quality

    img = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32)
    t = torch.from_numpy(img.transpose(2, 0, 1))[None]
    # torch uses unbiased var by default -> compare with tolerance
    assert metrics.colorfulness(t).item() == pytest.approx(quality.colorfulness(img), rel=0.02)


def test_lab_to_rgb_matches_opencv() -> None:
    img = np.random.default_rng(1).random((24, 24, 3), dtype=np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    rgb = metrics.lab_to_rgb(torch.from_numpy(lab.transpose(2, 0, 1))[None])[0].permute(1, 2, 0)
    np.testing.assert_allclose(rgb.numpy(), img, atol=5e-3)
    lightness = metrics.rgb_to_l(torch.from_numpy(img.transpose(2, 0, 1))[None])[0, 0]
    np.testing.assert_allclose(lightness.numpy(), lab[..., 0], atol=0.5)


def test_lab_to_rgb_is_differentiable() -> None:
    lab = torch.tensor([[[[50.0]], [[20.0]], [[-10.0]]]], requires_grad=True)
    metrics.lab_to_rgb(lab).sum().backward()
    assert lab.grad is not None and torch.isfinite(lab.grad).all()


def test_charbonnier_is_near_l1_for_large_errors() -> None:
    a, b = torch.zeros(10), torch.ones(10)
    assert charbonnier(a, b).item() == pytest.approx(1.0, abs=1e-3)


def test_latent_mask_keeps_thin_scratches() -> None:
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., :, 13] = 1  # 1-px vertical scratch
    lat = latent_mask(mask)
    assert lat.shape == (1, 1, 8, 8)
    assert lat[0, 0, :, 1].eq(1).all() and lat.sum() == 8
    # nearest-neighbour downsampling would have lost it entirely
    assert torch.nn.functional.interpolate(mask, size=(8, 8)).sum() == 0


@pytest.mark.parametrize("title", ["", "[Untitled]", "  "])
def test_caption_falls_back_to_generic(title: str) -> None:
    assert caption_for(title, 0.0, random.Random(0)) == GENERIC_PROMPT


def test_caption_uses_title_without_dropout() -> None:
    assert caption_for("Farm,  Iowa", 0.0, random.Random(0)) == "an old photograph, Farm, Iowa"
    assert caption_for("Farm", 1.0, random.Random(0)) == GENERIC_PROMPT
