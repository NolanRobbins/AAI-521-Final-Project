"""Unit tests for synthetic degradations: shape/dtype/range contracts and determinism."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from restoration.data import degradations as deg


@pytest.fixture
def clean() -> deg.FloatImage:
    """Smooth synthetic image with gradients and an edge (64x96)."""
    yy, xx = np.mgrid[0:64, 0:96].astype(np.float32)
    img = np.stack([xx / 95, yy / 63, 0.5 + 0.4 * (xx > 48)], axis=-1)
    return img.astype(np.float32)


def _assert_valid(img: deg.FloatImage, shape: tuple[int, ...]) -> None:
    assert img.shape == shape
    assert img.dtype == np.float32
    assert np.isfinite(img).all()
    assert img.min() >= 0.0 and img.max() <= 1.0


@settings(max_examples=25, deadline=None)
@given(sigma=st.floats(0, 0.5), seed=st.integers(0, 2**32 - 1))
def test_gaussian_noise_stays_in_range(sigma: float, seed: int) -> None:
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    _assert_valid(deg.add_gaussian_noise(img, sigma, np.random.default_rng(seed)), img.shape)


@settings(max_examples=25, deadline=None)
@given(strength=st.floats(0, 0.3), size=st.floats(0, 3), seed=st.integers(0, 2**32 - 1))
def test_film_grain_stays_in_range(strength: float, size: float, seed: int) -> None:
    img = np.random.default_rng(0).random((24, 24, 3), dtype=np.float32)
    _assert_valid(deg.add_film_grain(img, strength, size, np.random.default_rng(seed)), img.shape)


def test_film_grain_spares_pure_black_and_white() -> None:
    img = np.zeros((8, 16, 3), dtype=np.float32)
    img[:, 8:] = 1.0
    out = deg.add_film_grain(img, 0.3, 1.0, np.random.default_rng(0))
    np.testing.assert_allclose(out, img, atol=1e-6)


def test_noise_is_deterministic_given_seed(clean: deg.FloatImage) -> None:
    a = deg.make_denoise_pair(clean, np.random.default_rng(7))
    b = deg.make_denoise_pair(clean, np.random.default_rng(7))
    np.testing.assert_array_equal(a.inp, b.inp)


def test_denoise_pair_changes_input_not_target(clean: deg.FloatImage) -> None:
    pair = deg.make_denoise_pair(clean, np.random.default_rng(1))
    _assert_valid(pair.inp, clean.shape)
    np.testing.assert_array_equal(pair.target, clean)
    assert np.abs(pair.inp - clean).mean() > 0


@pytest.mark.parametrize("scale", [2, 4])
def test_sr_pair_shapes(clean: deg.FloatImage, scale: int) -> None:
    pair = deg.make_sr_pair(clean, np.random.default_rng(0), deg.SRConfig(scale=scale))
    h, w = pair.target.shape[:2]
    assert h % scale == 0 and w % scale == 0
    _assert_valid(pair.inp, (h // scale, w // scale, 3))


def test_downsample_rejects_indivisible() -> None:
    with pytest.raises(ValueError):
        deg.downsample(np.zeros((10, 10, 3), dtype=np.float32), 4)


def test_grayscale_has_equal_channels_and_preserves_lightness(clean: deg.FloatImage) -> None:
    gray = deg.to_grayscale(clean)
    _assert_valid(gray, clean.shape)
    assert np.abs(gray[..., 0] - gray[..., 1]).max() < 1e-2
    assert np.abs(gray[..., 1] - gray[..., 2]).max() < 1e-2


def test_jpeg_round_trip_is_close(clean: deg.FloatImage) -> None:
    out = deg.jpeg_compress(clean, 95)
    _assert_valid(out, clean.shape)
    assert np.abs(out - clean).mean() < 0.02


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_damage_mask_respects_hole_budget(seed: int) -> None:
    cfg = deg.DamageConfig()
    mask = deg.random_damage_mask(96, 128, np.random.default_rng(seed), cfg)
    assert mask.shape == (96, 128) and mask.dtype == np.bool_
    assert mask.mean() <= cfg.max_hole_fraction


def test_inpaint_pair_only_alters_masked_pixels(clean: deg.FloatImage) -> None:
    pair = deg.make_inpaint_pair(clean, np.random.default_rng(3))
    assert pair.mask is not None and pair.mask.any()
    np.testing.assert_array_equal(pair.inp[~pair.mask], clean[~pair.mask])


def test_inpaint_pair_resizes_external_mask(clean: deg.FloatImage) -> None:
    ext = np.zeros((32, 48), dtype=bool)
    ext[8:16, 8:24] = True
    pair = deg.make_inpaint_pair(clean, np.random.default_rng(0), external_mask=ext)
    assert pair.mask is not None and pair.mask.shape == clean.shape[:2]
    assert 0 < pair.mask.mean() < 0.2


@pytest.mark.parametrize("size", [32, 128])
def test_random_crop_size_and_padding(clean: deg.FloatImage, size: int) -> None:
    _assert_valid(deg.random_crop(clean, size, np.random.default_rng(0)), (size, size, 3))


def test_resize_max_side_never_upscales(clean: deg.FloatImage) -> None:
    assert deg.resize_max_side(clean, 1000).shape == clean.shape
    assert max(deg.resize_max_side(clean, 48).shape[:2]) == 48


def test_fade_reduces_contrast(clean: deg.FloatImage) -> None:
    faded = deg.fade(clean, 0.8, (0.8, 0.7, 0.5))
    _assert_valid(faded, clean.shape)
    assert faded.std() < clean.std()


def test_correct_fading_removes_cast_and_restores_contrast() -> None:
    rng = np.random.default_rng(0)
    base = rng.random((64, 64, 3), dtype=np.float32)
    faded = deg.fade(base, 0.8, (0.85, 0.55, 0.75))  # low contrast + magenta cast
    fixed = deg.correct_fading(faded, strength=1.0, chroma_gain=1.0)
    lab_faded = deg.cv2.cvtColor(faded, deg.cv2.COLOR_RGB2LAB)
    lab_fixed = deg.cv2.cvtColor(fixed, deg.cv2.COLOR_RGB2LAB)
    assert (
        np.abs(np.median(lab_fixed[..., 1:], axis=(0, 1))).max()
        < 0.5 * np.abs(np.median(lab_faded[..., 1:], axis=(0, 1))).max()
    )
    assert lab_fixed[..., 0].std() > lab_faded[..., 0].std()
    assert fixed.dtype == np.float32 and fixed.min() >= 0 and fixed.max() <= 1
