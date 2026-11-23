"""Synthetic degradations that turn clean photographs into training inputs.

Every function takes and returns float32 RGB arrays in ``[0, 1]`` with shape
``(H, W, 3)`` and draws randomness from an explicit :class:`numpy.random.Generator`, so
a pair is fully reproducible from ``(image, seed)``.

The degradations model what happens to analogue photographs:

* **Gaussian noise** -- sensor/scan noise.
* **Film grain** -- spatially correlated, luminance-dependent noise from silver/dye
  clouds; strongest in mid-tones (Newson et al., 2017 give a physical model; we use the
  common filtered-noise approximation).
* **Blur + downsampling** -- optics and low scan resolution (classical SR model
  ``y = (x * k)↓s + n``).
* **JPEG** -- compression of digitised copies.
* **Fading / colour cast** -- dye fading of chromogenic prints.
* **Scratches, cracks, blotches** -- physical damage, used as inpainting masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

FloatImage = NDArray[np.float32]
BoolMask = NDArray[np.bool_]
Range = tuple[float, float]


def _check(img: FloatImage) -> None:
    assert img.ndim == 3 and img.shape[2] == 3, f"expected HxWx3, got {img.shape}"
    assert img.dtype == np.float32, f"expected float32, got {img.dtype}"


def _uniform(rng: np.random.Generator, bounds: Range) -> float:
    lo, hi = bounds
    return float(rng.uniform(lo, hi))


def to_float(img: NDArray[np.uint8]) -> FloatImage:
    """uint8 ``[0,255]`` -> float32 ``[0,1]``."""
    return (img.astype(np.float32) / 255.0).astype(np.float32)


def to_uint8(img: FloatImage) -> NDArray[np.uint8]:
    """float32 ``[0,1]`` -> uint8 with rounding and clipping."""
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def luminance(img: FloatImage) -> NDArray[np.float32]:
    """Rec. 601 luma in ``[0,1]``, shape ``(H, W)``."""
    _check(img)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (img @ weights).astype(np.float32)


# --------------------------------------------------------------------------- primitives


def add_gaussian_noise(img: FloatImage, sigma: float, rng: np.random.Generator) -> FloatImage:
    """Add i.i.d. Gaussian noise with std ``sigma`` (in ``[0,1]`` units)."""
    _check(img)
    noisy = img + rng.standard_normal(img.shape, dtype=np.float32) * np.float32(sigma)
    return np.clip(noisy, 0, 1).astype(np.float32)


def add_film_grain(
    img: FloatImage,
    strength: float,
    grain_size: float,
    rng: np.random.Generator,
    chroma: float = 0.2,
) -> FloatImage:
    """Add correlated, luminance-dependent film grain.

    Args:
        img: Clean image.
        strength: Grain std at mid-grey, in ``[0,1]`` units.
        grain_size: Gaussian blur sigma (pixels) controlling grain clump size.
        rng: Random generator.
        chroma: Fraction of grain that is independent per channel (0 = monochrome).

    Returns:
        Grainy image.
    """
    _check(img)
    h, w = img.shape[:2]

    def field(channels: int) -> NDArray[np.float32]:
        noise = rng.standard_normal((h, w, channels), dtype=np.float32)
        if grain_size > 0:
            blurred = cv2.GaussianBlur(noise, (0, 0), grain_size)
            noise = np.asarray(blurred, dtype=np.float32).reshape(h, w, channels)
        std = float(noise.std())
        return np.asarray(noise / (std if std > 0 else 1.0), dtype=np.float32)

    grain = (1 - chroma) * field(1) + chroma * field(3)
    lum = luminance(img)[..., None]
    weight = 4.0 * lum * (1.0 - lum)  # peaks at mid-grey, vanishes at black/white
    out = img + np.float32(strength) * grain * weight
    return np.asarray(np.clip(out, 0, 1), dtype=np.float32)


def gaussian_blur(img: FloatImage, sigma: float) -> FloatImage:
    """Isotropic Gaussian blur; ``sigma <= 0`` is a no-op."""
    _check(img)
    if sigma <= 0:
        return img.copy()
    return cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float32)


def crop_to_multiple(img: FloatImage, multiple: int) -> FloatImage:
    """Crop bottom/right so both spatial dims are divisible by ``multiple``."""
    h, w = img.shape[:2]
    return img[: h - h % multiple, : w - w % multiple]


def downsample(img: FloatImage, scale: int) -> FloatImage:
    """Bicubic downsample by an integer factor (dims must be divisible by ``scale``)."""
    _check(img)
    h, w = img.shape[:2]
    if h % scale or w % scale:
        raise ValueError(f"image {h}x{w} not divisible by scale {scale}")
    out = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    return np.clip(out, 0, 1).astype(np.float32)


def jpeg_compress(img: FloatImage, quality: int) -> FloatImage:
    """Round-trip through JPEG at ``quality`` (1-100)."""
    _check(img)
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("JPEG decoding failed")
    return to_float(np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8))


def to_grayscale(img: FloatImage) -> FloatImage:
    """Grey RGB image from the CIELAB L channel (matches the colorizer's input)."""
    _check(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 1:] = 0
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1).astype(np.float32)


def fade(img: FloatImage, amount: float, tint: tuple[float, float, float]) -> FloatImage:
    """Simulate dye fading: lower contrast and blend toward a colour cast.

    Args:
        img: Clean image.
        amount: 0 (none) to 1 (fully washed out).
        tint: RGB colour of the cast (e.g. yellow-magenta for aged prints).

    Returns:
        Faded image.
    """
    _check(img)
    tint_arr = np.asarray(tint, dtype=np.float32)
    contrast = 1.0 - 0.5 * amount
    out = (img - 0.5) * contrast + 0.5
    out = out * (1 - 0.35 * amount) + tint_arr * 0.35 * amount
    return np.asarray(np.clip(out, 0, 1), dtype=np.float32)


# --------------------------------------------------------------------------- damage masks


@dataclass(frozen=True)
class DamageConfig:
    """Ranges for procedural physical-damage masks."""

    n_scratches: tuple[int, int] = (1, 8)
    scratch_width: tuple[int, int] = (1, 4)
    n_cracks: tuple[int, int] = (0, 3)
    n_blotches: tuple[int, int] = (0, 4)
    blotch_radius_frac: Range = (0.005, 0.03)
    max_hole_fraction: float = 0.25


def _random_walk(
    rng: np.random.Generator, start: NDArray[np.float64], heading: float, steps: int, step_len: float, wobble: float
) -> NDArray[np.int32]:
    points = [start]
    for _ in range(steps):
        heading += rng.normal(0, wobble)
        points.append(points[-1] + step_len * np.array([np.cos(heading), np.sin(heading)]))
    return np.round(np.array(points)).astype(np.int32)


def random_damage_mask(h: int, w: int, rng: np.random.Generator, cfg: DamageConfig = DamageConfig()) -> BoolMask:
    """Procedural mask of scratches, branching cracks and blotches (True = damaged).

    Scratches are long, near-vertical strokes (film transport scratches); cracks are
    wandering branched lines (emulsion cracking on prints); blotches are irregular
    ellipses (stains, missing emulsion).

    Args:
        h: Height.
        w: Width.
        rng: Random generator.
        cfg: Damage ranges.

    Returns:
        Boolean mask of shape ``(h, w)`` with hole fraction <= ``cfg.max_hole_fraction``.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    diag = float(np.hypot(h, w))

    for _ in range(int(rng.integers(cfg.n_scratches[0], cfg.n_scratches[1] + 1))):
        start = np.array([rng.uniform(0, w), rng.uniform(-0.1 * h, 0.3 * h)])
        heading = np.pi / 2 + rng.normal(0, 0.15)
        pts = _random_walk(rng, start, heading, steps=40, step_len=h / 40 * rng.uniform(0.5, 1.2), wobble=0.03)
        thickness = int(rng.integers(cfg.scratch_width[0], cfg.scratch_width[1] + 1))
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 1, thickness, lineType=cv2.LINE_AA)

    for _ in range(int(rng.integers(cfg.n_cracks[0], cfg.n_cracks[1] + 1))):
        start = np.array([rng.uniform(0, w), rng.uniform(0, h)])
        trunk = _random_walk(rng, start, rng.uniform(0, 2 * np.pi), 30, diag / 120, wobble=0.35)
        cv2.polylines(mask, [trunk.reshape(-1, 1, 2)], False, 1, 1, lineType=cv2.LINE_AA)
        for idx in rng.choice(len(trunk), size=min(3, len(trunk)), replace=False):
            branch = _random_walk(
                rng, trunk[idx].astype(np.float64), rng.uniform(0, 2 * np.pi), 12, diag / 150, wobble=0.4
            )
            cv2.polylines(mask, [branch.reshape(-1, 1, 2)], False, 1, 1, lineType=cv2.LINE_AA)

    for _ in range(int(rng.integers(cfg.n_blotches[0], cfg.n_blotches[1] + 1))):
        center = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        r = _uniform(rng, cfg.blotch_radius_frac) * diag
        axes = (max(1, int(r)), max(1, int(r * rng.uniform(0.4, 1.0))))
        cv2.ellipse(mask, center, axes, rng.uniform(0, 180), 0, 360, 1, -1)

    out = mask.astype(bool)
    if out.mean() > cfg.max_hole_fraction:  # rare: resample rather than clip shapes
        return random_damage_mask(h, w, rng, cfg)
    return out


def apply_damage(img: FloatImage, mask: BoolMask, rng: np.random.Generator) -> FloatImage:
    """Paint damaged pixels the way scratches look on scans (mostly bright, some dark)."""
    _check(img)
    assert mask.shape == img.shape[:2], "mask/image size mismatch"
    out = img.copy()
    base = 0.92 if rng.random() < 0.8 else 0.08
    fill = base + rng.normal(0, 0.03, size=(int(mask.sum()), 1)).astype(np.float32)
    out[mask] = np.clip(fill, 0, 1)
    return out


# --------------------------------------------------------------------------- pair builders


@dataclass(frozen=True)
class DenoiseConfig:
    """Ranges for denoising inputs (``[0,1]`` units)."""

    gaussian_sigma: Range = (2 / 255, 30 / 255)
    grain_strength: Range = (0.0, 0.10)
    grain_size: Range = (0.5, 1.5)
    p_jpeg: float = 0.4
    jpeg_quality: tuple[int, int] = (50, 95)


@dataclass(frozen=True)
class SRConfig:
    """Ranges for super-resolution inputs."""

    scale: int = 4
    blur_sigma: Range = (0.2, 1.6)
    noise_sigma: Range = (0.0, 8 / 255)
    p_jpeg: float = 0.4
    jpeg_quality: tuple[int, int] = (60, 95)


@dataclass(frozen=True)
class Pair:
    """A degraded input with its clean target (and optional hole mask)."""

    inp: FloatImage
    target: FloatImage
    mask: BoolMask | None = None


def make_denoise_pair(clean: FloatImage, rng: np.random.Generator, cfg: DenoiseConfig = DenoiseConfig()) -> Pair:
    """Grain + Gaussian noise (+ optional JPEG) on a clean image."""
    _check(clean)
    x = add_film_grain(clean, _uniform(rng, cfg.grain_strength), _uniform(rng, cfg.grain_size), rng)
    x = add_gaussian_noise(x, _uniform(rng, cfg.gaussian_sigma), rng)
    if rng.random() < cfg.p_jpeg:
        x = jpeg_compress(x, int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1)))
    return Pair(inp=x, target=clean)


def make_sr_pair(clean: FloatImage, rng: np.random.Generator, cfg: SRConfig = SRConfig()) -> Pair:
    """Blur -> bicubic downsample -> noise (-> JPEG); target is the cropped clean image."""
    _check(clean)
    hr = crop_to_multiple(clean, cfg.scale)
    lr = downsample(gaussian_blur(hr, _uniform(rng, cfg.blur_sigma)), cfg.scale)
    lr = add_gaussian_noise(lr, _uniform(rng, cfg.noise_sigma), rng)
    if rng.random() < cfg.p_jpeg:
        lr = jpeg_compress(lr, int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1)))
    return Pair(inp=lr, target=hr)


def make_colorization_pair(clean: FloatImage) -> Pair:
    """Grey (L-only) input with the original colour image as target."""
    _check(clean)
    return Pair(inp=to_grayscale(clean), target=clean)


def make_inpaint_pair(
    clean: FloatImage,
    rng: np.random.Generator,
    external_mask: BoolMask | None = None,
    cfg: DamageConfig = DamageConfig(),
) -> Pair:
    """Damage a clean image with a procedural or supplied (e.g. NVIDIA) mask.

    Args:
        clean: Clean image.
        rng: Random generator.
        external_mask: Optional mask (True = hole) resized to the image if needed.
        cfg: Procedural damage ranges (used when ``external_mask`` is None).

    Returns:
        Pair whose ``mask`` marks the damaged pixels.
    """
    _check(clean)
    h, w = clean.shape[:2]
    if external_mask is None:
        mask = random_damage_mask(h, w, rng, cfg)
    else:
        resized = cv2.resize(external_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = resized.astype(bool)
    return Pair(inp=apply_damage(clean, mask, rng), target=clean, mask=mask)


def resize_max_side(img: FloatImage, max_side: int) -> FloatImage:
    """Area-downscale so the longest side is at most ``max_side`` (never upscales)."""
    _check(img)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img.copy()
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def random_crop(img: FloatImage, size: int, rng: np.random.Generator) -> FloatImage:
    """Random ``size x size`` crop; reflect-pads images smaller than ``size``."""
    _check(img)
    h, w = img.shape[:2]
    if h < size or w < size:
        padded = cv2.copyMakeBorder(
            img, 0, max(0, size - h), 0, max(0, size - w), cv2.BORDER_REFLECT_101
        )
        img = np.asarray(padded, dtype=np.float32)
        h, w = img.shape[:2]
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return np.ascontiguousarray(img[y : y + size, x : x + size])
