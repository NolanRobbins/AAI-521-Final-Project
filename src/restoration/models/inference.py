"""Task-level inference on float32 RGB images in ``[0, 1]`` with shape ``(H, W, 3)``.

These functions are shared by evaluation, the restoration pipeline and the web app so
that every consumer pre/post-processes identically. Large scans are processed in
overlapping tiles to bound GPU memory; overlaps are averaged to hide seams.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image
from torch import nn

if TYPE_CHECKING:
    from diffusers import StableDiffusionInpaintPipeline
    from transformers import Swin2SRForImageSuperResolution

FloatImage = NDArray[np.float32]
GENERIC_PROMPT = "an old photograph, high quality, detailed"
SD_MIN_SIDE = 512
BoolMask = NDArray[np.bool_]
TensorFn = Callable[[torch.Tensor], torch.Tensor]


def _check_image(img: FloatImage) -> None:
    assert img.ndim == 3 and img.shape[2] == 3, f"expected HxWx3, got {img.shape}"
    assert img.dtype == np.float32, f"expected float32, got {img.dtype}"


def to_tensor(img: FloatImage, device: torch.device) -> torch.Tensor:
    """``(H, W, 3)`` numpy -> ``(1, 3, H, W)`` tensor on ``device``."""
    _check_image(img)
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].to(device)


def to_image(t: torch.Tensor) -> FloatImage:
    """``(1, 3, H, W)`` tensor -> clipped ``(H, W, 3)`` float32 numpy."""
    arr = t.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray(arr, dtype=np.float32)


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    """Reflect-pad the spatial dims of ``x`` up to a multiple of ``multiple``.

    Returns:
        ``(padded, original_height, original_width)``.
    """
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    if pad_h or pad_w:
        mode = "reflect" if pad_h < h and pad_w < w else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, h, w


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.inference_mode()
def tiled_apply(
    fn: TensorFn, x: torch.Tensor, *, tile: int, overlap: int, scale: int = 1
) -> torch.Tensor:
    """Apply ``fn`` over overlapping tiles and blend by averaging.

    Args:
        fn: Maps ``(1, C, h, w)`` to ``(1, C', h*scale, w*scale)``.
        x: Input ``(1, C, H, W)``.
        tile: Tile size in input pixels.
        overlap: Overlap between neighbouring tiles in input pixels.
        scale: Spatial upscaling factor of ``fn``.

    Returns:
        ``(1, C', H*scale, W*scale)`` output.
    """
    if overlap >= tile:
        raise ValueError("overlap must be smaller than tile")
    _, _, h, w = x.shape
    if h <= tile and w <= tile:
        return fn(x)
    stride = tile - overlap
    out: torch.Tensor | None = None
    weight = torch.zeros((1, 1, h * scale, w * scale), device=x.device)
    for y in _tile_starts(h, tile, stride):
        for xs in _tile_starts(w, tile, stride):
            patch = fn(x[..., y : y + tile, xs : xs + tile]).float()
            if out is None:
                out = torch.zeros((1, patch.shape[1], h * scale, w * scale), device=x.device)
            ys_, xs_ = y * scale, xs * scale
            out[..., ys_ : ys_ + patch.shape[-2], xs_ : xs_ + patch.shape[-1]] += patch
            weight[..., ys_ : ys_ + patch.shape[-2], xs_ : xs_ + patch.shape[-1]] += 1
    assert out is not None
    return out / weight


@torch.inference_mode()
def denoise(model: nn.Module, img: FloatImage, *, tile: int = 512, overlap: int = 32) -> FloatImage:
    """Blind denoising with SCUNet (input/output in ``[0, 1]``)."""
    device = next(model.parameters()).device

    def run(t: torch.Tensor) -> torch.Tensor:
        padded, h, w = pad_to_multiple(t, 64)
        out: torch.Tensor = model(padded)
        return out[..., :h, :w]

    return to_image(tiled_apply(run, to_tensor(img, device), tile=tile, overlap=overlap))


@torch.inference_mode()
def super_resolve(
    model: Swin2SRForImageSuperResolution,
    img: FloatImage,
    *,
    tile: int = 256,
    overlap: int = 16,
) -> FloatImage:
    """Swin2SR super-resolution by ``model.config.upscale``."""
    device = next(model.parameters()).device
    scale = int(model.config.upscale)
    window = int(model.config.window_size)

    def run(t: torch.Tensor) -> torch.Tensor:
        padded, h, w = pad_to_multiple(t, window)
        recon: torch.Tensor = model(pixel_values=padded).reconstruction
        return recon[..., : h * scale, : w * scale]

    x = to_tensor(img, device)
    return to_image(tiled_apply(run, x, tile=tile, overlap=overlap, scale=scale))


def rgb_to_lab(img: FloatImage) -> FloatImage:
    """RGB ``[0,1]`` -> CIELAB (L in ``[0,100]``, a/b roughly ``[-128,127]``)."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)


def lab_to_rgb(lab: FloatImage) -> FloatImage:
    """CIELAB -> RGB clipped to ``[0, 1]``."""
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1).astype(np.float32)


def gray_rgb_from_l(lab: FloatImage) -> FloatImage:
    """Grey RGB image carrying only the L channel (the DDColor input convention)."""
    gray_lab = np.concatenate([lab[..., :1], np.zeros_like(lab[..., 1:])], axis=-1)
    return lab_to_rgb(gray_lab)


@torch.inference_mode()
def colorize(model: nn.Module, img: FloatImage, input_size: int) -> FloatImage:
    """DDColor colorization: predict ab at ``input_size`` and merge with full-res L.

    Only chroma is predicted, so luminance detail of the input is preserved exactly; this
    matters because colorization runs before super-resolution in our pipeline.
    """
    _check_image(img)
    device = next(model.parameters()).device
    h, w = img.shape[:2]
    full_lab = rgb_to_lab(img)
    small = np.asarray(
        cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA), dtype=np.float32
    )
    model_in = to_tensor(gray_rgb_from_l(rgb_to_lab(small)), device)
    ab = model(model_in).float()
    ab = F.interpolate(ab, size=(h, w), mode="bilinear", align_corners=False)
    ab_np = ab[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return lab_to_rgb(np.concatenate([full_lab[..., :1], ab_np], axis=-1))


def _feathered(mask: BoolMask, dilate_px: int, feather_px: int) -> NDArray[np.float32]:
    m = mask.astype(np.uint8)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
        m = np.asarray(cv2.dilate(m, kernel), dtype=np.uint8)
    soft = m.astype(np.float32)
    if feather_px > 0:
        soft = np.asarray(cv2.GaussianBlur(soft, (0, 0), feather_px), dtype=np.float32)
        soft = np.maximum(soft, m.astype(np.float32))  # never fade inside the hole
    return soft[..., None]


def composite(
    original: FloatImage,
    filled: FloatImage,
    mask: BoolMask,
    dilate_px: int = 2,
    feather_px: int = 2,
) -> FloatImage:
    """Paste ``filled`` into ``original`` only inside a feathered ``mask``."""
    alpha = _feathered(mask, dilate_px, feather_px)
    return (original * (1 - alpha) + filled * alpha).astype(np.float32)


@torch.inference_mode()
def inpaint_lama(model: torch.jit.ScriptModule, img: FloatImage, mask: BoolMask) -> FloatImage:
    """Big-LaMa inpainting; ``mask`` is True where pixels are missing."""
    device = next(iter(model.parameters())).device
    x, h, w = pad_to_multiple(to_tensor(img, device), 8)
    m = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)
    m, _, _ = pad_to_multiple(m, 8)
    out = model(x, m)[..., :h, :w]
    return composite(img, to_image(out), mask, dilate_px=0, feather_px=0)


def inpaint_sd(
    pipe: StableDiffusionInpaintPipeline,
    img: FloatImage,
    mask: BoolMask,
    *,
    prompt: str = GENERIC_PROMPT,
    negative_prompt: str = "text, watermark, frame, border, blurry, artifacts",
    max_side: int = 768,
    steps: int = 30,
    guidance_scale: float = 6.0,
    seed: int = 0,
    latent_dilate_px: int = 3,
) -> FloatImage:
    """Stable Diffusion inpainting with composite-back of only the masked region.

    The image is resized so its longest side is ``max_side`` (multiple of 8) for the
    UNet, then the generated content is upsampled and composited into the original so
    unmasked pixels are never altered by the VAE round trip.

    SD's UNet sees the mask at 1/8 resolution, so 1-4 px scratches largely disappear
    from the latent mask and the model re-draws them as lines. The mask is therefore
    dilated by ``latent_dilate_px`` (at working resolution) *for generation only*; when
    compositing, only the true hole (+1 px, 2 px feather) is pasted back, because
    replacing the whole dilated band with VAE-decoded pixels measurably damages intact
    context (val, 12 images: SSIM 0.886 -> 0.915; see docs/decisions.md D8).
    """
    _check_image(img)
    h, w = img.shape[:2]
    # SD 1.5 was trained at 512 px: shrink large images to max_side, enlarge small ones
    # to a 512 px long side (generation quality collapses at much lower resolutions).
    target_long = min(max(max(h, w), SD_MIN_SIDE), max_side)
    scale = target_long / max(h, w)
    work_w, work_h = (max(8, int(round(d * scale / 8)) * 8) for d in (w, h))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    small = np.clip(cv2.resize(img, (work_w, work_h), interpolation=interp), 0, 1)
    small_mask = cv2.resize(
        mask.astype(np.uint8), (work_w, work_h), interpolation=cv2.INTER_NEAREST
    )
    if latent_dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * latent_dilate_px + 1,) * 2)
        small_mask = cv2.dilate(small_mask, kernel)
    runner = cast(Any, pipe)  # diffusers pipelines are untyped
    generator = torch.Generator(device=runner.device).manual_seed(seed)
    result = runner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=Image.fromarray((small * 255).round().astype(np.uint8)),
        mask_image=Image.fromarray(small_mask * 255),
        height=work_h,
        width=work_w,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    filled = cv2.resize(
        np.asarray(result, dtype=np.float32) / 255.0, (w, h), interpolation=cv2.INTER_CUBIC
    )
    return composite(img, np.clip(filled, 0, 1).astype(np.float32), mask, dilate_px=1, feather_px=2)
