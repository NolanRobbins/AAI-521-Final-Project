"""Image-quality metrics on batched ``(N, 3, H, W)`` tensors in ``[0, 1]``.

Metric choice per task (see the report's metrics section):

* **PSNR** -- pixel fidelity; the standard for denoising/SR but it rewards blur.
* **SSIM** -- local structure (Wang et al., 2004), 11x11 Gaussian window, sigma 1.5.
* **LPIPS** -- learned perceptual distance (Zhang et al., 2018, AlexNet); tracks human
  judgement far better for generative outputs (inpainting, colorization).
* **Masked PSNR** -- PSNR restricted to hole pixels, so inpainting scores are not
  dominated by the untouched region.
* **Colourfulness** and **CIEDE2000** -- colorization has many plausible answers, so we
  report how colourful outputs are relative to ground truth, and perceptual colour
  difference, rather than PSNR alone.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import deltaE_ciede2000, rgb2lab

EPS = 1e-10


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB, shape ``(N,)``."""
    mse = (pred.float() - target.float()).pow(2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / (mse + EPS))


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR over pixels where ``mask`` (``(N,1,H,W)``, 1 = hole) is set."""
    sq = (pred.float() - target.float()).pow(2) * mask
    denom = mask.flatten(1).sum(1) * pred.shape[1]
    mse = sq.flatten(1).sum(1) / denom.clamp_min(1)
    return 10 * torch.log10(1.0 / (mse + EPS))


def _gaussian_window(size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    return g.expand(channels, 1, size, size).contiguous()


def ssim(
    pred: torch.Tensor, target: torch.Tensor, window: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    """Per-image SSIM (mean over channels and valid positions), shape ``(N,)``."""
    pred, target = pred.float(), target.float()
    c = pred.shape[1]
    w = _gaussian_window(window, sigma, c, pred.device)
    c1, c2 = 0.01**2, 0.03**2

    def filt(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, w, groups=c)

    mu_x, mu_y = filt(pred), filt(target)
    sxx = filt(pred * pred) - mu_x**2
    syy = filt(target * target) - mu_y**2
    sxy = filt(pred * target) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2)
    return (num / den).flatten(1).mean(1)


@lru_cache(maxsize=2)
def _lpips_model(device: str) -> Any:
    import lpips

    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def lpips_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image LPIPS (AlexNet), shape ``(N,)``; lower is better. Differentiable."""
    model = _lpips_model(str(pred.device))
    out: torch.Tensor = model(pred.float() * 2 - 1, target.float() * 2 - 1)
    return out.flatten()


def colorfulness(img: torch.Tensor) -> torch.Tensor:
    """Hasler & Suesstrunk colourfulness per image (0-255 scale), shape ``(N,)``."""
    rgb = img.float() * 255
    rg = rgb[:, 0] - rgb[:, 1]
    yb = 0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2]
    std = torch.sqrt(rg.flatten(1).var(1) + yb.flatten(1).var(1))
    mean = torch.sqrt(rg.flatten(1).mean(1) ** 2 + yb.flatten(1).mean(1) ** 2)
    return std + 0.3 * mean


def ciede2000(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean CIEDE2000 colour difference per image, shape ``(N,)`` (CPU, numpy-based)."""
    out = []
    for p, t in zip(pred.detach().float().cpu(), target.detach().float().cpu(), strict=True):
        lab_p = rgb2lab(p.permute(1, 2, 0).clamp(0, 1).numpy())
        lab_t = rgb2lab(t.permute(1, 2, 0).clamp(0, 1).numpy())
        out.append(float(np.mean(deltaE_ciede2000(lab_t, lab_p))))  # type: ignore[no-untyped-call]
    return torch.tensor(out)


# ----------------------------------------------------------------- differentiable Lab


_D65 = (0.950456, 1.0, 1.088754)
_XYZ_TO_RGB = torch.tensor(
    [
        [3.240479, -1.537150, -0.498535],
        [-0.969256, 1.875992, 0.041556],
        [0.055648, -0.204043, 1.057311],
    ]
)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """Differentiable CIELAB (``L`` 0-100, ``a``/``b`` ~ +-128) -> sRGB in ``[0, 1]``.

    Matches OpenCV's float conversion (D65 white point), so losses can be computed in
    RGB space on DDColor's ``ab`` predictions.
    """
    lum, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (lum + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def finv(f: torch.Tensor) -> torch.Tensor:
        return torch.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29))

    xyz = torch.stack([finv(fx) * _D65[0], finv(fy) * _D65[1], finv(fz) * _D65[2]], dim=1)
    rgb_lin = torch.einsum("ij,njhw->nihw", _XYZ_TO_RGB.to(lab.device, lab.dtype), xyz)
    rgb_lin = rgb_lin.clamp_min(0)
    srgb = torch.where(
        rgb_lin <= 0.0031308, 12.92 * rgb_lin, 1.055 * rgb_lin.clamp_min(1e-8) ** (1 / 2.4) - 0.055
    )
    return srgb.clamp(0, 1)


def rgb_to_l(rgb: torch.Tensor) -> torch.Tensor:
    """CIELAB L channel (0-100) of sRGB ``[0,1]`` images, shape ``(N,1,H,W)``."""
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = 0.212671 * lin[:, 0] + 0.715160 * lin[:, 1] + 0.072169 * lin[:, 2]
    fy = torch.where(
        y > (6 / 29) ** 3, y.clamp_min(1e-8) ** (1 / 3), y / (3 * (6 / 29) ** 2) + 4 / 29
    )
    return (116 * fy - 16)[:, None]
