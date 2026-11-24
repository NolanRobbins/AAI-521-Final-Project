"""Classical (non-learned) baselines, one per task.

These are the "before deep learning" reference points in the comparison tables:

* **Denoise** -- non-local means (Buades et al., 2005), with the filter strength set
  from the Immerkaer noise estimate so it adapts per image.
* **Super-resolution** -- bicubic interpolation (Keys, 1981).
* **Colorize** -- none exists without learning; the grey input is the floor.
* **Inpaint** -- Telea's fast-marching method (Telea, 2004).
"""

from __future__ import annotations

import cv2
import numpy as np

from restoration.data import degradations as deg
from restoration.data.quality import estimate_noise_sigma

FloatImage = deg.FloatImage


def nlm_denoise(img: FloatImage) -> FloatImage:
    """Coloured non-local means with ``h`` scaled to the estimated noise level."""
    gray = np.asarray(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), dtype=np.float32)
    sigma = estimate_noise_sigma(gray) * 255
    h = float(np.clip(1.2 * sigma, 3, 30))
    out = cv2.fastNlMeansDenoisingColored(deg.to_uint8(img), None, h, h, 7, 21)
    return deg.to_float(np.asarray(out, dtype=np.uint8))


def bicubic_upscale(img: FloatImage, scale: int = 4) -> FloatImage:
    """Bicubic upsampling by ``scale``."""
    h, w = img.shape[:2]
    out = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return np.asarray(np.clip(out, 0, 1), dtype=np.float32)


def telea_inpaint(img: FloatImage, mask: deg.BoolMask, radius: int = 3) -> FloatImage:
    """OpenCV Telea inpainting (``mask`` True = missing)."""
    out = cv2.inpaint(deg.to_uint8(img), mask.astype(np.uint8), radius, cv2.INPAINT_TELEA)
    return deg.to_float(np.asarray(out, dtype=np.uint8))
