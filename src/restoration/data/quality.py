"""Per-image quality features for EDA and curation.

These are cheap, no-reference statistics used to (a) describe the corpus in the EDA and
(b) filter scans that are poor ground truth (blurry, very noisy, near-greyscale colour
slides, tiny derivatives). Learned no-reference scores (NIQE/BRISQUE) are added in the
EDA notebook via ``pyiqa`` because they need model weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

FloatImage = NDArray[np.float32]


def laplacian_variance(gray: NDArray[np.float32]) -> float:
    """Variance of the Laplacian -- a standard focus/sharpness measure (Pech-Pacheco 2000)."""
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def estimate_noise_sigma(gray: NDArray[np.float32]) -> float:
    """Fast noise std estimate (Immerkaer, 1996), in the units of ``gray``.

    Convolves with a Laplacian-difference kernel that cancels image structure to first
    order, leaving mostly noise.
    """
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = cv2.filter2D(gray, cv2.CV_32F, kernel)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(response).sum() / (6.0 * (w - 2) * (h - 2)))


def colorfulness(img: FloatImage) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric on ``[0,1]`` RGB (scaled to 0-255)."""
    rgb = img.astype(np.float32) * 255.0
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    std = np.hypot(rg.std(), yb.std())
    mean = np.hypot(rg.mean(), yb.mean())
    return float(std + 0.3 * mean)


def border_extent(
    gray: NDArray[np.float32], dark: float = 0.12, min_std: float = 0.04
) -> tuple[int, int, int, int]:
    """Detect uniform dark borders (slide mounts, film edges) on each side.

    A row/column counts as border while its mean is below ``dark`` or it is nearly
    uniform. Scanning stops at the first content row/column from each edge.

    Args:
        gray: Luma image in ``[0,1]``.
        dark: Mean-intensity threshold for border rows/columns.
        min_std: Rows/columns with lower std are treated as border.

    Returns:
        ``(top, bottom, left, right)`` border thickness in pixels.
    """

    def scan(lines: NDArray[np.float32]) -> int:
        means, stds = lines.mean(axis=1), lines.std(axis=1)
        is_border = (means < dark) | (stds < min_std)
        limit = len(is_border) // 4  # never crop more than a quarter per side
        count = 0
        while count < limit and is_border[count]:
            count += 1
        return count

    return scan(gray), scan(gray[::-1]), scan(gray.T), scan(gray.T[::-1])


@dataclass(frozen=True)
class QualityFeatures:
    """No-reference statistics for one image (computed at a fixed analysis size)."""

    width: int
    height: int
    aspect: float
    mean_luma: float
    std_luma: float
    sharpness: float
    noise_sigma: float
    colorfulness: float
    border_top: int
    border_bottom: int
    border_left: int
    border_right: int

    def as_dict(self) -> dict[str, float | int]:
        """Plain dict for DataFrame construction."""
        return asdict(self)


def compute_features(img: FloatImage, analysis_side: int = 1024) -> QualityFeatures:
    """Compute :class:`QualityFeatures` for an RGB ``[0,1]`` image.

    Sharpness and noise depend on scale, so statistics are measured after resizing the
    longest side to ``analysis_side``; border sizes are reported at original scale.

    Args:
        img: RGB image ``(H, W, 3)`` float32 in ``[0,1]``.
        analysis_side: Longest side used for scale-dependent statistics.

    Returns:
        Feature record.
    """
    h, w = img.shape[:2]
    scale = min(1.0, analysis_side / max(h, w))
    small = img
    if scale < 1:
        size = (round(w * scale), round(h * scale))
        small = np.asarray(cv2.resize(img, size, interpolation=cv2.INTER_AREA), dtype=np.float32)
    gray = np.asarray(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), dtype=np.float32)
    top, bottom, left, right = (round(v / scale) for v in border_extent(gray))
    return QualityFeatures(
        width=w,
        height=h,
        aspect=w / h,
        mean_luma=float(gray.mean()),
        std_luma=float(gray.std()),
        sharpness=laplacian_variance(gray),
        noise_sigma=estimate_noise_sigma(gray),
        colorfulness=colorfulness(small),
        border_top=top,
        border_bottom=bottom,
        border_left=left,
        border_right=right,
    )
