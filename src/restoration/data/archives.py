"""Single-archive datasets: NVIDIA irregular masks and DIV2K validation HR."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from restoration.utils.http import download_file, extract_zip

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def fetch_archive(session: requests.Session, url: str, out_dir: Path, archive_name: str) -> Path:
    """Download (once) and extract a zip archive.

    Args:
        session: HTTP session.
        url: Archive URL.
        out_dir: Source directory; the zip is kept in ``out_dir/_archive``.
        archive_name: File name for the downloaded zip.

    Returns:
        Directory the archive was extracted into (``out_dir/extracted``).
    """
    archive = download_file(session, url, out_dir / "_archive" / archive_name, timeout=600)
    logger.info("Archive ready: %s (%.1f MB)", archive, archive.stat().st_size / 2**20)
    return extract_zip(archive, out_dir / "extracted")


def list_images(root: Path) -> list[Path]:
    """Recursively list image files under ``root`` in sorted order."""
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def hole_ratio(mask_path: Path) -> float:
    """Fraction of hole pixels in a binary mask.

    The NVIDIA masks store valid pixels as white (255) and holes as black (0); we treat
    the minority value as the hole as a guard against inverted variants.

    Args:
        mask_path: Path to a mask image.

    Returns:
        Hole fraction in ``[0, 0.5]`` under the minority rule.
    """
    arr = np.asarray(Image.open(mask_path).convert("L")) > 127
    white = float(arr.mean())
    return min(white, 1.0 - white)


def build_mask_manifest(extracted: Path, out_dir: Path) -> pd.DataFrame:
    """Index the NVIDIA masks with their hole ratio for ratio-stratified sampling."""
    paths = list_images(extracted)
    if not paths:
        raise FileNotFoundError(f"No mask images found under {extracted}")
    rows = [
        {"mask_id": p.stem, "path": str(p.relative_to(out_dir)), "hole_ratio": hole_ratio(p)}
        for p in tqdm(paths, desc="masks")
    ]
    frame = pd.DataFrame(rows)
    frame["ratio_bin"] = pd.cut(
        frame["hole_ratio"], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], include_lowest=True
    ).astype(str)
    frame.to_csv(out_dir / "manifest.csv", index=False)
    logger.info("Masks: %d indexed; bins:\n%s", len(frame), frame["ratio_bin"].value_counts())
    return frame


def build_image_manifest(extracted: Path, out_dir: Path, source: str) -> pd.DataFrame:
    """Index an extracted image dataset (e.g. DIV2K) with image sizes."""
    rows = []
    for p in tqdm(list_images(extracted), desc=source):
        with Image.open(p) as img:
            width, height = img.size
        rows.append(
            {
                "image_id": f"{source}_{p.stem}",
                "source": source,
                "path": str(p.relative_to(out_dir)),
                "width": width,
                "height": height,
                "status": "ok",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "manifest.csv", index=False)
    logger.info("%s: %d images indexed", source, len(frame))
    return frame
