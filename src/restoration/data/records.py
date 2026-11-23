"""Image records, manifests and a parallel, resumable image downloader.

Every source crawler produces a list of :class:`ImageRecord`. The downloader fetches
them concurrently, validates each file decodes as an image, and writes a manifest CSV
that downstream curation/splitting reads. Re-running skips files already on disk.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from restoration.utils.http import RateLimiter, download_file

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.csv"


@dataclass
class ImageRecord:
    """Metadata for one source image.

    Attributes:
        image_id: Stable, filesystem-safe identifier unique within the source.
        source: Source name (e.g. ``"loc_fsa_color"``).
        url: Primary download URL.
        fallback_urls: URLs tried in order if ``url`` fails.
        title: Caption/title from the archive.
        date: Date string as provided by the archive.
        creator: Photographer, if known.
        license: Rights statement.
        group_hint: Archive-level grouping key (call number / NARA id) used to keep
            related frames in the same split.
        page_url: Human-readable catalog page.
        path: Image path relative to the source directory (set by the downloader).
        width: Width of the downloaded image in pixels (set by the downloader).
        height: Height of the downloaded image in pixels (set by the downloader).
        status: ``"ok"``, ``"failed"`` or ``"pending"``.
    """

    image_id: str
    source: str
    url: str
    fallback_urls: tuple[str, ...] = ()
    title: str = ""
    date: str = ""
    creator: str = ""
    license: str = ""
    group_hint: str = ""
    page_url: str = ""
    path: str = ""
    width: int = 0
    height: int = 0
    status: str = "pending"
    extra: dict[str, str] = field(default_factory=dict)


def _verify_image(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` if ``path`` decodes as an image, else raise."""
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        return img.size


def downscale_in_place(path: Path, max_side: int, jpeg_quality: int = 95) -> tuple[int, int]:
    """Shrink an image so its longest side is ``max_side``, overwriting it as JPEG.

    Area interpolation is used because it averages source pixels, which suppresses the
    8x8 block artifacts of the moderately compressed LoC IIIF renders.

    Args:
        path: Image file to rewrite.
        max_side: Target longest side; smaller images are left untouched.
        jpeg_quality: Quality for the re-encoded JPEG.

    Returns:
        Final ``(width, height)``.
    """
    with Image.open(path) as img:
        width, height = img.size
        if max(width, height) <= max_side:
            return width, height
        rgb = np.asarray(img.convert("RGB"))
    scale = max_side / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
    tmp = path.with_name(path.name + ".resize")
    Image.fromarray(resized).save(tmp, format="JPEG", quality=jpeg_quality, subsampling=0)
    tmp.replace(path)
    return new_size


def _fetch_one(
    record: ImageRecord,
    session: requests.Session,
    images_dir: Path,
    rate_limiter: RateLimiter,
    max_side: int | None,
) -> ImageRecord:
    dest = images_dir / f"{record.image_id}.jpg"
    errors: list[str] = []
    for url in (record.url, *record.fallback_urls):
        try:
            download_file(session, url, dest, rate_limiter=rate_limiter)
            record.width, record.height = _verify_image(dest)
            if max_side is not None:
                record.width, record.height = downscale_in_place(dest, max_side)
        except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
            errors.append(f"{url}: {exc}")
            dest.unlink(missing_ok=True)
            continue
        record.path = f"{images_dir.name}/{dest.name}"
        record.status = "ok"
        return record
    record.status = "failed"
    logger.warning("Failed %s after %d url(s): %s", record.image_id, len(errors), errors[-1])
    return record


def download_records(
    records: Sequence[ImageRecord],
    out_dir: Path,
    session: requests.Session,
    *,
    workers: int,
    requests_per_second: float,
    max_side: int | None = None,
) -> pd.DataFrame:
    """Download images for ``records`` into ``out_dir/images`` and write a manifest.

    Args:
        records: Records to fetch; ``image_id`` must be unique.
        out_dir: Source directory (``<data_root>/raw/<source>``).
        session: HTTP session.
        workers: Concurrent download threads.
        requests_per_second: Global request budget shared by all threads.
        max_side: If set, downscale images whose longest side exceeds this.

    Returns:
        The manifest as a DataFrame (also written to ``out_dir/manifest.csv``).
    """
    ids = [r.image_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image_id values in records")
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(requests_per_second)

    done: list[ImageRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_one, r, session, images_dir, limiter, max_side) for r in records]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=out_dir.name):
            done.append(fut.result())

    manifest = records_to_frame(done)
    manifest_path = out_dir / MANIFEST_NAME
    manifest.to_csv(manifest_path, index=False)
    n_ok = int((manifest["status"] == "ok").sum())
    logger.info("%s: %d/%d images ok -> %s", out_dir.name, n_ok, len(manifest), manifest_path)
    return manifest


def records_to_frame(records: Sequence[ImageRecord]) -> pd.DataFrame:
    """Convert records to a manifest DataFrame sorted by ``image_id``."""
    rows = []
    for rec in records:
        row = asdict(rec)
        row["fallback_urls"] = " ".join(rec.fallback_urls)
        extra = row.pop("extra")
        row.update({f"extra_{k}": v for k, v in extra.items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("image_id").reset_index(drop=True)


def load_manifest(source_dir: Path, only_ok: bool = True) -> pd.DataFrame:
    """Read a source manifest, optionally keeping only successfully downloaded rows."""
    path = source_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No manifest at {path}; run scripts/download_data.py first")
    frame = pd.read_csv(path, dtype={"image_id": str, "group_hint": str}, keep_default_na=False)
    if only_ok:
        frame = frame[frame["status"] == "ok"].reset_index(drop=True)
    return frame
