"""The ``denoise_inpaint`` test task and its Bringing Old Photos Back to Life baseline.

BOPBtL (Wan et al., 2020) removes grain and scratches but neither colorizes nor upscales,
so it is compared with our denoise + inpaint steps only. The test items are colour
test-split crops with film grain and then scratches/holes, quantized to 8 bit, because
BOPBtL runs outside this environment on exported 8-bit PNGs and every method must see the
same input bytes. BOPBtL's restored PNGs are read back by :func:`bopbtl_method` and
scored like any other method (decisions D12).

Layout under ``$AAI521_DATA_ROOT/outputs/bopbtl/``::

    inputs/<index:04d>.png    written by scripts/export_bopbtl_inputs.py
    manifest.csv              index, height, width, seed, limit, image_id
    outputs/<index:04d>.png   written by scripts/run_bopbtl_local.sh
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from restoration.config import CONFIG_DIR
from restoration.data import degradations as deg
from restoration.evaluation.harness import Method, TestItem, build_test_items

logger = logging.getLogger(__name__)


def build_items(limit: int, seed: int = 521) -> list[TestItem]:
    """Deterministic grain + damage test items at the inpainting working size.

    Crops, masks and the procedural/NVIDIA split by index parity come from the
    ``inpaint`` test items. Grain is applied to the clean crop, then the same mask's
    damage on top (scratches lie on the grainy print), all seeded by ``(seed, index)``.

    Args:
        limit: Number of test-split images.
        seed: Base seed.

    Returns:
        Items whose ``inp`` is an exact multiple of 1/255 and whose target is the clean crop.
    """
    with (CONFIG_DIR / "train" / "inpaint.yaml").open(encoding="utf-8") as fh:
        size = int(yaml.safe_load(fh)["patch_size"])
    items = []
    for it in build_test_items("inpaint", limit, size, seed):
        rng = np.random.default_rng([seed, 8, it.index])
        noisy = deg.make_denoise_pair(it.target, rng).inp
        damaged = deg.make_inpaint_pair(noisy, rng, external_mask=it.mask).inp
        inp = deg.to_float(deg.to_uint8(damaged))  # what BOPBtL reads from the PNG
        items.append(TestItem(it.index, inp, it.target, it.mask, it.mask_type))
    return items


def export_inputs(
    items: list[TestItem], root: Path, seed: int, limit: int, image_ids: list[str]
) -> Path:
    """Write ``inputs/<index:04d>.png`` and ``manifest.csv`` under ``root``.

    Returns:
        Path of the manifest.
    """
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for stale in inputs.glob("*.png"):  # BOPBtL processes the whole folder
        stale.unlink()
    rows = []
    for it, image_id in zip(items, image_ids, strict=True):
        Image.fromarray(deg.to_uint8(it.inp)).save(inputs / f"{it.index:04d}.png")
        h, w = it.inp.shape[:2]
        rows.append(
            {
                "index": it.index,
                "height": h,
                "width": w,
                "seed": seed,
                "limit": limit,
                "image_id": image_id,
            }
        )
    path = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def bopbtl_method(root: Path, items: list[TestItem], seed: int, limit: int) -> Method | None:
    """Loader for BOPBtL's restored images, or None (with a warning) if any are missing.

    Images whose size differs from the input are resized back to it (bicubic), since
    BOPBtL's ``full_size`` mode resizes to a multiple of 16; each one is logged.
    """
    out_dir = root / "outputs"
    missing = [it.index for it in items if not (out_dir / f"{it.index:04d}.png").is_file()]
    if missing:
        logger.warning(
            "BOPBtL outputs missing for %d/%d images in %s (first: %04d); skipping that row. "
            "Run scripts/export_bopbtl_inputs.py then scripts/run_bopbtl_local.sh.",
            len(missing),
            len(items),
            out_dir,
            missing[0],
        )
        return None
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        logger.warning("No %s; cannot check BOPBtL inputs match this run", manifest)
    else:
        m = pd.read_csv(manifest)
        if (int(m["seed"].iloc[0]), int(m["limit"].iloc[0])) != (seed, limit):
            logger.warning(
                "BOPBtL inputs were exported with seed=%s limit=%s but this run uses "
                "seed=%s limit=%s; outputs may not match these test items",
                m["seed"].iloc[0],
                m["limit"].iloc[0],
                seed,
                limit,
            )
        oldest = min((out_dir / f"{it.index:04d}.png").stat().st_mtime for it in items)
        if oldest < manifest.stat().st_mtime:
            logger.warning(
                "Some BOPBtL outputs predate %s; rerun BOPBtL on the new export", manifest
            )

    def run(sample: dict[str, Any]) -> deg.FloatImage:
        index = int(sample["index"])
        out = deg.to_float(np.asarray(Image.open(out_dir / f"{index:04d}.png").convert("RGB")))
        h, w = sample["inp"].shape[:2]
        if out.shape[:2] != (h, w):
            logger.warning(
                "BOPBtL %04d.png is %dx%d, input %dx%d; resized back (bicubic)",
                index,
                *out.shape[:2],
                h,
                w,
            )
            out = np.clip(cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)
        return np.ascontiguousarray(out, dtype=np.float32)

    return run
