"""Load every pretrained model and run it on a synthetically degraded crop.

This is a zero-shot sanity check (no fine-tuning): it proves the checkpoints load with
strict weight matching, inference runs on the available device, and outputs are finite
and correctly shaped. It also prints PSNR vs. the degraded input as a first, rough
signal and writes a side-by-side panel per task to ``$AAI521_DATA_ROOT/outputs/smoke``.

Usage:
    python scripts/smoke_models.py --image path/to/clean.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

from restoration.config import data_root, logs_dir, raw_dir
from restoration.data import degradations as deg
from restoration.models import inference, loaders
from restoration.utils.device import default_device, empty_cache, synchronize
from restoration.utils.logging import setup_logging
from restoration.utils.seed import seed_everything

logger = logging.getLogger("smoke_models")
FloatImage = deg.FloatImage


def _psnr(a: FloatImage, b: FloatImage) -> float:
    return float(peak_signal_noise_ratio(b, a, data_range=1.0))  # type: ignore[no-untyped-call]


def _save_panel(images: list[FloatImage], path: Path) -> None:
    h = min(im.shape[0] for im in images)
    resized = [
        np.asarray(
            Image.fromarray(deg.to_uint8(im)).resize((round(im.shape[1] * h / im.shape[0]), h))
        )
        for im in images
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(resized, axis=1)).save(path)


def _timed(fn: Callable[[], FloatImage]) -> tuple[FloatImage, float]:
    synchronize()
    start = time.perf_counter()
    out = fn()
    synchronize()
    return out, time.perf_counter() - start


def _default_image() -> Path:
    images = sorted((raw_dir("loc_fsa_color") / "images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError("No LoC images found; pass --image or run download_data.py")
    return images[0]


def run(image_path: Path, crop: int, seed: int) -> list[dict[str, float | str]]:
    """Run each task's primary and comparison models on one crop; return metric rows."""
    rng = np.random.default_rng(seed)
    full = deg.to_float(np.asarray(Image.open(image_path).convert("RGB")))
    clean = deg.random_crop(full, crop, rng)
    out_dir = data_root() / "outputs" / "smoke"
    device = default_device()
    rows: list[dict[str, float | str]] = []

    def record(
        task: str, model: str, inp: FloatImage, out: FloatImage, target: FloatImage, secs: float
    ) -> None:
        assert out.shape == target.shape, f"{model}: {out.shape} != {target.shape}"
        assert np.isfinite(out).all(), f"{model}: non-finite output"
        rows.append(
            {
                "task": task,
                "model": model,
                "psnr_in": _psnr(inp, target) if inp.shape == target.shape else float("nan"),
                "psnr_out": _psnr(out, target),
                "seconds": secs,
            }
        )
        _save_panel([inp, out, target], out_dir / f"{task}_{model}.png")

    # Denoise
    pair = deg.make_denoise_pair(clean, rng)
    scunet = loaders.load_scunet(device)
    out, secs = _timed(partial(inference.denoise, scunet, pair.inp))
    record("denoise", "scunet", pair.inp, out, pair.target, secs)
    del scunet

    # Inpaint
    pair = deg.make_inpaint_pair(clean, rng)
    assert pair.mask is not None
    lama = loaders.load_lama(device)
    out, secs = _timed(partial(inference.inpaint_lama, lama, pair.inp, pair.mask))
    record("inpaint", "lama", pair.inp, out, pair.target, secs)
    del lama
    sd = loaders.load_sd_inpaint(device)
    out, secs = _timed(
        partial(inference.inpaint_sd, sd, pair.inp, pair.mask, max_side=crop, steps=25)
    )
    record("inpaint", "sd15_inpaint", pair.inp, out, pair.target, secs)
    del sd

    # Colorize: DDColor relies on global scene context, so use the whole (downscaled)
    # photograph rather than a crop, which may be featureless sky.
    pair = deg.make_colorization_pair(deg.resize_max_side(full, crop))
    for name in ("colorize_tiny", "colorize"):
        model, size = loaders.load_ddcolor(name, device)
        out, secs = _timed(partial(inference.colorize, model, pair.inp, size))
        record("colorize", name, pair.inp, out, pair.target, secs)
        del model

    # Super-resolve (bicubic upsampling is the classical baseline)
    pair = deg.make_sr_pair(clean, rng)
    swin = loaders.load_swin2sr(device)
    out, secs = _timed(partial(inference.super_resolve, swin, pair.inp))
    h, w = pair.target.shape[:2]
    upsampled = cv2.resize(pair.inp, (w, h), interpolation=cv2.INTER_CUBIC)
    bicubic = np.asarray(np.clip(upsampled, 0, 1), dtype=np.float32)
    record("super_resolution", "bicubic", bicubic, bicubic, pair.target, 0.0)
    record("super_resolution", "swin2sr_x4", bicubic, out, pair.target, secs)
    del swin

    empty_cache()
    logger.info("Panels written to %s", out_dir)
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--seed", type=int, default=521)
    args = parser.parse_args(argv)
    setup_logging(logs_dir(), "smoke_models")
    seed_everything(args.seed)
    image = args.image or _default_image()
    logger.info("Image: %s | device: %s", image, default_device())
    rows = run(image, args.crop, args.seed)
    for row in rows:
        logger.info(
            "%-17s %-14s PSNR in %6.2f -> out %6.2f dB  (%.2fs)",
            row["task"],
            row["model"],
            row["psnr_in"],
            row["psnr_out"],
            row["seconds"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
