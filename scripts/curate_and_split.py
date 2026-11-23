"""Curate training sources and produce leakage-safe train/val/test splits.

Steps (per image of each training source):
    1. Compute no-reference quality features and a perceptual hash.
    2. Flag images failing curation rules (too small, near-greyscale, blurriest/noisiest
       percentiles per source, excessive borders) with a ``drop_reason``.
    3. Record border crop boxes so patch extraction can skip slide mounts.
    4. Group near-duplicates/related frames and split whole groups.

Writes ``$AAI521_DATA_ROOT/processed/curated.csv`` (all images, with ``keep`` and
``drop_reason``) and ``splits.csv`` (kept images only, with ``split``).

Usage:
    python scripts/curate_and_split.py [--workers 8]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from restoration.config import load_data_config, logs_dir, processed_dir, raw_dir
from restoration.data import degradations as deg
from restoration.data import quality, splits
from restoration.data.curation import drop_reasons
from restoration.data.records import load_manifest
from restoration.utils.logging import setup_logging

logger = logging.getLogger("curate_and_split")


def _features(path: Path) -> dict[str, float | int | str]:
    """Quality features + pHash for one image (runs in a worker process)."""
    img = deg.to_float(np.asarray(Image.open(path).convert("RGB")))
    row: dict[str, float | int | str] = dict(quality.compute_features(img).as_dict())
    row["phash_int"] = str(splits.phash_int(path))
    return row


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    setup_logging(logs_dir(), "curate_and_split")
    cfg = load_data_config()

    frames = []
    for source in cfg.curation.train_sources:
        src_dir = raw_dir(source)
        manifest = load_manifest(src_dir)
        paths = [src_dir / p for p in manifest["path"]]
        logger.info("%s: computing features for %d images", source, len(paths))
        feats = Parallel(n_jobs=args.workers)(delayed(_features)(p) for p in paths)
        feat_frame = pd.DataFrame(feats).drop(columns=["width", "height"])
        frames.append(pd.concat([manifest.reset_index(drop=True), feat_frame], axis=1))
    curated = pd.concat(frames, ignore_index=True)

    curated["drop_reason"] = drop_reasons(curated, cfg.curation)
    curated["keep"] = curated["drop_reason"] == ""
    out_dir = processed_dir()
    curated.to_csv(out_dir / "curated.csv", index=False)
    summary = curated.groupby(["source", "drop_reason"]).size().unstack(fill_value=0)
    logger.info('Curation summary ("" = kept):\n%s', summary.to_string())

    kept = curated[curated["keep"]].reset_index(drop=True)
    hashes = [int(h) for h in kept["phash_int"]]
    split_frame = splits.make_split_frame(
        kept,
        hashes,
        cfg.splits.phash_hamming_threshold,
        cfg.splits.val_fraction,
        cfg.splits.test_fraction,
        cfg.seed,
    )
    split_frame.to_csv(out_dir / "splits.csv", index=False)
    table = split_frame.groupby(["source", "split"]).size().unstack(fill_value=0)
    n_groups = split_frame["group"].nunique()
    logger.info("%d kept images in %d groups; split sizes:\n%s", len(kept), n_groups, table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
