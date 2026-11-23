"""Pack the curated dataset into one tar for Google Drive -> Colab transfer.

Reading thousands of JPEGs through the Drive FUSE mount is slow, so the Colab notebooks
copy this single archive to the VM's local disk and extract it there
(:func:`restoration.colab.setup`). Only images that survive curation are included,
plus the evaluation pools and inpainting masks.

Usage:
    python scripts/pack_dataset.py                       # -> $AAI521_DATA_ROOT/aai521_dataset.tar
    python scripts/pack_dataset.py --out D:/aai521_dataset.tar
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from restoration.config import data_root, logs_dir, processed_dir, raw_dir
from restoration.utils.logging import setup_logging

logger = logging.getLogger("pack_dataset")
EVAL_SOURCES = ("loc_fsa_bw", "div2k_valid", "nvidia_masks")


def files_to_pack() -> list[Path]:
    """Absolute paths of every file that goes into the archive."""
    root = data_root()
    files = [processed_dir() / "splits.csv", processed_dir() / "curated.csv"]
    splits = pd.read_csv(processed_dir() / "splits.csv", keep_default_na=False)
    for source, group in splits.groupby("source"):
        files.append(raw_dir(str(source)) / "manifest.csv")
        files.extend(raw_dir(str(source)) / p for p in group["path"])
    for source in EVAL_SOURCES:
        manifest = pd.read_csv(raw_dir(source) / "manifest.csv", keep_default_na=False)
        files.append(raw_dir(source) / "manifest.csv")
        if "status" in manifest.columns:  # mask manifest has no status column
            manifest = manifest[manifest["status"] == "ok"]
        paths = manifest["path"]
        files.extend(raw_dir(source) / p for p in paths)
    missing = [f for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files missing, e.g. {missing[:3]}")
    assert all(f.is_relative_to(root) for f in files)
    return files


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(logs_dir(), "pack_dataset")
    root = data_root()
    out: Path = args.out or root / "aai521_dataset.tar"
    files = files_to_pack()
    size_gb = sum(f.stat().st_size for f in files) / 2**30
    logger.info("Packing %d files (%.2f GB) -> %s", len(files), size_gb, out)
    tmp = out.with_name(out.name + ".part")
    with tarfile.open(tmp, "w") as tar:  # JPEGs are already compressed
        for f in tqdm(files, desc="pack"):
            tar.add(f, arcname=f.relative_to(root).as_posix())
    tmp.replace(out)
    logger.info("Wrote %s (%.2f GB). Upload it to Google Drive.", out, out.stat().st_size / 2**30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
