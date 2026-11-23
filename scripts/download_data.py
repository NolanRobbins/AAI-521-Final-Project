"""Download all project datasets into ``$AAI521_DATA_ROOT/raw/<source>``.

Resumable: images and archives already on disk are skipped, so the script can be
re-run after interruptions. Each source writes ``manifest.csv`` describing its images.

Usage:
    python scripts/download_data.py                          # every source
    python scripts/download_data.py --sources loc_fsa_color documerica
    python scripts/download_data.py --sources documerica --limit 50   # quick smoke test
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

import requests

from restoration.config import DataConfig, load_data_config, logs_dir, raw_dir
from restoration.data import archives, commons, loc
from restoration.data.records import download_records
from restoration.utils.http import build_session
from restoration.utils.logging import setup_logging
from restoration.utils.seed import seed_everything

logger = logging.getLogger("download_data")


def _loc(name: str) -> Callable[[DataConfig, requests.Session, int | None], None]:
    def run(cfg: DataConfig, session: requests.Session, limit: int | None) -> None:
        src = getattr(cfg.sources, name)
        records = loc.crawl_collection(
            session,
            src.collection,
            name,
            limit if limit is not None else src.limit,
            cfg.seed,
        )
        download_records(
            records,
            raw_dir(name),
            session,
            workers=src.workers,
            requests_per_second=src.requests_per_second,
            max_side=src.max_side,
        )

    return run


def _documerica(cfg: DataConfig, session: requests.Session, limit: int | None) -> None:
    src = cfg.sources.documerica
    records = commons.crawl_documerica(
        session,
        src.category,
        "documerica",
        src.max_side,
        limit if limit is not None else src.limit,
        cfg.seed,
    )
    download_records(
        records,
        raw_dir("documerica"),
        session,
        workers=src.workers,
        requests_per_second=src.requests_per_second,
        max_side=src.max_side,
    )


def _nvidia_masks(cfg: DataConfig, session: requests.Session, limit: int | None) -> None:
    out = raw_dir("nvidia_masks")
    extracted = archives.fetch_archive(
        session, str(cfg.sources.nvidia_masks.url), out, "test_mask.zip"
    )
    archives.build_mask_manifest(extracted, out)


def _div2k(cfg: DataConfig, session: requests.Session, limit: int | None) -> None:
    out = raw_dir("div2k_valid")
    extracted = archives.fetch_archive(
        session, str(cfg.sources.div2k_valid.url), out, "DIV2K_valid_HR.zip"
    )
    archives.build_image_manifest(extracted, out, "div2k_valid")


SOURCES: dict[str, Callable[[DataConfig, requests.Session, int | None], None]] = {
    "nvidia_masks": _nvidia_masks,
    "div2k_valid": _div2k,
    "loc_fsa_color": _loc("loc_fsa_color"),
    "loc_fsa_bw": _loc("loc_fsa_bw"),
    "documerica": _documerica,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=list(SOURCES))
    parser.add_argument(
        "--limit", type=int, default=None, help="Override per-source image limit (smoke tests)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Download the requested sources; returns a process exit code."""
    args = parse_args(argv)
    log_path = setup_logging(logs_dir(), "download_data")
    cfg = load_data_config()
    seed_everything(cfg.seed)
    session = build_session(pool_size=16)
    logger.info("Logging to %s", log_path)

    failed: list[str] = []
    for name in args.sources:
        logger.info("=== %s ===", name)
        try:
            SOURCES[name](cfg, session, args.limit)
        except Exception:  # one broken source should not abort the others
            logger.exception("Source %s failed", name)
            failed.append(name)
    if failed:
        logger.error("Failed sources: %s", ", ".join(failed))
        return 1
    logger.info("All sources complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
