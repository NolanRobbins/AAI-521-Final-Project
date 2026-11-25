"""Export the ``denoise_inpaint`` test inputs as 8-bit PNGs for BOPBtL (decisions D12).

Writes ``$AAI521_DATA_ROOT/outputs/bopbtl/inputs/<index:04d>.png`` and ``manifest.csv``.
Use the same ``--limit``/``--seed`` as ``evaluate.py --inpaint-limit``/``--seed``; the
loader warns if they differ.

Usage:
    python scripts/export_bopbtl_inputs.py --limit 8
    bash scripts/run_bopbtl_local.sh
    python scripts/evaluate.py --tasks denoise_inpaint --inpaint-limit 8
"""

from __future__ import annotations

import argparse
import logging
import sys

from restoration.config import data_root, logs_dir
from restoration.evaluation import denoise_inpaint as dni
from restoration.training.datasets import load_split
from restoration.utils.logging import setup_logging

logger = logging.getLogger("export_bopbtl_inputs")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=100, help="= evaluate.py --inpaint-limit")
    parser.add_argument("--seed", type=int, default=521)
    args = parser.parse_args(argv)
    setup_logging(logs_dir(), "export_bopbtl_inputs")
    items = dni.build_items(args.limit, args.seed)
    image_ids = load_split("test", limit=args.limit)["image_id"].astype(str).tolist()
    path = dni.export_inputs(
        items, data_root() / "outputs" / "bopbtl", args.seed, args.limit, image_ids
    )
    logger.info("Exported %d inputs; manifest %s", len(items), path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
