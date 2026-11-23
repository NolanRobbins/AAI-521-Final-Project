"""Fine-tune one task from a YAML config (also callable from the Colab notebooks).

Usage:
    python scripts/train.py --config configs/train/denoise.yaml
    python scripts/train.py --config configs/train/colorize.yaml --model-key colorize_tiny \\
        --run-name colorize_tiny_local --batch-size 4
    python scripts/train.py --config configs/train/denoise.yaml --max-steps 200 \\
        --train-limit 64 --val-limit 16 --run-name denoise_smoke      # quick sanity run

Re-running with the same ``--run-name`` resumes from ``last_state.pt``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from restoration.config import logs_dir
from restoration.training.sd_inpaint import InpaintLoraConfig, train_inpaint_lora
from restoration.training.trainer import TrainConfig, train
from restoration.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--model-key")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--val-every", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args(argv)

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    task = yaml.safe_load(args.config.read_text(encoding="utf-8"))["task"]
    if task == "inpaint":  # diffusion LoRA has its own objective and loop
        lora_cfg = InpaintLoraConfig.from_yaml(args.config, **overrides)
        setup_logging(logs_dir(), f"train_{lora_cfg.run_name}")
        out = train_inpaint_lora(lora_cfg)
    else:
        cfg = TrainConfig.from_yaml(args.config, **overrides)
        setup_logging(logs_dir(), f"train_{cfg.run_name}")
        out = train(cfg)
    print(f"Run artefacts: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
