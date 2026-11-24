"""Evaluate classical baselines, zero-shot and fine-tuned models on the held-out test split.

Writes per-image CSVs to ``$AAI521_DATA_ROOT/outputs/eval/<task>.csv``; notebook 03 turns
them into tables, bootstrap confidence intervals and figures.

Usage:
    python scripts/evaluate.py                                  # all tasks + end-to-end
    python scripts/evaluate.py --tasks denoise super_resolution --limit 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from restoration.config import CONFIG_DIR, checkpoint_root, data_root, logs_dir
from restoration.data import degradations as deg
from restoration.data.quality import estimate_noise_sigma
from restoration.evaluation import baselines, harness
from restoration.models import inference, loaders
from restoration.training.datasets import PairDatasetConfig, RestorationPairs
from restoration.utils.device import default_device, empty_cache
from restoration.utils.logging import setup_logging
from restoration.utils.seed import seed_everything

logger = logging.getLogger("evaluate")
TASKS = ("denoise", "super_resolution", "colorize", "inpaint", "end_to_end")


def train_cfg(task: str) -> dict[str, Any]:
    """The training YAML for a task (run name, patch size, options)."""
    with (CONFIG_DIR / "train" / f"{task}.yaml").open(encoding="utf-8") as fh:
        loaded: dict[str, Any] = yaml.safe_load(fh)
    return loaded


def finetuned_dir(task: str) -> Path | None:
    """Fine-tuned run directory if its best checkpoint exists."""
    run: Path = checkpoint_root() / str(train_cfg(task)["run_name"])
    marker = run / ("best_lora" if task == "inpaint" else "best.safetensors")
    if not marker.exists():
        logger.warning("No fine-tuned checkpoint for %s at %s; skipping that row", task, run)
        return None
    return run


def free() -> None:
    """Release cached GPU memory between models."""
    empty_cache()


def run_methods(
    task: str,
    items: list[harness.TestItem],
    factories: dict[str, Callable[[], harness.Method]],
    device: torch.device,
) -> Any:
    """Evaluate classical baselines, then each learned method (loaded one at a time)."""
    frames = [harness.evaluate_task(task, items, harness.classical_methods(task), device)]
    for name, factory in factories.items():
        method = factory()
        frames.append(harness.evaluate_task(task, items, {name: method}, device))
        del method
        free()
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def denoise_methods(device: torch.device) -> dict[str, Callable[[], harness.Method]]:
    def scunet(run: Path | None) -> Callable[[], harness.Method]:
        def make() -> harness.Method:
            model = loaders.load_scunet(device)
            if run is not None:
                loaders.load_finetuned_weights(model, run)
            return lambda s: inference.denoise(model, s["inp"])

        return make

    def drunet() -> harness.Method:
        model = loaders.load_drunet(device)

        @torch.inference_mode()
        def run(s: dict[str, Any]) -> deg.FloatImage:
            gray = np.asarray(cv2.cvtColor(s["inp"], cv2.COLOR_RGB2GRAY), dtype=np.float32)
            sigma = estimate_noise_sigma(gray)
            x, h, w = inference.pad_to_multiple(inference.to_tensor(s["inp"], device), 8)
            return inference.to_image(model(x, sigma)[..., :h, :w])

        return run

    factories = {"DRUNet (zero-shot)": drunet, "SCUNet (zero-shot)": scunet(None)}
    run = finetuned_dir("denoise")
    if run:
        factories["SCUNet (fine-tuned)"] = scunet(run)
    return factories


def sr_methods(device: torch.device) -> dict[str, Callable[[], harness.Method]]:
    def swin(run: Path | None) -> Callable[[], harness.Method]:
        def make() -> harness.Method:
            model = loaders.load_swin2sr(device)
            if run is not None:
                loaders.load_finetuned_weights(model, run)
            return lambda s: inference.super_resolve(model, s["inp"])

        return make

    factories = {"Swin2SR (zero-shot)": swin(None)}
    run = finetuned_dir("super_resolution")
    if run:
        factories["Swin2SR (fine-tuned)"] = swin(run)
    return factories


def colorize_methods(device: torch.device) -> dict[str, Callable[[], harness.Method]]:
    def ddcolor(key: str, run: Path | None) -> Callable[[], harness.Method]:
        def make() -> harness.Method:
            model, size = loaders.load_ddcolor(key, device)
            if run is not None:
                loaders.load_finetuned_weights(model, run)
            return lambda s: inference.colorize(model, s["inp"], size)

        return make

    factories = {
        "DDColor-tiny (zero-shot)": ddcolor("colorize_tiny", None),
        "DDColor-L (zero-shot)": ddcolor("colorize", None),
    }
    run = finetuned_dir("colorize")
    if run:
        factories["DDColor-L (fine-tuned)"] = ddcolor(train_cfg("colorize")["model_key"], run)
    return factories


def inpaint_methods(device: torch.device, steps: int) -> dict[str, Callable[[], harness.Method]]:
    def lama() -> harness.Method:
        model = loaders.load_lama(device)
        return lambda s: inference.inpaint_lama(model, s["inp"], s["mask"])

    def sd(lora: Path | None) -> Callable[[], harness.Method]:
        def make() -> harness.Method:
            pipe = loaders.load_sd_inpaint(device, dtype=torch.float16, lora_dir=lora)
            return lambda s: inference.inpaint_sd(
                pipe, s["inp"], s["mask"], steps=steps, seed=s["index"]
            )

        return make

    factories = {"Big-LaMa (zero-shot)": lama, "SD 1.5 inpaint (zero-shot)": sd(None)}
    run = finetuned_dir("inpaint")
    if run:
        factories["SD 1.5 inpaint + LoRA (fine-tuned)"] = sd(run / "best_lora")
    return factories


def end_to_end_items(limit: int, seed: int) -> list[harness.TestItem]:
    """Compound damage on 1024 px clean crops -> 256 px greyscale, grainy, scratched input.

    Damage is applied at the low resolution (as on a small old print); the target is the
    clean colour crop, fading-corrected when the colorizer was trained that way.
    """
    hr_size, scale = 1024, 4
    ds = RestorationPairs(
        PairDatasetConfig(task="denoise", split="test", patch_size=hr_size, limit=limit, seed=seed)
    )
    correct = bool(train_cfg("colorize").get("correct_target_fading", False))
    items = []
    for i in range(len(ds)):
        rng = np.random.default_rng([seed, 7, i])
        clean = np.ascontiguousarray(ds[i]["target"].permute(1, 2, 0).numpy())  # type: ignore[union-attr]
        lr = deg.downsample(deg.gaussian_blur(clean, 0.8), scale)
        noisy = deg.make_denoise_pair(lr, rng).inp
        damaged = deg.make_inpaint_pair(noisy, rng)
        target = deg.correct_fading(clean) if correct else clean
        items.append(
            harness.TestItem(i, deg.to_grayscale(damaged.inp), target, damaged.mask, "compound")
        )
    return items


def end_to_end_methods(device: torch.device, steps: int) -> dict[str, Callable[[], harness.Method]]:
    from restoration.pipeline import TASK_ORDER, PipelineConfig, RestorationPipeline

    def chain(finetuned: bool) -> Callable[[], harness.Method]:
        def make() -> harness.Method:
            pipe = RestorationPipeline(
                PipelineConfig(use_finetuned=finetuned, inpaint_steps=steps, device=device)
            )
            return lambda s: pipe.run(s["inp"], TASK_ORDER, mask=s["mask"], seed=s["index"]).image

        return make

    return {"pipeline (zero-shot)": chain(False), "pipeline (fine-tuned)": chain(True)}


def classical_chain(s: dict[str, Any]) -> deg.FloatImage:
    """NLM -> Telea -> (no colour) -> bicubic."""
    x = baselines.nlm_denoise(s["inp"])
    x = baselines.telea_inpaint(x, s["mask"])
    return baselines.bicubic_upscale(x)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--limit", type=int, default=200, help="test images per task")
    parser.add_argument("--inpaint-limit", type=int, default=100)
    parser.add_argument("--inpaint-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=521)
    args = parser.parse_args(argv)
    setup_logging(logs_dir(), "evaluate")
    seed_everything(args.seed)
    device = default_device()
    out_dir = data_root() / "outputs" / "eval"

    for task in args.tasks:
        logger.info("=== %s ===", task)
        if task == "denoise":
            items = harness.build_test_items(
                "denoise", args.limit, train_cfg("denoise")["patch_size"], args.seed
            )
            frame = run_methods(task, items, denoise_methods(device), device)
        elif task == "super_resolution":
            items = harness.build_test_items(
                "super_resolution", args.limit, train_cfg(task)["patch_size"], args.seed
            )
            frame = run_methods(task, items, sr_methods(device), device)
        elif task == "colorize":
            c = train_cfg("colorize")
            items = harness.build_test_items(
                "colorize",
                args.limit,
                c["patch_size"],
                args.seed,
                correct_target_fading=bool(c.get("correct_target_fading", False)),
            )
            frame = run_methods(task, items, colorize_methods(device), device)
        elif task == "inpaint":
            items = harness.build_test_items(
                "inpaint", args.inpaint_limit, train_cfg("inpaint")["patch_size"], args.seed
            )
            frame = run_methods(task, items, inpaint_methods(device, args.inpaint_steps), device)
        else:
            items = end_to_end_items(args.inpaint_limit, args.seed)
            import pandas as pd

            classical = harness.evaluate_task(
                task, items, {"classical chain": classical_chain}, device
            )
            learned = [
                harness.evaluate_task(task, items, {name: factory()}, device)
                for name, factory in end_to_end_methods(device, args.inpaint_steps).items()
            ]
            frame = pd.concat([classical, *learned], ignore_index=True)
        path = harness.save(frame, out_dir, task)
        logger.info("Saved %s\n%s", path, harness.summarise(frame).round(4).to_string())
        free()
    return 0


if __name__ == "__main__":
    sys.exit(main())
