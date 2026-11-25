"""Held-out test-set evaluation: baselines vs zero-shot vs fine-tuned, per image.

For every task we build a *fixed* set of test pairs (the test split, degradations seeded
by image index, identical for every method) and score every method on every image.
Per-image rows make paired statistics possible: the fine-tuned vs zero-shot difference
is reported with a paired bootstrap 95 % confidence interval, which is far tighter (and
more honest) than comparing two independent means.

Inpainting rows also carry the mask type (procedural / NVIDIA) and the hole fraction, so
results can be stratified, since classical diffusion-style fill and generative fill behave
very differently on thin scratches vs large holes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from restoration.data import degradations as deg
from restoration.evaluation import baselines, metrics
from restoration.training.datasets import PairDatasetConfig, RestorationPairs, Task

logger = logging.getLogger(__name__)

FloatImage = deg.FloatImage
Method = Callable[[dict[str, Any]], FloatImage]
LOWER_IS_BETTER = {"lpips", "ciede2000"}


@dataclass(frozen=True)
class TestItem:
    """One test pair as numpy arrays plus metadata."""

    index: int
    inp: FloatImage
    target: FloatImage
    mask: np.ndarray | None = None
    mask_type: str = ""


def _hwc(t: torch.Tensor) -> FloatImage:
    return np.ascontiguousarray(t.permute(1, 2, 0).numpy(), dtype=np.float32)


def build_test_items(
    task: Task, limit: int, patch_size: int, seed: int = 521, **kwargs: Any
) -> list[TestItem]:
    """Deterministic test pairs for ``task`` (test split, seeded degradations).

    For inpainting, half the images use procedural damage and half NVIDIA masks; the
    split is decided by index parity so both mask families are equally represented.
    """
    items = []
    if task == "inpaint":
        procedural = RestorationPairs(
            PairDatasetConfig(
                task=task,
                split="test",
                patch_size=patch_size,
                limit=limit,
                seed=seed,
                nvidia_mask_prob=0.0,
            )
        )
        nvidia = RestorationPairs(
            PairDatasetConfig(
                task=task,
                split="test",
                patch_size=patch_size,
                limit=limit,
                seed=seed,
                nvidia_mask_prob=1.0,
            )
        )
        for i in range(len(procedural)):
            ds, kind = (nvidia, "nvidia") if i % 2 else (procedural, "procedural")
            s = ds[i]
            items.append(
                TestItem(i, _hwc(s["inp"]), _hwc(s["target"]), s["mask"][0].numpy() > 0.5, kind)  # type: ignore[arg-type,union-attr]
            )
        return items
    ds = RestorationPairs(
        PairDatasetConfig(
            task=task, split="test", patch_size=patch_size, limit=limit, seed=seed, **kwargs
        )
    )
    for i in range(len(ds)):
        s = ds[i]
        items.append(TestItem(i, _hwc(s["inp"]), _hwc(s["target"])))  # type: ignore[arg-type]
    return items


def score(pred: FloatImage, item: TestItem, task: str, device: torch.device) -> dict[str, float]:
    """All metrics for one prediction."""
    p = torch.from_numpy(pred).permute(2, 0, 1)[None].to(device).clamp(0, 1)
    t = torch.from_numpy(item.target).permute(2, 0, 1)[None].to(device)
    row = {
        "psnr": float(metrics.psnr(p, t)),
        "ssim": float(metrics.ssim(p, t)),
        "lpips": float(metrics.lpips_distance(p, t)),
    }
    if task == "colorize":
        row["ciede2000"] = float(metrics.ciede2000(p, t))
        row["colorfulness"] = float(metrics.colorfulness(p))
        row["colorfulness_gt"] = float(metrics.colorfulness(t))
    if task in ("inpaint", "denoise_inpaint") and item.mask is not None:
        m = torch.from_numpy(item.mask.astype(np.float32))[None, None].to(device)
        row["psnr_hole"] = float(metrics.masked_psnr(p, t, m))
    return row


def evaluate_task(
    task: str, items: list[TestItem], methods: dict[str, Method], device: torch.device
) -> pd.DataFrame:
    """Score every method on every item; one row per (method, image)."""
    rows = []
    for name, method in methods.items():
        start = time.perf_counter()
        for item in items:
            sample = {"inp": item.inp, "mask": item.mask, "index": item.index}
            pred = method(sample)
            assert pred.shape == item.target.shape, f"{name}: {pred.shape} vs {item.target.shape}"
            row: dict[str, Any] = {
                "task": task,
                "method": name,
                "image": item.index,
                **score(pred, item, task, device),
            }
            if item.mask is not None:
                row["mask_type"] = item.mask_type
                row["hole_fraction"] = float(item.mask.mean())
            rows.append(row)
        secs = (time.perf_counter() - start) / max(1, len(items))
        logger.info("%-16s %-24s %.3f s/image", task, name, secs)
        for row in rows[-len(items) :]:
            row["seconds_per_image"] = secs
    return pd.DataFrame(rows)


def paired_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    method_a: str,
    method_b: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of (a - b) over images with a percentile bootstrap 95 % CI."""
    a = frame[frame["method"] == method_a].set_index("image")[metric]
    b = frame[frame["method"] == method_b].set_index("image")[metric]
    diff = (a - b.reindex(a.index)).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarise(frame: pd.DataFrame, metric_cols: list[str] | None = None) -> pd.DataFrame:
    """Mean ± std per method (rows ordered as they appear)."""
    cols = metric_cols or [
        c for c in ("psnr", "ssim", "lpips", "psnr_hole", "ciede2000", "colorfulness") if c in frame
    ]
    order = list(dict.fromkeys(frame["method"]))
    grouped = frame.groupby("method")[cols + ["seconds_per_image"]].agg(["mean", "std"])
    return grouped.loc[order]


def classical_methods(task: str) -> dict[str, Method]:
    """The no-learning reference(s) for a task."""
    if task == "denoise":
        return {
            "noisy input": lambda s: s["inp"],
            "non-local means": lambda s: baselines.nlm_denoise(s["inp"]),
        }
    if task == "super_resolution":
        return {"bicubic": lambda s: baselines.bicubic_upscale(s["inp"])}
    if task == "colorize":
        return {"grey input": lambda s: s["inp"]}
    if task == "inpaint":
        return {"OpenCV Telea": lambda s: baselines.telea_inpaint(s["inp"], s["mask"])}
    raise ValueError(task)


def save(frame: pd.DataFrame, out_dir: Path, task: str) -> Path:
    """Write per-image results to ``out_dir/<task>.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task}.csv"
    frame.to_csv(path, index=False)
    return path
