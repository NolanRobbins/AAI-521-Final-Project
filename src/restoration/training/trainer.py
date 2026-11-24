"""Generic supervised fine-tuning loop shared by denoising, super-resolution and colorization.

Design points:

* **Step 0 = zero-shot evaluation.** Validation runs before the first update, so every
  run records the pretrained model's score next to the classical baseline -- the
  "before fine-tuning" column of our comparison table comes for free.
* **Resumable.** ``last_state.pt`` (model + optimiser + scheduler + step) is written at
  every validation, so a disconnected Colab session continues where it stopped.
* **Safe artefacts.** The best weights are saved as ``best.safetensors`` (no pickle),
  alongside ``config.json`` and an append-only ``metrics.csv``.
* **Mixed precision.** bf16 autocast where supported (Ampere+, incl. RTX 50xx / A100),
  fp16 + GradScaler otherwise (Colab T4).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml
from pydantic import BaseModel, Field
from safetensors.torch import save_model
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from restoration.config import checkpoint_root
from restoration.evaluation import metrics
from restoration.training.datasets import PairDatasetConfig, RestorationPairs, Task
from restoration.training.tasks import ADAPTERS, Batch, TaskAdapter
from restoration.utils.device import default_device
from restoration.utils.seed import seed_everything

logger = logging.getLogger(__name__)

Precision = Literal["auto", "bf16", "fp16", "fp32"]


class TrainConfig(BaseModel):
    """Fine-tuning hyper-parameters (one YAML per task in ``configs/train/``)."""

    model_config = {"extra": "forbid"}

    task: Task
    model_key: str
    run_name: str
    patch_size: int = Field(256, gt=0)
    batch_size: int = Field(8, gt=0)
    grad_accum: int = Field(1, gt=0)
    num_workers: int = Field(4, ge=0)
    lr: float = Field(1e-5, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    max_steps: int = Field(5000, gt=0)
    warmup_steps: int = Field(200, ge=0)
    val_every: int = Field(500, gt=0)
    val_limit: int | None = Field(64, gt=0)
    train_limit: int | None = Field(None, gt=0)
    sources: list[str] | None = None
    pixel_loss: Literal["l1", "charbonnier"] = "charbonnier"
    lpips_weight: float = Field(0.0, ge=0)
    colorfulness_weight: float = Field(0.0, ge=0)
    correct_target_fading: bool = False
    grad_clip: float = Field(1.0, gt=0)
    precision: Precision = "auto"
    select_metric: Literal["psnr", "lpips"] = "psnr"
    seed: int = 521

    @classmethod
    def from_yaml(cls, path: Path, **overrides: Any) -> TrainConfig:
        """Load a YAML config, applying keyword overrides (e.g. ``max_steps=200``)."""
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(raw)


def run_dir(run_name: str) -> Path:
    """Checkpoint directory for a run under the data root."""
    path = checkpoint_root() / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_precision(precision: Precision, device: torch.device) -> torch.dtype | None:
    # MPS stays fp32: AMP gave no speed-up there and DDColor fails under it (decisions D11).
    if device.type != "cuda" or precision == "fp32":
        return None
    if precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def _cycle(loader: DataLoader[Any]) -> Iterator[Batch]:
    while True:
        yield from loader


def _to_device(batch: dict[str, Any], device: torch.device) -> Batch:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}


def _lr_lambda(warmup: int, total: int) -> Any:
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return fn


@torch.no_grad()
def evaluate(
    adapter: TaskAdapter,
    model: nn.Module | None,
    loader: DataLoader[Any],
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> dict[str, float]:
    """Mean PSNR/SSIM/LPIPS (+ colourfulness for colorization) over a loader.

    Args:
        adapter: Task adapter.
        model: Model to evaluate, or ``None`` to score the classical baseline.
        loader: Validation loader.
        device: Device.
        autocast_dtype: Mixed-precision dtype or ``None``.

    Returns:
        Metric name -> mean value.
    """
    if model is not None:
        model.eval()
    sums: dict[str, float] = {}
    count = 0
    for raw in loader:
        batch = _to_device(raw, device)
        with torch.autocast(
            device.type, dtype=autocast_dtype or torch.float32, enabled=autocast_dtype is not None
        ):
            pred = adapter.baseline(batch) if model is None else adapter.predict(model, batch)
        pred = pred.float().clamp(0, 1)
        target = batch["target"]
        values = {
            "psnr": metrics.psnr(pred, target),
            "ssim": metrics.ssim(pred, target),
            "lpips": metrics.lpips_distance(pred, target),
        }
        if adapter.task == "colorize":
            values["colorfulness"] = metrics.colorfulness(pred)
            values["colorfulness_gt"] = metrics.colorfulness(target)
        for name, val in values.items():
            sums[name] = sums.get(name, 0.0) + float(val.sum())
        count += target.shape[0]
    return {k: v / count for k, v in sums.items()}


class MetricsLog:
    """Append-only CSV with a fixed superset of columns (train and val rows differ)."""

    FIELDS: tuple[str, ...] = (
        "step",
        "kind",
        "loss",
        "lr",
        "psnr",
        "ssim",
        "lpips",
        "colorfulness",
        "colorfulness_gt",
    )

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, row: dict[str, Any]) -> None:
        exists = self.path.is_file()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDS, restval="")
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def build_adapter(cfg: TrainConfig) -> TaskAdapter:
    """Instantiate the adapter for ``cfg.task``."""
    return ADAPTERS[cfg.task](
        cfg.model_key,
        pixel_loss=cfg.pixel_loss,
        lpips_weight=cfg.lpips_weight,
        colorfulness_weight=cfg.colorfulness_weight,
    )


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Train (shuffled, infinite via cycling) and fixed validation loaders."""

    def dataset(split: Literal["train", "val"], limit: int | None) -> RestorationPairs:
        return RestorationPairs(
            PairDatasetConfig(
                task=cfg.task,
                split=split,
                patch_size=cfg.patch_size,
                sources=cfg.sources,
                seed=cfg.seed,
                limit=limit,
                correct_target_fading=cfg.correct_target_fading,
            )
        )

    train_ds, val_ds = dataset("train", cfg.train_limit), dataset("val", cfg.val_limit)
    loader_kw: dict[str, Any] = {
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.num_workers > 0,
    }
    train = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **loader_kw
    )
    val = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kw)
    return train, val


def train(cfg: TrainConfig, device: torch.device | None = None) -> Path:
    """Fine-tune ``cfg.model_key`` on ``cfg.task``; returns the run directory.

    Args:
        cfg: Training configuration.
        device: Device (defaults to CUDA, then MPS, then CPU).

    Returns:
        Directory containing ``best.safetensors``, ``last_state.pt``, ``config.json``
        and ``metrics.csv``.
    """
    seed_everything(cfg.seed)
    device = device or default_device()
    if device.type == "cuda":
        # Fail fast with OOM instead of letting the Windows driver silently spill into
        # system RAM, which slows training by an order of magnitude.
        torch.cuda.set_per_process_memory_fraction(
            0.95, device.index if device.index is not None else torch.cuda.current_device()
        )
    out = run_dir(cfg.run_name)
    (out / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    log = MetricsLog(out / "metrics.csv")

    adapter = build_adapter(cfg)
    model = adapter.build_model(device)
    train_loader, val_loader = build_loaders(cfg)
    amp_dtype = resolve_precision(cfg.precision, device)
    scaler = GradScaler(enabled=amp_dtype == torch.float16)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(cfg.warmup_steps, cfg.max_steps)
    )
    logger.info(
        "%s | %s | %.1fM trainable params | precision=%s | train=%d val=%d images",
        cfg.run_name,
        cfg.model_key,
        sum(p.numel() for p in params) / 1e6,
        amp_dtype,
        len(cast(RestorationPairs, train_loader.dataset)),
        len(cast(RestorationPairs, val_loader.dataset)),
    )

    step, best = 0, cast(float | None, None)
    state_path = out / "last_state.pt"
    best_path = out / "best.safetensors"
    if state_path.is_file():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, best = int(state["step"]), cast(float | None, state["best"])
        logger.info("Resumed from step %d (best %s=%s)", step, cfg.select_metric, best)
    else:
        baseline = evaluate(adapter, None, val_loader, device, amp_dtype)
        log.write({"step": -1, "kind": "baseline", **baseline})
        logger.info("baseline   %s", _fmt(baseline))

    def is_better(value: float) -> bool:
        if best is None:
            return True
        return value > best if cfg.select_metric == "psnr" else value < best

    def validate() -> None:
        nonlocal best
        val = evaluate(adapter, model, val_loader, device, amp_dtype)
        kind = "zero_shot" if step == 0 else "val"
        log.write({"step": step, "kind": kind, **val})
        score = val[cfg.select_metric]
        if is_better(score):
            best = score
            save_model(model, str(best_path))
        logger.info(
            "step %6d %-9s %s%s", step, kind, _fmt(val), "  *best*" if best == score else ""
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "best": best,
            },
            state_path,
        )

    if step == 0:
        validate()
    batches = _cycle(train_loader)
    model.train()
    start, running = time.perf_counter(), 0.0
    while step < cfg.max_steps:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            batch = _to_device(next(batches), device)
            with torch.autocast(
                device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None
            ):
                pred = adapter.predict(model, batch)
            terms = adapter.loss(pred.float(), batch)
            scaled = scaler.scale(terms["total"] / cfg.grad_accum)
            scaled.backward()  # type: ignore[no-untyped-call]
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        running += float(terms["total"])
        if step % 50 == 0:
            rate = 50 / (time.perf_counter() - start)
            logger.info(
                "step %6d loss %.4f lr %.2e (%.2f it/s)",
                step,
                running / 50,
                scheduler.get_last_lr()[0],
                rate,
            )
            log.write(
                {
                    "step": step,
                    "kind": "train",
                    "loss": running / 50,
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            start, running = time.perf_counter(), 0.0
        if step % cfg.val_every == 0 or step == cfg.max_steps:
            validate()
            model.train()

    summary = {"run_name": cfg.run_name, "best_" + cfg.select_metric: best, "steps": step}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def _fmt(values: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in values.items())
