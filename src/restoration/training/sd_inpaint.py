"""LoRA fine-tuning of Stable Diffusion 1.5 Inpainting on archive photographs.

Training objective (standard latent-diffusion epsilon prediction, Rombach et al., 2022):

    z0 = E(x),  z_m = E(x * (1 - m)),  m' = maxpool_8(m)
    z_t = sqrt(a_t) z0 + sqrt(1 - a_t) eps,   t ~ U{0..T-1}
    L = || eps - UNet([z_t, m', z_m], t, c) ||^2

Only low-rank adapters on the UNet attention projections are trained (Hu et al., 2021):
~3M of 860M parameters, so it fits a 16 GB GPU and checkpoints are a few MB.

Two choices are specific to old-photo damage:

* The pixel mask is downsampled to latent resolution with **max-pooling**, not nearest
  neighbour, so 1-4 px scratches still mark their 8x8 latent cell as "hole". This is
  the training-time counterpart of the mask dilation in
  :func:`restoration.models.inference.inpaint_sd`.
* Captions are the archive titles (``"an old photograph, <title>"``) with 50% dropout
  to the generic prompt the web app uses, so the adapter works with and without one.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, Field
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from restoration.evaluation import metrics
from restoration.models import inference
from restoration.models.inference import GENERIC_PROMPT
from restoration.models.registry import model_path
from restoration.training.datasets import PairDatasetConfig, RestorationPairs
from restoration.training.trainer import MetricsLog, resolve_precision, run_dir
from restoration.utils.device import default_device
from restoration.utils.seed import seed_everything

logger = logging.getLogger(__name__)


class InpaintLoraConfig(BaseModel):
    """Hyper-parameters for ``configs/train/inpaint.yaml``."""

    model_config = {"extra": "forbid"}

    task: Literal["inpaint"] = "inpaint"
    model_key: str = "inpaint"
    run_name: str = "inpaint_sd15_lora"
    patch_size: int = Field(512, gt=0)
    batch_size: int = Field(2, gt=0)
    grad_accum: int = Field(2, gt=0)
    num_workers: int = Field(4, ge=0)
    lr: float = Field(1e-4, gt=0)
    lora_rank: int = Field(16, gt=0)
    lora_alpha: int = Field(16, gt=0)
    max_steps: int = Field(3000, gt=0)
    warmup_steps: int = Field(100, ge=0)
    val_every: int = Field(500, gt=0)
    val_limit: int = Field(16, gt=0)
    val_inference_steps: int = Field(20, gt=0)
    train_limit: int | None = Field(None, gt=0)
    caption_dropout: float = Field(0.5, ge=0, le=1)
    nvidia_mask_prob: float = Field(0.5, ge=0, le=1)
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    gradient_checkpointing: bool = True
    seed: int = 521

    @classmethod
    def from_yaml(cls, path: Path, **overrides: Any) -> InpaintLoraConfig:
        """Load YAML with keyword overrides."""
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(raw)


def caption_for(title: str, dropout: float, rng: random.Random) -> str:
    """Training caption: archive title or (with prob ``dropout``) the generic prompt."""
    title = " ".join(title.split())
    if not title or title.lower() in {"[untitled]", "untitled"} or rng.random() < dropout:
        return GENERIC_PROMPT
    return f"an old photograph, {title}"


def latent_mask(mask: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Downsample a ``(N,1,H,W)`` hole mask to latent size; any hole pixel marks its cell."""
    return F.max_pool2d(mask, kernel_size=factor, stride=factor)


def _load_components(dtype: torch.dtype, device: torch.device) -> dict[str, Any]:
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    root = model_path("inpaint")
    kw: dict[str, Any] = {"torch_dtype": dtype, "variant": "fp16"}
    vae = AutoencoderKL.from_pretrained(root, subfolder="vae", **kw).to(device)
    unet = UNet2DConditionModel.from_pretrained(root, subfolder="unet", **kw).to(device)
    text_encoder: Any = CLIPTextModel.from_pretrained(root, subfolder="text_encoder", **kw)
    text_encoder.to(device)
    tokenizer = CLIPTokenizer.from_pretrained(root, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(root, subfolder="scheduler")
    for module in (vae, text_encoder, unet):
        module.requires_grad_(False)
    return {
        "vae": vae,
        "unet": unet,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "noise_scheduler": noise_scheduler,
        "root": root,
    }


def _build_pipeline(parts: dict[str, Any]) -> Any:
    from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline

    pipe: Any = StableDiffusionInpaintPipeline(
        vae=parts["vae"],
        text_encoder=parts["text_encoder"],
        tokenizer=parts["tokenizer"],
        unet=parts["unet"],
        scheduler=DDIMScheduler.from_pretrained(parts["root"], subfolder="scheduler"),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def telea_baseline(inp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV Telea inpainting (fast marching; Telea, 2004) -- the classical baseline."""
    img8 = np.clip(np.round(inp * 255), 0, 255).astype(np.uint8)
    out = cv2.inpaint(img8, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return out.astype(np.float32) / 255.0


@torch.no_grad()
def evaluate_inpaint(
    pipe: Any | None, loader: DataLoader[Any], steps: int, device: torch.device
) -> dict[str, float]:
    """Full-image PSNR/SSIM/LPIPS plus hole-only PSNR; ``pipe=None`` scores Telea."""
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        preds = []
        for i in range(batch["inp"].shape[0]):
            inp = batch["inp"][i].permute(1, 2, 0).numpy()
            mask = batch["mask"][i, 0].numpy() > 0.5
            if pipe is None:
                preds.append(telea_baseline(inp, mask))
            else:
                preds.append(
                    inference.inpaint_sd(
                        pipe, inp, mask, max_side=inp.shape[0], steps=steps, seed=i
                    )
                )
        pred = torch.from_numpy(np.stack(preds)).permute(0, 3, 1, 2).to(device)
        target, mask_t = batch["target"].to(device), batch["mask"].to(device)
        values = {
            "psnr": metrics.psnr(pred, target),
            "ssim": metrics.ssim(pred, target),
            "lpips": metrics.lpips_distance(pred, target),
            "psnr_hole": metrics.masked_psnr(pred, target, mask_t),
        }
        for k, v in values.items():
            sums[k] = sums.get(k, 0.0) + float(v.sum())
        count += target.shape[0]
    return {k: v / count for k, v in sums.items()}


def save_lora(unet: Any, directory: Path) -> None:
    """Write the UNet LoRA in diffusers format (``pytorch_lora_weights.safetensors``)."""
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionInpaintPipeline.save_lora_weights(directory, unet_lora_layers=layers)


class _InpaintLog(MetricsLog):
    FIELDS: tuple[str, ...] = ("step", "kind", "loss", "lr", "psnr", "ssim", "lpips", "psnr_hole")


def train_inpaint_lora(cfg: InpaintLoraConfig, device: torch.device | None = None) -> Path:
    """Fine-tune SD 1.5 inpainting with LoRA; returns the run directory.

    Artefacts: ``best_lora/`` and ``last_lora/`` (diffusers LoRA format),
    ``last_state.pt`` (optimiser/step for resuming), ``config.json``, ``metrics.csv``.
    """
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict

    seed_everything(cfg.seed)
    device = device or default_device()
    amp_dtype = resolve_precision(cfg.precision, device)
    weight_dtype = amp_dtype or torch.float32
    out = run_dir(cfg.run_name)
    (out / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    log = _InpaintLog(out / "metrics.csv")

    parts = _load_components(weight_dtype, device)
    unet, vae, text_encoder, tokenizer = (
        parts["unet"],
        parts["vae"],
        parts["text_encoder"],
        parts["tokenizer"],
    )
    noise_scheduler = parts["noise_scheduler"]
    unet.add_adapter(
        LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    cast_training_params(unet, dtype=torch.float32)  # LoRA params in fp32, base stays low-precision
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-2)

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=amp_dtype == torch.float16)

    common = {
        "task": "inpaint",
        "patch_size": cfg.patch_size,
        "seed": cfg.seed,
        "nvidia_mask_prob": cfg.nvidia_mask_prob,
    }
    train_ds = RestorationPairs(PairDatasetConfig(split="train", limit=cfg.train_limit, **common))  # type: ignore[arg-type]
    val_ds = RestorationPairs(PairDatasetConfig(split="val", limit=cfg.val_limit, **common))  # type: ignore[arg-type]
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    logger.info(
        "%s | LoRA r=%d | %.2fM trainable params | precision=%s | train=%d val=%d",
        cfg.run_name,
        cfg.lora_rank,
        sum(p.numel() for p in params) / 1e6,
        amp_dtype,
        len(train_ds),
        len(val_ds),
    )

    step, best = 0, cast(float | None, None)
    state_path = out / "last_state.pt"
    if state_path.is_file():
        state = torch.load(state_path, map_location=device, weights_only=False)
        set_peft_model_state_dict(unet, state["lora"])
        cast_training_params(unet, dtype=torch.float32)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, best = int(state["step"]), state["best"]
        logger.info("Resumed from step %d (best lpips=%s)", step, best)
    else:
        base = evaluate_inpaint(None, val_loader, cfg.val_inference_steps, device)
        log.write({"step": -1, "kind": "baseline_telea", **base})
        logger.info("telea      %s", " ".join(f"{k}={v:.4f}" for k, v in base.items()))

    def validate() -> None:
        nonlocal best
        from peft.utils import get_peft_model_state_dict

        unet.eval()
        pipe = _build_pipeline(parts)
        with torch.autocast(device.type, dtype=weight_dtype, enabled=amp_dtype is not None):
            val = evaluate_inpaint(pipe, val_loader, cfg.val_inference_steps, device)
        kind = "zero_shot" if step == 0 else "val"
        log.write({"step": step, "kind": kind, **val})
        improved = best is None or val["lpips"] < best
        if improved:
            best = val["lpips"]
            save_lora(unet, out / "best_lora")
        logger.info(
            "step %6d %-9s %s%s",
            step,
            kind,
            " ".join(f"{k}={v:.4f}" for k, v in val.items()),
            "  *best*" if improved else "",
        )
        save_lora(unet, out / "last_lora")
        torch.save(
            {
                "lora": get_peft_model_state_dict(unet),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "best": best,
            },
            state_path,
        )
        unet.train()

    if step == 0:
        validate()  # LoRA B matrices start at zero, so this is the zero-shot model

    rng = random.Random(cfg.seed + step)
    sf = float(vae.config.scaling_factor)
    batches: Iterator[dict[str, Any]] = iter(())
    unet.train()
    running = 0.0
    while step < cfg.max_steps:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(train_loader)
                batch = next(batches)
            image = batch["target"].to(device, weight_dtype) * 2 - 1
            mask = batch["mask"].to(device, weight_dtype)
            captions = [caption_for(t, cfg.caption_dropout, rng) for t in batch["caption"]]
            with torch.no_grad():
                latents = vae.encode(image).latent_dist.sample() * sf
                masked = vae.encode(image * (1 - mask)).latent_dist.sample() * sf
                ids = tokenizer(
                    captions,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(device)
                text = text_encoder(ids)[0]
            noise = torch.randn_like(latents)
            t = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            )
            noisy = noise_scheduler.add_noise(latents, noise, t)
            model_in = torch.cat([noisy, latent_mask(mask), masked], dim=1)
            with torch.autocast(device.type, dtype=weight_dtype, enabled=amp_dtype is not None):
                pred = unet(model_in, t, encoder_hidden_states=text).sample
            loss = F.mse_loss(pred.float(), noise.float())
            scaler.scale(loss / cfg.grad_accum).backward()
            running += float(loss) / cfg.grad_accum
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        if step % 25 == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info("step %6d loss %.4f lr %.2e", step, running / 25, lr)
            log.write({"step": step, "kind": "train", "loss": running / 25, "lr": lr})
            running = 0.0
        if step % cfg.val_every == 0 or step == cfg.max_steps:
            validate()

    (out / "summary.json").write_text(
        json.dumps({"run_name": cfg.run_name, "best_lpips": best, "steps": step}, indent=2)
    )
    return out
