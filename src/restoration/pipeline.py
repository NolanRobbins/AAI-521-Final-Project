"""End-to-end restoration: denoise -> inpaint -> colorize -> super-resolve.

The order is fixed (see notebook 01): noise must go before the generative steps would
turn it into texture, damage is filled before colour is predicted, and colour is
predicted before upsampling because DDColor reuses the input lightness at full
resolution.

Models load lazily on first use and are cached, so the web app only pays for the tasks
users select. Each task uses the fine-tuned checkpoint when one exists under
``$AAI521_CHECKPOINT_ROOT/<run_name>`` and otherwise falls back to the pretrained weights,
so the app works before training has finished.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
import yaml

from restoration.config import CONFIG_DIR, checkpoint_root
from restoration.data import degradations as deg
from restoration.models import inference, loaders

logger = logging.getLogger(__name__)

TaskName = Literal["denoise", "inpaint", "colorize", "super_resolution"]
TASK_ORDER: tuple[TaskName, ...] = ("denoise", "inpaint", "colorize", "super_resolution")
FloatImage = deg.FloatImage
BoolMask = deg.BoolMask


def default_run_names() -> dict[TaskName, str]:
    """Run names from ``configs/train/*.yaml`` (where fine-tuned weights are saved)."""
    names: dict[TaskName, str] = {}
    for task in TASK_ORDER:
        with (CONFIG_DIR / "train" / f"{task}.yaml").open(encoding="utf-8") as fh:
            names[task] = str(yaml.safe_load(fh)["run_name"])
    return names


def detect_scratches(img: FloatImage, sensitivity: float = 0.5) -> BoolMask:
    """Heuristic damage mask: thin bright/dark line structures and specks.

    Morphological top-hats with long thin structuring elements respond to scratches
    narrower than the element (Gonzalez & Woods, ch. 9). This is a classical stand-in
    for a learned scratch detector (cf. Wan et al., 2020). It catches high-contrast
    scratches and dust but misses low-contrast cracks, so the app also accepts a
    user-painted mask.

    Args:
        img: RGB image in ``[0, 1]``.
        sensitivity: 0 (few detections) to 1 (many).

    Returns:
        Boolean damage mask.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    responses = []
    for ksize in [(1, 25), (25, 1), (7, 7)]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        responses.append(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel))  # bright damage
        responses.append(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))  # dark damage
    response = np.max(np.stack(responses), axis=0)
    threshold = np.percentile(response, 99.5 - 2.5 * sensitivity) + 0.02
    mask = (response > threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(4, int(img.shape[0] * img.shape[1] * 2e-6))
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    mask = keep[labels].astype(np.uint8)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask.astype(bool)


@dataclass
class PipelineConfig:
    """Options for :class:`RestorationPipeline`.

    Attributes:
        use_finetuned: Prefer fine-tuned checkpoints when available.
        colorize_model: Registry key for DDColor (``"colorize"`` or ``"colorize_tiny"``).
        inpaint_steps: Diffusion sampling steps.
        sr_max_input_side: Downscale inputs above this before ×4 SR (bounds output size
            and memory; a 1024 px input already yields 4096 px).
        run_names: Override task -> run directory name.
        device: Torch device.
    """

    use_finetuned: bool = True
    colorize_model: str = "colorize"
    inpaint_steps: int = 30
    sr_max_input_side: int = 1024
    run_names: dict[TaskName, str] = field(default_factory=default_run_names)
    device: torch.device = field(default_factory=loaders.default_device)


@dataclass
class RestorationResult:
    """Output of :meth:`RestorationPipeline.run`."""

    image: FloatImage
    stages: dict[str, FloatImage]
    seconds: dict[str, float]
    weights: dict[str, str]
    mask: BoolMask | None = None


class RestorationPipeline:
    """Lazily loaded chain of the four restoration models."""

    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        """Create the pipeline (no models are loaded until first use)."""
        self.cfg = cfg or PipelineConfig()
        self._models: dict[str, Any] = {}
        self.weights: dict[str, str] = {}

    def _run_dir(self, task: TaskName) -> Path | None:
        if not self.cfg.use_finetuned:
            return None
        run = checkpoint_root() / self.cfg.run_names[task]
        marker = run / ("best_lora" if task == "inpaint" else "best.safetensors")
        return run if marker.exists() else None

    def _model(self, task: TaskName) -> Any:
        if task in self._models:
            return self._models[task]
        device = self.cfg.device
        run = self._run_dir(task)
        model: Any
        if task == "denoise":
            model = loaders.load_scunet(device)
        elif task == "super_resolution":
            model = loaders.load_swin2sr(device)
        elif task == "colorize":
            model = loaders.load_ddcolor(self.cfg.colorize_model, device)
        else:
            lora = run / "best_lora" if run else None
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            model = loaders.load_sd_inpaint(device, dtype=dtype, lora_dir=lora)
        if run is not None and task != "inpaint":
            net = model[0] if task == "colorize" else model
            loaders.load_finetuned_weights(net, run)
        self.weights[task] = f"fine-tuned ({run.name})" if run else "pretrained"
        logger.info("Loaded %s: %s", task, self.weights[task])
        self._models[task] = model
        return model

    def run(
        self,
        img: FloatImage,
        tasks: Iterable[TaskName],
        mask: BoolMask | None = None,
        auto_mask: bool = False,
        prompt: str = inference.GENERIC_PROMPT,
        seed: int = 0,
    ) -> RestorationResult:
        """Restore ``img`` with the selected tasks, always in :data:`TASK_ORDER`.

        Args:
            img: RGB float32 image in ``[0, 1]``.
            tasks: Subset of task names (order ignored).
            mask: Damage mask (True = missing) for inpainting.
            auto_mask: Detect scratches heuristically if no mask is given.
            prompt: Text prompt for inpainting.
            seed: Diffusion seed.

        Returns:
            Final image, per-stage outputs, timings and which weights were used.
        """
        selected = set(tasks)
        unknown = selected - set(TASK_ORDER)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        out = np.ascontiguousarray(img, dtype=np.float32)
        stages: dict[str, FloatImage] = {"input": out}
        seconds: dict[str, float] = {}
        used_mask = mask
        for task in TASK_ORDER:
            if task not in selected:
                continue
            start = time.perf_counter()
            if task == "denoise":
                out = inference.denoise(self._model(task), out)
            elif task == "inpaint":
                if used_mask is None and auto_mask:
                    used_mask = detect_scratches(out)
                if used_mask is None or not used_mask.any():
                    logger.info("Inpainting skipped: empty mask")
                    continue
                if used_mask.shape != out.shape[:2]:
                    used_mask = cv2.resize(
                        used_mask.astype(np.uint8),
                        out.shape[1::-1],
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                out = inference.inpaint_sd(
                    self._model(task),
                    out,
                    used_mask,
                    prompt=prompt,
                    steps=self.cfg.inpaint_steps,
                    seed=seed,
                )
            elif task == "colorize":
                model, size = self._model(task)
                out = inference.colorize(model, out, size)
            else:
                out = deg.resize_max_side(out, self.cfg.sr_max_input_side)
                out = inference.super_resolve(self._model(task), out)
            seconds[task] = time.perf_counter() - start
            stages[task] = out
        return RestorationResult(out, stages, seconds, dict(self.weights), used_mask)

    def unload(self) -> None:
        """Free all loaded models (and cached GPU memory)."""
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
