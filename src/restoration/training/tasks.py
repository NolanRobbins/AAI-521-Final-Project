"""Per-task adapters: how to build, run and score each model during fine-tuning.

An adapter hides model-specific I/O so a single training loop serves every task:

* :meth:`TaskAdapter.predict` maps a batch to an RGB prediction in ``[0, 1]`` that is
  directly comparable to ``batch["target"]`` (differentiable).
* :meth:`TaskAdapter.baseline` gives the classical/no-op reference for the same batch
  (degraded input, bicubic upsampling, grey image), reported at step 0.
* :meth:`TaskAdapter.loss` combines the pixel loss with optional perceptual and
  colourfulness terms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from restoration.evaluation import metrics
from restoration.models import loaders

Batch = dict[str, torch.Tensor]


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier (smooth L1) loss -- robust, standard for restoration (Lai et al., 2017)."""
    return torch.sqrt((pred - target).pow(2) + eps**2).mean()


class TaskAdapter(ABC):
    """Model-specific glue for the generic trainer."""

    task: str

    def __init__(
        self,
        model_key: str,
        pixel_loss: Literal["l1", "charbonnier"] = "charbonnier",
        lpips_weight: float = 0.0,
        colorfulness_weight: float = 0.0,
    ) -> None:
        """Configure the adapter.

        Args:
            model_key: Registry key in ``configs/models.yaml``.
            pixel_loss: Pixel reconstruction loss.
            lpips_weight: Weight of the LPIPS perceptual term.
            colorfulness_weight: Weight of the colourfulness-matching term.
        """
        self.model_key = model_key
        self.pixel_loss = pixel_loss
        self.lpips_weight = lpips_weight
        self.colorfulness_weight = colorfulness_weight

    @abstractmethod
    def build_model(self, device: torch.device) -> nn.Module:
        """Load the pretrained model onto ``device``."""

    @abstractmethod
    def predict(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """RGB prediction ``(N, 3, H, W)`` aligned with ``batch["target"]``."""

    def baseline(self, batch: Batch) -> torch.Tensor:
        """No-learning reference output (default: the degraded input itself)."""
        return batch["inp"]

    def loss(self, pred: torch.Tensor, batch: Batch) -> dict[str, torch.Tensor]:
        """Weighted training loss; ``"total"`` is optimised, the rest are logged."""
        target = batch["target"]
        l1 = self.pixel_loss == "l1"
        pixel = F.l1_loss(pred, target) if l1 else charbonnier(pred, target)
        terms = {"pixel": pixel}
        total = pixel
        if self.lpips_weight > 0:
            terms["lpips"] = metrics.lpips_distance(pred, target).mean()
            total = total + self.lpips_weight * terms["lpips"]
        if self.colorfulness_weight > 0:
            # Normalised gap in colourfulness: counteracts the L1 pull toward sepia means.
            gap = (metrics.colorfulness(pred) - metrics.colorfulness(target)).abs() / 100
            terms["colorfulness"] = gap.mean()
            total = total + self.colorfulness_weight * terms["colorfulness"]
        terms["total"] = total
        return terms


class DenoiseAdapter(TaskAdapter):
    """SCUNet: noisy RGB -> clean RGB (patches are multiples of 64 by construction)."""

    task = "denoise"

    def build_model(self, device: torch.device) -> nn.Module:
        """Pretrained SCUNet."""
        return loaders.load_scunet(device)

    def predict(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """Denoised patch."""
        out: torch.Tensor = model(batch["inp"])
        return out


class SuperResolutionAdapter(TaskAdapter):
    """Swin2SR x4: LR RGB -> HR RGB."""

    task = "super_resolution"

    def build_model(self, device: torch.device) -> nn.Module:
        """Pretrained Swin2SR."""
        return loaders.load_swin2sr(device)

    def predict(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """Upscaled patch."""
        out: torch.Tensor = model(pixel_values=batch["inp"]).reconstruction
        return out

    def baseline(self, batch: Batch) -> torch.Tensor:
        """Bicubic upsampling to the target size."""
        size = batch["target"].shape[-2:]
        return F.interpolate(batch["inp"], size=size, mode="bicubic", align_corners=False).clamp(
            0, 1
        )


class ColorizeAdapter(TaskAdapter):
    """DDColor: grey (L-only) RGB -> Lab ``ab``; recombined with the input L as RGB."""

    task = "colorize"

    def build_model(self, device: torch.device) -> nn.Module:
        """Pretrained DDColor (large or tiny depending on ``model_key``)."""
        model, _ = loaders.load_ddcolor(self.model_key, device)
        return model

    def predict(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """Colourised RGB using the input's own lightness."""
        ab: torch.Tensor = model(batch["inp"])
        lightness = metrics.rgb_to_l(batch["inp"])
        return metrics.lab_to_rgb(torch.cat([lightness, ab.float()], dim=1))


ADAPTERS: dict[str, type[TaskAdapter]] = {
    "denoise": DenoiseAdapter,
    "super_resolution": SuperResolutionAdapter,
    "colorize": ColorizeAdapter,
}
