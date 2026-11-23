"""Construct each pretrained model from its cached Hugging Face snapshot.

All loaders resolve weights via :func:`restoration.models.registry.model_path`, so they
never touch the network; run ``scripts/download_models.py`` first. State dicts are
loaded with ``strict=True`` so a mismatched checkpoint fails loudly instead of silently
leaving layers randomly initialised.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from restoration.models.registry import model_path

if TYPE_CHECKING:
    from diffusers import StableDiffusionInpaintPipeline
    from transformers import Swin2SRForImageSuperResolution

logger = logging.getLogger(__name__)


def default_device() -> torch.device:
    """CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _single_file(snapshot: Path, pattern: str) -> Path:
    matches = sorted(snapshot.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern!r} in {snapshot}, found {matches}")
    return matches[0]


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "params" in ckpt:  # BasicSR-style wrapper
        ckpt = ckpt["params"]
    if not isinstance(ckpt, dict):
        raise TypeError(f"{path} does not contain a state dict")
    return ckpt


def load_scunet(device: torch.device | None = None) -> nn.Module:
    """SCUNet blind denoiser (``configs/models.yaml: denoise``)."""
    from deepinv.models import SCUNet

    weights = _single_file(model_path("denoise"), "*.pth")
    model = SCUNet(in_nc=3, pretrained=str(weights), device="cpu")
    return cast(nn.Module, model.to(device or default_device()).eval())


def load_drunet(device: torch.device | None = None) -> nn.Module:
    """DRUNet non-blind denoiser (``denoise_drunet``); forward takes ``(x, sigma)``."""
    from deepinv.models import DRUNet

    weights = _single_file(model_path("denoise_drunet"), "*.pth")
    model = DRUNet(in_channels=3, out_channels=3, pretrained=str(weights), device="cpu")
    return cast(nn.Module, model.to(device or default_device()).eval())


def load_swin2sr(device: torch.device | None = None) -> Swin2SRForImageSuperResolution:
    """Swin2SR x4 real-world super-resolution (``super_resolution``)."""
    from transformers import Swin2SRForImageSuperResolution

    model = Swin2SRForImageSuperResolution.from_pretrained(model_path("super_resolution"))
    model.to(device or default_device())  # type: ignore[arg-type]  # transformers' typed `to`
    model.eval()  # type: ignore[no-untyped-call]
    return model


def load_ddcolor(
    name: str = "colorize", device: torch.device | None = None
) -> tuple[nn.Module, int]:
    """DDColor colorizer.

    Args:
        name: Registry key (``"colorize"`` for ConvNeXt-L, ``"colorize_tiny"``).
        device: Target device.

    Returns:
        ``(model, input_size)`` where ``input_size`` is the square resolution the
        checkpoint was trained at.
    """
    from restoration.models.third_party.ddcolor import DDColor

    snapshot = model_path(name)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    model = DDColor(**config)
    model.load_state_dict(_state_dict(_single_file(snapshot, "pytorch_model.bin")), strict=True)
    input_size = int(config["input_size"][0])
    return model.to(device or default_device()).eval(), input_size


def load_sd_inpaint(
    device: torch.device | None = None, dtype: torch.dtype = torch.float16
) -> StableDiffusionInpaintPipeline:
    """Stable Diffusion 1.5 inpainting pipeline (``inpaint``), fp16, no safety checker.

    The safety checker is omitted because inputs are archival photographs restored
    locally, and it would blank legitimate historical content (false positives).
    """
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(  # type: ignore[no-untyped-call]
        model_path("inpaint"),
        torch_dtype=dtype,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    return cast("StableDiffusionInpaintPipeline", pipe.to(device or default_device()))


def load_lama(device: torch.device | None = None) -> torch.jit.ScriptModule:
    """Big-LaMa TorchScript inpainter (``inpaint_lama``); forward takes ``(image, mask)``."""
    weights = _single_file(model_path("inpaint_lama"), "big-lama.pt")
    target = device or default_device()
    model = torch.jit.load(str(weights), map_location=target)  # type: ignore[no-untyped-call]
    return cast(torch.jit.ScriptModule, model.eval())
