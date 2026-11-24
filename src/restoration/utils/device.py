"""Accelerator selection shared by inference, training and scripts (CUDA -> MPS -> CPU)."""

from __future__ import annotations

import torch


def default_device() -> torch.device:
    """Best available device: CUDA, then Apple-silicon MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_cache() -> None:
    """Release cached allocator memory on whichever accelerator is present."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def synchronize() -> None:
    """Wait for queued accelerator kernels, so wall-clock timings are accurate."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
