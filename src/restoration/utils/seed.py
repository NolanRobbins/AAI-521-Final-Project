"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs.

    Args:
        seed: Non-negative integer seed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # torch is optional for data-only workflows
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
