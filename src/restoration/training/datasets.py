"""PyTorch datasets that synthesise degraded/clean training pairs on the fly.

Pairs are generated at load time from the curated clean images, so every epoch sees
fresh degradations (an effectively unlimited augmentation) while validation sees a
*fixed* set: validation randomness is seeded by ``(seed, index)``.

Each item is a dict of CHW float32 tensors in ``[0, 1]`` (plus task extras):

* ``denoise``          -- ``inp`` noisy patch, ``target`` clean patch
* ``super_resolution`` -- ``inp`` LR patch, ``target`` HR patch (``scale`` x larger)
* ``colorize``         -- whole image resized to a square: ``inp`` grey (L-only) RGB,
  ``target`` colour RGB, ``target_ab`` Lab chroma (DDColor's output space)
* ``inpaint``          -- ``inp`` damaged patch, ``target`` clean, ``mask`` (1 = hole),
  ``caption`` archive title
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from restoration.config import processed_dir, raw_dir
from restoration.data import degradations as deg

Task = Literal["denoise", "super_resolution", "colorize", "inpaint"]
Split = Literal["train", "val", "test"]
Sample = dict[str, torch.Tensor | str]

BORDER_MARGIN_PX = 8  # extra trim beyond detected borders (anti-aliased mount edges)


def load_split(
    split: Split, sources: list[str] | None = None, limit: int | None = None
) -> pd.DataFrame:
    """Rows of ``processed/splits.csv`` for one split, with absolute image paths.

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.
        sources: Optional subset of sources to keep.
        limit: Optional cap on rows (deterministic: first ``limit`` by ``image_id``).

    Returns:
        DataFrame with an added ``abs_path`` column.
    """
    path = processed_dir() / "splits.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run scripts/curate_and_split.py")
    frame = pd.read_csv(path, keep_default_na=False)
    frame = frame[frame["split"] == split]
    if sources is not None:
        frame = frame[frame["source"].isin(sources)]
    frame = frame.sort_values("image_id").reset_index(drop=True)
    if limit is not None:
        frame = frame.head(limit)
    frame["abs_path"] = [
        str(raw_dir(s) / p) for s, p in zip(frame["source"], frame["path"], strict=True)
    ]
    if frame.empty:
        raise ValueError(f"No rows for split={split!r}, sources={sources}")
    return frame


def load_clean(row: pd.Series) -> deg.FloatImage:
    """Read an image and trim detected borders (slide mounts, film edges)."""
    img = deg.to_float(np.asarray(Image.open(row["abs_path"]).convert("RGB")))
    h, w = img.shape[:2]
    top = int(row.get("border_top", 0)) + BORDER_MARGIN_PX
    bottom = int(row.get("border_bottom", 0)) + BORDER_MARGIN_PX
    left = int(row.get("border_left", 0)) + BORDER_MARGIN_PX
    right = int(row.get("border_right", 0)) + BORDER_MARGIN_PX
    if top + bottom < h // 2 and left + right < w // 2:
        img = img[top : h - bottom, left : w - right]
    return np.ascontiguousarray(img)


def augment(img: deg.FloatImage, rng: np.random.Generator) -> deg.FloatImage:
    """Random flips / 90-degree rotations (the dihedral group D4)."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    k = int(rng.integers(0, 4))
    return np.ascontiguousarray(np.rot90(img, k))


def to_chw(img: np.ndarray) -> torch.Tensor:
    """HWC (or HW) numpy -> CHW float32 tensor."""
    arr = img[..., None] if img.ndim == 2 else img
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)).astype(np.float32))


@dataclass
class PairDatasetConfig:
    """Options for :class:`RestorationPairs`.

    Attributes:
        task: Which pair type to synthesise.
        split: Data split.
        patch_size: Training crop size (target resolution). For ``colorize`` this is the
            square resolution the whole image is resized to.
        scale_jitter: Random pre-crop downscale range, so patches cover different
            spatial scales of the 2048 px scans (train split only).
        sr_scale: Super-resolution factor.
        sources: Optional source subset.
        limit: Optional cap on images (smoke runs).
        seed: Base seed for deterministic validation sampling.
        nvidia_mask_prob: For ``inpaint``, probability of using an NVIDIA mask instead
            of procedural scratches/cracks/blotches.
        correct_target_fading: For ``colorize``, apply
            :func:`~restoration.data.degradations.correct_fading` to the colour target.
    """

    task: Task
    split: Split
    patch_size: int = 256
    scale_jitter: tuple[float, float] = (0.5, 1.0)
    sr_scale: int = 4
    sources: list[str] | None = None
    limit: int | None = None
    seed: int = 521
    nvidia_mask_prob: float = 0.5
    correct_target_fading: bool = False


class RestorationPairs(Dataset[Sample]):
    """On-the-fly degraded/clean pairs for one task and split."""

    def __init__(self, cfg: PairDatasetConfig) -> None:
        """Index the split (and NVIDIA masks for inpainting)."""
        self.cfg = cfg
        self.frame = load_split(cfg.split, cfg.sources, cfg.limit)
        self.train = cfg.split == "train"
        self.mask_paths: list[Path] = []
        if cfg.task == "inpaint" and cfg.nvidia_mask_prob > 0:
            masks_dir = raw_dir("nvidia_masks")
            manifest = pd.read_csv(masks_dir / "manifest.csv")
            self.mask_paths = [masks_dir / p for p in manifest["path"]]

    def __len__(self) -> int:
        """Number of source images."""
        return len(self.frame)

    def _rng(self, index: int) -> np.random.Generator:
        if self.train:
            # DataLoader reseeds torch per worker and epoch, so this differs every draw.
            return np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
        return np.random.default_rng([self.cfg.seed, index])

    def _patch(self, img: deg.FloatImage, size: int, rng: np.random.Generator) -> deg.FloatImage:
        if self.train:
            lo, hi = self.cfg.scale_jitter
            factor = float(rng.uniform(lo, hi))
            short = min(img.shape[:2])
            factor = max(factor, size / short)  # never shrink below the patch size
            if factor < 1:
                img = deg.resize_max_side(img, round(max(img.shape[:2]) * factor))
            return augment(deg.random_crop(img, size, rng), rng)
        return deg.random_crop(img, size, rng)

    def _nvidia_mask(self, size: int, rng: np.random.Generator) -> deg.BoolMask:
        path = self.mask_paths[int(rng.integers(len(self.mask_paths)))]
        arr = np.asarray(Image.open(path).convert("L")) > 127
        hole = ~arr if arr.mean() > 0.5 else arr  # holes are the minority value
        hole = augment(hole[..., None].astype(np.float32), rng)[..., 0] > 0.5
        return cv2.resize(
            hole.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    def __getitem__(self, index: int) -> Sample:
        """Synthesize the pair for image ``index``."""
        row = self.frame.iloc[index]
        rng = self._rng(index)
        clean = load_clean(row)
        task, size = self.cfg.task, self.cfg.patch_size

        if task == "denoise":
            pair = deg.make_denoise_pair(self._patch(clean, size, rng), rng)
            return {"inp": to_chw(pair.inp), "target": to_chw(pair.target)}

        if task == "super_resolution":
            sr_cfg = deg.SRConfig(scale=self.cfg.sr_scale)
            pair = deg.make_sr_pair(self._patch(clean, size, rng), rng, sr_cfg)
            return {"inp": to_chw(pair.inp), "target": to_chw(pair.target)}

        if task == "colorize":
            # Whole image (global context matters for colour), squashed to a square
            # exactly as DDColor does at inference.
            if self.train and rng.random() < 0.5:
                clean = np.ascontiguousarray(clean[:, ::-1])  # no rotations: sky stays up
            square = cv2.resize(clean, (size, size), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            if self.cfg.correct_target_fading:
                square = deg.correct_fading(square)
            lab = cv2.cvtColor(square, cv2.COLOR_RGB2LAB)
            return {
                "inp": to_chw(deg.to_grayscale(square)),
                "target": to_chw(square),
                "target_ab": to_chw(lab[..., 1:]),
            }

        if task == "inpaint":
            patch = self._patch(clean, size, rng)
            use_nvidia = bool(self.mask_paths) and rng.random() < self.cfg.nvidia_mask_prob
            ext = self._nvidia_mask(size, rng) if use_nvidia else None
            pair = deg.make_inpaint_pair(patch, rng, external_mask=ext)
            assert pair.mask is not None
            return {
                "inp": to_chw(pair.inp),
                "target": to_chw(pair.target),
                "mask": to_chw(pair.mask.astype(np.float32)),
                "caption": str(row.get("title", "")),
            }

        raise ValueError(f"Unknown task {task!r}")
