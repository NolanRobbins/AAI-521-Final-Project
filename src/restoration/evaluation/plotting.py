"""Shared matplotlib styling and image-grid helpers for the notebooks and report figures.

Colours follow a validated categorical palette (fixed order, colour-blind safe adjacent
pairs; checked with the dataviz validator). Colour follows the entity: a source or model
keeps its colour in every figure.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

#: Fixed entity -> colour mapping used across every figure.
SOURCE_COLORS = {
    "loc_fsa_color": SERIES[0],
    "documerica": SERIES[1],
    "div2k_valid": SERIES[2],
    "loc_fsa_bw": SERIES[3],
}
SOURCE_LABELS = {
    "loc_fsa_color": "FSA/OWI colour (1939-44)",
    "documerica": "DOCUMERICA (1972-77)",
    "div2k_valid": "DIV2K (modern)",
    "loc_fsa_bw": "FSA/OWI B&W negatives",
}
METHOD_COLORS = {"baseline": SERIES[2], "zero_shot": SERIES[0], "fine_tuned": SERIES[1]}


def apply_style() -> None:
    """Recessive axes and grid, thin lines, readable type."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "lines.linewidth": 2,
            "font.size": 10,
            "figure.dpi": 110,
            "savefig.bbox": "tight",
        }
    )


def image_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    ncols: int = 4,
    size: float = 3.0,
    suptitle: str | None = None,
) -> Figure:
    """Show images (HxWx3 float/uint8 or HxW mask) in a grid without axes.

    Args:
        images: Images to show.
        titles: Optional per-image captions.
        ncols: Grid columns.
        size: Inches per cell.
        suptitle: Optional figure title.

    Returns:
        The matplotlib figure.
    """
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    flat: list[Axes] = list(axes.ravel())
    for i, ax in enumerate(flat):
        ax.axis("off")
        if i >= len(images):
            continue
        img = images[i]
        ax.imshow(
            np.clip(img, 0, 1) if img.dtype != np.uint8 else img,
            cmap="gray" if img.ndim == 2 else None,
        )
        if titles is not None:
            ax.set_title(titles[i], fontsize=9, color=INK_2, fontweight="normal")
    if suptitle:
        fig.suptitle(suptitle, color=INK, fontweight="semibold")
    fig.tight_layout()
    return fig


def plot_training_run(
    metrics_csv: str | Path, metrics: Sequence[str] = ("psnr", "lpips"), title: str | None = None
) -> Figure:
    """Validation curves for one run, with classical-baseline and zero-shot references.

    Args:
        metrics_csv: ``metrics.csv`` written by the trainer.
        metrics: Validation metrics to plot (one panel each).
        title: Figure title.

    Returns:
        The matplotlib figure.
    """
    import pandas as pd

    frame = pd.read_csv(metrics_csv)
    val = frame[frame["kind"].isin(["zero_shot", "val"])].sort_values("step")
    baseline = frame[frame["kind"].str.startswith("baseline")]
    train = frame[frame["kind"] == "train"]
    fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(5 * (len(metrics) + 1), 3.6))
    ax0 = axes[0]
    ax0.plot(train["step"], train["loss"], color=SERIES[6], linewidth=1.5)
    ax0.set_title("Training loss", loc="left")
    ax0.set_xlabel("step")
    for ax, name in zip(axes[1:], metrics, strict=True):
        ax.plot(
            val["step"],
            val[name],
            color=METHOD_COLORS["fine_tuned"],
            marker="o",
            markersize=4,
            label="fine-tuning",
        )
        zero = val[val["kind"] == "zero_shot"]
        if not zero.empty:
            ax.axhline(
                float(zero[name].iloc[0]),
                color=METHOD_COLORS["zero_shot"],
                linestyle="--",
                linewidth=1.2,
                label="zero-shot",
            )
        if not baseline.empty and name in baseline:
            ax.axhline(
                float(baseline[name].iloc[0]),
                color=METHOD_COLORS["baseline"],
                linestyle=":",
                linewidth=1.5,
                label="classical baseline",
            )
        better = "higher" if name in ("psnr", "ssim", "psnr_hole") else "lower"
        ax.set_title(f"Validation {name.upper()} ({better} is better)", loc="left")
        ax.set_xlabel("step")
    axes[1].legend(loc="best")
    if title:
        fig.suptitle(title, color=INK, fontweight="semibold", x=0.01, ha="left")
    fig.tight_layout()
    return fig
