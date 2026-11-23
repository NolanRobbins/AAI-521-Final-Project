"""Rule-based curation of archive scans before they are used as ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from restoration.config import CurationConfig


def drop_reasons(frame: pd.DataFrame, cfg: CurationConfig) -> pd.Series:
    """Return the first failing rule per image ("" if the image is kept).

    Percentile thresholds are computed per ``source`` so archives with different scan
    pipelines are each judged against their own distribution.
    """
    reasons = pd.Series("", index=frame.index, dtype=object)

    def flag(mask: pd.Series, reason: str) -> None:
        reasons[(reasons == "") & mask] = reason

    flag(frame[["width", "height"]].min(axis=1) < cfg.min_side, "too_small")
    border_v = (frame["border_top"] + frame["border_bottom"]) / frame["height"]
    border_h = (frame["border_left"] + frame["border_right"]) / frame["width"]
    flag((border_v > cfg.max_border_fraction) | (border_h > cfg.max_border_fraction), "borders")
    flag(frame["colorfulness"] < cfg.min_colorfulness, "near_greyscale")
    by_source = frame.groupby("source")
    sharp_min = by_source["sharpness"].transform(
        lambda s: np.percentile(s, cfg.sharpness_min_percentile)
    )
    noise_max = by_source["noise_sigma"].transform(
        lambda s: np.percentile(s, cfg.noise_max_percentile)
    )
    flag(frame["sharpness"] < sharp_min, "blurry")
    flag(frame["noise_sigma"] > noise_max, "noisy")
    return reasons
