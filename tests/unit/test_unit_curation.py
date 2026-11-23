"""Unit tests for curation rules."""

from __future__ import annotations

import pandas as pd

from restoration.config import CurationConfig
from restoration.data.curation import drop_reasons


def _frame() -> pd.DataFrame:
    n = 20
    frame = pd.DataFrame(
        {
            "source": ["a"] * n,
            "width": [2048] * n,
            "height": [1400] * n,
            "border_top": [0] * n,
            "border_bottom": [0] * n,
            "border_left": [0] * n,
            "border_right": [0] * n,
            "colorfulness": [30.0] * n,
            "sharpness": [float(100 + i) for i in range(n)],
            "noise_sigma": [0.01] * n,
        }
    )
    frame.loc[1, "width"] = 800
    frame.loc[2, "border_top"] = 700  # 50% of the height
    frame.loc[3, "colorfulness"] = 2.0
    frame.loc[0, "sharpness"] = 1.0
    frame.loc[4, "noise_sigma"] = 0.5
    return frame


def test_each_rule_flags_its_image_and_keeps_the_rest() -> None:
    cfg = CurationConfig(train_sources=["a"], sharpness_min_percentile=5, noise_max_percentile=95)
    reasons = drop_reasons(_frame(), cfg)
    assert reasons[0] == "blurry"
    assert reasons[1] == "too_small"
    assert reasons[2] == "borders"
    assert reasons[3] == "near_greyscale"
    assert reasons[4] == "noisy"
    assert (reasons[5:] == "").all()


def test_first_failing_rule_wins() -> None:
    frame = _frame()
    frame.loc[1, "colorfulness"] = 0.0  # also greyscale, but too_small is checked first
    assert drop_reasons(frame, CurationConfig(train_sources=["a"]))[1] == "too_small"
