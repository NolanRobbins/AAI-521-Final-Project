"""Unit tests for leakage-safe grouping and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from restoration.data import splits


def test_near_duplicate_pairs_threshold() -> None:
    hashes = [0b0000, 0b0011, 0b1111_0000, 0b0001]
    assert splits.near_duplicate_pairs(hashes, threshold=1) == [(0, 3), (1, 3)]
    assert set(splits.near_duplicate_pairs(hashes, threshold=2)) == {(0, 1), (0, 3), (1, 3)}


def test_assign_groups_is_transitive_across_keys_and_hashes() -> None:
    # 0~1 by hash, 1~2 by key, 3 isolated -> {0,1,2}, {3}
    hashes = [0, 1, 0xFFFF_FFFF_0000_0000, 0x00FF_00FF_00FF_00FF]
    keys = ["", "a", "a", "b"]
    groups = splits.assign_groups(hashes, [keys], threshold=1)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


@settings(max_examples=30, deadline=None)
@given(
    group_sizes=st.lists(st.integers(1, 6), min_size=12, max_size=80),
    seed=st.integers(0, 10_000),
)
def test_split_groups_never_straddles_and_covers_all(group_sizes: list[int], seed: int) -> None:
    groups = np.repeat(np.arange(len(group_sizes)), group_sizes).astype(np.int64)
    assigned = splits.split_groups(groups, 0.1, 0.1, seed)
    assert set(assigned) <= set(splits.SPLITS)
    for gid in np.unique(groups):
        assert len(set(assigned[groups == gid])) == 1


def test_split_fractions_are_close_for_singletons() -> None:
    groups = np.arange(1000, dtype=np.int64)
    assigned = splits.split_groups(groups, 0.1, 0.2, seed=0)
    assert (assigned == "val").sum() == 100
    assert (assigned == "test").sum() == 200


def test_make_split_frame_groups_identical_captions_within_source_only() -> None:
    frame = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "source": ["loc", "loc", "documerica"],
            "title": ["Farm, Iowa.", "farm iowa", "Farm Iowa"],
            "group_hint": ["", "", ""],
        }
    )
    out = splits.make_split_frame(frame, [0, 2**63, 2**40], 0, 0.3, 0.3, seed=0)
    assert out.loc[0, "group"] == out.loc[1, "group"]
    assert out.loc[2, "group"] != out.loc[0, "group"]
    assert list(out.columns[-3:]) == ["phash", "group", "split"]
