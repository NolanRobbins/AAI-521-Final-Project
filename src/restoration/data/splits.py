"""Leakage-safe train/val/test splits.

Archive collections contain many near-identical frames of the same scene (bracketed
exposures, consecutive shots). If one lands in train and its twin in test, test scores
are inflated. We therefore build *groups* of related images and split whole groups:

1. Images whose perceptual hashes (pHash, 64-bit) differ by at most ``threshold`` bits.
2. Images sharing an archive grouping key (``group_hint``), e.g. the NARA id or an
   identical normalised caption within the same source.

Groups are the connected components of the union of both relations.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

SPLITS = ("train", "val", "test")


def phash_int(path: Path) -> int:
    """64-bit perceptual hash of an image as an unsigned int."""
    with Image.open(path) as img:
        return int(str(imagehash.phash(img.convert("L"), hash_size=8)), 16)


class UnionFind:
    """Disjoint-set forest with path halving and union by size."""

    def __init__(self, n: int) -> None:
        """Create ``n`` singleton sets."""
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, i: int) -> int:
        """Return the representative of ``i``'s set."""
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        """Merge the sets containing ``a`` and ``b``."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def near_duplicate_pairs(hashes: Sequence[int], threshold: int) -> list[tuple[int, int]]:
    """All index pairs ``(i, j), i < j`` with Hamming distance <= ``threshold``.

    Vectorised row-by-row (O(n^2) popcounts, ~1 s for 10k images).
    """
    arr: NDArray[np.uint64] = np.asarray(hashes, dtype=np.uint64)
    pairs: list[tuple[int, int]] = []
    for i in range(len(arr) - 1):
        dist = np.bitwise_count(np.bitwise_xor(arr[i + 1 :], arr[i]))
        for offset in np.nonzero(dist <= threshold)[0]:
            pairs.append((i, i + 1 + int(offset)))
    return pairs


def normalise_caption(text: str) -> str:
    """Lower-case caption with punctuation/whitespace collapsed (for exact-match grouping)."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def assign_groups(
    hashes: Sequence[int], key_columns: Sequence[Sequence[str]], threshold: int
) -> NDArray[np.int64]:
    """Connected-component group id per image.

    Args:
        hashes: pHash per image.
        key_columns: Zero or more grouping-key sequences (one key per image each);
            images sharing a non-empty key in any column are linked.
        threshold: Max Hamming distance for near-duplicates.

    Returns:
        Group id per image (ids are component representatives).
    """
    n = len(hashes)
    if any(len(col) != n for col in key_columns):
        raise ValueError("every key column must have one entry per hash")
    uf = UnionFind(n)
    for i, j in near_duplicate_pairs(hashes, threshold):
        uf.union(i, j)
    for col in key_columns:
        first_with_key: dict[str, int] = {}
        for i, key in enumerate(col):
            if key:
                uf.union(first_with_key.setdefault(key, i), i)
    return np.array([uf.find(i) for i in range(n)], dtype=np.int64)


def split_groups(
    groups: NDArray[np.int64], val_fraction: float, test_fraction: float, seed: int
) -> NDArray[np.str_]:
    """Assign whole groups to train/val/test to approximate the target image fractions.

    Groups are shuffled with ``seed`` and greedily filled into test, then val, then train,
    so the split is deterministic and no group straddles two splits.

    Args:
        groups: Group id per image.
        val_fraction: Target fraction of images in val.
        test_fraction: Target fraction of images in test.
        seed: Shuffle seed.

    Returns:
        Split name per image.
    """
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")
    unique, counts = np.unique(groups, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(unique))
    n = len(groups)
    targets = {"test": test_fraction * n, "val": val_fraction * n}
    filled = {"test": 0, "val": 0}
    assignment: dict[int, str] = {}
    for idx in order:
        gid, size = int(unique[idx]), int(counts[idx])
        for split in ("test", "val"):
            fits = filled[split] + size <= targets[split]
            # allow one oversized group into an empty split so tiny datasets still get one
            first_fit = filled[split] == 0 and size <= 2 * targets[split]
            if fits or first_fit:
                assignment[gid] = split
                filled[split] += size
                break
        else:
            assignment[gid] = "train"
    return np.array([assignment[int(g)] for g in groups])


def make_split_frame(
    frame: pd.DataFrame,
    hashes: Sequence[int],
    threshold: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Add ``phash``, ``group`` and ``split`` columns to a curated manifest.

    Keys are namespaced by ``source`` so identical captions or ids from different
    archives are not merged.
    """

    def keyed(column: str, transform: Callable[[str], str] = str.strip) -> list[str]:
        return [
            f"{src}|{column}|{transform(str(val))}" if str(val).strip() else ""
            for src, val in zip(frame["source"], frame[column], strict=True)
        ]

    groups = assign_groups(
        hashes, [keyed("group_hint"), keyed("title", normalise_caption)], threshold
    )
    out = frame.copy()
    out["phash"] = [f"{h:016x}" for h in hashes]
    out["group"] = groups
    out["split"] = split_groups(groups, val_fraction, test_fraction, seed)
    return out
