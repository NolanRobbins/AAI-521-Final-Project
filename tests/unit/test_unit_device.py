"""Device selection order: CUDA -> MPS -> CPU."""

from __future__ import annotations

import pytest
import torch

from restoration.utils.device import default_device


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_default_device_order(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert default_device().type == expected
