"""Web app + pipeline integration tests with stub models (no GPU, no weights)."""

from __future__ import annotations

import io
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from restoration import pipeline as pl
from restoration.pipeline import PipelineConfig, RestorationPipeline, RestorationResult, TaskName
from restoration.webapp import create_app


class FakePipeline(RestorationPipeline):
    """Records calls and returns a brightened image instead of running models."""

    def __init__(self) -> None:
        super().__init__(
            PipelineConfig(use_finetuned=True, run_names={t: t for t in pl.TASK_ORDER})
        )
        self.calls: list[tuple[list[str], bool]] = []

    def run(self, img: np.ndarray, tasks: Iterable[TaskName], **kwargs: Any) -> RestorationResult:  # type: ignore[override]
        tasks = list(tasks)
        mask = kwargs.get("mask")
        self.calls.append((tasks, mask is not None))
        out = np.clip(img + 0.1, 0, 1).astype(np.float32)
        return RestorationResult(
            out, {t: out for t in tasks}, {t: 0.01 for t in tasks}, {t: "stub" for t in tasks}, mask
        )


def _png(arr: np.ndarray, mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("AAI521_DATA_ROOT", str(tmp_path))
    fake = FakePipeline()
    app = create_app(fake)
    app.config["TESTING"] = True
    with app.test_client() as c:
        c.fake = fake  # type: ignore[attr-defined]
        yield c


def test_index_lists_tasks_in_pipeline_order(client: Any) -> None:
    html = client.get("/").get_data(as_text=True)
    positions = [html.index(f'value="{t}"') for t in pl.TASK_ORDER]
    assert positions == sorted(positions)


def test_restore_round_trip_with_painted_mask(client: Any) -> None:
    img = (np.random.default_rng(0).random((60, 80, 3)) * 255).astype(np.uint8)
    mask = np.zeros((30, 40, 4), dtype=np.uint8)
    mask[10:20, 10:20] = (255, 64, 64, 150)  # painted stroke (half-resolution canvas)
    resp = client.post(
        "/api/restore",
        data={
            "image": (io.BytesIO(_png(img)), "photo.png"),
            "mask": (io.BytesIO(_png(mask, "RGBA")), "mask.png"),
            "tasks": ["colorize", "denoise"],
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["output"].startswith("data:image/jpeg;base64,")
    assert data["size"]["input"] == [60, 80]
    assert client.fake.calls == [(["colorize", "denoise"], True)]


@pytest.mark.parametrize(
    ("form", "error"),
    [({"tasks": ["denoise"]}, "No image"), ({"tasks": ["bogus"]}, "Select at least one task")],
)
def test_restore_validation_errors(client: Any, form: dict[str, Any], error: str) -> None:
    if "No image" not in error:
        form = {**form, "image": (io.BytesIO(_png(np.zeros((8, 8, 3), np.uint8))), "x.png")}
    resp = client.post("/api/restore", data=form, content_type="multipart/form-data")
    assert resp.status_code == 400 and error in resp.get_json()["error"]


def test_unreadable_image_is_rejected(client: Any) -> None:
    resp = client.post(
        "/api/restore",
        data={"image": (io.BytesIO(b"not an image"), "x.png"), "tasks": ["denoise"]},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_feedback_is_appended_to_csv(client: Any, tmp_path: Path) -> None:
    for rating in ("4", "5"):
        assert (
            client.post("/api/feedback", json={"overall": rating, "comment": "nice"}).status_code
            == 200
        )
    lines = (tmp_path / "outputs" / "user_feedback.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("timestamp,") and len(lines) == 3


def test_pipeline_runs_tasks_in_fixed_order(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    for name in ("denoise", "super_resolve"):
        monkeypatch.setattr(pl.inference, name, lambda m, x, _n=name, **k: (order.append(_n), x)[1])
    monkeypatch.setattr(pl.inference, "colorize", lambda m, x, s: (order.append("colorize"), x)[1])
    monkeypatch.setattr(
        pl.inference, "inpaint_sd", lambda p, x, m, **k: (order.append("inpaint"), x)[1]
    )
    pipe = RestorationPipeline(
        PipelineConfig(use_finetuned=False, run_names={t: t for t in pl.TASK_ORDER})
    )
    monkeypatch.setattr(pipe, "_model", lambda task: (None, 512) if task == "colorize" else None)
    img = np.zeros((16, 16, 3), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:6] = True
    result = pipe.run(img, ["super_resolution", "colorize", "inpaint", "denoise"], mask=mask)
    assert order == ["denoise", "inpaint", "colorize", "super_resolve"]
    assert list(result.stages) == ["input", "denoise", "inpaint", "colorize", "super_resolution"]


def test_pipeline_rejects_unknown_task() -> None:
    pipe = RestorationPipeline(PipelineConfig(run_names={t: t for t in pl.TASK_ORDER}))
    with pytest.raises(ValueError, match="Unknown tasks"):
        pipe.run(np.zeros((4, 4, 3), np.float32), ["sharpen"])  # type: ignore[list-item]


def test_detect_scratches_finds_a_thin_line() -> None:
    rng = np.random.default_rng(0)
    img = (0.4 + 0.05 * rng.random((200, 200, 3))).astype(np.float32)
    img[20:180, 100:102] = 0.95  # 2-px bright vertical scratch
    mask = pl.detect_scratches(img)
    assert mask[20:180, 100:102].mean() > 0.9
    assert mask.mean() < 0.1
