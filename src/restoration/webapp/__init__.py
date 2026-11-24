"""Flask web app: upload a photo, pick restoration tasks, compare before/after.

Endpoints:
    GET  /               single-page UI (upload, mask brush, task selection, results)
    POST /api/restore    multipart: image, optional mask PNG, tasks, options -> JSON
    POST /api/feedback   JSON user-testing ratings -> appended to feedback.csv
    GET  /api/health     liveness + device info

Run locally:  ``python -m restoration.webapp``  (http://127.0.0.1:5000)
On Colab:     see notebooks/04_web_app.ipynb
"""

from __future__ import annotations

import base64
import csv
import io
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from restoration.config import data_root
from restoration.data import degradations as deg
from restoration.pipeline import TASK_ORDER, PipelineConfig, RestorationPipeline

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 2**20
MAX_INPUT_SIDE = 2048  # larger uploads are downscaled before processing
FEEDBACK_FIELDS = (
    "timestamp",
    "tasks",
    "weights",
    "overall",
    "denoise",
    "inpaint",
    "colorize",
    "super_resolution",
    "comment",
)


def encode_image(img: np.ndarray, fmt: str = "JPEG", max_side: int | None = None) -> str:
    """Float ``[0,1]`` RGB (or bool mask) -> base64 data URL."""
    arr = img.astype(np.float32) if img.dtype == np.bool_ else img
    if max_side is not None and arr.ndim == 3:
        arr = deg.resize_max_side(arr, max_side)
    pil = Image.fromarray(deg.to_uint8(arr) if arr.ndim == 3 else (arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def decode_upload(data: bytes) -> np.ndarray:
    """Uploaded file bytes -> float RGB image capped at :data:`MAX_INPUT_SIDE`."""
    with Image.open(io.BytesIO(data)) as pil:
        rgb = np.asarray(pil.convert("RGB"))
    return deg.resize_max_side(deg.to_float(rgb), MAX_INPUT_SIDE)


def decode_mask(data: bytes, shape: tuple[int, int]) -> np.ndarray | None:
    """Painted mask PNG (any non-transparent, bright stroke = damage) -> bool mask."""
    with Image.open(io.BytesIO(data)) as pil:
        rgba = np.asarray(
            pil.convert("RGBA").resize((shape[1], shape[0]), Image.Resampling.NEAREST)
        )
    mask = (rgba[..., 3] > 0) & (rgba[..., :3].max(axis=-1) > 127)
    return mask if mask.any() else None


def create_app(pipeline: RestorationPipeline | None = None) -> Flask:
    """Application factory.

    Args:
        pipeline: Pre-built pipeline (tests inject a fake); defaults to a lazily loaded one.

    Returns:
        Configured Flask app.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    pipelines: dict[bool, RestorationPipeline] = {}
    if pipeline is not None:
        pipelines[pipeline.cfg.use_finetuned] = pipeline
    gpu_lock = threading.Lock()  # one restoration at a time on the GPU
    feedback_path = data_root() / "outputs" / "user_feedback.csv"

    def get_pipeline(finetuned: bool) -> RestorationPipeline:
        if finetuned not in pipelines:
            # Keep only one pipeline's models resident to bound GPU memory.
            for other in pipelines.values():
                other.unload()
            pipelines.clear()
            pipelines[finetuned] = RestorationPipeline(PipelineConfig(use_finetuned=finetuned))
        return pipelines[finetuned]

    @app.get("/")
    def index() -> str:
        return render_template("index.html", tasks=TASK_ORDER)

    @app.get("/api/health")
    def health() -> Any:
        import torch

        return jsonify(status="ok", cuda=torch.cuda.is_available())

    @app.post("/api/restore")
    def restore() -> Any:
        upload = request.files.get("image")
        if upload is None:
            return jsonify(error="No image uploaded"), 400
        tasks = [t for t in request.form.getlist("tasks") if t in TASK_ORDER]
        if not tasks:
            return jsonify(error="Select at least one task"), 400
        try:
            img = decode_upload(upload.read())
        except (UnidentifiedImageError, OSError):
            return jsonify(error="Could not read the image file"), 400
        mask = None
        if "mask" in request.files:
            mask = decode_mask(request.files["mask"].read(), img.shape[:2])
        finetuned = request.form.get("finetuned", "true") == "true"
        auto_mask = request.form.get("auto_mask", "false") == "true"
        prompt = request.form.get("prompt") or None
        start = time.perf_counter()
        with gpu_lock:
            pipe = get_pipeline(finetuned)
            kwargs: dict[str, Any] = {"mask": mask, "auto_mask": auto_mask}
            if prompt:
                kwargs["prompt"] = prompt
            result = pipe.run(img, tasks, **kwargs)
        total = time.perf_counter() - start
        return jsonify(
            input=encode_image(img, max_side=1600),
            output=encode_image(result.image, max_side=2400),
            output_full=encode_image(result.image, fmt="PNG"),
            stages={
                k: encode_image(v, max_side=800) for k, v in result.stages.items() if k != "input"
            },
            mask=encode_image(result.mask, fmt="PNG") if result.mask is not None else None,
            seconds={k: round(v, 2) for k, v in result.seconds.items()},
            total_seconds=round(total, 2),
            weights=result.weights,
            size={"input": list(img.shape[:2]), "output": list(result.image.shape[:2])},
        )

    @app.post("/api/feedback")
    def feedback() -> Any:
        payload = request.get_json(silent=True) or {}
        row = {k: str(payload.get(k, ""))[:2000] for k in FEEDBACK_FIELDS}
        row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        new = not feedback_path.is_file()
        with feedback_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FEEDBACK_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        return jsonify(status="saved")

    @app.errorhandler(413)
    def too_large(_: Exception) -> Any:
        return jsonify(error=f"File too large (max {MAX_UPLOAD_BYTES // 2**20} MB)"), 413

    return app
