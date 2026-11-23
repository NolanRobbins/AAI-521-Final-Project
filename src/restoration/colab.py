"""One-call environment setup for the Google Colab notebooks.

Layout on Colab::

    /content/drive/MyDrive/<drive_dir>/aai521_dataset.tar   (uploaded once; see pack_dataset.py)
    /content/drive/MyDrive/<drive_dir>/checkpoints/         (persists across sessions)
    /content/aai521_data/                                    (fast local copy, per session)

Usage in a notebook (after ``pip install -e .``)::

    from restoration import colab
    colab.setup()
"""

from __future__ import annotations

import os
import shutil
import tarfile
import time
from pathlib import Path

from restoration.config import CHECKPOINT_ROOT_ENV, DATA_ROOT_ENV

LOCAL_DATA = Path("/content/aai521_data")
DRIVE_ROOT = Path("/content/drive/MyDrive")


def in_colab() -> bool:
    """True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def setup(drive_dir: str = "aai521", archive: str = "aai521_dataset.tar") -> dict[str, str]:
    """Mount Drive, stage the dataset on local disk and point env vars at it.

    Idempotent: re-running in the same session skips the copy/extract.
    Outside Colab it is a no-op that reports the current configuration, so the same
    notebook runs locally.

    Args:
        drive_dir: Folder under ``MyDrive`` holding the archive and checkpoints.
        archive: Dataset archive file name produced by ``scripts/pack_dataset.py``.

    Returns:
        The resolved data and checkpoint roots.
    """
    if not in_colab():
        from restoration.config import checkpoint_root, data_root

        return {"data_root": str(data_root()), "checkpoint_root": str(checkpoint_root())}

    from google.colab import drive

    drive.mount("/content/drive")
    base = DRIVE_ROOT / drive_dir
    ckpt = base / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    os.environ[DATA_ROOT_ENV] = str(LOCAL_DATA)
    os.environ[CHECKPOINT_ROOT_ENV] = str(ckpt)
    # Hugging Face weights also cached on Drive: ~3.5 GB downloaded once, not per session.
    os.environ.setdefault("HF_HOME", str(base / "hf_cache"))

    marker = LOCAL_DATA / ".extracted"
    if not marker.exists():
        src = base / archive
        if not src.is_file():
            raise FileNotFoundError(
                f"{src} not found. Run scripts/pack_dataset.py locally and upload the tar "
                f"to MyDrive/{drive_dir}/, or run scripts/download_data.py in Colab."
            )
        start = time.time()
        LOCAL_DATA.mkdir(parents=True, exist_ok=True)
        local_tar = Path("/content") / archive
        shutil.copyfile(src, local_tar)
        with tarfile.open(local_tar) as tar:
            tar.extractall(LOCAL_DATA, filter="data")
        local_tar.unlink()
        marker.touch()
        print(f"Dataset staged to {LOCAL_DATA} in {time.time() - start:.0f}s")
    return {"data_root": str(LOCAL_DATA), "checkpoint_root": str(ckpt)}
