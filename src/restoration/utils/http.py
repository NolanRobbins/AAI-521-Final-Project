"""HTTP helpers: a polite retrying session, a rate limiter and atomic downloads."""

from __future__ import annotations

import logging
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Archives (LoC, Wikimedia) ask automated clients to identify themselves.
USER_AGENT = (
    "AAI521-PhotoRestoration/0.1 "
    "(student research; +https://github.com/NolanRobbins/AAI-521-Final-Project)"
)


def build_session(
    user_agent: str = USER_AGENT,
    total_retries: int = 5,
    backoff_factor: float = 1.0,
    pool_size: int = 16,
) -> requests.Session:
    """Create a session that retries transient failures with exponential backoff.

    Args:
        user_agent: ``User-Agent`` header sent with every request.
        total_retries: Maximum retries per request.
        backoff_factor: Backoff multiplier between retries (seconds).
        pool_size: Connection pool size; should be >= the number of worker threads.

    Returns:
        Configured :class:`requests.Session`.
    """
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504, 520, 521, 522, 524),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_size, pool_maxsize=pool_size)
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class RateLimiter:
    """Thread-safe limiter enforcing a minimum interval between calls."""

    def __init__(self, max_per_second: float) -> None:
        """Create a limiter.

        Args:
            max_per_second: Maximum sustained call rate across all threads.
        """
        if max_per_second <= 0:
            raise ValueError(f"max_per_second must be positive, got {max_per_second}")
        self._interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def wait(self) -> None:
        """Block until the caller may issue its next request."""
        with self._lock:
            now = time.monotonic()
            delay = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._interval
        if delay > 0:
            time.sleep(delay)


_TRANSIENT_BODY_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.JSONDecodeError,
)


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, str | int] | None = None,
    *,
    attempts: int = 4,
    backoff: float = 2.0,
    timeout: float = 60.0,
    allow_redirects: bool = True,
) -> dict[str, Any]:
    """GET a JSON object, retrying failures that ``urllib3.Retry`` does not cover.

    ``Retry`` handles connection errors and retryable status codes, but not a response
    whose body is cut off mid-read (``IncompleteRead``) or is truncated JSON, both of
    which archive APIs produce under load.

    Args:
        session: HTTP session.
        url: Endpoint.
        params: Query parameters.
        attempts: Total attempts for transient body errors.
        backoff: Base for exponential backoff between attempts (seconds).
        timeout: Per-request timeout.
        allow_redirects: Passed to ``requests``; when False a redirect raises.

    Returns:
        Decoded JSON object.

    Raises:
        RuntimeError: If redirects are disallowed and the server redirects.
        requests.RequestException: If all attempts fail.
    """
    for attempt in range(1, attempts + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout, allow_redirects=allow_redirects)
            if resp.is_redirect and not allow_redirects:
                raise RuntimeError(f"{url} redirected to {resp.headers.get('location')}")
            resp.raise_for_status()
            payload = resp.json()
        except _TRANSIENT_BODY_ERRORS as exc:
            if attempt == attempts:
                raise
            delay = backoff**attempt
            logger.warning("GET %s failed (%s); retry %d in %.0fs", url, exc, attempt, delay)
            time.sleep(delay)
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object from {url}")
        return payload
    raise AssertionError("unreachable")


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    timeout: float = 120.0,
    rate_limiter: RateLimiter | None = None,
    chunk_size: int = 1 << 16,
) -> Path:
    """Stream ``url`` to ``dest`` atomically, skipping files that already exist.

    The payload is written to ``<dest>.part`` and renamed on success, so an interrupted
    run never leaves a truncated file that a later run would mistake for a finished one.

    Args:
        session: HTTP session (see :func:`build_session`).
        url: Source URL.
        dest: Destination path.
        timeout: Per-request timeout in seconds.
        rate_limiter: Optional limiter shared across worker threads.
        chunk_size: Streaming chunk size in bytes.

    Returns:
        ``dest``.

    Raises:
        requests.HTTPError: On a non-2xx response after retries.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    if rate_limiter is not None:
        rate_limiter.wait()
    try:
        with session.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size):
                    fh.write(chunk)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def extract_zip(archive: Path, out_dir: Path, marker_name: str = ".extracted") -> Path:
    """Extract a zip archive once, guarded by a marker file.

    Args:
        archive: Path to the ``.zip`` file.
        out_dir: Extraction directory.
        marker_name: File written into ``out_dir`` after a successful extraction.

    Returns:
        ``out_dir``.
    """
    marker = out_dir / marker_name
    if marker.exists():
        logger.info("Already extracted: %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise zipfile.BadZipFile(f"Corrupt member {bad!r} in {archive}")
        zf.extractall(out_dir)
    marker.touch()
    logger.info("Extracted %s -> %s", archive.name, out_dir)
    return out_dir
