"""Library of Congress (loc.gov) collection crawler for the FSA/OWI photographs.

The loc.gov JSON API lists items with small derivative JPEGs (<= 1024 px). The full
scans are multi-megapixel TIFF masters (40-100 MB each). The LoC IIIF server renders a
master as a full-resolution JPEG (``full/full``) but returns HTTP 500 for any server-side
scaling request, so we fetch ``full/full`` and downscale locally (see
:func:`restoration.data.records.download_records`). If a master is not exposed via IIIF
we fall back to the 1024 px service derivative.

The search API refuses to paginate beyond 100,000 results (``index-depth`` redirect), so
random page sampling of the 170k-item B&W collection is capped at that depth.

API reference: https://www.loc.gov/apis/json-and-yaml/
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Iterator
from typing import Any

import requests

from restoration.data.records import ImageRecord
from restoration.utils.http import get_json

logger = logging.getLogger(__name__)

API_ROOT = "https://www.loc.gov/collections"
IIIF_ROOT = "https://tile.loc.gov/image-services/iiif"
SERVICE_ROOT = "https://tile.loc.gov/storage-services/service"
MAX_RESULT_DEPTH = 100_000
_SERVICE_RE = re.compile(
    r"https://tile\.loc\.gov/storage-services/service/(?P<path>(?:[\w-]+/)+)"
    r"(?P<stem>[0-9a-z]+?)(?P<suffix>_150px|[tvr])?\.(?:jpg|gif)"
)


def iiif_urls_from_service_url(service_url: str) -> tuple[str, ...]:
    """Derive download URLs (IIIF master first, then service JPEG) from a derivative URL.

    Example:
        ``.../storage-services/service/pnp/fsac/1a33000/1a33800/1a33849v.jpg`` becomes
        ``.../iiif/master:pnp:fsac:1a33000:1a33800:1a33849u/full/full/0/default.jpg``.

    Args:
        service_url: Any derivative URL from an item's ``image_url`` list.

    Returns:
        Candidate URLs in preference order; empty if the URL is not a P&P derivative.
    """
    match = _SERVICE_RE.match(service_url.split("#", 1)[0])
    if match is None:
        return ()
    path = match.group("path")
    segments = ":".join(path.strip("/").split("/"))
    stem = match.group("stem")
    return (
        f"{IIIF_ROOT}/master:{segments}:{stem}u/full/full/0/default.jpg",
        f"{SERVICE_ROOT}/{path}{stem}v.jpg",
    )


def _first_str(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def parse_item(item: dict[str, Any], source: str) -> ImageRecord | None:
    """Convert one loc.gov search result into an :class:`ImageRecord`.

    Args:
        item: A dict from the API's ``results`` list.
        source: Source name stored on the record.

    Returns:
        The record, or ``None`` if the item is access-restricted or has no usable image.
    """
    if item.get("access_restricted"):
        return None
    urls: tuple[str, ...] = ()
    for image_url in item.get("image_url") or []:
        urls = iiif_urls_from_service_url(image_url)
        if urls:
            break
    if not urls:
        return None
    lccn = _first_str(item.get("number_lccn")) or item.get("id", "").rstrip("/").split("/")[-1]
    if not lccn:
        return None
    return ImageRecord(
        image_id=f"loc_{lccn}",
        source=source,
        url=urls[0],
        fallback_urls=urls[1:],
        title=_first_str(item.get("title")),
        date=_first_str(item.get("date")),
        creator=_first_str(item.get("contributor")),
        license="No known restrictions on publication (LoC P&P)",
        group_hint=_first_str(item.get("shelf_id")).replace(" [P&P]", ""),
        page_url=_first_str(item.get("url") or item.get("id")),
        extra={"location": "; ".join(item.get("location") or [])},
    )


def _fetch_page(
    session: requests.Session, collection: str, page: int, per_page: int
) -> dict[str, Any]:
    params: dict[str, str | int] = {
        "fo": "json",
        "c": per_page,
        "sp": page,
        "at": "results,pagination",
    }
    # Redirects are disallowed: past the depth cap loc.gov redirects to an HTML error page.
    return get_json(session, f"{API_ROOT}/{collection}/", params, allow_redirects=False)


def iter_collection(
    session: requests.Session,
    collection: str,
    source: str,
    *,
    per_page: int = 100,
    pages: list[int] | None = None,
) -> Iterator[ImageRecord]:
    """Yield downloadable records from a loc.gov collection.

    Args:
        session: HTTP session.
        collection: Collection slug (e.g. ``"fsa-owi-color-photographs"``).
        source: Source name stored on records.
        per_page: Results per API page (max 1000; 100 keeps responses small).
        pages: Specific 1-based pages to fetch; ``None`` walks the whole collection.

    Yields:
        One record per unrestricted item with an image.
    """
    page_iter: Iterator[int] = iter(pages) if pages is not None else _count_from(1)
    for page in page_iter:
        try:
            payload = _fetch_page(session, collection, page, per_page)
        except (requests.RequestException, RuntimeError) as exc:
            if pages is None:
                raise  # a full walk must not silently lose pages
            logger.warning("Skipping sampled page %d: %s", page, exc)
            continue
        results = payload.get("results") or []
        for item in results:
            record = parse_item(item, source)
            if record is not None:
                yield record
        if pages is None and not (payload.get("pagination") or {}).get("next"):
            return


def _count_from(start: int) -> Iterator[int]:
    n = start
    while True:
        yield n
        n += 1


def collection_size(session: requests.Session, collection: str) -> int:
    """Return the number of items the API reports for a collection."""
    payload = _fetch_page(session, collection, page=1, per_page=1)
    return int(payload["pagination"]["of"])


def crawl_collection(
    session: requests.Session,
    collection: str,
    source: str,
    limit: int | None,
    seed: int,
    per_page: int = 100,
) -> list[ImageRecord]:
    """Crawl a collection, optionally as a seeded random sample of pages.

    Small collections (or ``limit=None``) are crawled in full. For huge collections such
    as the 170k-item B&W negatives, we sample random result pages so the candidate pool
    spans the whole archive rather than its first few hundred catalog entries.

    Args:
        session: HTTP session.
        collection: Collection slug.
        source: Source name.
        limit: Maximum number of records, or ``None`` for all.
        seed: RNG seed for page sampling.
        per_page: Results per page.

    Returns:
        De-duplicated records, at most ``limit`` of them.
    """
    total = collection_size(session, collection)
    n_pages = min(-(-total // per_page), MAX_RESULT_DEPTH // per_page)
    logger.info("%s: %d items; %d pages reachable", collection, total, n_pages)

    pages: list[int] | None = None
    if limit is not None and limit < total:
        rng = random.Random(seed)
        # Restricted items are dropped, so oversample pages by 2x.
        wanted = min(n_pages, max(1, 2 * -(-limit // per_page)))
        pages = sorted(rng.sample(range(1, n_pages + 1), wanted))

    records: dict[str, ImageRecord] = {}
    for record in iter_collection(session, collection, source, per_page=per_page, pages=pages):
        records.setdefault(record.image_id, record)
    selected = list(records.values())
    if limit is not None and len(selected) > limit:
        selected = random.Random(seed).sample(selected, limit)
    logger.info("%s: %d downloadable records selected", collection, len(selected))
    return selected
