"""Wikimedia Commons crawler for the NARA DOCUMERICA series (EPA, 1972-1977).

NARA mirrors DOCUMERICA to Commons, where each photo usually appears twice (a ``.jpg``
and a ``.tif`` carrying the same ``NARA - <id>`` suffix). We list the category, dedupe on
the NARA id (preferring JPEG), take a seeded sample, then resolve server-rendered
thumbnails at a standard Wikimedia width.

API reference: https://www.mediawiki.org/wiki/API:Categorymembers and API:Imageinfo
"""

from __future__ import annotations

import html
import logging
import random
import re
from collections.abc import Iterator, Sequence
from typing import Any

import requests

from restoration.data.records import ImageRecord
from restoration.utils.http import get_json

logger = logging.getLogger(__name__)

API_URL = "https://commons.wikimedia.org/w/api.php"
_NARA_ID_RE = re.compile(r"NARA - (\d+)")
_TAG_RE = re.compile(r"<[^>]+>")
_IMAGE_EXT_PRIORITY = {".jpg": 0, ".jpeg": 0, ".png": 1, ".tif": 2, ".tiff": 2}
IMAGEINFO_BATCH = 50  # MediaWiki limit for anonymous clients


def nara_id(title: str) -> str | None:
    """Extract the NARA identifier from a Commons file title, if present."""
    match = _NARA_ID_RE.search(title)
    return match.group(1) if match else None


def _extension(title: str) -> str:
    dot = title.rfind(".")
    return title[dot:].lower() if dot >= 0 else ""


def clean_metadata(value: str) -> str:
    """Strip HTML tags/entities and hidden QuickStatements from extmetadata values."""
    value = re.sub(r'<div style="display: none;">.*?</div>', "", value, flags=re.S)
    return " ".join(html.unescape(_TAG_RE.sub(" ", value)).split())


def iter_category_files(session: requests.Session, category: str) -> Iterator[str]:
    """Yield every file title in a Commons category (non-recursive).

    Args:
        session: HTTP session.
        category: Full category title including the ``Category:`` prefix.

    Yields:
        File titles (``"File:..."``).
    """
    params: dict[str, str | int] = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "file",
        "cmlimit": "500",
        "format": "json",
        "formatversion": "2",
    }
    while True:
        payload = get_json(session, API_URL, params)
        if "error" in payload:
            raise RuntimeError(f"Commons API error: {payload['error']}")
        for member in payload["query"]["categorymembers"]:
            yield str(member["title"])
        cont = payload.get("continue")
        if not cont:
            return
        params.update({k: str(v) for k, v in cont.items()})


def dedupe_by_nara_id(titles: Sequence[str]) -> dict[str, str]:
    """Map NARA id -> preferred file title (JPEG over PNG over TIFF).

    Titles without a NARA id or with non-image extensions are dropped.
    """
    best: dict[str, str] = {}
    for title in titles:
        ext = _extension(title)
        ident = nara_id(title)
        if ident is None or ext not in _IMAGE_EXT_PRIORITY:
            continue
        current = best.get(ident)
        if current is None or _IMAGE_EXT_PRIORITY[ext] < _IMAGE_EXT_PRIORITY[_extension(current)]:
            best[ident] = title
    return best


def resolve_records(
    session: requests.Session,
    titles_by_id: dict[str, str],
    source: str,
    max_side: int,
) -> list[ImageRecord]:
    """Resolve thumbnail URLs and metadata for the given files.

    Args:
        session: HTTP session.
        titles_by_id: NARA id -> Commons file title.
        source: Source name stored on records.
        max_side: Requested thumbnail width (use a standard Wikimedia width).

    Returns:
        Records with ``url`` pointing at a JPEG thumbnail (or the original if smaller).
    """
    id_by_title = {t: i for i, t in titles_by_id.items()}
    titles = list(titles_by_id.values())
    records: list[ImageRecord] = []
    for start in range(0, len(titles), IMAGEINFO_BATCH):
        batch = titles[start : start + IMAGEINFO_BATCH]
        payload = get_json(
            session,
            API_URL,
            {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iiurlwidth": max_side,
                "iiextmetadatafilter": "DateTimeOriginal|Artist|LicenseShortName|ObjectName",
                "format": "json",
                "formatversion": "2",
            },
        )
        for page in payload["query"]["pages"]:
            info_list = page.get("imageinfo")
            if not info_list:
                continue
            info = info_list[0]
            meta = {k: clean_metadata(v.get("value", "")) for k, v in info.get("extmetadata", {}).items()}
            ident = id_by_title.get(page["title"])
            if ident is None:
                continue
            # Commons returns the original when it is already narrower than max_side;
            # TIFF originals must still go through the thumbnailer to get a JPEG.
            url = info.get("thumburl") or info["url"]
            records.append(
                ImageRecord(
                    image_id=f"documerica_{ident}",
                    source=source,
                    url=url,
                    title=meta.get("ObjectName", page["title"]),
                    date=meta.get("DateTimeOriginal", ""),
                    creator=meta.get("Artist", ""),
                    license=meta.get("LicenseShortName", "Public domain"),
                    group_hint=ident,
                    page_url=str(info.get("descriptionurl", "")),
                    extra={"commons_title": page["title"], "original_mime": str(info.get("mime", ""))},
                )
            )
    return records


def crawl_documerica(
    session: requests.Session,
    category: str,
    source: str,
    max_side: int,
    limit: int | None,
    seed: int,
) -> list[ImageRecord]:
    """List, dedupe, sample and resolve DOCUMERICA images.

    Args:
        session: HTTP session.
        category: Commons category title.
        source: Source name.
        max_side: Thumbnail width.
        limit: Number of unique photos to sample, or ``None`` for all.
        seed: Sampling seed.

    Returns:
        Downloadable records.
    """
    titles = list(iter_category_files(session, category))
    by_id = dedupe_by_nara_id(titles)
    logger.info("Commons: %d files -> %d unique NARA ids", len(titles), len(by_id))
    ids = sorted(by_id)
    if limit is not None and limit < len(ids):
        ids = sorted(random.Random(seed).sample(ids, limit))
    records = resolve_records(session, {i: by_id[i] for i in ids}, source, max_side)
    logger.info("Commons: resolved %d records", len(records))
    return records
