"""Unit tests for archive record parsing (no network)."""

from __future__ import annotations

from typing import Any

import pytest

from restoration.data import commons, loc

SERVICE = "https://tile.loc.gov/storage-services/service/pnp/fsac/1a33000/1a33800/1a33849v.jpg#h=715&w=1024"


@pytest.mark.parametrize(
    "url",
    [SERVICE, SERVICE.replace("v.jpg", "_150px.jpg"), SERVICE.replace("v.jpg", "t.gif")],
)
def test_iiif_urls_from_any_derivative(url: str) -> None:
    master, service = loc.iiif_urls_from_service_url(url)
    assert master == (
        "https://tile.loc.gov/image-services/iiif/"
        "master:pnp:fsac:1a33000:1a33800:1a33849u/full/full/0/default.jpg"
    )
    assert (
        service
        == "https://tile.loc.gov/storage-services/service/pnp/fsac/1a33000/1a33800/1a33849v.jpg"
    )


def test_iiif_urls_reject_foreign_urls() -> None:
    assert loc.iiif_urls_from_service_url("https://example.com/a.jpg") == ()


def _item(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": "http://www.loc.gov/item/2017877351/",
        "title": "Commuters waiting for the bus",
        "number_lccn": ["2017877351"],
        "image_url": [SERVICE],
        "date": "1939-01-01",
        "contributor": ["delano, jack"],
        "shelf_id": "LC-USF35-1 [P&P]",
        "location": ["lowell", "massachusetts"],
        "access_restricted": False,
    }
    item.update(overrides)
    return item


def test_parse_item_fields() -> None:
    rec = loc.parse_item(_item(), "loc_fsa_color")
    assert rec is not None
    assert rec.image_id == "loc_2017877351"
    assert rec.group_hint == "LC-USF35-1"
    assert rec.creator == "delano, jack"
    assert rec.fallback_urls and rec.url.endswith("/full/full/0/default.jpg")


@pytest.mark.parametrize("override", [{"access_restricted": True}, {"image_url": []}])
def test_parse_item_skips_unusable(override: dict[str, Any]) -> None:
    assert loc.parse_item(_item(**override), "loc_fsa_color") is None


def test_dedupe_prefers_jpeg_over_tiff() -> None:
    titles = [
        "File:A scene - NARA - 550302.tif",
        "File:A scene - NARA - 550302.jpg",
        "File:Other - NARA - 7.tif",
        "File:No id here.jpg",
        "File:Doc - NARA - 9.pdf",
    ]
    assert commons.dedupe_by_nara_id(titles) == {
        "550302": "File:A scene - NARA - 550302.jpg",
        "7": "File:Other - NARA - 7.tif",
    }


def test_clean_metadata_strips_html_and_hidden_divs() -> None:
    raw = 'June 1973<div style="display: none;">date QS:P571,+1973</div>'
    assert commons.clean_metadata(raw) == "June 1973"
    assert commons.clean_metadata("<bdi><a href='x'>Jim &amp; Olive</a></bdi>") == "Jim & Olive"
