"""Check Tina's saved-HTML extraction corrections and reviewed cache repair."""

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "status/physics-kahniashvili-cache-repair-2026-10-24.json"


class _SavedHTMLTitles(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.titles: dict[str, str] = {}
        self.current_id = ""
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a" and "gsc_a_at" in (attributes.get("class") or "").split():
            self.current_id = parse_qs(
                urlsplit(attributes.get("href") or "").query
            ).get("citation_for_view", [""])[0]
            self.parts = []
        elif (
            self.current_id
            and tag == "svg"
            and "gs_fsvg" in (attributes.get("class") or "").split()
        ):
            self.parts.append(attributes.get("aria-label") or "")

    def handle_data(self, data: str) -> None:
        if self.current_id:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.current_id:
            self.titles[self.current_id] = "".join(self.parts).strip()
            self.current_id = ""


def test_tina_row_116_cache_title_repair_preserves_capture() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    [repair] = ledger["repairs"]
    assert repair["capture_row"] == 116
    assert repair["cache_field"] == "title"
    assert repair["author_pub_id"] == "au_cLfoAAAAJ:VL0QpB8kHFEC"

    receipt = json.loads((ROOT / ledger["capture_receipt"]).read_text())
    for kind in ("json", "html"):
        source_path = ROOT / receipt[f"source_{kind}_path"]
        assert (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            == receipt[f"source_{kind}_sha256"]
        )

    capture = json.loads((ROOT / receipt["source_json_path"]).read_text())
    raw_row = next(row for row in capture["works"] if row["index"] == 116)
    assert raw_row["title"] == repair["captured_value"]
    assert parse_qs(urlsplit(raw_row["citation_url"]).query)["citation_for_view"] == [
        repair["author_pub_id"]
    ]

    cache = json.loads(
        (ROOT / "data/authors/au_cLfoAAAAJ.json").read_text(encoding="utf-8")
    )
    [cached_row] = [
        row
        for row in cache["publications"]
        if row["author_pub_id"] == repair["author_pub_id"]
    ]
    assert cached_row["capture_row_index"] == 116
    assert cached_row["title"] == repair["repaired_value"]
    assert len(cache["publications"]) == len(capture["works"]) == 134

    review = json.loads(
        (ROOT / "status/physics-kahniashvili-source-review-2026-10-24.json").read_text()
    )
    assert repair["author_pub_id"] not in {
        row["author_pub_id"] for row in review["proposed_exclusions"]
    }
    assert repair["author_pub_id"] in {
        row["author_pub_id"] for row in review["reviewed_retained"]
    }


def test_tina_svg_title_extraction_corrections_match_saved_html() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert ledger["raw_capture_and_receipt_unchanged"] is False
    assert ledger["source_html_unchanged"] is True
    assert (
        ledger["original_source_commit"] == "cd147263c153682e7004a62c0d97d9e79a646730"
    )
    assert ledger["original_source_json_sha256"] == (
        "6a6936548887252c66e7c9c70befdad3f7753e00f7d75ed85a0ab4a4b1b54d2b"
    )

    receipt = json.loads((ROOT / ledger["capture_receipt"]).read_text())
    raw_path = ROOT / receipt["source_json_path"]
    html_path = ROOT / receipt["source_html_path"]
    assert (
        hashlib.sha256(raw_path.read_bytes()).hexdigest()
        == receipt["source_json_sha256"]
        == ledger["corrected_source_json_sha256"]
    )
    assert (
        hashlib.sha256(html_path.read_bytes()).hexdigest()
        == receipt["source_html_sha256"]
        == "08a8d5b3bd39a8c7e12ec66a1d21295dc5b690d99338ec4240bfc6a1acabb232"
    )
    cache = json.loads((ROOT / "data/authors/au_cLfoAAAAJ.json").read_text())
    assert (
        cache["browser_capture"]["source_json_sha256"] == receipt["source_json_sha256"]
    )
    assert (
        cache["browser_capture"]["source_html_sha256"] == receipt["source_html_sha256"]
    )

    raw = json.loads(raw_path.read_text())
    parser = _SavedHTMLTitles()
    parser.feed(html_path.read_text())
    corrections = ledger["extraction_corrections"]
    assert {(row["capture_row"], row["author_pub_id"]) for row in corrections} == {
        (67, "au_cLfoAAAAJ:EUQCXRtRnyEC"),
        (110, "au_cLfoAAAAJ:j3f4tGmQtD8C"),
    }
    for correction in corrections:
        index = correction["capture_row"]
        publication_id = correction["author_pub_id"]
        source_row = raw["works"][index - 1]
        [cached_row] = [
            row for row in cache["publications"] if row["capture_row_index"] == index
        ]
        assert source_row["index"] == index
        assert parse_qs(urlsplit(source_row["citation_url"]).query)[
            "citation_for_view"
        ] == [publication_id]
        assert cached_row["author_pub_id"] == publication_id
        assert (
            correction["original_json_value"]
            != correction["corrected_json_and_cache_value"]
        )
        assert (
            source_row["title"]
            == cached_row["title"]
            == parser.titles[publication_id]
            == correction["corrected_json_and_cache_value"]
        )
        assert correction["saved_html_svg_aria_label"] in source_row["title"]
