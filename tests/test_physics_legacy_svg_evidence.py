"""Keep the seven published Physics title repairs bound to saved capture bytes."""

import hashlib
import json
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "status/physics-seven-legacy-svg-title-repairs-2026-10-25.json"
EXPECTED = {
    "8AqWr_4AAAAJ": (272, 23, 23),
    "JbSQsXAAAAAJ": (156, 21, 21),
    "jrRg4PAAAAAJ": (65, 11, 11),
    "jcb81yAAAAAJ": (184, 1, 1),
    "xKrmD5IAAAAJ": (44, 2, 1),
    "O0NnwB0AAAAJ": (46, 1, 0),
    "79uTAy8AAAAJ": (328, 2, 2),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


class _SavedTitles(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.titles: list[str] = []
        self.citation_ids: list[str] = []
        self.svg_rows: set[int] = set()
        self._parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "gsc_a_at" in classes:
            ids = parse_qs(urlparse(attributes.get("href") or "").query).get(
                "citation_for_view", []
            )
            assert len(ids) == 1
            self.citation_ids.append(ids[0])
            self._parts = []
            self._in_title = True
        elif self._in_title and tag == "svg" and "gs_fsvg" in classes:
            self.svg_rows.add(len(self.citation_ids))
            self._parts.append((attributes.get("aria-label") or "").strip())

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self.titles.append("".join(self._parts).strip())
            self._in_title = False


def test_seven_legacy_repairs_preserve_every_source_and_other_cache_field() -> None:
    review = json.loads(LEDGER.read_text())
    assert review["schema_version"] == 1
    assert review["review_state"] == "saved_html_svg_title_repair"
    assert review["scope"] == "seven_published_legacy_physics_profiles"
    assert review["raw_capture_and_receipt_unchanged"] is True
    assert review["profile_count"] == len(EXPECTED)
    assert review["svg_mismatch_count"] == 61
    assert review["cache_title_change_count"] == 59
    assert {item["scholar_id"] for item in review["profiles"]} == set(EXPECTED)

    entries = review["entries"]
    assert len(entries) == 61
    assert len({(item["scholar_id"], item["capture_row"]) for item in entries}) == 61
    assert sum(item["cache_change_required"] for item in entries) == 59

    for profile in review["profiles"]:
        scholar_id = profile["scholar_id"]
        count, mismatches, changes = EXPECTED[scholar_id]
        assert (
            profile["publication_count"],
            profile["svg_mismatch_count"],
            profile["cache_title_change_count"],
        ) == (count, mismatches, changes)
        cache_path = ROOT / profile["cache_path"]
        json_path = ROOT / profile["source_json_path"]
        html_path = ROOT / profile["source_html_path"]
        receipt_path = ROOT / profile["capture_receipt"]
        assert _sha256(cache_path) == profile["repaired_cache_sha256"]
        assert _sha256(json_path) == profile["source_json_sha256"]
        assert _sha256(html_path) == profile["source_html_sha256"]
        receipt = json.loads(receipt_path.read_text())
        assert receipt["source_json_sha256"] == profile["source_json_sha256"]
        assert receipt["source_html_sha256"] == profile["source_html_sha256"]

        cache = json.loads(cache_path.read_text())
        source = json.loads(json_path.read_text())
        html = _SavedTitles()
        html.feed(html_path.read_text())
        assert len(cache["publications"]) == len(source["works"]) == count
        assert len(html.titles) == len(html.citation_ids) == count
        by_row = {row["capture_row_index"]: row for row in cache["publications"]}
        assert set(by_row) == set(range(1, count + 1))
        repairs = {
            item["capture_row"]: item
            for item in entries
            if item["scholar_id"] == scholar_id
        }
        actual_mismatches = set()
        for index, source_row in enumerate(source["works"], start=1):
            cached = by_row[index]
            assert source_row["index"] == index
            assert cached["author_pub_id"] == html.citation_ids[index - 1]
            assert cached["pub_url"] == source_row["citation_url"]
            if _normalized(source_row["title"]) == _normalized(html.titles[index - 1]):
                continue
            actual_mismatches.add(index)
            entry = repairs[index]
            assert index in html.svg_rows
            assert entry["author_pub_id"] == cached["author_pub_id"]
            assert entry["captured_value"] == source_row["title"]
            assert entry["repaired_value"] == cached["title"] == html.titles[index - 1]
            assert entry["source_json_sha256"] == profile["source_json_sha256"]
            assert entry["source_html_sha256"] == profile["source_html_sha256"]
            assert entry["cache_change_required"] == (
                entry["prior_cache_value"] != cached["title"]
            )
        assert actual_mismatches == set(repairs)

        for index, entry in repairs.items():
            by_row[index]["title"] = entry["prior_cache_value"]
        baseline = (json.dumps(cache, indent=2, ensure_ascii=False) + "\n").encode()
        assert hashlib.sha256(baseline).hexdigest() == profile["baseline_cache_sha256"]
