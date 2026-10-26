"""Keep the seven published Physics title repairs bound to saved capture bytes."""

import hashlib
import json
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyarrow.parquet as parquet

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


def test_repaired_snapshot_preserves_every_other_row_and_field() -> None:
    ledger = json.loads(LEDGER.read_text())
    preservation = ledger["snapshot_preservation"]
    snapshot_path = ROOT / "snapshots/cmu-engineering-publications.parquet"
    manifest = json.loads(
        (
            ROOT / "snapshots/cmu-engineering-publications.parquet.manifest.json"
        ).read_text()
    )
    successful = json.loads((ROOT / "status/last-successful-upload.json").read_text())
    assert preservation["baseline_snapshot_sha256"] == successful["raw_snapshot_sha256"]
    assert (
        preservation["repaired_snapshot_sha256"]
        == manifest["snapshot_sha256"]
        == _sha256(snapshot_path)
    )
    assert (
        preservation["source_rows_before_and_after"]
        == manifest["publication_rows"]
        == 59434
    )
    assert (
        preservation["curated_non_scholar_rows_preserved"]
        == manifest["curated_work_rows"]
        == 43
    )
    assert preservation["only_changed_columns"] == ["title", "source_record_json"]

    repairs = {
        item["author_pub_id"]: item
        for item in ledger["entries"]
        if item["cache_change_required"]
    }
    assert len(repairs) == preservation["changed_row_count"] == 59
    assert (
        hashlib.sha256("\n".join(sorted(repairs)).encode()).hexdigest()
        == preservation["changed_author_pub_ids_sha256"]
    )
    repaired = hashlib.sha256()
    reconstructed_baseline = hashlib.sha256()
    seen: set[str] = set()
    curated_rows = 0
    table = parquet.read_table(snapshot_path)
    assert table.num_rows == 59434
    for row in table.to_pylist():
        if row["scholar_id"] == "":
            curated_rows += 1
        repaired.update(
            (
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            ).encode()
        )
        before = dict(row)
        pub_id = row["author_pub_id"]
        if pub_id in repairs:
            assert pub_id not in seen
            seen.add(pub_id)
            item = repairs[pub_id]
            assert row["title"] == item["repaired_value"]
            record = json.loads(row["source_record_json"])
            assert record["title"] == item["repaired_value"]
            before["title"] = item["prior_cache_value"]
            record["title"] = item["prior_cache_value"]
            before["source_record_json"] = json.dumps(
                record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        reconstructed_baseline.update(
            (
                json.dumps(
                    before,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            ).encode()
        )
    assert seen == set(repairs)
    assert curated_rows == 43
    assert repaired.hexdigest() == preservation["repaired_content_sha256"]
    assert reconstructed_baseline.hexdigest() == preservation["baseline_content_sha256"]
