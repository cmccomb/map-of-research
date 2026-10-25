"""Keep Chatterjee's saved-HTML title restoration tied to source provenance."""

import hashlib
import json
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCHOLAR_ID = "wovjAPsAAAAJ"


def _read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def test_chatterjee_svg_title_restoration_keeps_capture_evidence_and_hold() -> None:
    status = _read_json("status/physics-eleven-profile-source-2026-10-24.json")
    ledger = _read_json(status["chatterjee_svg_title_repair_path"])
    receipt = _read_json(ledger["capture_receipt"])
    raw = _read_json(receipt["source_json_path"])
    cache = _read_json(f"data/authors/{SCHOLAR_ID}.json")

    assert status["ingestion_ready"] is False
    assert status["source_files_preserve_raw_rows"] is False
    assert ledger["review_state"] == "saved_html_extraction_corrected"
    assert ledger["scholar_id"] == receipt["scholar_id"] == SCHOLAR_ID
    assert ledger["original_source_commit"] == (
        "1da9ddf3b925f2a5ac3ba01054acc7ee0b6aec4d"
    )
    assert ledger["original_source_json_sha256"] == (
        "e9e6701454c954abaed4a020c786ab7d589ef535683ac936dc973ec0b6fd54d0"
    )
    assert receipt["source_json_sha256"] == ledger["corrected_source_json_sha256"]
    assert _sha256(receipt["source_json_path"]) == receipt["source_json_sha256"]
    assert _sha256(receipt["source_html_path"]) == receipt["source_html_sha256"]
    assert receipt["source_html_sha256"] == ledger["source_html_sha256_unchanged"]
    assert (
        cache["browser_capture"]["source_json_sha256"]
        == (receipt["source_json_sha256"])
    )
    assert (
        cache["browser_capture"]["source_html_sha256"]
        == (receipt["source_html_sha256"])
    )
    assert len(raw["works"]) == len(cache["publications"]) == 90
    by_index = {row["capture_row_index"]: row for row in cache["publications"]}
    assert set(by_index) == set(range(1, 91))
    assert {entry["capture_row"] for entry in ledger["extraction_corrections"]} == {
        34,
        36,
        39,
        88,
    }
    for index, source_row in enumerate(raw["works"], start=1):
        cached_row = by_index[index]
        assert source_row["index"] == index
        assert parse_qs(urlsplit(source_row["citation_url"]).query)[
            "citation_for_view"
        ] == [cached_row["author_pub_id"]]
        assert source_row["title"] == cached_row["title"]
        assert _normalized(source_row["authors"]) == _normalized(cached_row["author"])
        assert _normalized(source_row["venue"]) == _normalized(cached_row["venue"])
        assert _normalized(source_row["year"]) == _normalized(cached_row["pub_year"])
    for correction in ledger["extraction_corrections"]:
        row = raw["works"][correction["capture_row"] - 1]
        assert (
            by_index[correction["capture_row"]]["author_pub_id"]
            == (correction["author_pub_id"])
        )
        assert row["title"] == correction["corrected_json_and_cache_value"]
