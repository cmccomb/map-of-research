"""Bind Dodelson's derived title repair to the saved capture provenance."""

import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCHOLAR_ID = "kcXOB1MAAAAJ"
BASE = "physics-scott-dodelson-2026-10-24"
LEDGER = ROOT / "status/physics-dodelson-svg-title-repair-2026-10-25.json"
EXPECTED_REPAIR_IDS = {
    36: "kcXOB1MAAAAJ:__bU50VfleQC",
    51: "kcXOB1MAAAAJ:Zph67rFs4hoC",
    57: "kcXOB1MAAAAJ:UebtZRa9Y70C",
    116: "kcXOB1MAAAAJ:dAp6zn-oMfAC",
    131: "kcXOB1MAAAAJ:oi2SiIJ9l4AC",
    141: "kcXOB1MAAAAJ:tHtfpZlB6tUC",
    170: "kcXOB1MAAAAJ:IsPWOBWtZBwC",
    184: "kcXOB1MAAAAJ:ziOE8S1-AIUC",
    239: "kcXOB1MAAAAJ:_tF6a-HnqWAC",
    246: "kcXOB1MAAAAJ:KTwcwpFFj4wC",
    267: "kcXOB1MAAAAJ:aEW5N-EHWIMC",
    274: "kcXOB1MAAAAJ:WWeOtg8bX_EC",
    292: "kcXOB1MAAAAJ:YsMSGLbcyi4C",
    327: "kcXOB1MAAAAJ:M_lZXyI38BkC",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _citation_id(url: str) -> str:
    [value] = parse_qs(urlsplit(url).query)["citation_for_view"]
    return value


def test_dodelson_title_repair_preserves_capture_provenance_and_hold() -> None:
    ledger = _json(LEDGER)
    assert ledger["scholar_id"] == SCHOLAR_ID
    assert ledger["source_release_hold"] is True
    assert ledger["source_branch_base_commit"] == (
        "1da9ddf3b925f2a5ac3ba01054acc7ee0b6aec4d"
    )
    assert ledger["original_source_json_blob_sha1"] == (
        "d6b1262d8e57a4f155ef2cc110e6d3b0d548001f"
    )

    batch_path = ROOT / ledger["original_guarded_batch_receipt_path"]
    batch = _json(batch_path)
    assert _sha256(batch_path) == ledger["original_guarded_batch_receipt_sha256"]
    [original_dodelson] = [
        row for row in batch["profile_capture"] if row["scholar_id"] == SCHOLAR_ID
    ]
    assert (
        original_dodelson["source_json_sha256"]
        == (ledger["original_source_json_sha256"])
    )

    raw_path = ROOT / ledger["source_json_path"]
    html_path = ROOT / ledger["source_html_path"]
    receipt = _json(ROOT / ledger["individual_capture_receipt_path"])
    cache = _json(ROOT / f"data/authors/{SCHOLAR_ID}.json")
    assert _sha256(raw_path) == ledger["repaired_source_json_sha256"]
    assert (
        receipt["source_json_sha256"]
        == cache["browser_capture"]["source_json_sha256"]
        == ledger["repaired_source_json_sha256"]
    )
    assert (
        _sha256(html_path)
        == receipt["source_html_sha256"]
        == ledger["source_html_sha256"]
    )

    raw = _json(raw_path)
    raw_rows = raw["works"]
    cache_rows = cache["publications"]
    assert len(raw_rows) == len(cache_rows) == receipt["publication_count"] == 443
    cache_by_index = {row["capture_row_index"]: row for row in cache_rows}
    assert set(cache_by_index) == set(range(1, 444))
    assert len({row["author_pub_id"] for row in cache_rows}) == 443

    repairs = ledger["repairs"]
    assert {row["capture_row"]: row["author_pub_id"] for row in repairs} == (
        EXPECTED_REPAIR_IDS
    )
    for repair in repairs:
        index = repair["capture_row"]
        source_row = raw_rows[index - 1]
        cache_row = cache_by_index[index]
        assert source_row["index"] == index
        assert repair["field"] == "title"
        assert (
            repair["original_derived_json_value"] != repair["restored_saved_html_value"]
        )
        assert repair["original_cache_value"] != repair["restored_saved_html_value"]
        assert _citation_id(source_row["citation_url"]) == repair["author_pub_id"]
        assert cache_row["author_pub_id"] == repair["author_pub_id"]
        assert (
            source_row["title"]
            == cache_row["title"]
            == repair["restored_saved_html_value"]
        )

    assert ledger["unresolved_capture_rows"] == [435, 436]
    for index, title, citation_id in (
        (435, "Book• Third Edition• 2025", "kcXOB1MAAAAJ:3lUAU8Oskd0C"),
        (436, "Book• Second Edition• 2020", "kcXOB1MAAAAJ:0sTkTiv_uMkC"),
    ):
        assert raw_rows[index - 1]["title"] == title
        assert _citation_id(raw_rows[index - 1]["citation_url"]) == citation_id

    source_status = _json(ROOT / "status/physics-eleven-profile-source-2026-10-24.json")
    assert source_status["ingestion_ready"] is False
