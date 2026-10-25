"""Keep the reviewed Breivik byline repair tied to unchanged capture evidence."""

import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCHOLAR_ID = "AcW2EXwAAAAJ"
AUTHOR_PUB_ID = f"{SCHOLAR_ID}:roLk4NBRz8UC"
CAPTURED_AUTHOR = "K Breivik, S Chatterjee, J Andrews"
REPAIRED_AUTHOR = "J Andrews, S Chatterjee, K Breivik"
SOURCE_JSON_SHA256 = "89cc02ae3a6319b05ea905baf715a207d41254c3935fea15500c98a2b0c38253"
SOURCE_HTML_SHA256 = "67563a99261141eb6ab016622605ddf4f6c8c16fda577888afa64a13a0be758d"
BATCH_SHA256 = "88a1db96e95183879bcf489ce561c34daa506d3594f37500716a716506ad4dac"


def _read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_breivik_row115_author_repair_keeps_capture_evidence() -> None:
    ledger = _read_json("status/physics-breivik-row115-cache-repair-2026-10-24.json")
    assert ledger["schema_version"] == 1
    assert ledger["review_state"] == "source_reviewed_cache_repaired"
    assert ledger["raw_capture_and_receipt_unchanged"] is True
    assert ledger["capture_receipt"] == (
        "status/browser-capture-physics-katelyn-breivik-2026-10-24.json"
    )
    assert len(ledger["repairs"]) == 1
    repair = ledger["repairs"][0]
    assert {
        key: repair[key]
        for key in (
            "scholar_id",
            "capture_row",
            "author_pub_id",
            "cache_field",
            "captured_value",
            "repaired_value",
            "evidence_url",
        )
    } == {
        "scholar_id": SCHOLAR_ID,
        "capture_row": 115,
        "author_pub_id": AUTHOR_PUB_ID,
        "cache_field": "author",
        "captured_value": CAPTURED_AUTHOR,
        "repaired_value": REPAIRED_AUTHOR,
        "evidence_url": "https://aas.org/files/aas233_abstracts.pdf",
    }

    receipt = _read_json(ledger["capture_receipt"])
    assert receipt["scholar_id"] == SCHOLAR_ID
    assert receipt["publication_count"] == 131
    assert receipt["source_json_sha256"] == SOURCE_JSON_SHA256
    assert receipt["source_html_sha256"] == SOURCE_HTML_SHA256
    assert _sha256(receipt["source_json_path"]) == SOURCE_JSON_SHA256
    assert _sha256(receipt["source_html_path"]) == SOURCE_HTML_SHA256
    batch_path = "status/browser-capture-physics-breivik-katoch-batch-2026-10-24.json"
    assert _sha256(batch_path) == BATCH_SHA256

    capture = _read_json(receipt["source_json_path"])
    cache = _read_json(f"data/authors/{SCHOLAR_ID}.json")
    assert len(capture["works"]) == len(cache["publications"]) == 131
    by_index = {row["capture_row_index"]: row for row in cache["publications"]}
    assert set(by_index) == set(range(1, 132))
    for index, source_row in enumerate(capture["works"], start=1):
        cached_row = by_index[index]
        assert source_row["index"] == index
        assert parse_qs(urlsplit(source_row["citation_url"]).query)[
            "citation_for_view"
        ] == [cached_row["author_pub_id"]]
        assert source_row["authors"] == (
            CAPTURED_AUTHOR if index == 115 else cached_row["author"]
        )
    assert by_index[115]["author_pub_id"] == AUTHOR_PUB_ID
    assert by_index[115]["author"] == REPAIRED_AUTHOR
    assert by_index[115]["venue"].endswith("112.01, 2019")
    assert by_index[117]["author"] == REPAIRED_AUTHOR
