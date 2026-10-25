"""Lock the AAS abstract byline correction to the saved Breivik capture."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHOLAR_ID = "AcW2EXwAAAAJ"
ROW116_ID = f"{SCHOLAR_ID}:qjMakFHDy7sC"
ROW121_ID = f"{SCHOLAR_ID}:ufrVoPGSRksC"
CAPTURED_AUTHOR = "A Polzin, A Geller, A Miller, K Breivik"
REPAIRED_AUTHOR = "A Geller, A Polzin, K Breivik, A Miller"


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_breivik_row116_repair_preserves_capture_and_hold() -> None:
    status = _json("status/physics-breivik-source-2026-10-24.json")
    ledger_path = status["cache_author_row116_repair_path"]
    ledger = _json(ledger_path)
    assert ledger["schema_version"] == 1
    assert ledger["review_state"] == "source_reviewed_cache_repaired"
    assert ledger["raw_capture_and_receipt_unchanged"] is True
    assert status["ingestion_ready"] is False
    assert status["snapshot_release_hold"] is True

    [repair] = ledger["repairs"]
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
        "capture_row": 116,
        "author_pub_id": ROW116_ID,
        "cache_field": "author",
        "captured_value": CAPTURED_AUTHOR,
        "repaired_value": REPAIRED_AUTHOR,
        "evidence_url": "https://aas.org/files/aas233_abstracts.pdf",
    }

    receipt = _json(ledger["capture_receipt"])
    assert receipt["scholar_id"] == SCHOLAR_ID
    assert receipt["publication_count"] == 131
    assert _sha256(receipt["source_json_path"]) == receipt["source_json_sha256"]
    assert _sha256(receipt["source_html_path"]) == receipt["source_html_sha256"]

    capture = _json(receipt["source_json_path"])
    cache = _json(f"data/authors/{SCHOLAR_ID}.json")
    assert len(capture["works"]) == len(cache["publications"]) == 131
    raw_row = capture["works"][115]
    rows = {row["author_pub_id"]: row for row in cache["publications"]}
    repaired_row = rows[ROW116_ID]
    assert raw_row["index"] == repaired_row["capture_row_index"] == 116
    assert raw_row["authors"] == CAPTURED_AUTHOR
    assert repaired_row["author"] == REPAIRED_AUTHOR
    assert repaired_row["venue"].endswith("363.17, 2019")
    assert rows[ROW121_ID]["capture_row_index"] == 121
    assert rows[ROW121_ID]["author"] == CAPTURED_AUTHOR
