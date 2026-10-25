"""Keep Tina row 116's source title and reviewed cache repair distinct."""

import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "status/physics-kahniashvili-cache-repair-2026-10-24.json"


def test_tina_row_116_cache_title_repair_preserves_capture() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert ledger["raw_capture_and_receipt_unchanged"] is True
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
