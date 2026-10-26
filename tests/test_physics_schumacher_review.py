"""The Schumacher review partitions an unchanged complete saved capture."""

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids_digest(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def test_schumacher_review_accounts_for_every_saved_row():
    root = Path(__file__).resolve().parents[1]
    ledger = json.loads(
        (root / "status/physics-schumacher-release-review-2026-10-26.json").read_text()
    )
    capture_path = root / ledger["source_capture_path"]
    cache_path = root / ledger["source_cache_path"]
    capture = json.loads(capture_path.read_text())
    cache = json.loads(cache_path.read_text())
    status = json.loads(
        (root / "status/physics-nine-held-source-2026-10-25.json").read_text()
    )
    profile = next(
        item
        for item in status["profiles"]
        if item["scholar_id"] == ledger["scholar_id"]
    )

    assert ledger["person_id"] == "person-reinhardschumacher-35cd3c6f"
    assert ledger["scholar_id"] == "LibSWRMAAAAJ"
    assert _sha256(capture_path) == ledger["source_capture_sha256"]
    assert _sha256(cache_path) == ledger["source_cache_sha256"]
    assert cache["publication_count"] == ledger["source_rows"] == 543
    assert cache["browser_capture"]["more_remain"] is False
    by_row = {work["index"]: work for work in capture["works"]}
    assert set(by_row) == set(range(1, 544))

    excluded = set(ledger["reviewed_exclusion_rows"])
    held = set(ledger["provisional_hold_rows"])
    ordinary = set(ledger["ordinary_gate_capture_indices"])
    assert excluded.isdisjoint(held)
    assert excluded.isdisjoint(ordinary)
    assert held.isdisjoint(ordinary)
    assert excluded | held | ordinary == set(by_row)
    assert (len(excluded), len(held), len(ordinary)) == (26, 258, 259)
    pub_ids = {
        row: by_row[row]["citation_url"].split("citation_for_view=", 1)[1]
        for row in by_row
    }
    assert len(set(pub_ids.values())) == 543
    assert (
        _ids_digest({pub_ids[row] for row in excluded})
        == ledger["reviewed_exclusion_ids_sha256"]
    )
    assert (
        _ids_digest({pub_ids[row] for row in held})
        == ledger["provisional_hold_ids_sha256"]
    )
    assert {513, 514, 521, 527, 528, 529} <= held
    assert {1, 51, 150, 207, 233, 261, 268, 295, 332, 351} <= ordinary
    assert profile["state"] == "reviewed_for_paired_release"
    assert profile["paired_release_review_path"] == (
        "status/physics-schumacher-release-review-2026-10-26.json"
    )
    assert status["release_state"] == "held"
    paired_status = json.loads(
        (
            root / "status/physics-cremonesi-schumacher-source-2026-10-24.json"
        ).read_text()
    )
    decision = paired_status["paired_profile_release_ready"][ledger["scholar_id"]]
    assert paired_status["map_quality_hold"] is True
    assert "wJ_gQLoAAAAJ" in paired_status["paired_profile_release_ready"]
    assert "wJ_gQLoAAAAJ" in paired_status["map_quality_hold_scholar_ids"]
    assert decision["map_quality_hold"] is False
    assert decision["producer_review_ready"] is True
    assert decision["unresolved_exact_ids"] == []
    assert decision["reviewed_exclusion_count"] == len(excluded)
    assert decision["provisional_hold_count"] == len(held)
    assert decision["review_ledger_path"] == profile["paired_release_review_path"]
    assert (
        _sha256(root / decision["review_ledger_path"])
        == decision["review_ledger_sha256"]
    )
