"""The Palmese paired draft partitions a complete unchanged source capture."""

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids_sha256(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def test_palmese_review_accounts_for_every_saved_row():
    root = Path(__file__).resolve().parents[1]
    ledger = json.loads(
        (root / "status/physics-palmese-release-review-2026-10-26.json").read_text()
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

    assert ledger["person_id"] == "person-antonellapalmese-5378b53d"
    assert ledger["scholar_id"] == "6pPZKd0AAAAJ"
    assert _sha256(capture_path) == ledger["source_capture_sha256"]
    assert _sha256(cache_path) == ledger["source_cache_sha256"]
    assert cache["publication_count"] == ledger["source_rows"] == 624
    assert cache["browser_capture"]["more_remain"] is False
    assert capture["more_remain"] is False
    rows = {work["index"]: work for work in capture["works"]}
    assert set(rows) == set(range(1, 625))
    ids = {
        row: work["citation_url"].split("citation_for_view=", 1)[1]
        for row, work in rows.items()
    }
    assert len(set(ids.values())) == 624

    excluded = set(ledger["reviewed_exclusion_rows"])
    held = set(ledger["provisional_hold_rows"])
    ordinary = set(ledger["ordinary_gate_capture_indices"])
    assert excluded == {338, 531, 544}
    assert excluded.isdisjoint(held)
    assert excluded.isdisjoint(ordinary)
    assert held.isdisjoint(ordinary)
    assert excluded | held | ordinary == set(rows)
    assert (len(excluded), len(held), len(ordinary)) == (3, 346, 275)
    assert (
        _ids_sha256({ids[row] for row in excluded})
        == ledger["reviewed_exclusion_ids_sha256"]
    )
    assert (
        _ids_sha256({ids[row] for row in held}) == ledger["provisional_hold_ids_sha256"]
    )
    assert {27, 43, 177, 539, 585, 586, 615, 616, 618, 619, 624} <= held
    assert {25, 58, 70, 86, 118, 128, 145, 175, 198, 221} <= ordinary
    decisions = ledger["hold_decisions"]
    assert {item["capture_row_index"] for item in decisions} == held
    assert all(
        item["author_pub_id"] == ids[item["capture_row_index"]]
        and item["citation_url"] == rows[item["capture_row_index"]]["citation_url"]
        and item["reason"]
        for item in decisions
    )
    assert profile["state"] == "reviewed_for_paired_release"
    assert profile["paired_release_review_path"] == (
        "status/physics-palmese-release-review-2026-10-26.json"
    )
    assert profile["reviewed_exclusion_count"] == len(excluded)
    assert profile["provisional_hold_count"] == len(held)
    assert profile["ordinary_gate_rows"] == len(ordinary)
    assert status["release_state"] == "held"
