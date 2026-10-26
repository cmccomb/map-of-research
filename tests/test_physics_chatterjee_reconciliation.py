"""The already published Chatterjee profile has a complete source partition."""

import hashlib
import json
from pathlib import Path


def _digest(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def test_chatterjee_source_status_matches_complete_capture():
    root = Path(__file__).resolve().parents[1]
    ledger_path = (
        root / "status/physics-chatterjee-publication-reconciliation-2026-10-26.json"
    )
    ledger = json.loads(ledger_path.read_text())
    capture_path = root / ledger["source_capture_path"]
    cache_path = root / ledger["source_cache_path"]
    capture = json.loads(capture_path.read_text())
    cache = json.loads(cache_path.read_text())
    status = json.loads(
        (root / "status/physics-nine-held-source-2026-10-25.json").read_text()
    )
    profile = next(
        item for item in status["profiles"] if item["scholar_id"] == "wovjAPsAAAAJ"
    )

    assert ledger["person_id"] == "person-shubhayuchatterjee-34929dff"
    assert (
        hashlib.sha256(capture_path.read_bytes()).hexdigest()
        == ledger["source_capture_sha256"]
    )
    assert (
        hashlib.sha256(cache_path.read_bytes()).hexdigest()
        == ledger["source_cache_sha256"]
    )
    assert cache["publication_count"] == ledger["source_rows"] == 90
    assert capture["more_remain"] is False
    by_row = {work["index"]: work for work in capture["works"]}
    assert set(by_row) == set(range(1, 91))
    ids = {
        row: work["citation_url"].split("citation_for_view=", 1)[1]
        for row, work in by_row.items()
    }
    assert len(set(ids.values())) == 90

    excluded = set(ledger["reviewed_exclusion_rows"])
    held = set(ledger["provisional_hold_rows"])
    ordinary = set(ledger["ordinary_gate_capture_indices"])
    assert excluded == {62}
    assert held == {65, 66, 73, 86}
    assert excluded.isdisjoint(held)
    assert excluded.isdisjoint(ordinary)
    assert held.isdisjoint(ordinary)
    assert excluded | held | ordinary == set(by_row)
    assert (len(excluded), len(held), len(ordinary)) == (1, 4, 85)
    assert (
        _digest({ids[row] for row in excluded})
        == ledger["reviewed_exclusion_ids_sha256"]
    )
    assert _digest({ids[row] for row in held}) == ledger["provisional_hold_ids_sha256"]
    assert profile["state"] == "published_on_hub_parent"
    assert (
        profile["paired_release_review_path"]
        == ledger_path.relative_to(root).as_posix()
    )
    assert profile["published_hub_revision"] == ledger["published_hub_revision"]
    assert profile["reviewed_exclusion_count"] == len(excluded)
    assert profile["provisional_hold_count"] == len(held)
    assert profile["ordinary_gate_rows"] == len(ordinary)
    assert status["release_state"] == "held"
