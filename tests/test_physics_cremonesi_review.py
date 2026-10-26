"""Cremonesi's source handoff accounts for the complete unchanged capture."""

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids_digest(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def test_cremonesi_review_partitions_every_saved_row_and_keeps_global_hold():
    root = Path(__file__).resolve().parents[1]
    ledger_path = root / "status/physics-cremonesi-release-review-2026-10-26.json"
    ledger = json.loads(ledger_path.read_text())
    capture_path = root / ledger["source_capture_path"]
    cache_path = root / ledger["source_cache_path"]
    capture = json.loads(capture_path.read_text())
    cache = json.loads(cache_path.read_text())
    status = json.loads(
        (
            root / "status/physics-cremonesi-schumacher-source-2026-10-24.json"
        ).read_text()
    )

    assert ledger["person_id"] == "person-matteocremonesi-b2aba740"
    assert ledger["scholar_id"] == "wJ_gQLoAAAAJ"
    assert ledger["primary_academic_home"]["department"] == "Physics"
    assert _sha256(capture_path) == ledger["source_capture_sha256"]
    assert _sha256(cache_path) == ledger["source_cache_sha256"]
    assert cache["publication_count"] == ledger["source_rows"] == 1963
    assert cache["browser_capture"]["more_remain"] is False
    by_row = {work["index"]: work for work in capture["works"]}
    assert len(by_row) == 1963
    assert set(by_row) == set(range(1, 1964))
    ids = {
        row: work["citation_url"].split("citation_for_view=", 1)[1]
        for row, work in by_row.items()
    }
    assert len(set(ids.values())) == 1963

    excluded = set(ledger["reviewed_exclusion_rows"])
    held = set(ledger["provisional_hold_rows"])
    ordinary = set(ledger["ordinary_gate_capture_indices"])
    assert excluded.isdisjoint(held)
    assert excluded.isdisjoint(ordinary)
    assert held.isdisjoint(ordinary)
    assert excluded | held | ordinary == set(by_row)
    assert (len(excluded), len(held), len(ordinary)) == (80, 1087, 796)
    assert (
        _ids_digest({ids[row] for row in excluded})
        == ledger["reviewed_exclusion_ids_sha256"]
    )
    assert (
        _ids_digest({ids[row] for row in held}) == ledger["provisional_hold_ids_sha256"]
    )
    assert held == {row for rows in ledger["hold_categories"].values() for row in rows}
    assert {68, 1430, 1926, 1962} <= held
    assert {183, 801, 1170, 1246} <= held
    assert {
        item["author_pub_id"] for item in ledger["specific_unresolved_rows"].values()
    } == set(status["unresolved_exact_ids"])
    assert set(ledger["verified_retain_ids_from_prior_followup"]).isdisjoint(
        {ids[row] for row in excluded}
    )

    decision = status["paired_profile_release_ready"][ledger["scholar_id"]]
    assert decision["reviewed_exclusion_count"] == len(excluded)
    assert decision["provisional_hold_count"] == len(held)
    assert decision["unresolved_exact_ids"] == []
    assert decision["review_ledger_path"] == str(ledger_path.relative_to(root))
    assert decision["review_ledger_sha256"] == _sha256(ledger_path)
    assert status["map_quality_hold"] is True
    assert ledger["scholar_id"] in status["map_quality_hold_scholar_ids"]
    assert status["producer_review_ready"] is False
