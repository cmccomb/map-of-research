"""Protect the Marion Oliver source review's identity and release holds."""

import hashlib
import json
import shutil

import pytest

from scripts.validate_oliver_source_review import (
    EXPECTED_DOIS,
    REVIEW_ROOT,
    load_held_candidates,
    validate_review,
)


def test_exact_publication_candidates_remain_held() -> None:
    manifest, rows = validate_review()
    loaded = load_held_candidates()

    assert manifest["lifetime_complete"] is False
    assert manifest["verified_scholar_id"] is None
    assert {row["doi"] for row in rows} == EXPECTED_DOIS
    assert len(loaded) == 3
    assert sum("Williams" in row["authors_as_deposited"] for row in loaded) == 2
    assert all(row["canonical_work_id"] is None for row in loaded)
    assert all(row["canonical_author_ids"] == [] for row in loaded)
    assert all(row["scholar_publication_id"] is None for row in loaded)
    assert all(row["release_eligible"] is False for row in loaded)
    assert all(row["map_eligible"] is False for row in loaded)
    assert all(row["collaboration_eligible"] is False for row in loaded)


def test_medical_namesake_cannot_be_promoted_by_review_edit(tmp_path) -> None:
    root = tmp_path / "review"
    shutil.copytree(REVIEW_ROOT, root)
    collision_path = root / "name-collision-review.json"
    collisions = json.loads(collision_path.read_text())
    medical = next(
        item
        for item in collisions
        if item["openalex_author_id"].endswith("A5023340417")
    )
    medical["review_decision"] = "included_source_candidate"
    collision_path.write_text(json.dumps(collisions))

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data = collision_path.read_bytes()
    manifest["files"][collision_path.name]["bytes"] = len(data)
    manifest["files"][collision_path.name]["sha256"] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        ValueError, match="medical namesake clusters cannot be included"
    ):
        validate_review(root)
