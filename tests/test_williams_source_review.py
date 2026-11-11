"""Guard the unreleased Williams source observations against accidental promotion."""

import csv
import hashlib
import json
import shutil

import pytest

from scripts.validate_williams_source_review import (
    REVIEW_ROOT,
    load_held_candidates,
    validate_review,
)


def test_frozen_observations_have_no_graph_or_publication_path():
    manifest, candidates = validate_review()
    loaded = load_held_candidates()

    assert manifest["verified_scholar_id"] is None
    assert manifest["lifetime_complete"] is False
    assert len(candidates) == len(loaded) == 65
    assert {row["source_candidate_id"] for row in loaded} == {
        row["candidate_id"] for row in candidates
    }
    assert all(
        row["canonical_work_id"] is None
        and row["canonical_author_ids"] == []
        and row["scholar_publication_id"] is None
        and row["release_eligible"] is False
        and row["map_eligible"] is False
        and row["collaboration_eligible"] is False
        for row in loaded
    )


def test_reviewed_book_creators_cannot_become_collaboration_authors():
    loaded = {row["doi"]: row for row in load_held_candidates() if row["doi"]}

    physics_today = loaded["10.1063/1.2915016"]
    asme = loaded["10.1115/1.3162673"]
    assert "Billington" in physics_today["authors_as_deposited"]
    assert "Perzyna" in asme["authors_as_deposited"]
    assert physics_today["reviewed_genre"] == asme["reviewed_genre"] == "book_review"
    assert physics_today["canonical_author_ids"] == []
    assert asme["canonical_author_ids"] == []


def test_changed_review_genre_fails_even_if_manifest_hash_is_recomputed(tmp_path):
    review = tmp_path / "review"
    shutil.copytree(REVIEW_ROOT, review)
    crosswalk = review / "candidate-crosswalk-65.csv"
    with crosswalk.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    for row in rows:
        if row["doi"] == "10.1063/1.2915016":
            row["reviewed_genre"] = "research_article"
    with crosswalk.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = review / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data = crosswalk.read_bytes()
    manifest["files"][crosswalk.name]["sha256"] = hashlib.sha256(data).hexdigest()
    manifest["files"][crosswalk.name]["bytes"] = len(data)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="review bylines must remain under review"):
        validate_review(review)
