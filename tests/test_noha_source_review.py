"""Keep Noha Abdelghany observations and possible ElGarem aliases held."""

import csv
import hashlib
import json
import shutil

import pytest

from scripts.validate_noha_source_review import (
    REVIEW_ROOT,
    load_held_candidates,
    validate_review,
)


def test_confirmed_and_alias_observations_remain_source_only() -> None:
    manifest, rows = validate_review()
    loaded = load_held_candidates()

    assert manifest["verified_google_scholar_id"] is None
    assert manifest["lifetime_complete"] is False
    assert len(rows) == len(loaded) == 5
    assert [row["attribution_status"] for row in loaded] == [
        "confirmed_person_candidate",
        "confirmed_person_candidate",
        "confirmed_person_candidate",
        "unresolved_alias_candidate",
        "unresolved_alias_candidate",
    ]
    assert [row["genre"] for row in loaded[:3]] == [
        "research_article",
        "research_article",
        "doctoral_dissertation",
    ]
    assert all(row["canonical_work_id"] is None for row in loaded)
    assert all(row["canonical_author_ids"] == [] for row in loaded)
    assert all(row["scholar_publication_id"] is None for row in loaded)
    assert all(row["release_eligible"] is False for row in loaded)
    assert all(row["map_eligible"] is False for row in loaded)
    assert all(row["collaboration_eligible"] is False for row in loaded)


def test_alias_cannot_be_promoted_with_only_crosswalk_edit(tmp_path) -> None:
    root = tmp_path / "review"
    shutil.copytree(REVIEW_ROOT, root)
    crosswalk = root / "candidate-crosswalk-5.csv"
    with crosswalk.open(newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        rows = list(reader)
    rows[3]["attribution_status"] = "confirmed_person_candidate"
    with crosswalk.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data = crosswalk.read_bytes()
    manifest["files"][crosswalk.name]["bytes"] = len(data)
    manifest["files"][crosswalk.name]["sha256"] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unresolved alias cannot be promoted"):
        validate_review(root)
