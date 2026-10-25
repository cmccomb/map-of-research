"""Keep Dylan Quintana's incomplete career review out of the release graph."""

import csv
import hashlib
import json
import shutil

import pytest

from scripts.validate_dylan_source_review import (
    REVIEW_ROOT,
    load_held_candidates,
    validate_review,
)


def _rehash(root, name):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data = (root / name).read_bytes()
    manifest["files"][name]["bytes"] = len(data)
    manifest["files"][name]["sha256"] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_all_discovery_rows_and_versions_remain_held() -> None:
    manifest, rows = validate_review()
    loaded = load_held_candidates()

    assert manifest["verified_google_scholar_id"] is None
    assert manifest["lifetime_complete"] is False
    assert len(rows) == len(loaded) == 16
    assert sum(bool(row["openalex_id"]) for row in rows) == 11
    assert [row["source_record_id"] for row in loaded] == [
        row["source_record_id"] for row in rows
    ]
    assert all(row["canonical_work_id"] is None for row in loaded)
    assert all(row["canonical_author_ids"] == [] for row in loaded)
    assert all(row["scholar_publication_id"] is None for row in loaded)
    assert all(row["release_eligible"] is False for row in loaded)
    assert all(row["map_eligible"] is False for row in loaded)
    assert all(row["collaboration_eligible"] is False for row in loaded)


def test_a_discovery_only_aps_abstract_cannot_be_promoted_by_csv_edit(tmp_path) -> None:
    root = tmp_path / "review"
    shutil.copytree(REVIEW_ROOT, root)
    path = root / "candidate-crosswalk-16.csv"
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = list(reader)
    next(row for row in rows if row["source_record_id"] == "W2564317063")[
        "attribution_status"
    ] = "confirmed_primary"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _rehash(root, path.name)

    with pytest.raises(ValueError, match="known source conflicts or holds changed"):
        validate_review(root)


def test_a_raw_openalex_observation_cannot_be_lost_by_csv_edit(tmp_path) -> None:
    root = tmp_path / "review"
    shutil.copytree(REVIEW_ROOT, root)
    path = root / "candidate-crosswalk-16.csv"
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = list(reader)
    rows = [row for row in rows if row["source_record_id"] != "W2900132984"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _rehash(root, path.name)

    with pytest.raises(ValueError, match="crosswalk manifestation count changed"):
        validate_review(root)
