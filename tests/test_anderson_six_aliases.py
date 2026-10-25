"""Keep Theresa Anderson title variants exact and out of the release graph."""

import csv
import hashlib
import json
import shutil

import pytest

from scripts.validate_anderson_six_aliases import (
    REVIEW_ROOT,
    load_held_aliases,
    validate_review,
)


def _rehash(root, name):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data = (root / name).read_bytes()
    manifest["files"][name]["bytes"] = len(data)
    manifest["files"][name]["sha256"] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_six_distinct_published_work_aliases_remain_held() -> None:
    manifest, rows = validate_review()
    loaded = load_held_aliases()

    assert manifest["verified_google_scholar_id"] is None
    assert manifest["lifetime_complete"] is False
    assert len(rows) == len(loaded) == 6
    assert sum(bool(row["publisher_doi"]) for row in rows) == 5
    assert [row["source_ordinal"] for row in loaded] == [21, 22, 23, 24, 25, 33]
    assert all(row["canonical_work_id"] is None for row in loaded)
    assert all(row["canonical_author_ids"] == [] for row in loaded)
    assert all(row["scholar_publication_id"] is None for row in loaded)
    assert all(row["release_eligible"] is False for row in loaded)
    assert all(row["map_eligible"] is False for row in loaded)
    assert all(row["collaboration_eligible"] is False for row in loaded)


def test_a_doi_cannot_be_reassigned_by_resolution_edit(tmp_path) -> None:
    root = tmp_path / "review"
    shutil.copytree(REVIEW_ROOT, root)
    path = root / "alias-resolution-six.json"
    rows = json.loads(path.read_text())
    rows[2]["publisher_doi"] = rows[1]["publisher_doi"]
    path.write_text(json.dumps(rows))
    _rehash(root, path.name)

    with pytest.raises(ValueError, match="DOI alias collision"):
        validate_review(root)


def test_an_original_cv_title_cannot_be_silently_replaced(tmp_path) -> None:
    root = tmp_path / "review"
    shutil.copytree(REVIEW_ROOT, root)
    path = root / "author-site-cv-raw-six.csv"
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = list(reader)
    rows[2]["cv_citation"] = rows[2]["source_citation"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _rehash(root, path.name)

    with pytest.raises(ValueError, match="raw author-page or CV title changed"):
        validate_review(root)
