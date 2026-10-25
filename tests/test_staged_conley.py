"""Conley's full visible bibliography must remain attributed and release held."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/clinton-conley-2026-10-25"
)
SPEC = importlib.util.spec_from_file_location(
    "load_staged_conley", SOURCE / "load_staged.py"
)
assert SPEC is not None and SPEC.loader is not None
LOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOADER)
load_staged_conley_rows = LOADER.load_staged_conley_rows


def test_conley_source_union_preserves_current_and_historical_entries() -> None:
    rows = load_staged_conley_rows(SOURCE)
    assert len(rows) == len({row["source_native_id"] for row in rows}) == 36
    assert [row["source_kind"] for row in rows].count("researcher_owned_site") == 31
    assert [row["source_kind"] for row in rows].count(
        "publisher_verified_omission"
    ) == 2
    assert [row["source_kind"] for row in rows].count(
        "researcher_owned_historical_site"
    ) == 3
    assert sum("arxiv.org/abs/" in row["source_work_url"] for row in rows) == 7
    assert all(row["person_id"] == LOADER.PERSON_ID for row in rows)
    assert all(row["primary_department_id"] == "math" for row in rows)
    assert all(
        row["scholar_id"] is None and row["citation_count"] is None for row in rows
    )
    assert all(row["release_state"] == "source_only_release_held" for row in rows)
    assert all(
        not row["map_eligible"] and not row["collaboration_eligible"] for row in rows
    )
    assert all(row["year"] is None for row in rows[:31])


def test_conley_versions_genres_and_namesake_hold_are_kept_distinct() -> None:
    rows = load_staged_conley_rows(SOURCE)
    assert rows[20]["doi"] is None  # PDF prints only a placeholder DOI.
    assert "placeholder DOI" in rows[20]["row_hold"]
    assert rows[26]["publication_type"] == "preliminary_draft"
    assert "preliminary draft" in rows[26]["row_hold"]
    assert "different title" in rows[29]["row_hold"]
    assert rows[29]["title"].startswith("Defining non-empty small sets")
    assert rows[29]["source_record"]["primary_work"]["first_page_title"].startswith(
        "Defining small sets from"
    )
    chapter, review = rows[31:33]
    assert chapter["doi"] == LOADER.CHAPTER_DOI
    assert chapter["publication_type"] == "book_chapter"
    assert chapter["year"] is None
    assert chapter["year_facts"] == {"crossref_online": 2017, "book_catalog": 2018}
    assert review["doi"] == LOADER.REVIEW_DOI
    assert review["publication_type"] == "book_review"
    assert review["authors"] == "Clinton T. Conley"
    assert review["year"] == 2016
    assert all(row["doi"] != LOADER.CHEMISTRY_HOLD_DOI for row in rows)
    lecture, matching, smooth = rows[33:]
    assert lecture["title"] == "Measurable 3-colorings of acyclic graphs"
    assert lecture["title_as_listed"].startswith("Baire measurable 3-colorings")
    assert lecture["publication_type"] == "draft_lecture_note"
    assert lecture["related_current_ordinal"] is None
    assert matching["related_current_ordinal"] == 13
    assert smooth["related_current_ordinal"] == 14
    assert (
        matching["version_relation"]
        == smooth["version_relation"]
        == ("candidate_precursor_unresolved")
    )
    assert all(row["year"] is None and row["row_hold"] for row in rows[33:])
    assert all(row["doi"] != LOADER.CHEMISTRY_HOLD_DOI for row in rows)


def test_conley_pinned_source_tampering_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "source"
    shutil.copytree(SOURCE, copied)
    page = copied / "author-site.html"
    page.write_bytes(page.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum changed"):
        load_staged_conley_rows(copied)


def test_conley_version_hold_survives_rehashed_review(tmp_path: Path) -> None:
    copied = tmp_path / "source"
    shutil.copytree(SOURCE, copied)
    review_path = copied / "pdf-first-page-review.csv"
    with review_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[30 - 8]["version_review"] = "researcher_pdf_first_page_matches_list"
    with review_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["source_files"]:
        if item["file"] == "pdf-first-page-review.csv":
            item["sha256"] = hashlib.sha256(review_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="row 30 version hold changed"):
        load_staged_conley_rows(copied)
