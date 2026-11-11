"""The source-only loader must retain all evidence and release holds."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/robin-neumayer-2026-10-25"
)
SPEC = importlib.util.spec_from_file_location(
    "load_staged_neumayer", SOURCE / "load_staged.py"
)
assert SPEC is not None and SPEC.loader is not None
LOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOADER)
CV_URL = LOADER.CV_URL
SITE_URL = LOADER.SITE_URL
NONCOLLABORATION_GENRES = LOADER.NONCOLLABORATION_GENRES
OBERWOLFACH_REPORT_URLS = LOADER.OBERWOLFACH_REPORT_URLS
load_staged_neumayer_rows = LOADER.load_staged_neumayer_rows


def test_staged_neumayer_preserves_every_source_and_release_hold() -> None:
    rows = load_staged_neumayer_rows(SOURCE)

    assert len(rows) == len({row["observation_id"] for row in rows}) == 31
    assert sum(row["source_kind"] == "researcher_owned_site" for row in rows) == 30
    assert sum(row["source_kind"] == "researcher_cv" for row in rows) == 1
    assert sum(row["doi"] is not None for row in rows) == 18
    assert sum(row["row_hold"] is not None for row in rows) == 1
    assert all(row["scholar_id"] is None for row in rows)
    assert all(row["citation_count"] is None for row in rows)
    assert all(row["release_state"] == "source_only_release_held" for row in rows)
    assert all(not row["map_eligible"] for row in rows)
    assert all(not row["collaboration_eligible"] for row in rows)
    assert all(
        not row["candidate_collaboration_eligible"]
        for row in rows
        if row["reviewed_genre"] in NONCOLLABORATION_GENRES
    )
    for row in rows:
        raw = row["source_record"]
        assert (
            row["observation_id"]
            == raw["author_site_observation_or_cv_only_record"]["observation_id"]
        )
        assert row["observation_id"] == raw["cv_crosswalk"]["observation_id"]
        assert row["observation_id"] == raw["genre_doi_review"]["observation_id"]


def test_staged_neumayer_keeps_cv_only_and_undated_note_distinct() -> None:
    rows = load_staged_neumayer_rows(SOURCE)
    yamabe = rows[29]
    cv_only = rows[30]

    assert yamabe["source_kind"] == "researcher_owned_site"
    assert yamabe["source_url"] == SITE_URL
    assert CV_URL not in yamabe["source_verification_urls"]
    assert yamabe["source_record"]["cv_crosswalk"]["source_relation"] == "site_only"
    assert yamabe["year"] is None
    assert yamabe["year_basis"] == "publication_date_not_stated"
    assert yamabe["event_or_lecture_year"] == 2018
    assert yamabe["publication_type"] == "expository_lecture_notes"

    assert cv_only["source_kind"] == "researcher_cv"
    assert cv_only["source_url"] == CV_URL
    assert SITE_URL not in cv_only["source_verification_urls"]
    assert cv_only["source_record"]["cv_crosswalk"]["source_relation"] == "cv_only"
    assert cv_only["year"] == 2023
    assert cv_only["pages"] == "1649-1650"
    assert cv_only["publication_type"] == "oberwolfach_talk_summary"
    assert cv_only["authors"] == "Robin Neumayer"
    assert cv_only["row_review_state"] == "source_only_primary_resolved"


def test_staged_neumayer_never_promotes_joint_work_notes_to_bylines() -> None:
    rows = load_staged_neumayer_rows(SOURCE)
    for number in (24, 26, 31):
        row = rows[number - 1]
        assert row["authors"] == "Robin Neumayer"
        assert row["publication_type"] == "oberwolfach_talk_summary"
        assert not row["candidate_collaboration_eligible"]
        assert OBERWOLFACH_REPORT_URLS[number] in row["source_verification_urls"]
        assert row["year"] == {24: 2022, 26: 2019, 31: 2023}[number]
        assert (
            row["pages"] == {24: "1563-1565", 26: "2046-2047", 31: "1649-1650"}[number]
        )
    assert rows[16]["authors"].endswith("R. Neumayer")
    assert rows[26]["year"] == 2020
    assert rows[26]["event_or_lecture_year"] == 2018
    assert rows[26]["doi"] == "10.1007/978-3-030-38230-8_20"
    assert rows[7]["year"] == 2022
    assert rows[7]["publication_type"] == "research_preprint"
    assert rows[7]["row_hold"] == "publisher_placement_unverified"


def test_staged_neumayer_fails_closed_if_a_pinned_source_changes(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "source"
    shutil.copytree(SOURCE, copied)
    source = copied / "author-site.html"
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum changed"):
        load_staged_neumayer_rows(copied)

    shutil.copytree(SOURCE, copied, dirs_exist_ok=True)
    report = copied / "resolution/ems-owr-2023-29.pdf"
    report.write_bytes(report.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum changed"):
        load_staged_neumayer_rows(copied)
