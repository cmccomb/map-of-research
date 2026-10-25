"""Larsson's staged bibliography preserves all sources behind a graph hold."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/martin-larsson-2026-10-25"
)
SPEC = importlib.util.spec_from_file_location(
    "load_staged_larsson", SOURCE / "load_staged.py"
)
assert SPEC is not None and SPEC.loader is not None
LOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOADER)
load_staged_larsson_rows = LOADER.load_staged_larsson_rows


def test_staged_larsson_preserves_all_74_with_global_graph_hold() -> None:
    rows = load_staged_larsson_rows(SOURCE)
    assert len(rows) == len({row["observation_id"] for row in rows}) == 74
    assert sum(row["source_kind"] == "researcher_owned_site" for row in rows) == 63
    assert (
        sum(
            row["source_kind"] == "publisher_registered_journal_discussion"
            for row in rows
        )
        == 2
    )
    assert sum(row["source_kind"] == "arxiv_with_ssrn_preprint" for row in rows) == 1
    assert sum(row["source_kind"] == "research_preprint" for row in rows) == 4
    assert sum(row["source_kind"] == "masters_thesis" for row in rows) == 1
    assert sum(row["source_kind"] == "withdrawn_preprint" for row in rows) == 1
    assert sum(row["source_kind"] == "magazine_article" for row in rows) == 2
    assert all(row["primary_department_id"] == "math" for row in rows)
    assert all(row["scholar_id"] is None for row in rows)
    assert all(row["canonical_title"] is None for row in rows)
    assert all(row["canonical_authors"] is None for row in rows)
    assert all(row["citation_count"] is None for row in rows)
    assert all(row["release_state"] == "source_only_release_held" for row in rows)
    assert all(not row["release_eligible"] for row in rows)
    assert all(not row["map_eligible"] for row in rows)
    assert all(not row["collaboration_eligible"] for row in rows)
    assert "10.2139/ssrn.2040907" not in {row["doi"] for row in rows}


def test_staged_larsson_does_not_invent_site_years_or_bylines() -> None:
    rows = load_staged_larsson_rows(SOURCE)
    site = rows[:63]
    assert all(row["year"] is None for row in site)
    assert all(row["canonical_authors"] is None for row in site)
    assert all(row["doi"] is None for row in site)
    assert all(
        row["source_record"]["index"] == index for index, row in enumerate(site, 1)
    )
    assert site[21]["source_title_as_listed"] == "Robust"
    assert site[21]["source_citation_as_listed"].startswith(
        "Robust asymptotic growth in stochastic portfolio theory"
    )
    assert site[0]["row_hold"] == "article_link_points_to_row_11"
    assert site[49]["row_hold"] == "article_link_points_to_row_51"
    assert site[35]["row_hold"] == "book_chapter_manifestation_unpaired"


def test_staged_larsson_discussions_and_preprint_keep_distinct_dates() -> None:
    rows = load_staged_larsson_rows(SOURCE)
    discussions = {row["doi"]: row for row in rows[63:65]}
    assert discussions["10.1093/jrsssb/qkae061"]["year"] == 2024
    assert discussions["10.1093/jrsssb/qkad120"]["year"] == 2023
    for row in discussions.values():
        assert row["journal_issue_year"] == 2024
        assert row["authors_as_registered"][0] == "Martin Larsson"
        assert row["year_basis"] == "publisher_registered_online_year"
    assert discussions["10.1093/jrsssb/qkae061"]["authors_as_registered"] == [
        "Martin Larsson",
        "Aaditya Ramdas",
        "Johannes Ruf",
    ]
    preprint = rows[65]
    assert preprint["doi"] == "10.2139/ssrn.3170978"
    assert preprint["year"] is None
    assert preprint["source_years"] == {
        "arxiv_first_submission": 2017,
        "ssrn_registration": 2018,
    }
    assert preprint["authors_as_registered"] == [
        "Damir Filipovic",
        "Martin Larsson",
        "Tony Ware",
    ]


def test_staged_larsson_additional_artifacts_keep_genres_and_source_dates() -> None:
    rows = {row["observation_id"]: row for row in load_staged_larsson_rows(SOURCE)}
    assert rows["arxiv:1008.1754"]["reviewed_genre"] == "masters_thesis"
    assert rows["arxiv:1008.1754"]["source_years"] == {
        "arxiv_first_posted": "2010-08-10"
    }
    withdrawn = rows["arxiv:1104.5272"]
    assert withdrawn["reviewed_genre"] == "withdrawn_preprint"
    assert withdrawn["row_hold"] == "withdrawn_work_no_collaboration_edge"
    assert withdrawn["collaboration_eligible"] is False
    magazine = rows[
        "ercim-news:145:optimal-gambling-strategies-when-testing-by-betting"
    ]
    assert magazine["reviewed_genre"] == "magazine_article"
    assert magazine["source_years"] == {"published_date": "2026-07-01"}
    assert magazine["canonical_title"] is None
    assert magazine["canonical_authors"] is None
    assert rows["arxiv:2608.09840"]["authors_as_registered"] == ["Martin Larsson"]


def test_staged_larsson_rejects_changed_source_or_release_gate(tmp_path: Path) -> None:
    copied = tmp_path / "source"
    shutil.copytree(SOURCE, copied)
    source = copied / "author-site.html"
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum changed"):
        load_staged_larsson_rows(copied)
    shutil.copy2(SOURCE / "author-site.html", source)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_eligible"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="release hold changed"):
        load_staged_larsson_rows(copied)


def test_staged_larsson_rejects_rehashed_extra_work(tmp_path: Path) -> None:
    copied = tmp_path / "source"
    shutil.copytree(SOURCE, copied)
    omissions_path = copied / "omissions.json"
    omissions = json.loads(omissions_path.read_text(encoding="utf-8"))
    omissions["observations"][2]["doi"] = "10.2139/ssrn.2040907"
    omissions_path.write_text(json.dumps(omissions), encoding="utf-8")
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files_sha256"]["omissions.json"] = hashlib.sha256(
        omissions_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="omitted-work review changed"):
        load_staged_larsson_rows(copied)
