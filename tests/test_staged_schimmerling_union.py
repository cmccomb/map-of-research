"""Schimmerling's 47 source records stay whole, typed, and release-held."""

from __future__ import annotations

import csv
import json
import runpy
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/source_reviews/ernest-schimmerling-2026-10-25"


def _loader():
    return runpy.run_path(str(SOURCE / "load_staged.py"))


def test_staged_union_preserves_all_source_observations_and_genres() -> None:
    rows = _loader()["load_staged_rows"]()
    assert len(rows) == len({row["source_native_id"] for row in rows}) == 47
    assert Counter(row["source_type"] for row in rows) == {
        "cmu_elements": 45,
        "author_site": 1,
        "researcher_orcid": 1,
    }
    assert sum(row["research_candidate"] for row in rows) == 40
    assert sum(row["collaboration_candidate"] for row in rows) == 26
    assert all(not row["release_eligible"] for row in rows)
    assert all(not row["map_eligible"] for row in rows)
    assert all(not row["collaboration_eligible"] for row in rows)
    assert all(not row["citation_count_known"] for row in rows)
    assert all(row["source_authors"] for row in rows)
    genres = {row["source_native_id"]: row["reviewed_genre"] for row in rows}
    assert genres["cmu-elements:1550:object:327734"] == "article_review"
    assert genres["cmu-elements:1550:object:767686"] == "edited_volume"
    assert genres["author-site:schimmerling:other-publications:4"] == "appendix"
    assert genres["orcid:0000-0001-6740-4809:work:30000450"] == (
        "doctoral_dissertation"
    )
    nonresearch = [row for row in rows if not row["research_candidate"]]
    assert len(nonresearch) == 7
    assert all(not row["collaboration_candidate"] for row in nonresearch)
    thesis = rows[-1]
    article = next(
        row
        for row in rows
        if row["source_native_id"] == "cmu-elements:1550:object:66288"
    )
    assert thesis["source_title"].casefold() == article["source_title"].casefold()
    assert thesis["source_year"] == 1992
    assert article["source_year"] == 1995
    assert thesis["source_doi"] == ""
    assert article["source_doi"] == "10.1016/0168-0072(94)00036-3"
    assert thesis["version_review_pending"]


def test_source_hash_change_rejects_batch(tmp_path: Path) -> None:
    from shutil import copytree

    source = tmp_path / "review"
    copytree(SOURCE, source)
    path = source / "elements-45.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(AssertionError):
        _loader()["load_staged_rows"](source)


def test_partial_union_rejected_after_source_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _loader()
    original = loader["_csv"]

    def partial(path: Path) -> list[dict[str, str]]:
        rows = original(path)
        return rows[:-1] if path.name == "elements-45.csv" else rows

    loader["load_staged_rows"].__globals__["_csv"] = partial
    with pytest.raises(ValueError, match="whole-person source union"):
        loader["load_staged_rows"]()


def test_source_manifest_keeps_publication_hold() -> None:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    assert manifest["status"] == "review_only_no_snapshot_no_release"
    assert manifest["scholar_account_status"] == "no_verified_account"
    assert manifest["counts"]["union_source_manifestations"] == 47
    with (SOURCE / "author-elements-crosswalk-34.csv").open(newline="") as handle:
        crosswalk = list(csv.DictReader(handle))
    assert len(crosswalk) == 34
    assert sum(row["relation"] == "author_only" for row in crosswalk) == 1
