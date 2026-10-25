"""All Cummings source observations stay intact and outside the public graph."""

from __future__ import annotations

import runpy
from collections import Counter
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/james-cummings-2026-10-25"
)


def _loader():
    return runpy.run_path(str(SOURCE / "load_staged.py"))


def test_all_214_observations_are_held_with_roles_preserved() -> None:
    rows = _loader()["load_staged_rows"]()
    assert len(rows) == len({row["source_native_id"] for row in rows}) == 214
    assert Counter(row["source_type"] for row in rows) == {
        "cmu_elements": 108,
        "researcher_site": 51,
        "researcher_orcid": 51,
        "publisher_registration": 4,
    }
    assert all(
        not row[field]
        for row in rows
        for field in (
            "release_eligible",
            "map_eligible",
            "collaboration_eligible",
            "citation_count_known",
        )
    )
    by_id = {row["source_native_id"]: row for row in rows}
    assert by_id["cmu-elements:2104:object:619986"]["contributor_role"] == "editor"
    assert by_id["cmu-elements:2104:object:327935"]["reviewed_genre"] == ("book_review")
    assert by_id["researcher-site:jcumming:papers:14"]["reviewed_genre"] == (
        "unpublished_talk_slides"
    )
    assert by_id["researcher-site:jcumming:papers:14"]["contributor_role"] == (
        "presenter"
    )
    assert by_id["doi:10.1017/cbo9781139208574"]["contributor_role"] == ("editor")
    chapter = by_id["doi:10.1007/978-94-015-8988-8_2"]
    assert chapter["reviewed_genre"] == "book_chapter"
    assert chapter["contributor_role"] == "author"
    assert by_id["cmu-elements:2104:object:341749"]["reviewed_genre"] == (
        "presentation"
    )


def test_pinned_source_change_rejects_batch(tmp_path: Path) -> None:
    from shutil import copytree

    source = tmp_path / "review"
    copytree(SOURCE, source)
    path = source / "elements-108.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="pinned source changed"):
        _loader()["load_staged_rows"](source)


def test_partial_union_rejected_after_provenance_check() -> None:
    loader = _loader()
    original = loader["_csv"]

    def partial(path: Path) -> list[dict[str, str]]:
        rows = original(path)
        return rows[:-1] if path.name == "researcher-crosswalk-51.csv" else rows

    loader["load_staged_rows"].__globals__["_csv"] = partial
    with pytest.raises(ValueError, match="whole-source observation hold"):
        loader["load_staged_rows"]()
