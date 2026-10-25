from collections import Counter
from pathlib import Path
from shutil import copytree

import pytest

from scripts.validate_weston_source_review import (
    REVIEW,
    ROOT,
    load_held_observations,
    validate,
)


def test_weston_source_review_is_complete_and_held():
    assert validate() == {
        "cmu_scholars": 29,
        "orcid_groups": 24,
        "exact_doi_overlaps": 23,
        "cmu_dois_absent_orcid": 3,
        "cmu_non_doi_not_in_orcid": 2,
    }


def test_weston_source_review_rejects_changed_raw_capture(tmp_path: Path):
    review = tmp_path / "review"
    copytree(REVIEW, review)
    source = review / "orcid-works.json"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source digest changed"):
        validate(review, ROOT / "registry")


def test_weston_visible_sources_stage_as_53_held_observations():
    rows = load_held_observations()
    assert len(rows) == len({row["source_observation_id"] for row in rows}) == 53
    assert Counter(row["source_type"] for row in rows) == {
        "cmu_scholars": 29,
        "researcher_orcid": 24,
    }
    assert all(
        row["attribution_review_pending"]
        and not row["career_scope_verified"]
        and not row["release_eligible"]
        and not row["map_eligible"]
        and not row["collaboration_eligible"]
        for row in rows
    )
    cmu = {row["doi_as_source"] for row in rows[:29] if row["doi_as_source"]}
    orcid = {row["doi_as_source"] for row in rows[29:] if row["doi_as_source"]}
    assert len(cmu) == 26
    assert len(orcid) == len(cmu & orcid) == 23
    assert sum(row["genre_as_source"] == "patent" for row in rows[:29]) == 1
