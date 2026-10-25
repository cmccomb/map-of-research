"""The exact Chemistry review replaces only its 19 historical region flips."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from map_of_research.chem_region_review import (
    CHEM_REVIEW_SHA256,
    DECISION_COUNTS,
    DEFAULT_CHEM_REGION_REVIEW_PATH,
    load_chem_region_review,
)


def test_checked_in_chem_review_is_exact_and_preserves_source() -> None:
    path = DEFAULT_CHEM_REGION_REVIEW_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CHEM_REVIEW_SHA256
    source = json.loads(path.read_text())
    decisions = source["decisions"]
    repair = json.loads(
        Path("data/curated/region-remediation-bernhard.json").read_text()
    )
    math = json.loads(Path("data/curated/scholar-region-review.json").read_text())
    assert len(decisions) == len({item["work_id"] for item in decisions}) == 19
    assert len({item["author_pub_id"] for item in decisions}) == 19
    assert Counter((item["transition"], item["decision"]) for item in decisions) == (
        DECISION_COUNTS
    )
    assert {item["work_id"] for item in decisions}.isdisjoint(
        {item["work_id"] for item in math["decisions"]}
    )
    for item in decisions:
        key = (
            "restore_eligible_work_ids"
            if item["transition"] == "restore"
            else "restore_outlier_work_ids"
        )
        assert item["work_id"] in repair[key]
        assert item["saved_scholar_url"].startswith("https://scholar.google.com/")
        assert len(item["source_record_sha256"]) == 64
        assert len(item["work_fingerprint_sha256"]) == 64
    assert {
        item["reason_code"] for item in decisions if item["decision"] == "exclude"
    } == {
        "false_profile_attribution",
        "corrupted_duplicate",
    }
    assert {item["work_id"] for item in decisions if item.get("output_type")} == {
        "work-4e63e66d343de69522c77ffde7a55853382c39d01a9379ca47f4fa4eb939e802",
        "work-f07078afd572186934d04127dcd8526c089e5089ad69b1c4504702ac49c4329b",
    }
    assert all(
        item["output_type"] == "conference_presentation"
        for item in decisions
        if item.get("output_type")
    )


def test_modified_chem_review_fails_before_accessing_source(tmp_path: Path) -> None:
    changed = tmp_path / "chem-region-review.json"
    changed.write_bytes(DEFAULT_CHEM_REGION_REVIEW_PATH.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest bytes changed"):
        load_chem_region_review(
            changed,
            snapshot_frame=None,
            snapshot_sha256="f8d7ffec683eb1830dadc77e612cbad7f3fa18ff199696041a71a9b8edbcefa2",
            cache_dir=tmp_path,
            registry=None,
        )
