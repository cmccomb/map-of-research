"""Exact source and region gates for the three McComb coauthor rebindings."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pandas
import pytest

from map_of_research import coauthor_binding_approval as review
from map_of_research.region_continuity import (
    RegionReleaseDiffError,
    reconcile_region_decisions,
)


def _rows() -> tuple[list[dict], list[dict]]:
    changes = review.load_json(review.DEFAULT_APPROVAL_PATH)["binding_changes"]
    prior, current = [], []
    for item in changes:
        person_id, scholar_id, author_pub_id, source_type = item["observation"].split(
            "/"
        )
        row = {
            "person_id": person_id,
            "scholar_id": scholar_id,
            "author_pub_id": author_pub_id,
            "source_type": source_type,
            "work_id": item["prior_work_id"],
            "map_eligible": True,
            "collaboration_eligible": True,
            "title": "A reviewed McComb coauthored work",
            "source_record_json": "{}",
        }
        prior.append(row)
        current.append({**row, "work_id": item["current_work_id"]})
    return prior, current


def _validate(monkeypatch, prior: list[dict], current: list[dict], **overrides):
    original_sha = review.sha256_file

    def source_sha(path: Path) -> str:
        return review.CACHE_SHA256 if path.name == "cache.json" else original_sha(path)

    monkeypatch.setattr(review, "sha256_file", source_sha)
    arguments = {
        "prior_revision": review.PRIOR_REVISION,
        "source_snapshot_sha256": review.SNAPSHOT_SHA256,
        "curated_works_path": Path("data/curated/works.json"),
        "mccomb_cache_path": Path("cache.json"),
        "martelaro_review_path": review.DEFAULT_MARTELARO_REVIEW_PATH,
        "akinci_review_path": review.DEFAULT_AKINCI_REVIEW_PATH,
        "prior_profiles": prior,
        "current_profiles": current,
    }
    arguments.update(overrides)
    return review.validate_exact_coauthor_bindings(
        review.DEFAULT_APPROVAL_PATH, **arguments
    )


def _work(work_id: str) -> dict:
    return {
        "work_id": work_id,
        "title": "A reviewed McComb coauthored work",
        "authors": "Christopher McComb and colleague",
        "year": 2025,
        "map_eligible": True,
        "collaboration_eligible": True,
        "map_region_outlier": False,
        "map_region_documented_override": False,
        "map_exclusion_reasons": [],
        "department_ids": ["meche"],
        "quality_assessment_version": "quality-v1",
    }


def test_exact_three_bindings_pass_with_unchanged_source(monkeypatch):
    prior, current = _rows()
    approved = _validate(monkeypatch, prior, current)
    assert len(approved.binding_changes) == 3
    _, receipt = reconcile_region_decisions(
        pandas.DataFrame([_work(row["work_id"]) for row in current]),
        pandas.DataFrame(current),
        [_work(row["work_id"]) for row in prior],
        prior,
        prior_revision=review.PRIOR_REVISION,
        coauthor_binding_approval=approved,
    )
    assert len(receipt["approved_coauthor_binding_changes"]) == 3
    assert receipt["unapproved_mapped_binding_changes"] == []
    assert receipt["unused_coauthor_binding_approvals"] == []


def test_changed_snapshot_or_observation_is_rejected(monkeypatch):
    prior, current = _rows()
    with pytest.raises(ValueError, match="reviewed source"):
        _validate(monkeypatch, prior, current, source_snapshot_sha256="0" * 64)
    altered = copy.deepcopy(current)
    altered[0]["work_id"] = "work-unreviewed"
    with pytest.raises(ValueError, match="exact reviewed transition"):
        _validate(monkeypatch, prior, altered)
    wrong_identity = copy.deepcopy(current)
    wrong_identity[0]["author_pub_id"] = "0P9w_S0AAAAJ:wrong-owner-row"
    with pytest.raises(ValueError, match="exact reviewed transition"):
        _validate(monkeypatch, prior, wrong_identity)


def test_fourth_binding_is_rejected_even_with_exact_three_approval(monkeypatch):
    prior, current = _rows()
    approved = _validate(monkeypatch, prior, current)
    extra = {
        **prior[0],
        "author_pub_id": "0P9w_S0AAAAJ:unreviewed",
        "work_id": "work-extra-prior",
    }
    extra_after = {**extra, "work_id": "work-extra-current"}
    with pytest.raises(RegionReleaseDiffError) as error:
        reconcile_region_decisions(
            pandas.DataFrame(
                [_work(row["work_id"]) for row in [*current, extra_after]]
            ),
            pandas.DataFrame([*current, extra_after]),
            [_work(row["work_id"]) for row in [*prior, extra]],
            [*prior, extra],
            prior_revision=review.PRIOR_REVISION,
            coauthor_binding_approval=approved,
        )
    assert len(error.value.receipt["unapproved_mapped_binding_changes"]) == 1


def test_region_rejects_unreviewed_approval_hash(monkeypatch):
    prior, current = _rows()
    approved = _validate(monkeypatch, prior, current)
    with pytest.raises(ValueError, match="not the reviewed set"):
        reconcile_region_decisions(
            pandas.DataFrame([_work(row["work_id"]) for row in current]),
            pandas.DataFrame(current),
            [_work(row["work_id"]) for row in prior],
            prior,
            prior_revision=review.PRIOR_REVISION,
            coauthor_binding_approval=replace(approved, manifest_sha256="unreviewed"),
        )


def test_region_rejects_unused_one_of_three_approvals(monkeypatch):
    prior, current = _rows()
    approved = _validate(monkeypatch, prior, current)
    unchanged = copy.deepcopy(current)
    unchanged[0]["work_id"] = prior[0]["work_id"]
    with pytest.raises(RegionReleaseDiffError) as error:
        reconcile_region_decisions(
            pandas.DataFrame([_work(row["work_id"]) for row in unchanged]),
            pandas.DataFrame(unchanged),
            [_work(row["work_id"]) for row in prior],
            prior,
            prior_revision=review.PRIOR_REVISION,
            coauthor_binding_approval=approved,
        )
    assert len(error.value.receipt["unused_coauthor_binding_approvals"]) == 1
