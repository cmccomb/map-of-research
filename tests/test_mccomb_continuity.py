"""One-time review must match the exact source and graph transition."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from map_of_research import mccomb_continuity as review


def _rows() -> tuple[list[dict], list[dict]]:
    manifest = json.loads(review.DEFAULT_APPROVAL_PATH.read_text())
    prior: list[dict] = []
    current: list[dict] = []
    targets: set[str] = set()
    for loss in manifest["losses"]:
        prior.append(
            {
                "person_id": review.PERSON_ID,
                "scholar_id": review.SCHOLAR_ID,
                "author_pub_id": loss["prior_author_pub_id"],
                "source_type": "scholar",
                "work_id": loss["prior_work_id"],
                "map_eligible": True,
                "collaboration_eligible": True,
            }
        )
        for index in range(loss["prior_work_observation_count"] - 1):
            other = {
                **prior[-1],
                "person_id": f"person-other-{loss['prior_author_pub_id']}-{index}",
            }
            prior.append(other)
            current.append(other.copy())
        target_id = loss["target_author_pub_id"]
        if target_id not in targets:
            current.append(
                {
                    **prior[0],
                    "author_pub_id": target_id,
                    "work_id": loss["target_work_id"],
                }
            )
            targets.add(target_id)
    binding = manifest["binding_changes"][0]
    for rows, work_id, year in (
        (prior, binding["prior_work_id"], binding["prior_year"]),
        (current, binding["current_work_id"], binding["current_year"]),
    ):
        rows.append(
            {
                "person_id": review.PERSON_ID,
                "scholar_id": review.SCHOLAR_ID,
                "author_pub_id": binding["author_pub_id"],
                "source_type": "scholar",
                "work_id": work_id,
                "year": year,
                "title": "Smoothing the Rough Edges",
                "map_eligible": True,
                "collaboration_eligible": True,
            }
        )
    return prior, current


def _validate(
    monkeypatch: pytest.MonkeyPatch,
    prior: list[dict],
    current: list[dict],
    *,
    snapshot_sha256: str = review.SNAPSHOT_SHA256,
):
    real_sha = review.sha256_file

    def source_sha(path: Path) -> str:
        return review.CACHE_SHA256 if path.name == "cache.json" else real_sha(path)

    monkeypatch.setattr(review, "sha256_file", source_sha)
    return review.validate_exact_mccomb_continuity(
        review.DEFAULT_APPROVAL_PATH,
        prior_revision=review.PRIOR_REVISION,
        source_snapshot_commit=review.SNAPSHOT_COMMIT,
        source_snapshot_sha256=snapshot_sha256,
        cache_path=Path("cache.json"),
        crosswalk_path=review.DEFAULT_CROSSWALK_PATH,
        prior_profiles=prior,
        current_profiles=current,
    )


def test_exact_mccomb_manifest_accepts_only_reviewed_rekeys(monkeypatch):
    prior, current = _rows()
    approved = _validate(monkeypatch, prior, current)
    assert len(approved.loss_identities) == 3
    assert len(approved.binding_changes) == 1

    current[0]["work_id"] = "work-unreviewed"
    with pytest.raises(ValueError, match="another faculty edge"):
        _validate(monkeypatch, prior, current)


def test_exact_mccomb_manifest_rejects_changed_source(monkeypatch):
    prior, current = _rows()
    with pytest.raises(ValueError, match="reviewed source"):
        _validate(monkeypatch, prior, current, snapshot_sha256="0" * 64)
