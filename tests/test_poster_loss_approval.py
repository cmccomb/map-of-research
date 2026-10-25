"""A parent-pinned approval can explain only the McManus poster point."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pandas
import pytest

import map_of_research.poster_loss_approval as approval
from map_of_research.profile_releases import ProfileRelease
from map_of_research.provisional_holds import ProvisionalHold
from map_of_research.region_continuity import (
    RegionReleaseDiffError,
    reconcile_region_decisions,
)

PARENT = "a" * 40
SOURCE = "b" * 64
RELEASES = "c" * 64
HOLDS = "d" * 64
REASON = "The organizer record does not verify the complete Scholar byline."
EVIDENCE = (
    "https://scholar.google.com/citations?view_op=view_citation&"
    "user=28yZqbcAAAAJ&citation_for_view=28yZqbcAAAAJ:HKviVsUxM5wC"
)


def _hold() -> ProvisionalHold:
    return ProvisionalHold(
        scholar_id=approval.POSTER_ID[1],
        author_pub_id=approval.POSTER_ID[2],
        capture_row_index=approval.POSTER_CAPTURE_ROW_INDEX,
        reason=REASON,
        evidence_url=EVIDENCE,
        reviewed_at="2026-10-25",
    )


def _release() -> ProfileRelease:
    return ProfileRelease(
        person_id=approval.POSTER_ID[0],
        scholar_id=approval.POSTER_ID[1],
        state="released",
        expected_source_rows=62,
        cache_sha256=approval.POSTER_CACHE_SHA256,
        reason="Complete profile with one poster held",
        reviewed_at="2026-10-25",
        expected_provisional_hold_rows=1,
        provisional_hold_ids_sha256=approval.POSTER_HELD_IDS_SHA256,
    )


def _document() -> dict:
    return approval.approval_document(
        prior_revision=PARENT,
        source_snapshot_sha256=SOURCE,
        profile_releases_sha256=RELEASES,
        provisional_holds_sha256=HOLDS,
        hold=_hold(),
    )


def _load(tmp_path, document=None) -> approval.PosterLossApproval:
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(document or _document()), encoding="utf-8")
    return approval.load_poster_loss_approval(
        path,
        prior_revision=PARENT,
        source_snapshot_sha256=SOURCE,
        profile_releases_sha256=RELEASES,
        provisional_holds_sha256=HOLDS,
        release_decisions=(_release(),),
        validated_holds={(approval.POSTER_ID[1], approval.POSTER_ID[2]): _hold()},
    )


def _poster_row(*, eligible: bool) -> dict:
    return {
        **dict(
            zip(
                ("person_id", "scholar_id", "author_pub_id", "source_type"),
                approval.POSTER_ID,
                strict=True,
            )
        ),
        "work_id": approval.POSTER_WORK_ID,
        "source_record_json": "poster-raw",
        "map_eligible": eligible,
        "map_exclusion_reasons": [] if eligible else ["attribution_provisional_hold"],
        "review_attribution_status": "" if eligible else "held",
        "review_attribution_evidence_url": "" if eligible else EVIDENCE,
        "title": "A poster",
        "authors": "A Author",
        "year": 2021,
        "venue": "Rustbelt RNA",
        "quality_assessment_version": "quality-v1",
    }


def _work(*, eligible: bool) -> dict:
    return {
        "work_id": approval.POSTER_WORK_ID,
        "map_eligible": eligible,
        "map_region_outlier": False,
        "map_exclusion_reasons": [] if eligible else ["attribution_provisional_hold"],
        "department_ids": ["bio"],
        "quality_assessment_version": "quality-v1",
        "title": "A poster",
        "authors": "A Author",
        "year": 2021,
        "venue": "Rustbelt RNA",
    }


def _reconcile(approved=None, *, prior_rows=None, current_rows=None, prior_works=None):
    return reconcile_region_decisions(
        pandas.DataFrame([_work(eligible=False)]),
        pandas.DataFrame(current_rows or [_poster_row(eligible=False)]),
        prior_works or [_work(eligible=True)],
        prior_rows or [_poster_row(eligible=True)],
        prior_revision=PARENT,
        approved_poster_loss=approved,
    )


def test_approval_requires_immutable_parent_and_exact_source_manifests(
    tmp_path,
) -> None:
    loaded = _load(tmp_path)
    assert loaded.prior_revision == PARENT
    assert (
        loaded.manifest_sha256
        == hashlib.sha256((tmp_path / "approval.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("approval_id", "generic-hold-loss"),
        ("prior_revision", "e" * 40),
        ("source_snapshot_sha256", "e" * 64),
        ("profile_releases_sha256", "e" * 64),
        ("provisional_holds_sha256", "e" * 64),
        ("source_cache_sha256", "e" * 64),
        ("expected_loss_count", 2),
    ],
)
def test_approval_rejects_parent_source_or_count_drift(tmp_path, field, value) -> None:
    document = _document()
    document[field] = value
    with pytest.raises(ValueError, match="McManus poster approval"):
        _load(tmp_path, document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("person_id", "person-other"),
        ("scholar_id", "other"),
        ("author_pub_id", "other:paper"),
        ("source_type", "curated_verified"),
        ("prior_work_id", "other-work"),
        ("source_record_sha256", "e" * 64),
        ("capture_row_index", 59),
        ("reason", "A different reason"),
        ("evidence_url", "https://scholar.google.com/citations?other=1"),
    ],
)
def test_approval_rejects_any_other_loss_or_hold(tmp_path, field, value) -> None:
    document = _document()
    document["loss"][field] = value
    with pytest.raises(ValueError, match="McManus poster approval"):
        _load(tmp_path, document)


def test_approval_rejects_extra_losses_and_unverified_release_decisions(
    tmp_path,
) -> None:
    document = _document()
    document["losses"] = [document["loss"], {"author_pub_id": "other"}]
    with pytest.raises(ValueError, match="exactly one loss"):
        _load(tmp_path, document)

    path = tmp_path / "approval.json"
    path.write_text(json.dumps(_document()), encoding="utf-8")
    with pytest.raises(ValueError, match="validated hold"):
        approval.load_poster_loss_approval(
            path,
            prior_revision=PARENT,
            source_snapshot_sha256=SOURCE,
            profile_releases_sha256=RELEASES,
            provisional_holds_sha256=HOLDS,
            release_decisions=(replace(_release(), state="held"),),
            validated_holds={(approval.POSTER_ID[1], approval.POSTER_ID[2]): _hold()},
        )


def test_exact_approved_loss_is_separate_from_reviewed_exclusions(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        approval,
        "POSTER_SOURCE_SHA256",
        hashlib.sha256(b"poster-raw").hexdigest(),
    )
    approved = _load(tmp_path)
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained losses"):
        _reconcile()
    _, receipt = _reconcile(approved)
    assert receipt["reviewed_losses"] == []
    assert receipt["approved_one_time_losses"] == ["/".join(approval.POSTER_ID)]
    assert receipt["poster_loss_approval_sha256"] == approved.manifest_sha256
    assert receipt["unexplained_losses"] == []

    sibling_prior = _poster_row(eligible=True)
    sibling_prior["person_id"] = "person-other"
    sibling_prior["author_pub_id"] = "other:paper"
    sibling_prior["work_id"] = approval.POSTER_WORK_ID
    sibling_current = {**sibling_prior, "map_eligible": False}
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained losses"):
        _reconcile(
            approved,
            prior_rows=[_poster_row(eligible=True), sibling_prior],
            current_rows=[_poster_row(eligible=False), sibling_current],
        )


def test_candidate_must_retain_the_unmapped_poster_work(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        approval,
        "POSTER_SOURCE_SHA256",
        hashlib.sha256(b"poster-raw").hexdigest(),
    )
    approved = _load(tmp_path)
    with pytest.raises(ValueError, match="exact prior and held row"):
        approved.require_exact_loss(
            prior_revision=PARENT,
            losses={approval.POSTER_ID},
            previous_rows={approval.POSTER_ID: _poster_row(eligible=True)},
            current_rows={approval.POSTER_ID: _poster_row(eligible=False)},
            previous_works={approval.POSTER_WORK_ID: _work(eligible=True)},
            final_works={},
        )


@pytest.mark.parametrize(
    "change",
    [
        "unused",
        "wrong_parent",
        "prior_work",
        "prior_source",
        "current_source",
        "status",
        "evidence",
        "prior_not_visible",
    ],
)
def test_approval_cannot_be_reused_or_cover_drift(
    monkeypatch, tmp_path, change
) -> None:
    monkeypatch.setattr(
        approval,
        "POSTER_SOURCE_SHA256",
        hashlib.sha256(b"poster-raw").hexdigest(),
    )
    approved = _load(tmp_path)
    old = _poster_row(eligible=True)
    current = _poster_row(eligible=False)
    old_work = _work(eligible=True)
    if change == "unused":
        current["map_eligible"] = True
    elif change == "wrong_parent":
        approved = replace(approved, prior_revision="e" * 40)
    elif change == "prior_work":
        old["work_id"] = "other-work"
    elif change == "prior_source":
        old["source_record_json"] = "changed"
    elif change == "current_source":
        current["source_record_json"] = "changed"
    elif change == "status":
        current["review_attribution_status"] = "released"
    elif change == "evidence":
        current["review_attribution_evidence_url"] = "https://scholar.google.com/other"
    elif change == "prior_not_visible":
        old_work["map_eligible"] = False
    with pytest.raises(ValueError, match="exact prior and held row"):
        _reconcile(
            approved,
            prior_rows=[old],
            current_rows=[current],
            prior_works=[old_work],
        )
