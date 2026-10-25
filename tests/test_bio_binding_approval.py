"""The Bio 38f approval covers only 11 reviewed work rekeys."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas
import pytest

from map_of_research import bio_binding_approval as approval
from map_of_research.region_continuity import (
    RegionReleaseDiffError,
    reconcile_region_decisions,
)

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data/curated"
FROZEN_BIO_CURATED = ROOT / "tests/fixtures/bio_approved_works.json"


def _inputs() -> dict:
    return {
        "approval_path": CURATED / "bio-exact-binding-approval-dcb.json",
        "prior_revision": approval.PRIOR_REVISION,
        "source_snapshot_commit": approval.SOURCE_COMMIT,
        "source_snapshot_sha256": approval.SOURCE_SHA256,
        "doi_aliases_path": CURATED / "scholar-doi-aliases.json",
        "exclusions_path": CURATED / "scholar-exclusions.json",
        "person_publication_holds_path": CURATED / "person-publication-holds.json",
        "curated_works_path": FROZEN_BIO_CURATED,
        "profile_releases_path": CURATED / "scholar-profile-releases.json",
        "provisional_holds_path": CURATED / "scholar-provisional-holds.json",
        "poster_loss_approval_path": CURATED / "bio-mcmanus-poster-loss-approval.json",
    }


def _work(work_id: str) -> dict:
    return {
        "work_id": work_id,
        "map_eligible": True,
        "map_region_outlier": False,
        "quality_assessment_version": "quality-v1",
        "department_ids": ["bio"],
    }


def _candidate():
    approved = approval.load_bio_binding_approval(**_inputs())
    previous = []
    current = []
    prior_works = []
    current_works = []
    for identity, old_id, new_id, _, _ in sorted(approved.binding_changes):
        base = {
            "person_id": identity[0],
            "scholar_id": identity[1],
            "author_pub_id": identity[2],
            "source_type": identity[3],
            "source_record_json": '{"title":"Reviewed Bio publication"}',
            "title": "Reviewed Bio publication",
            "authors": "A. Researcher",
            "year": 2020,
            "map_eligible": True,
            "collaboration_eligible": True,
        }
        previous.append({**base, "work_id": old_id})
        current.append({**base, "work_id": new_id})
        prior_works.append(_work(old_id))
        current_works.append(_work(new_id))
    return approved, previous, current, prior_works, current_works


def _reconcile(approved, previous, current, prior_works, current_works):
    return reconcile_region_decisions(
        pandas.DataFrame(current_works),
        pandas.DataFrame(current),
        prior_works,
        previous,
        prior_revision=approval.PRIOR_REVISION,
        bio_binding_approval=approved,
    )


def test_exact_review_matches_all_11_bio_bindings():
    assert approval.sha256_file(FROZEN_BIO_CURATED) == json.loads(
        (CURATED / "bio-exact-binding-approval-dcb.json").read_text(encoding="utf-8")
    )["curated_works_sha256"]
    approved, previous, current, prior_works, current_works = _candidate()
    assert len(approved.binding_changes) == 11
    assert approval.binding_digest(approved.binding_changes) == approval.BINDINGS_SHA256
    _, receipt = _reconcile(approved, previous, current, prior_works, current_works)
    assert len(receipt["approved_bio_binding_changes"]) == 11
    assert receipt["unapproved_mapped_binding_changes"] == []
    assert receipt["unused_bio_binding_approvals"] == []
    assert receipt["bio_binding_approval_sha256"] == approval.APPROVAL_SHA256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prior_revision", "0" * 40),
        ("source_snapshot_commit", "0" * 40),
        ("source_snapshot_sha256", "0" * 64),
    ],
)
def test_approval_rejects_stale_parent_or_source(field, value):
    inputs = _inputs()
    inputs[field] = value
    with pytest.raises(ValueError, match="different release"):
        approval.load_bio_binding_approval(**inputs)


def _load_changed_document(tmp_path, monkeypatch, change):
    inputs = _inputs()
    document = json.loads(inputs["approval_path"].read_text())
    change(document)
    copy_path = tmp_path / "approval.json"
    copy_path.write_text(json.dumps(document), encoding="utf-8")
    inputs["approval_path"] = copy_path
    monkeypatch.setattr(
        approval, "APPROVAL_SHA256", hashlib.sha256(copy_path.read_bytes()).hexdigest()
    )
    return approval.load_bio_binding_approval(**inputs)


def test_approval_rejects_manifest_or_evidence_drift(tmp_path, monkeypatch):
    inputs = _inputs()
    copy_path = tmp_path / "approval.json"
    copy_path.write_bytes(inputs["approval_path"].read_bytes() + b" ")
    inputs["approval_path"] = copy_path
    with pytest.raises(ValueError, match="manifest changed"):
        approval.load_bio_binding_approval(**inputs)

    document = json.loads(copy_path.read_text())
    document["approved_binding_changes"][0]["evidence_url"] = (
        "https://example.com/unreviewed"
    )
    copy_path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        approval, "APPROVAL_SHA256", hashlib.sha256(copy_path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="lacks exact reviewed evidence"):
        approval.load_bio_binding_approval(**inputs)


def test_approval_requires_poster_approval():
    inputs = _inputs()
    inputs["poster_loss_approval_path"] = None
    with pytest.raises(ValueError, match="requires the exact poster approval"):
        approval.load_bio_binding_approval(**inputs)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda doc: doc.pop("reviewed_at"), "schema changed"),
        (
            lambda doc: doc.__setitem__("profile_releases_sha256", "0" * 64),
            "evidence or release binding changed",
        ),
        (
            lambda doc: doc["approved_binding_changes"].pop(),
            "decision set changed",
        ),
        (
            lambda doc: doc["approved_binding_changes"][0].__setitem__(
                "current_work_id", doc["approved_binding_changes"][0]["prior_work_id"]
            ),
            "row differs",
        ),
        (
            lambda doc: doc["approved_binding_changes"].__setitem__(
                1, copy.deepcopy(doc["approved_binding_changes"][0])
            ),
            "identities changed",
        ),
        (
            lambda doc: doc["approved_binding_changes"][0].__setitem__(
                "current_work_id", "work-reviewed-but-different"
            ),
            "exact decision set changed",
        ),
    ],
)
def test_approval_rejects_changed_schema_or_exact_decision(
    tmp_path, monkeypatch, change, message
):
    with pytest.raises(ValueError, match=message):
        _load_changed_document(tmp_path, monkeypatch, change)


def test_continuity_rejects_forged_bio_approval():
    approved, previous, current, prior_works, current_works = _candidate()
    with pytest.raises(ValueError, match="not the reviewed set"):
        _reconcile(object(), previous, current, prior_works, current_works)
    with pytest.raises(ValueError, match="not the reviewed set"):
        reconcile_region_decisions(
            pandas.DataFrame(current_works),
            pandas.DataFrame(current),
            prior_works,
            previous,
            prior_revision="a" * 40,
            bio_binding_approval=approved,
        )


def test_approval_rejects_missing_or_extra_binding():
    approved, previous, current, prior_works, current_works = _candidate()
    missing = copy.deepcopy(current)
    missing[0]["work_id"] = previous[0]["work_id"]
    current_works.append(_work(previous[0]["work_id"]))
    with pytest.raises(RegionReleaseDiffError) as error:
        _reconcile(approved, previous, missing, prior_works, current_works)
    assert len(error.value.receipt["unused_bio_binding_approvals"]) == 1

    extra_previous = {
        **previous[0],
        "person_id": "person-unreviewed",
        "author_pub_id": "unreviewed:one",
        "scholar_id": "unreviewed",
        "work_id": "work-old-unreviewed",
    }
    extra_current = {**extra_previous, "work_id": "work-new-unreviewed"}
    with pytest.raises(RegionReleaseDiffError) as error:
        _reconcile(
            approved,
            [*previous, extra_previous],
            [*current, extra_current],
            [*prior_works, _work("work-old-unreviewed")],
            [*current_works, _work("work-new-unreviewed")],
        )
    assert len(error.value.receipt["unapproved_mapped_binding_changes"]) == 1


def test_approval_rejects_changed_binding_tuple():
    approved, previous, current, prior_works, current_works = _candidate()
    current[0]["collaboration_eligible"] = False
    with pytest.raises(RegionReleaseDiffError) as error:
        _reconcile(approved, previous, current, prior_works, current_works)
    assert len(error.value.receipt["unapproved_mapped_binding_changes"]) == 1
    assert len(error.value.receipt["unused_bio_binding_approvals"]) == 1
