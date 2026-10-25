"""The Sullivan follow-up approves only eight source-bound continuity deltas."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from map_of_research import sullivan_continuity_addendum as addendum

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/curated/sullivan-continuity-addendum-bb88.json"
ROWS = ROOT / "tests/fixtures/sullivan_continuity_addendum_rows.json"


@pytest.fixture
def inputs():
    return {
        "addendum_path": MANIFEST,
        "prior_revision": addendum.PRIOR_REVISION,
        "source_snapshot_commit": addendum.SOURCE_COMMIT,
        "source_snapshot_sha256": addendum.SOURCE_SHA256,
        **json.loads(ROWS.read_text(encoding="utf-8")),
    }


def test_exact_hosted_losses_work_rekeys_and_curated_targets_pass(inputs):
    decision = addendum.validate_sullivan_continuity_addendum(**inputs)
    assert len(decision.loss_identities) == 5
    assert len(decision.binding_changes) == 3
    assert len(decision.required_curated_work_ids) == 2
    assert addendum.decision_set_digests(decision) == (
        addendum.LOSS_IDS_SHA256,
        addendum.BINDINGS_SHA256,
        addendum.TARGETS_SHA256,
    )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("source_commit", "different release"),
        ("prior_record", "added loss differs"),
        ("loss_eligible", "added loss differs"),
        ("loss_review_evidence", "added loss differs"),
        ("binding_work", "work binding differs"),
        ("binding_collaboration", "work binding differs"),
        ("agu_target_work", "curated target differs"),
        ("thesis_target_collaboration", "curated target differs"),
    ],
)
def test_unreviewed_source_or_endpoint_change_fails(inputs, change, expected):
    values = copy.deepcopy(inputs)
    if change == "source_commit":
        values["source_snapshot_commit"] = "0" * 40
    elif change == "prior_record":
        values["prior_profiles"][0]["source_record_json"] = "changed"
    elif change == "loss_eligible":
        row = next(
            row
            for row in values["current_profiles"]
            if row["author_pub_id"] == "wXSgFr4AAAAJ:9c2xU6iGI7YC"
        )
        row["map_eligible"] = True
    elif change == "loss_review_evidence":
        row = next(
            row
            for row in values["current_profiles"]
            if row["author_pub_id"] == "wXSgFr4AAAAJ:9c2xU6iGI7YC"
        )
        row["review_evidence_url"] = "https://example.org/other"
    elif change == "binding_work":
        row = next(
            row
            for row in values["current_profiles"]
            if row["author_pub_id"] == "wXSgFr4AAAAJ:NJ774b8OgUMC"
        )
        row["work_id"] = "unreviewed-work"
    elif change == "binding_collaboration":
        row = next(
            row
            for row in values["current_profiles"]
            if row["author_pub_id"] == "wXSgFr4AAAAJ:NJ774b8OgUMC"
        )
        row["collaboration_eligible"] = False
    elif change == "agu_target_work":
        row = next(
            row
            for row in values["current_profiles"]
            if row["source_native_id"] == "agu2017:a13j-05"
        )
        row["work_id"] = "unreviewed-work"
    else:
        row = next(
            row
            for row in values["current_profiles"]
            if row["source_native_id"] == "escholarship:qt89c072pv"
        )
        row["collaboration_eligible"] = False
    with pytest.raises(ValueError, match=expected):
        addendum.validate_sullivan_continuity_addendum(**values)


def test_addendum_bytes_are_pinned(inputs, tmp_path):
    changed = tmp_path / "addendum.json"
    changed.write_bytes(MANIFEST.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest changed"):
        addendum.validate_sullivan_continuity_addendum(
            **{**inputs, "addendum_path": changed}
        )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("schema", "schema changed"),
        ("release", "release binding changed"),
        ("decision_count", "decision set changed"),
        ("identity", "identities changed"),
        ("exact_digest", "exact decision set changed"),
    ],
)
def test_addendum_document_mutations_fail_closed(
    inputs, monkeypatch, tmp_path, change, expected
):
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if change == "schema":
        document["unexpected"] = True
    elif change == "release":
        document["source_snapshot_commit"] = "0" * 40
    elif change == "decision_count":
        document["additional_approved_losses"].pop()
    elif change == "identity":
        document["required_curated_targets"][0]["source_native_id"] = "other"
    else:
        monkeypatch.setattr(addendum, "LOSS_IDS_SHA256", "0" * 64)
    path = tmp_path / "altered-addendum.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        addendum, "ADDENDUM_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match=expected):
        addendum.validate_sullivan_continuity_addendum(
            **{**inputs, "addendum_path": path}
        )


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("source_rows", "duplicate identities"),
        ("current_works", "work IDs are duplicated"),
    ],
)
def test_duplicate_source_or_work_identity_fails(inputs, group, expected):
    values = copy.deepcopy(inputs)
    values[group].append(copy.deepcopy(values[group][0]))
    with pytest.raises(ValueError, match=expected):
        addendum.validate_sullivan_continuity_addendum(**values)
