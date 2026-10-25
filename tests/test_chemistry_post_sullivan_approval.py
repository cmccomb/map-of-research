"""The next Chemistry release stays blocked until its real parent is pinned."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas
import pytest

from map_of_research import chemistry_post_sullivan_approval as approval_module
from map_of_research import region_continuity
from map_of_research.chemistry_post_sullivan_approval import (
    SOURCE_COMMIT,
    SOURCE_SHA256,
    validate_chemistry_post_sullivan_approval,
)
from map_of_research.io import sha256_file
from map_of_research.region_continuity import (
    RegionReleaseDiffError,
    reconcile_region_decisions,
)

REVISION = "a" * 40
SOURCE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    SOURCE_ROOT / "data/curated/chemistry-post-sullivan-exact-continuity.json"
)
REAL_PARENT_HASHER = approval_module._hosted_parent_hashes


def _work(work_id: str, eligible: bool = True) -> dict:
    return {
        "work_id": work_id,
        "map_eligible": eligible,
        "collaboration_eligible": eligible,
        "map_region_outlier": False,
        "map_exclusion_reasons": [],
        "department_ids": ["chem"],
        "quality_assessment_version": "quality-v1",
    }


@pytest.fixture
def reviewed_case(tmp_path, monkeypatch):
    document = json.loads(MANIFEST_PATH.read_text())
    document["status"] = "reviewed_exact_parent"
    document["prior_revision"] = REVISION
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for scholar_id in approval_module.SCHOLAR_COUNTS:
        cache = cache_dir / f"{scholar_id}.json"
        cache.write_text(json.dumps({"scholar_id": scholar_id}))
        document["scholar_cache_sha256"][scholar_id] = sha256_file(cache)
    monkeypatch.setattr(
        approval_module,
        "SCHOLAR_COUNTS",
        {"vF51LxIAAAAJ": 2, "zKKBwB4AAAAJ": 2, "FRtd8_QAAAAJ": 1, "ILambBMAAAAJ": 2},
    )
    old_rows = []
    new_rows = []
    old_works = {}
    new_works = {}
    for entry in document["affected_observations"]:
        raw = json.dumps({"scholar_observation": entry["author_pub_id"]})
        entry["source_record_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        old = {
            "person_id": entry["person_id"],
            "scholar_id": entry["scholar_id"],
            "author_pub_id": entry["author_pub_id"],
            "source_type": "scholar",
            "source_record_json": raw,
            "work_id": entry["prior_work_id"],
            "map_eligible": entry["prior_map_eligible"],
            "collaboration_eligible": entry["prior_collaboration_eligible"],
            "map_exclusion_reasons": [],
            "review_exclusion_code": "",
        }
        now = {
            **old,
            "work_id": entry["candidate_work_id"],
            "map_eligible": entry["candidate_map_eligible"],
            "collaboration_eligible": entry["candidate_collaboration_eligible"],
        }
        old_rows.append(old)
        new_rows.append(now)
        old_works.setdefault(entry["prior_work_id"], _work(entry["prior_work_id"]))
        new_works.setdefault(
            entry["candidate_work_id"], _work(entry["candidate_work_id"])
        )
        review = tmp_path / entry["review_path"]
        review.parent.mkdir(parents=True, exist_ok=True)
        review.write_text("Reviewed original sources.\n")
    manifest = tmp_path / "data/curated/approval.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(document, indent=2) + "\n")
    monkeypatch.setattr(approval_module, "APPROVAL_SHA256", sha256_file(manifest))
    monkeypatch.setattr(
        region_continuity,
        "CHEMISTRY_POST_SULLIVAN_APPROVAL_SHA256",
        sha256_file(manifest),
    )
    monkeypatch.setattr(
        approval_module,
        "_hosted_parent_hashes",
        lambda **kwargs: document["expected_parent_sha256_by_path"],
    )
    return {
        "manifest": manifest,
        "document": document,
        "cache_dir": cache_dir,
        "old_rows": old_rows,
        "new_rows": new_rows,
        "source_rows": copy.deepcopy(old_rows),
        "old_works": list(old_works.values()),
        "new_works": list(new_works.values()),
    }


def _validate(case):
    return validate_chemistry_post_sullivan_approval(
        approval_path=case["manifest"],
        prior_revision=REVISION,
        source_snapshot_commit=SOURCE_COMMIT,
        source_snapshot_sha256=SOURCE_SHA256,
        hf_token="test-token",
        repo_id="test/repo",
        cache_dir=case["cache_dir"],
        prior_works=case["old_works"],
        prior_profiles=case["old_rows"],
        source_rows=case["source_rows"],
        current_works=case["new_works"],
        current_profiles=case["new_rows"],
    )


def _rewrite_manifest(case, monkeypatch):
    case["manifest"].write_text(json.dumps(case["document"], indent=2) + "\n")
    digest = sha256_file(case["manifest"])
    monkeypatch.setattr(approval_module, "APPROVAL_SHA256", digest)
    monkeypatch.setattr(
        region_continuity, "CHEMISTRY_POST_SULLIVAN_APPROVAL_SHA256", digest
    )


def test_checked_in_candidate_cannot_approve_without_actual_hub_parent(monkeypatch):
    monkeypatch.setattr(
        approval_module,
        "_hosted_parent_hashes",
        lambda **kwargs: pytest.fail("inactive gate contacted the Hub"),
    )
    with pytest.raises(ValueError, match="awaits actual Hub parent"):
        approval_module._load_approved_document(
            MANIFEST_PATH,
            prior_revision=REVISION,
            source_snapshot_commit=SOURCE_COMMIT,
            source_snapshot_sha256=SOURCE_SHA256,
        )


def test_reviewed_seven_transition_set_passes_only_on_exact_parent(reviewed_case):
    approval = _validate(reviewed_case)
    assert len(approval.loss_identities) == 1
    assert len(approval.gain_identities) == 1
    assert len(approval.binding_changes) == 5


def test_wrong_hosted_parent_hash_rejects_release(reviewed_case, monkeypatch):
    monkeypatch.setattr(
        approval_module,
        "_hosted_parent_hashes",
        lambda **kwargs: {"README.md": "0" * 64},
    )
    with pytest.raises(ValueError, match="Hosted Sullivan parent differs"):
        _validate(reviewed_case)


def test_changed_cache_or_raw_source_rejects_release(reviewed_case):
    cache = reviewed_case["cache_dir"] / "FRtd8_QAAAAJ.json"
    cache.write_text("other cache")
    with pytest.raises(ValueError, match="complete Scholar cache changed"):
        _validate(reviewed_case)
    cache.write_text(json.dumps({"scholar_id": "FRtd8_QAAAAJ"}))
    reviewed_case["new_rows"][0]["source_record_json"] = "changed raw Scholar row"
    with pytest.raises(ValueError, match="Frozen source record changed"):
        _validate(reviewed_case)


def test_extra_reviewed_loss_fails_both_exact_gate_and_region_gate(reviewed_case):
    approval = _validate(reviewed_case)
    additional = {
        **reviewed_case["old_rows"][0],
        "person_id": "person-unrelated",
        "scholar_id": "unrelated",
        "author_pub_id": "unrelated:row",
        "work_id": "work-unrelated",
    }
    reviewed_case["old_rows"].append(additional)
    reviewed_case["source_rows"].append(copy.deepcopy(additional))
    reviewed_case["new_rows"].append(
        {
            **additional,
            "map_eligible": False,
            "collaboration_eligible": False,
            "review_exclusion_code": "not_a_publication",
            "map_exclusion_reasons": ["reviewed_not_a_publication"],
        }
    )
    reviewed_case["old_works"].append(_work("work-unrelated"))
    reviewed_case["new_works"].append(_work("work-unrelated", eligible=False))
    with pytest.raises(ValueError, match="unreviewed or unused row transition"):
        _validate(reviewed_case)
    with pytest.raises(RegionReleaseDiffError) as error:
        reconcile_region_decisions(
            pandas.DataFrame(reviewed_case["new_works"]),
            pandas.DataFrame(reviewed_case["new_rows"]),
            reviewed_case["old_works"],
            reviewed_case["old_rows"],
            prior_revision=REVISION,
            chemistry_post_sullivan_approval=approval,
        )
    assert any(
        "person-unrelated" in item for item in error.value.receipt["unexplained_losses"]
    )


def test_missing_reviewed_gain_fails_with_unused_approval(reviewed_case):
    approval = _validate(reviewed_case)
    gained = next(iter(approval.gain_identities))
    row = next(
        row for row in reviewed_case["new_rows"] if row["author_pub_id"] == gained[2]
    )
    row["map_eligible"] = False
    row["collaboration_eligible"] = False
    with pytest.raises(RegionReleaseDiffError) as error:
        reconcile_region_decisions(
            pandas.DataFrame(reviewed_case["new_works"]),
            pandas.DataFrame(reviewed_case["new_rows"]),
            reviewed_case["old_works"],
            reviewed_case["old_rows"],
            prior_revision=REVISION,
            chemistry_post_sullivan_approval=approval,
        )
    assert error.value.receipt["unused_chemistry_approvals"] == ["/".join(gained)]


def test_different_source_snapshot_cannot_rekey_without_exact_approval(reviewed_case):
    # A different source SHA skips the one-time Chemistry approval in publisher.
    # The ordinary graph gate must still reject the five changed mapped works.
    with pytest.raises(RegionReleaseDiffError) as error:
        reconcile_region_decisions(
            pandas.DataFrame(reviewed_case["new_works"]),
            pandas.DataFrame(reviewed_case["new_rows"]),
            reviewed_case["old_works"],
            reviewed_case["old_rows"],
            prior_revision=REVISION,
        )
    assert len(error.value.receipt["unapproved_mapped_binding_changes"]) == 5


def test_manifest_bytes_and_schema_are_pinned(reviewed_case, monkeypatch):
    reviewed_case["manifest"].write_text(reviewed_case["manifest"].read_text() + " ")
    with pytest.raises(ValueError, match="approval manifest changed"):
        _validate(reviewed_case)
    reviewed_case["document"].pop("reviewed_at")
    _rewrite_manifest(reviewed_case, monkeypatch)
    with pytest.raises(ValueError, match="approval schema changed"):
        _validate(reviewed_case)


def test_source_commit_and_snapshot_hash_are_pinned(reviewed_case):
    with pytest.raises(ValueError, match="source binding changed"):
        validate_chemistry_post_sullivan_approval(
            approval_path=reviewed_case["manifest"],
            prior_revision=REVISION,
            source_snapshot_commit="b" * 40,
            source_snapshot_sha256=SOURCE_SHA256,
            hf_token="test-token",
            repo_id="test/repo",
            cache_dir=reviewed_case["cache_dir"],
            prior_works=reviewed_case["old_works"],
            prior_profiles=reviewed_case["old_rows"],
            source_rows=reviewed_case["source_rows"],
            current_works=reviewed_case["new_works"],
            current_profiles=reviewed_case["new_rows"],
        )


def test_parent_file_inventory_and_count_contract_are_exact(reviewed_case, monkeypatch):
    reviewed_case["document"]["expected_parent_sha256_by_path"].pop("README.md")
    _rewrite_manifest(reviewed_case, monkeypatch)
    with pytest.raises(ValueError, match="parent file set changed"):
        _validate(reviewed_case)
    reviewed_case["document"]["expected_parent_sha256_by_path"]["README.md"] = "a" * 64
    reviewed_case["document"]["raw_profile_rows_before_after"] = [966, 967]
    _rewrite_manifest(reviewed_case, monkeypatch)
    with pytest.raises(ValueError, match="approval counts changed"):
        _validate(reviewed_case)


def test_hosted_parent_hashes_reads_immutable_files_and_rejects_missing(
    tmp_path, monkeypatch
):
    for relative_path in approval_module.PARENT_PATHS:
        file = tmp_path / relative_path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(relative_path.encode())
    calls = []

    def download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=download),
    )
    hashes = REAL_PARENT_HASHER(
        prior_revision=REVISION, hf_token="test-token", repo_id="test/repo"
    )
    assert set(hashes) == approval_module.PARENT_PATHS
    assert calls[0][1]["revision"] == REVISION
    (tmp_path / "README.md").unlink()
    with pytest.raises(ValueError, match="parent artifact is missing README.md"):
        REAL_PARENT_HASHER(
            prior_revision=REVISION, hf_token="test-token", repo_id="test/repo"
        )


def test_duplicate_and_missing_source_identities_fail(reviewed_case):
    reviewed_case["old_rows"].append(copy.deepcopy(reviewed_case["old_rows"][0]))
    with pytest.raises(ValueError, match="duplicate observation identities"):
        _validate(reviewed_case)
    reviewed_case["old_rows"].pop()
    reviewed_case["source_rows"].pop()
    with pytest.raises(ValueError, match="frozen source identity set"):
        _validate(reviewed_case)


def test_profile_counts_must_preserve_all_four_researchers(reviewed_case):
    for label in ("old_rows", "new_rows", "source_rows"):
        reviewed_case[label].pop()
    with pytest.raises(ValueError, match="prior raw profile counts changed"):
        _validate(reviewed_case)


def test_decision_schema_and_identity_order_are_exact(reviewed_case, monkeypatch):
    reviewed_case["document"]["affected_observations"][0].pop("review_path")
    _rewrite_manifest(reviewed_case, monkeypatch)
    with pytest.raises(ValueError, match="decision schema changed"):
        _validate(reviewed_case)
    reviewed_case["document"]["affected_observations"][0]["review_path"] = (
        "registry/reviews/2026-10-25-chem-gil-archive-provisional-hold.md"
    )
    entries = reviewed_case["document"]["affected_observations"]
    entries[0], entries[1] = entries[1], entries[0]
    _rewrite_manifest(reviewed_case, monkeypatch)
    with pytest.raises(ValueError, match="not sorted and unique"):
        _validate(reviewed_case)


def test_duplicate_work_and_observation_mismatch_fail(reviewed_case):
    reviewed_case["old_works"].append(copy.deepcopy(reviewed_case["old_works"][0]))
    with pytest.raises(ValueError, match="duplicate work IDs"):
        _validate(reviewed_case)
    reviewed_case["old_works"].pop()
    reviewed_case["old_rows"][0]["work_id"] = "work-unreviewed"
    with pytest.raises(ValueError, match="reviewed observation differs"):
        _validate(reviewed_case)


def test_held_gain_and_binding_require_exact_visibility(reviewed_case, monkeypatch):
    gil_work = reviewed_case["document"]["affected_observations"][0]["prior_work_id"]
    old_gil_work = next(
        row for row in reviewed_case["old_works"] if row["work_id"] == gil_work
    )
    old_gil_work["map_eligible"] = False
    with pytest.raises(ValueError, match="held loss changed"):
        _validate(reviewed_case)
    old_gil_work["map_eligible"] = True

    gain_entry = next(
        entry
        for entry in reviewed_case["document"]["affected_observations"]
        if entry["decision"] == "approved_gain_and_binding_change"
    )
    prior_gain = next(
        row
        for row in reviewed_case["old_rows"]
        if row["author_pub_id"] == gain_entry["author_pub_id"]
    )
    prior_gain["map_eligible"] = True
    prior_gain["collaboration_eligible"] = True
    gain_entry["prior_map_eligible"] = True
    gain_entry["prior_collaboration_eligible"] = True
    _rewrite_manifest(reviewed_case, monkeypatch)
    with pytest.raises(ValueError, match="gain changed"):
        _validate(reviewed_case)
    prior_gain["map_eligible"] = False
    prior_gain["collaboration_eligible"] = False
    gain_entry["prior_map_eligible"] = False
    gain_entry["prior_collaboration_eligible"] = False
    _rewrite_manifest(reviewed_case, monkeypatch)

    binding_entry = next(
        entry
        for entry in reviewed_case["document"]["affected_observations"]
        if entry["decision"] == "approved_binding_change"
    )
    old_binding_work = next(
        row
        for row in reviewed_case["old_works"]
        if row["work_id"] == binding_entry["prior_work_id"]
    )
    old_binding_work["map_eligible"] = False
    with pytest.raises(ValueError, match="work binding changed"):
        _validate(reviewed_case)


def test_unknown_decision_or_wrong_decision_count_rejected(reviewed_case, monkeypatch):
    entry = reviewed_case["document"]["affected_observations"][0]
    entry["decision"] = "unknown"
    _rewrite_manifest(reviewed_case, monkeypatch)
    with pytest.raises(ValueError, match="decision is unknown"):
        _validate(reviewed_case)
    entry["decision"] = "approved_held_loss"
    binding_entry = reviewed_case["document"]["affected_observations"][1]
    binding_entry["decision"] = "approved_held_loss"
    binding_entry["candidate_map_eligible"] = False
    binding_entry["candidate_collaboration_eligible"] = False
    matching = next(
        row
        for row in reviewed_case["new_rows"]
        if row["author_pub_id"] == binding_entry["author_pub_id"]
    )
    matching["map_eligible"] = False
    matching["collaboration_eligible"] = False
    _rewrite_manifest(reviewed_case, monkeypatch)
    with pytest.raises(ValueError, match="exact decision count changed"):
        _validate(reviewed_case)


def test_decision_digest_and_reconciler_reject_forged_approval(
    reviewed_case, monkeypatch
):
    monkeypatch.setattr(approval_module, "DECISION_SET_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="exact decision set changed"):
        _validate(reviewed_case)
    monkeypatch.setattr(
        approval_module,
        "DECISION_SET_SHA256",
        region_continuity.CHEMISTRY_POST_SULLIVAN_DECISION_SET_SHA256,
    )
    valid = _validate(reviewed_case)
    forged = replace(valid, manifest_sha256="0" * 64)
    with pytest.raises(ValueError, match="not the reviewed set"):
        reconcile_region_decisions(
            pandas.DataFrame(reviewed_case["new_works"]),
            pandas.DataFrame(reviewed_case["new_rows"]),
            reviewed_case["old_works"],
            reviewed_case["old_rows"],
            prior_revision=REVISION,
            chemistry_post_sullivan_approval=forged,
        )
