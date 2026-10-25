"""The Sullivan transition permits exactly the reviewed bb88 row losses."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy
import pytest

from map_of_research import sullivan_loss_approval as approval
from map_of_research.provisional_holds import load_provisional_holds

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data/curated"
FIXTURE = ROOT / "tests/fixtures/sullivan_exact_loss_rows.json"
FROZEN_CURATED = ROOT / "tests/fixtures/sullivan_approved_works.json"
REVIEW = ROOT / "registry/reviews/2026-10-25-chem-sullivan-prior-id-crosswalk.json"


@pytest.fixture
def verified_rows(monkeypatch, tmp_path):
    """Use recorded hosted prior rows without a network dependency in CI."""

    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    real_sha256_file = approval.sha256_file
    cache = tmp_path / f"{approval.SULLIVAN_ID}.json"
    cache.write_text("test fixture: checked in the full no-write preview")
    approval_manifest = json.loads(
        (CURATED / "sullivan-exact-losses-bb88.json").read_text(encoding="utf-8")
    )
    # The producer tree can advance to the Bio review after the Sullivan source
    # was frozen. The release passes paths from that pinned source checkout.
    pinned_source_hashes = {
        CURATED / "scholar-profile-releases.json": approval_manifest[
            "profile_releases_sha256"
        ],
        CURATED / "scholar-provisional-holds.json": approval_manifest[
            "provisional_holds_sha256"
        ],
        CURATED / "works.json": approval_manifest["curated_works_sha256"],
        CURATED / "scholar-exclusions.json": approval_manifest[
            "reviewed_exclusions_sha256"
        ],
    }
    expected_curated_sha = approval_manifest["curated_works_sha256"]
    assert approval.sha256_file(FROZEN_CURATED) == expected_curated_sha

    def fixture_sha(path):
        if Path(path) == cache:
            return approval_manifest["sullivan_cache_sha256"]
        if Path(path) in pinned_source_hashes:
            return pinned_source_hashes[Path(path)]
        return real_sha256_file(path)

    monkeypatch.setattr(approval, "sha256_file", fixture_sha)
    profiles = load_provisional_holds(CURATED / "scholar-provisional-holds.json")
    holds = {
        (item.scholar_id, item.author_pub_id): item
        for profile in profiles
        if profile.scholar_id == approval.SULLIVAN_ID
        for item in profile.held_observations
    }
    return {
        "approval_path": CURATED / "sullivan-exact-losses-bb88.json",
        "prior_revision": approval.PRIOR_REVISION,
        "source_snapshot_commit": approval.SOURCE_COMMIT,
        "source_snapshot_sha256": approval.SOURCE_SHA256,
        "cache_path": cache,
        "profile_releases_path": CURATED / "scholar-profile-releases.json",
        "provisional_holds_path": CURATED / "scholar-provisional-holds.json",
        "crosswalk_path": REVIEW,
        "curated_works_path": FROZEN_CURATED,
        "exclusions_path": CURATED / "scholar-exclusions.json",
        "prior_works": rows["prior_works"],
        "prior_profiles": rows["prior_profiles"],
        "source_rows": rows["source_rows"],
        "current_profiles": rows["current_profiles"],
        "validated_holds": holds,
    }


def test_exact_hosted_rows_and_candidate_dispositions_pass(verified_rows):
    result = approval.validate_sullivan_exact_losses(**verified_rows)
    assert len(result.identities) == 22
    assert len(result.held_identities) == 18
    assert len(result.rekey_identities) == 4
    assert result.manifest_sha256 == approval.APPROVAL_SHA256


def test_historical_approval_rejects_current_extended_curated_file(
    verified_rows, tmp_path
):
    current = tmp_path / "current-works.json"
    current.write_bytes((CURATED / "works.json").read_bytes())
    inputs = {**verified_rows, "curated_works_path": current}
    with pytest.raises(ValueError, match="source changed: curated_works_sha256"):
        approval.validate_sullivan_exact_losses(**inputs)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("prior_revision", "different release"),
        ("source_snapshot_commit", "different release"),
        ("source_snapshot_sha256", "different release"),
        ("prior_work_id", "differs from Hub parent"),
        ("prior_source_record", "differs from Hub parent"),
        ("held_evidence", "held loss lacks exact evidence"),
        ("raw_held_missing", "held loss lacks exact evidence"),
        ("rekey_target_missing", "rekey target absent"),
        ("rekey_prior_missing", "missing from Hub parent"),
        ("prior_duplicate", "duplicate identities"),
        ("raw_duplicate", "duplicate identities"),
        ("current_duplicate", "duplicate identities"),
        ("rekey_old_raw", "rekey loss lacks exact evidence"),
        ("false_target_mapped", "excluded rekey target changed"),
        ("egu_target_unmapped", "duplicate rekey target not mapped"),
        ("egu_target_no_collaboration", "duplicate rekey target not mapped"),
        ("egu_target_new_work", "duplicate rekey target not mapped"),
        ("agu_target_unmapped", "curated rekey target absent"),
        ("agu_target_no_evidence", "curated rekey target absent"),
    ],
)
def test_any_unreviewed_parent_or_candidate_change_fails(
    verified_rows, change, expected
):
    inputs = copy.deepcopy(verified_rows)
    if change in {"prior_revision", "source_snapshot_commit", "source_snapshot_sha256"}:
        inputs[change] = "0" * (64 if change == "source_snapshot_sha256" else 40)
    elif change == "prior_work_id":
        inputs["prior_profiles"][0]["work_id"] = "work-changed"
    elif change == "prior_source_record":
        inputs["prior_profiles"][0]["source_record_json"] = "changed"
    elif change == "held_evidence":
        held = next(
            row
            for row in inputs["current_profiles"]
            if row.get("review_attribution_status") == "held"
        )
        held["review_attribution_evidence_url"] = "https://example.org/unreviewed"
    elif change == "raw_held_missing":
        held_id = next(
            row["author_pub_id"]
            for row in inputs["current_profiles"]
            if row.get("review_attribution_status") == "held"
        )
        inputs["source_rows"] = [
            row for row in inputs["source_rows"] if row["author_pub_id"] != held_id
        ]
    elif change == "rekey_target_missing":
        inputs["current_profiles"] = [
            row
            for row in inputs["current_profiles"]
            if row["author_pub_id"] != "wXSgFr4AAAAJ:j7_hQOaDUrUC"
        ]
    elif change == "rekey_prior_missing":
        inputs["prior_profiles"] = [
            row
            for row in inputs["prior_profiles"]
            if row["author_pub_id"] != "wXSgFr4AAAAJ:Ehil0879vHcC"
        ]
    elif change == "prior_duplicate":
        inputs["prior_profiles"].append(inputs["prior_profiles"][0])
    elif change == "raw_duplicate":
        inputs["source_rows"].append(inputs["source_rows"][0])
    elif change == "current_duplicate":
        inputs["current_profiles"].append(inputs["current_profiles"][0])
    elif change == "rekey_old_raw":
        old = next(
            row
            for row in inputs["prior_profiles"]
            if row["author_pub_id"] == "wXSgFr4AAAAJ:Ehil0879vHcC"
        )
        inputs["source_rows"].append(old)
    elif change == "false_target_mapped":
        target = next(
            row
            for row in inputs["current_profiles"]
            if row["author_pub_id"] == "wXSgFr4AAAAJ:XUvXOeBm_78C"
        )
        target["map_eligible"] = True
    elif change.startswith("egu_target_"):
        target = next(
            row
            for row in inputs["current_profiles"]
            if row["author_pub_id"] == "wXSgFr4AAAAJ:j7_hQOaDUrUC"
        )
        if change == "egu_target_unmapped":
            target["map_eligible"] = False
        elif change == "egu_target_no_collaboration":
            target["collaboration_eligible"] = False
        else:
            target["work_id"] = "new-work-without-lowry"
    elif change.startswith("agu_target_"):
        target = next(
            row
            for row in inputs["current_profiles"]
            if row.get("source_native_id") == "agu2017:a13j-05"
        )
        if change == "agu_target_unmapped":
            target["map_eligible"] = False
        else:
            raw = next(
                row
                for row in inputs["source_rows"]
                if row.get("source_native_id") == "agu2017:a13j-05"
            )
            raw["source_url"] = ""
            raw["source_verification_urls"] = []
    with pytest.raises(ValueError, match=expected):
        approval.validate_sullivan_exact_losses(**inputs)


def test_manifest_bytes_and_source_manifests_are_immutable(verified_rows, tmp_path):
    inputs = dict(verified_rows)
    approval_copy = tmp_path / "approval.json"
    approval_copy.write_bytes(Path(inputs["approval_path"]).read_bytes() + b" ")
    inputs["approval_path"] = approval_copy
    with pytest.raises(ValueError, match="approval manifest changed"):
        approval.validate_sullivan_exact_losses(**inputs)

    inputs["approval_path"] = verified_rows["approval_path"]
    inputs["profile_releases_path"] = approval_copy
    with pytest.raises(ValueError, match="source changed: profile_releases_sha256"):
        approval.validate_sullivan_exact_losses(**inputs)


def test_array_and_missing_reason_values_are_handled():
    assert approval._values(None) == []
    assert approval._values(numpy.array(["attribution_provisional_hold"])) == [
        "attribution_provisional_hold"
    ]


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("document_not_object", "approval schema changed"),
        ("document_missing_field", "approval schema changed"),
        ("wrong_release_binding", "release binding changed"),
        ("wrong_entry_count", "needs exactly 22 rows"),
        ("wrong_entry_schema", "entry schema changed"),
        ("duplicate_identity", "sorted and unique"),
        ("wrong_identity_digest", "identity set changed"),
        ("another_profile", "includes another profile"),
        ("wrong_decision", "loss decision changed"),
    ],
)
def test_approval_document_validation_fails_closed(
    verified_rows, monkeypatch, tmp_path, change, expected
):
    inputs = dict(verified_rows)
    source = json.loads(Path(inputs["approval_path"]).read_text(encoding="utf-8"))
    if change == "document_not_object":
        source = []
    elif change == "document_missing_field":
        source.pop("reviewed_at")
    elif change == "wrong_release_binding":
        source["source_snapshot_commit"] = "0" * 40
    elif change == "wrong_entry_count":
        source["approved_losses"].pop()
    elif change == "wrong_entry_schema":
        source["approved_losses"][0]["unexpected"] = "value"
    elif change == "duplicate_identity":
        source["approved_losses"][1]["author_pub_id"] = source["approved_losses"][0][
            "author_pub_id"
        ]
    elif change == "wrong_identity_digest":
        monkeypatch.setattr(approval, "APPROVED_IDS_SHA256", "0" * 64)
    elif change == "another_profile":
        source["approved_losses"][-1]["person_id"] = "person-tz-other"
        identities = [approval._identity(row) for row in source["approved_losses"]]
        assert identities == sorted(identities)
        digest = hashlib.sha256(
            "\n".join("/".join(identity) for identity in identities).encode()
        ).hexdigest()
        monkeypatch.setattr(approval, "APPROVED_IDS_SHA256", digest)
    elif change == "wrong_decision":
        source["approved_losses"][0]["decision"] = "unreviewed"
    approval_copy = tmp_path / "approval.json"
    approval_copy.write_text(json.dumps(source), encoding="utf-8")
    inputs["approval_path"] = approval_copy
    real_sha256_file = approval.sha256_file

    def pinned_sha(path):
        return (
            approval.APPROVAL_SHA256
            if Path(path) == approval_copy
            else real_sha256_file(path)
        )

    monkeypatch.setattr(approval, "sha256_file", pinned_sha)
    with pytest.raises(ValueError, match=expected):
        approval.validate_sullivan_exact_losses(**inputs)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("wrong_owner", "crosswalk differs"),
        ("missing_rekey", "four rekeys"),
        ("wrong_old_digest", "rekey loss lacks exact evidence"),
        ("wrong_target_kind", "rekey target type changed"),
        ("wrong_target_class", "rekey classification changed"),
        ("wrong_target_title", "rekey target absent"),
        ("wrong_agu_evidence", "curated rekey target absent"),
    ],
)
def test_crosswalk_target_validation_fails_closed(
    verified_rows, monkeypatch, tmp_path, change, expected
):
    inputs = dict(verified_rows)
    crosswalk = json.loads(Path(inputs["crosswalk_path"]).read_text(encoding="utf-8"))
    entries = crosswalk["entries"]
    if change == "wrong_owner":
        crosswalk["person_id"] = "person-other"
    elif change == "missing_rekey":
        entries.pop()
    elif change == "wrong_old_digest":
        entries[0]["legacy_row_sha256"] = "0" * 64
    elif change == "wrong_target_kind":
        entries[0]["target_kind"] = "unreviewed"
    elif change == "wrong_target_class":
        target = next(
            row for row in entries if row["classification"] == "duplicate_rekeyed"
        )
        target["classification"] = "unreviewed"
    elif change == "wrong_target_title":
        target = next(
            row for row in entries if row["classification"] == "duplicate_rekeyed"
        )
        target["target_title"] = "Different abstract"
    elif change == "wrong_agu_evidence":
        target = next(row for row in entries if row["target_kind"] == "curated_native")
        target["evidence_url"] = "https://example.org/different"
        approved = json.loads(Path(inputs["approval_path"]).read_text(encoding="utf-8"))
        for row in approved["approved_losses"]:
            if row["author_pub_id"] == target["prior_author_pub_id"]:
                row["evidence_url"] = target["evidence_url"]
        approval_copy = tmp_path / "approval.json"
        approval_copy.write_text(json.dumps(approved), encoding="utf-8")
        inputs["approval_path"] = approval_copy
    crosswalk_copy = tmp_path / "crosswalk.json"
    crosswalk_copy.write_text(json.dumps(crosswalk), encoding="utf-8")
    inputs["crosswalk_path"] = crosswalk_copy
    real_sha256_file = approval.sha256_file
    approved = json.loads((CURATED / "sullivan-exact-losses-bb88.json").read_text())

    def pinned_sha(path):
        if Path(path) == crosswalk_copy:
            return approved["prior_id_crosswalk_sha256"]
        if (
            Path(path) == inputs["approval_path"]
            and Path(path) != CURATED / "sullivan-exact-losses-bb88.json"
        ):
            return approval.APPROVAL_SHA256
        return real_sha256_file(path)

    monkeypatch.setattr(approval, "sha256_file", pinned_sha)
    with pytest.raises(ValueError, match=expected):
        approval.validate_sullivan_exact_losses(**inputs)
