"""The first failed upload must not erase the prior successful release."""

from __future__ import annotations

import json
import runpy
import sys

import pytest

from map_of_research.atomic_release import RELEASE_PATHS
from map_of_research.release_baseline import pin_prior_release

REVISION = "a" * 40


def test_first_upload_pins_success_before_a_failure_and_retry(
    tmp_path, monkeypatch, capsys
):
    receipt = {"status": "success", "artifact_commit": REVISION}
    latest = tmp_path / "last-upload.json"
    latest.write_text(json.dumps(receipt) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["python -m map_of_research.release_baseline", "--status-dir", str(tmp_path)],
    )
    with monkeypatch.context() as patch:
        patch.delitem(sys.modules, "map_of_research.release_baseline")
        runpy.run_module("map_of_research.release_baseline", run_name="__main__")
    assert capsys.readouterr().out == f"revision={REVISION}\n"
    saved = tmp_path / "last-successful-upload.json"
    assert saved.read_bytes() == latest.read_bytes()

    latest.write_text(json.dumps({"status": "failure"}) + "\n")
    assert pin_prior_release(tmp_path) == REVISION
    assert json.loads(saved.read_text()) == receipt


@pytest.mark.parametrize(
    "receipt",
    [
        {"status": "failure", "artifact_commit": REVISION},
        {"status": "success", "artifact_commit": "main"},
        {"status": "success"},
    ],
)
def test_unvalidated_receipt_cannot_be_pinned(tmp_path, receipt):
    (tmp_path / "last-upload.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        pin_prior_release(tmp_path)
    assert not (tmp_path / "last-successful-upload.json").exists()


def test_invalid_saved_success_is_not_replaced_by_later_receipt(tmp_path):
    (tmp_path / "last-successful-upload.json").write_text(
        json.dumps({"status": "failure", "artifact_commit": REVISION})
    )
    (tmp_path / "last-upload.json").write_text(
        json.dumps({"status": "success", "artifact_commit": "b" * 40})
    )
    with pytest.raises(ValueError, match="not successful"):
        pin_prior_release(tmp_path)


def test_missing_previous_receipt_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        pin_prior_release(tmp_path)


def test_ambiguous_atomic_write_blocks_retry_until_exact_review(tmp_path):
    latest = tmp_path / "last-upload.json"
    successful = tmp_path / "last-successful-upload.json"
    successful.write_text(
        json.dumps({"status": "success", "artifact_commit": REVISION})
    )
    intent = {
        "commit_nonce": "b" * 32,
        "parent_revision": REVISION,
        "size_by_path": {path: 1 for path in RELEASE_PATHS},
        "sha256_by_path": {path: "c" * 64 for path in RELEASE_PATHS},
    }
    failure = {
        "status": "failure",
        "atomic_release_intent": intent,
        "atomic_release_reconciliation": {
            **intent,
            "write_outcome": "unknown",
            "observed_head_revision": REVISION,
        },
    }
    latest.write_text(json.dumps(failure))
    with pytest.raises(ValueError, match="awaits exact reviewed reconciliation"):
        pin_prior_release(tmp_path)

    resolution = {
        "outcome": "reconciled_no_write",
        "commit_nonce": intent["commit_nonce"],
        "parent_revision": REVISION,
        "observed_head_revision": REVISION,
        "matching_commit_ids": [],
        "reviewer": "reviewed-pr-123",
        "reviewed_at_utc": "2026-10-25T12:00:00Z",
        "evidence_url": "https://example.test/review/123",
    }
    failure["atomic_release_resolution"] = resolution
    latest.write_text(json.dumps(failure))
    assert pin_prior_release(tmp_path) == REVISION

    failure["atomic_release_resolution"] = {
        **resolution,
        "commit_nonce": "wrong-nonce",
    }
    latest.write_text(json.dumps(failure))
    with pytest.raises(ValueError, match="awaits exact reviewed reconciliation"):
        pin_prior_release(tmp_path)

    committed = "d" * 40
    successful.write_text(
        json.dumps({"status": "success", "artifact_commit": committed})
    )
    failure["atomic_release_resolution"] = {
        **resolution,
        "outcome": "verified_success",
        "artifact_revision": committed,
        "verified_sha256_by_path": intent["sha256_by_path"],
        "verified_size_by_path": intent["size_by_path"],
    }
    latest.write_text(json.dumps(failure))
    assert pin_prior_release(tmp_path) == committed
    failure["atomic_release_resolution"]["verified_size_by_path"] = {}
    latest.write_text(json.dumps(failure))
    with pytest.raises(ValueError, match="differs from intent"):
        pin_prior_release(tmp_path)


def test_verified_atomic_success_with_staged_intent_allows_next_upload(tmp_path):
    parent = REVISION
    committed = "d" * 40
    intent = {
        "commit_nonce": "b" * 32,
        "parent_revision": parent,
        "size_by_path": {path: 1 for path in RELEASE_PATHS},
        "sha256_by_path": {path: "c" * 64 for path in RELEASE_PATHS},
    }
    receipt = {
        "status": "success",
        "artifact_commit": committed,
        "atomic_release_intent": intent,
        "atomic_release": {
            **intent,
            "artifact_revision": committed,
            "hosted_file_bytes_match_prewrite": True,
        },
    }
    (tmp_path / "last-upload.json").write_text(json.dumps(receipt))
    (tmp_path / "last-successful-upload.json").write_text(json.dumps(receipt))
    assert pin_prior_release(tmp_path) == committed
    receipt["atomic_release"]["hosted_file_bytes_match_prewrite"] = False
    (tmp_path / "last-upload.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="differs from staged intent"):
        pin_prior_release(tmp_path)


def test_malformed_or_misreviewed_atomic_intent_cannot_clear_retry_gate(tmp_path):
    (tmp_path / "last-successful-upload.json").write_text(
        json.dumps({"status": "success", "artifact_commit": REVISION})
    )
    latest = tmp_path / "last-upload.json"
    failure = {"status": "failure", "atomic_release_intent": None}
    latest.write_text(json.dumps(failure))
    with pytest.raises(ValueError, match="intent is malformed"):
        pin_prior_release(tmp_path)

    intent = {
        "commit_nonce": "b" * 32,
        "parent_revision": REVISION,
        "size_by_path": {path: 1 for path in RELEASE_PATHS},
        "sha256_by_path": {path: "c" * 64 for path in RELEASE_PATHS},
    }
    failure["atomic_release_intent"] = {**intent, "size_by_path": {}}
    latest.write_text(json.dumps(failure))
    with pytest.raises(ValueError, match="lacks seven staged files"):
        pin_prior_release(tmp_path)

    resolution = {
        "outcome": "reconciled_no_write",
        "commit_nonce": intent["commit_nonce"],
        "parent_revision": REVISION,
        "observed_head_revision": "d" * 40,
        "matching_commit_ids": [],
        "reviewer": "reviewed-pr-123",
        "reviewed_at_utc": "2026-10-25T12:00:00Z",
        "evidence_url": "https://example.test/review/123",
    }
    failure["atomic_release_intent"] = intent
    failure["atomic_release_resolution"] = resolution
    latest.write_text(json.dumps(failure))
    with pytest.raises(ValueError, match="differs from the pinned parent"):
        pin_prior_release(tmp_path)

    failure["atomic_release_resolution"] = {
        **resolution,
        "outcome": "unreviewed_retry",
        "observed_head_revision": REVISION,
    }
    latest.write_text(json.dumps(failure))
    with pytest.raises(ValueError, match="awaits exact reviewed reconciliation"):
        pin_prior_release(tmp_path)


def test_saved_success_can_be_pinned_without_a_latest_receipt(tmp_path):
    (tmp_path / "last-successful-upload.json").write_text(
        json.dumps({"status": "success", "artifact_commit": REVISION})
    )
    assert pin_prior_release(tmp_path) == REVISION
