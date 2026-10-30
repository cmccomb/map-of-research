"""The publisher stages complete release files before its only Hub call."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas
import pytest

import map_of_research.publisher as publisher
from map_of_research.atomic_release import RELEASE_PATHS
from map_of_research.presentation_drift import PresentationDriftError
from map_of_research.release_baseline import pin_prior_release


def _bio_registry():
    return SimpleNamespace(
        memberships=[
            SimpleNamespace(person_id="bio-person", included=True, department_id="bio")
        ]
    )


def _prior_bio_people():
    return [{"person_id": "bio-person", "included_department_ids": ["bio"]}]


def test_manual_bio_dispatch_passes_release_intent() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/upload.yml"
    ).read_text(encoding="utf-8")
    assert (
        "REQUIRE_BIO_RELEASE: ${{ github.event_name == 'workflow_dispatch' "
        "&& inputs.require_bio_release || false }}"
    ) in workflow
    assert "BIO_RELEASE_ARGS+=(--require-bio-release)" in workflow
    assert '"${BIO_RELEASE_ARGS[@]}"' in workflow
    assert "MAP_DURABLE_INTENT_REPO: receipt" in workflow
    assert "MAP_DURABLE_INTENT_BRANCH: automation/map-snapshot" in workflow


def test_manual_additive_dispatch_requires_fixed_review_file() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/upload.yml"
    ).read_text(encoding="utf-8")
    assert "use_additive_presentation_review:" in workflow
    review_path = "source/data/curated/scholar-additive-placement-review.json"
    assert f"ADDITIVE_REVIEW={review_path}" in workflow
    assert 'test -s "$ADDITIVE_REVIEW"' in workflow
    assert (
        'ADDITIVE_REVIEW_ARGS+=(--additive-placement-review "$ADDITIVE_REVIEW")'
        in workflow
    )
    assert '"${ADDITIVE_REVIEW_ARGS[@]}" \\' in workflow


def test_upload_workflow_separates_frozen_source_from_current_receipts() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/upload.yml"
    ).read_text(encoding="utf-8")
    source_checkout = workflow.split("- name: Check out collected snapshot", 1)[
        1
    ].split("- name: Check out current upload receipts", 1)[0]
    receipt_checkout = workflow.split("- name: Check out current upload receipts", 1)[
        1
    ].split("- name: Check out reviewed prior upload receipt", 1)[0]
    assert (
        "SNAPSHOT_REF: ${{ github.event_name == 'workflow_dispatch' "
        "&& inputs.snapshot_ref || github.sha }}"
    ) in workflow
    assert "ref: ${{ env.SNAPSHOT_REF }}" in source_checkout
    assert "path: snapshot" in source_checkout
    assert "persist-credentials: false" in source_checkout
    # Physics and Breivik review gates read source-pinned status evidence.
    assert "status\n" in source_checkout
    assert "ref: automation/map-snapshot" in receipt_checkout
    assert "path: receipt" in receipt_checkout
    assert "persist-credentials: true" in receipt_checkout
    assert "status\n" in receipt_checkout
    assert "--status-dir receipt/status" in workflow
    assert "--current-receipt receipt/status/last-upload.json" in workflow
    assert 'MAP_SOURCE_COMMIT="$(git -C snapshot rev-parse HEAD)"' in workflow
    assert "--status-file receipt/status/last-upload.json" in workflow
    assert "working-directory: receipt" in workflow
    assert "snapshot/status/last-upload.json" not in workflow


def test_retry_uses_frozen_source_and_current_failure_receipt(
    monkeypatch, tmp_path
) -> None:
    """A failed run may advance status without invalidating its frozen input."""

    def git(*args):
        return subprocess.run(
            ["git", *map(str, args)], check=True, capture_output=True, text=True
        ).stdout.strip()

    remote = tmp_path / "remote.git"
    initial = tmp_path / "initial"
    snapshot = tmp_path / "snapshot"
    receipt = tmp_path / "receipt"
    git("init", "--bare", remote)
    git("clone", remote, initial)
    git("-C", initial, "checkout", "-b", "automation/map-snapshot")
    (initial / "snapshots").mkdir()
    (initial / "snapshots" / "raw.parquet").write_bytes(b"frozen input")
    (initial / "status").mkdir()
    success = {"status": "success", "artifact_commit": "a" * 40}
    (initial / "status" / "last-upload.json").write_text(json.dumps(success))
    (initial / "status" / "last-successful-upload.json").write_text(json.dumps(success))
    git("-C", initial, "add", ".")
    git(
        "-C",
        initial,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.test",
        "commit",
        "-m",
        "Frozen source and successful release",
    )
    frozen_commit = git("-C", initial, "rev-parse", "HEAD")
    git("-C", initial, "push", "origin", "HEAD:automation/map-snapshot")
    git("clone", remote, snapshot)
    git("-C", snapshot, "checkout", frozen_commit)
    failure = {"status": "failure", "workflow_url": "https://github.test/run/1"}
    (initial / "status" / "last-upload.json").write_text(json.dumps(failure))
    git("-C", initial, "add", "status/last-upload.json")
    git(
        "-C",
        initial,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.test",
        "commit",
        "-m",
        "Record failed upload",
    )
    failed_receipt_commit = git("-C", initial, "rev-parse", "HEAD")
    git("-C", initial, "push", "origin", "HEAD:automation/map-snapshot")
    git("clone", "--branch", "automation/map-snapshot", remote, receipt)

    assert git("-C", snapshot, "rev-parse", "HEAD") == frozen_commit
    assert git("-C", receipt, "rev-parse", "HEAD") == failed_receipt_commit
    assert pin_prior_release(receipt / "status") == "a" * 40
    monkeypatch.setenv("MAP_DURABLE_INTENT_REPO", str(receipt))
    monkeypatch.setenv("MAP_DURABLE_INTENT_BRANCH", "automation/map-snapshot")
    intent = {
        "release_context": {
            "producer_commit": "c" * 40,
            "snapshot_commit": frozen_commit,
        }
    }
    publisher._persist_atomic_intent(
        intent,
        status={"workflow_url": "https://github.test/run/2"},
        status_path=receipt / "status" / "last-upload.json",
    )
    saved = json.loads(
        git(
            "--git-dir",
            remote,
            "show",
            "automation/map-snapshot:status/last-upload.json",
        )
    )
    assert saved["workflow_url"] == "https://github.test/run/2"
    assert saved["atomic_release_intent"] == intent
    assert git("-C", snapshot, "rev-parse", "HEAD") == frozen_commit
    assert (snapshot / "snapshots" / "raw.parquet").read_bytes() == b"frozen input"


def test_atomic_intent_is_pushed_before_hub_write_and_keeps_success_baseline(
    monkeypatch, tmp_path
) -> None:
    remote = tmp_path / "remote.git"
    checkout = tmp_path / "snapshot"

    def git(*args):
        return subprocess.run(
            ["git", *map(str, args)], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--bare", remote)
    git("clone", remote, checkout)
    git("-C", checkout, "checkout", "-b", "automation/map-snapshot")
    status_path = checkout / "status" / "last-upload.json"
    status_path.parent.mkdir()
    status_path.write_text('{"status":"success"}')
    successful_path = checkout / "status" / "last-successful-upload.json"
    successful_path.write_text('{"artifact_commit":"old"}')
    git("-C", checkout, "add", "status")
    git(
        "-C",
        checkout,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.test",
        "commit",
        "-m",
        "Initial receipt",
    )
    git("-C", checkout, "push", "origin", "HEAD:automation/map-snapshot")
    monkeypatch.setenv("MAP_DURABLE_INTENT_REPO", str(checkout))
    monkeypatch.setenv("MAP_DURABLE_INTENT_BRANCH", "automation/map-snapshot")
    status = {"workflow_url": "https://github.test/actions/runs/1"}
    intent = {
        "intent_status": "before_write",
        "commit_nonce": "unique-nonce",
        "parent_revision": "a" * 40,
        "release_context": {
            "producer_commit": "c" * 40,
            "snapshot_commit": "d" * 40,
        },
        "size_by_path": {"README.md": 10},
        "sha256_by_path": {"README.md": "b" * 64},
    }
    publisher._persist_atomic_intent(intent, status=status, status_path=status_path)
    saved = json.loads(
        git(
            "--git-dir",
            remote,
            "show",
            "automation/map-snapshot:status/last-upload.json",
        )
    )
    assert saved["status"] == "failure"
    assert saved["atomic_release_intent"] == intent
    assert json.loads(
        git(
            "--git-dir",
            remote,
            "show",
            "automation/map-snapshot:status/last-successful-upload.json",
        )
    ) == {"artifact_commit": "old"}

    returned = {**intent, "intent_status": "commit_returned_unverified"}
    returned["artifact_revision"] = "c" * 40
    publisher._persist_atomic_intent(returned, status=status, status_path=status_path)
    saved = json.loads(
        git(
            "--git-dir",
            remote,
            "show",
            "automation/map-snapshot:status/last-upload.json",
        )
    )
    assert saved["atomic_release_intent"]["artifact_revision"] == "c" * 40

    # An intervening source push must stop the next checkpoint before any Hub call.
    other = tmp_path / "other"
    git("clone", "--branch", "automation/map-snapshot", remote, other)
    (other / "status" / "unrelated.txt").write_text("advanced")
    git("-C", other, "add", "status/unrelated.txt")
    git(
        "-C",
        other,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.test",
        "commit",
        "-m",
        "Advance snapshot branch",
    )
    git("-C", other, "push", "origin", "HEAD:automation/map-snapshot")
    with pytest.raises(RuntimeError, match="Snapshot branch moved"):
        publisher._persist_atomic_intent(
            {**intent, "intent_status": "before_write", "commit_nonce": "new"},
            status=status,
            status_path=status_path,
        )


@pytest.mark.parametrize(
    ("branch", "status_location", "message"),
    [
        ("wrong-branch", "correct", "requires the snapshot branch"),
        ("automation/map-snapshot", "outside", "outside snapshot checkout"),
        ("automation/map-snapshot", "wrong-path", "must checkpoint"),
    ],
)
def test_atomic_intent_rejects_unreviewed_checkpoint_location(
    monkeypatch, tmp_path, branch, status_location, message
) -> None:
    repo = tmp_path / "snapshot"
    repo.mkdir()
    status_path = {
        "correct": repo / "status" / "last-upload.json",
        "outside": tmp_path / "last-upload.json",
        "wrong-path": repo / "status" / "other.json",
    }[status_location]
    monkeypatch.setenv("MAP_DURABLE_INTENT_REPO", str(repo))
    monkeypatch.setenv("MAP_DURABLE_INTENT_BRANCH", branch)
    with pytest.raises(ValueError, match=message):
        publisher._persist_atomic_intent(
            {
                "parent_revision": "a" * 40,
                "release_context": {
                    "producer_commit": "c" * 40,
                    "snapshot_commit": "d" * 40,
                },
            },
            status={},
            status_path=status_path,
        )


@pytest.mark.parametrize("context", ["invalid", {"producer_commit": "c" * 40}])
def test_atomic_intent_requires_producer_and_source_ids(
    monkeypatch, tmp_path, context
) -> None:
    repo = tmp_path / "snapshot"
    repo.mkdir()
    monkeypatch.setenv("MAP_DURABLE_INTENT_REPO", str(repo))
    monkeypatch.setenv("MAP_DURABLE_INTENT_BRANCH", "automation/map-snapshot")
    with pytest.raises(ValueError):
        publisher._persist_atomic_intent(
            {"release_context": context},
            status={},
            status_path=repo / "status" / "last-upload.json",
        )


@pytest.mark.parametrize("prior_revision", [None, "main", "a" * 39])
def test_unpinned_publish_cannot_reach_any_hub_write(
    monkeypatch, tmp_path, prior_revision
) -> None:
    monkeypatch.setattr(
        publisher,
        "_to_hub_dataset",
        lambda *args: pytest.fail("unversioned dataset write"),
    )
    monkeypatch.setattr(
        publisher,
        "_publish_atomic_release",
        lambda *args, **kwargs: pytest.fail("unversioned atomic write"),
    )
    with pytest.raises(ValueError, match="revision"):
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
            prior_revision=prior_revision,
        )


def test_pinned_publish_without_intent_sink_cannot_reach_hub(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        publisher,
        "_require_current_dataset_revision",
        lambda **kwargs: pytest.fail("Hub revision probe reached"),
    )
    monkeypatch.setattr(
        publisher,
        "_publish_atomic_release",
        lambda *args, **kwargs: pytest.fail("Hub write reached"),
    )
    with pytest.raises(ValueError, match="durable prewrite intent sink"):
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
            prior_revision="a" * 40,
        )


def _stage(monkeypatch, tmp_path, *, bio=False, gate_status="pass", pending=False):
    readme = tmp_path / "parent-README.md"
    readme.write_text("---\nlicense: mit\n---\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda *args, **kwargs: str(readme)),
    )
    calls = []

    def map_files(works, registry, *, maps_dir, **kwargs):
        maps_dir.mkdir(parents=True)
        artifact = {"point_count": 2, "points": []}
        artifact_path = maps_dir / "publications.json"
        manifest_path = maps_dir / "manifest.json"
        artifact_path.write_text(json.dumps(artifact))
        manifest_path.write_text(json.dumps(kwargs))
        calls.append(("map", kwargs))
        return artifact_path, manifest_path, artifact

    class Dataset:
        def __init__(self, frame):
            self.frame = frame

        def to_parquet(self, path):
            self.frame.to_parquet(path, index=False)

    def commit(files, **kwargs):
        assert set(files) == RELEASE_PATHS
        assert all(path.is_file() and path.stat().st_size for path in files.values())
        assert (
            json.loads(files["maps/manifest.json"].read_text())["dataset_commits"] == {}
        )
        calls.append(("commit", kwargs))
        return "b" * 40, {"mode": "single_parent_commit"}

    monkeypatch.setattr(publisher, "_write_map_artifacts", map_files)
    monkeypatch.setattr(publisher, "_to_hub_dataset", Dataset)
    monkeypatch.setattr(publisher, "build_updated_dataset_card", lambda *args: "card")
    monkeypatch.setattr(publisher, "commit_atomic_release", commit)
    manifest = tmp_path / "repair.json"
    if pending:
        manifest.write_text(json.dumps({"restore_eligible_work_ids": ["work"]}))
    if bio:
        monkeypatch.setattr(
            publisher,
            "load_bio_baselines",
            lambda **kwargs: {
                "repair_works": [
                    {
                        "work_id": "work",
                        "keyword_id": "keyword-unclassified" if pending else "other",
                        "detail_keyword_id": (
                            "keyword-unclassified-pending-review"
                            if pending
                            else "other"
                        ),
                        "map_eligible": True,
                    }
                ]
            },
        )
        monkeypatch.setattr(
            publisher,
            "preserve_pending_review_topics",
            lambda works, baseline, review: calls.append(("preserve", review)) or works,
        )
        monkeypatch.setattr(
            publisher,
            "validate_bio_release_prewrite",
            lambda **kwargs: calls.append(("gate", kwargs)) or {"status": gate_status},
        )
    tables = {
        name: pandas.DataFrame({"value": [name]}) for name in publisher.DATASET_CONFIGS
    }
    return calls, tables, manifest


def test_direct_atomic_stage_without_intent_sink_stops_before_hub(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="durable prewrite intent sink"):
        publisher._publish_atomic_release(
            tables,
            _bio_registry(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={"snapshot_sha256": "other"},
            hf_token="token",
            prior_revision="a" * 40,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={"prior_revision": "a" * 40},
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
        )
    assert calls == []


def test_material_presentation_drift_stops_before_atomic_commit(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        publisher,
        "_pinned_parent_map",
        lambda **kwargs: ({"points": []}, "parent-sha"),
    )
    monkeypatch.setattr(
        publisher,
        "compare_presentation",
        lambda *args, **kwargs: {
            "material_reasons": ["broad_semantic_topic_churn"],
            "report_sha256": "report-sha",
        },
    )
    with pytest.raises(PresentationDriftError, match="lacks a reviewed approval"):
        publisher._publish_atomic_release(
            tables,
            _bio_registry(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={"snapshot_sha256": "other"},
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={"prior_revision": "a" * 40},
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
            enforce_presentation_drift=True,
        )
    assert [event for event, _ in calls] == ["map"]


def test_pinned_parent_map_rejects_invalid_payload(monkeypatch, tmp_path) -> None:
    path = tmp_path / "parent-map.json"
    path.write_text("[]", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda *args, **kwargs: str(path)),
    )
    with pytest.raises(ValueError, match="not an object"):
        publisher._pinned_parent_map(prior_revision="a" * 40, hf_token="token")
    path.write_text('{"points": []}', encoding="utf-8")
    parent, digest = publisher._pinned_parent_map(
        prior_revision="a" * 40, hf_token="token"
    )
    assert parent == {"points": []}
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_reviewed_presentation_gate_is_in_manifest_and_atomic_intent(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        publisher,
        "_pinned_parent_map",
        lambda **kwargs: ({"points": []}, "parent-sha"),
    )
    monkeypatch.setattr(
        publisher,
        "compare_presentation",
        lambda *args, **kwargs: {"material_reasons": [], "report_sha256": "digest"},
    )
    publisher._publish_atomic_release(
        tables,
        _bio_registry(),
        source_frame=pandas.DataFrame({"source": [1]}),
        source_manifest={"snapshot_sha256": "other"},
        hf_token="token",
        prior_revision="a" * 40,
        atomic_intent_sink=lambda intent: None,
        generated_at_utc="2026-10-25T00:00:00+00:00",
        continuity_receipt={"prior_revision": "a" * 40},
        repair_manifest_path=manifest,
        provisional_holds={},
        prior_profiles=[],
        prior_people=_prior_bio_people(),
        enforce_presentation_drift=True,
    )
    assert [event for event, _ in calls] == ["map", "commit"]
    assert calls[-1][1]["prewrite_gate"]["presentation_drift"]["status"] == (
        "pass_no_material_drift"
    )


def test_deletion_only_preservation_is_reported_before_atomic_commit(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        publisher,
        "_pinned_parent_map",
        lambda **kwargs: ({"points": []}, "parent-sha"),
    )
    monkeypatch.setattr(
        publisher,
        "build_map_artifact",
        lambda *args, **kwargs: {"points": []},
    )
    receipt = {"status": "parent_presentation_preserved_for_deletions"}
    monkeypatch.setattr(
        publisher,
        "preserve_deletion_only_presentation",
        lambda works, prior, parent, candidate: (works, receipt),
    )
    monkeypatch.setattr(
        publisher,
        "compare_presentation",
        lambda *args, **kwargs: {"material_reasons": [], "report_sha256": "digest"},
    )
    publisher._publish_atomic_release(
        tables,
        _bio_registry(),
        source_frame=pandas.DataFrame({"source": [1]}),
        source_manifest={"snapshot_sha256": "other"},
        hf_token="token",
        prior_revision="a" * 40,
        atomic_intent_sink=lambda intent: None,
        generated_at_utc="2026-10-25T00:00:00+00:00",
        continuity_receipt={"prior_revision": "a" * 40},
        repair_manifest_path=manifest,
        provisional_holds={},
        prior_profiles=[],
        prior_people=_prior_bio_people(),
        prior_works=[],
        enforce_presentation_drift=True,
        preserve_deletion_presentation=True,
    )
    assert [event for event, _ in calls] == ["map", "commit"]
    assert calls[-1][1]["prewrite_gate"]["presentation_preservation"] == receipt


def test_reviewed_additive_path_binds_selected_ids_and_stages_receipt(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path)
    tables["profile_publications"] = pandas.DataFrame(
        {"author_pub_id": ["owner:row-1"], "value": ["profile"]}
    )
    parent = {"points": []}
    monkeypatch.setattr(
        publisher, "_pinned_parent_map", lambda **kwargs: (parent, "parent-sha")
    )
    candidate = {"points": [{"work_id": "new"}]}
    monkeypatch.setattr(
        publisher, "build_map_artifact", lambda *args, **kwargs: candidate
    )
    received = {}
    receipt = {"status": "parent_frozen_new_points_reviewed"}

    def freeze(works, prior, pinned, proposal, **kwargs):
        received.update(kwargs)
        assert pinned is parent and proposal is candidate and prior == []
        return works, receipt

    monkeypatch.setattr(publisher, "preserve_reviewed_additive_presentation", freeze)
    monkeypatch.setattr(
        publisher,
        "require_preserved_additive_artifact",
        lambda pinned, staged, item: (
            None
            if pinned is parent and item is receipt
            else pytest.fail("Additive staged-map gate received the wrong inputs")
        ),
    )
    monkeypatch.setattr(
        publisher,
        "compare_presentation",
        lambda *args, **kwargs: {"material_reasons": [], "report_sha256": "digest"},
    )
    review_path = tmp_path / "placement.json"
    review_path.write_text("{}", encoding="utf-8")
    publisher._publish_atomic_release(
        tables,
        _bio_registry(),
        source_frame=pandas.DataFrame({"source": [1]}),
        source_manifest={"snapshot_sha256": "source-sha"},
        hf_token="token",
        prior_revision="a" * 40,
        atomic_intent_sink=lambda intent: None,
        generated_at_utc="2026-10-30T00:00:00+00:00",
        continuity_receipt={"prior_revision": "a" * 40},
        repair_manifest_path=manifest,
        provisional_holds={},
        prior_profiles=[],
        prior_people=_prior_bio_people(),
        prior_works=[],
        enforce_presentation_drift=True,
        preserve_deletion_presentation=True,
        additive_placement_review_path=review_path,
    )
    assert received == {
        "prior_revision": "a" * 40,
        "source_snapshot_sha256": "source-sha",
        "selected_ids_sha256": publisher.ids_sha256({"owner:row-1"}),
        "parent_map_sha256": "parent-sha",
        "review_path": review_path,
    }
    assert [event for event, _ in calls] == ["map", "commit"]
    assert calls[-1][1]["prewrite_gate"]["presentation_preservation"] == receipt


@pytest.mark.parametrize("bad_ids", [None, ["same", "same"], [None], [""]])
def test_additive_selected_id_boundary_stops_before_stage(
    monkeypatch, tmp_path, bad_ids
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path)
    if bad_ids is not None:
        tables["profile_publications"] = pandas.DataFrame({"author_pub_id": bad_ids})
    monkeypatch.setattr(
        publisher,
        "_pinned_parent_map",
        lambda **kwargs: ({"points": []}, "parent-sha"),
    )
    with pytest.raises(ValueError, match="selected source IDs|duplicate source IDs"):
        publisher._publish_atomic_release(
            tables,
            _bio_registry(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={"snapshot_sha256": "source-sha"},
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
            generated_at_utc="2026-10-30T00:00:00+00:00",
            continuity_receipt={"prior_revision": "a" * 40},
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
            prior_works=[],
            enforce_presentation_drift=True,
            additive_placement_review_path=tmp_path / "placement.json",
        )
    assert calls == []


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("disabled", "requires pinned presentation review"),
        ("bio", "restricted to ordinary releases"),
        ("refit", "refit approvals cannot be combined"),
        ("no_parent", "needs pinned parent works"),
    ],
)
def test_additive_incompatible_release_modes_stop_before_stage(
    monkeypatch, tmp_path, mode: str, message: str
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        publisher,
        "_pinned_parent_map",
        lambda **kwargs: ({"points": []}, "parent-sha"),
    )
    with pytest.raises(ValueError, match=message):
        publisher._publish_atomic_release(
            tables,
            _bio_registry(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={
                "snapshot_sha256": publisher.BIO_SOURCE_SHA256
                if mode == "bio"
                else "source-sha"
            },
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
            generated_at_utc="2026-10-30T00:00:00+00:00",
            continuity_receipt={"prior_revision": "a" * 40},
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
            prior_works=None if mode == "no_parent" else [],
            enforce_presentation_drift=mode != "disabled",
            presentation_drift_approval_path=tmp_path / "refit.json"
            if mode == "refit"
            else None,
            additive_placement_review_path=tmp_path / "placement.json",
        )
    assert calls == []


def test_additive_rejects_historical_comparison_before_source_reads(tmp_path) -> None:
    with pytest.raises(ValueError, match="exact parent comparison"):
        publisher.publish_snapshot(
            tmp_path / "missing.parquet",
            tmp_path / "missing.json",
            hf_token="token",
            prior_revision="a" * 40,
            region_comparison_revision="b" * 40,
            atomic_intent_sink=lambda intent: None,
            additive_placement_review_path=tmp_path / "placement.json",
        )


def test_deletion_preservation_missing_parent_works_stops_before_stage(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        publisher,
        "_pinned_parent_map",
        lambda **kwargs: ({"points": []}, "parent-sha"),
    )
    with pytest.raises(ValueError, match="needs pinned parent works"):
        publisher._publish_atomic_release(
            tables,
            _bio_registry(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={"snapshot_sha256": "other"},
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={"prior_revision": "a" * 40},
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
            enforce_presentation_drift=True,
            preserve_deletion_presentation=True,
        )
    assert calls == []


def test_presentation_gate_rejects_invalid_staged_manifest_before_hub(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        publisher,
        "_pinned_parent_map",
        lambda **kwargs: ({"points": []}, "parent-sha"),
    )
    monkeypatch.setattr(
        publisher,
        "compare_presentation",
        lambda *args, **kwargs: {"material_reasons": []},
    )
    original_load = publisher.load_json
    monkeypatch.setattr(
        publisher,
        "load_json",
        lambda path: [] if path.name == "manifest.json" else original_load(path),
    )
    with pytest.raises(ValueError, match="Staged map manifest is not an object"):
        publisher._publish_atomic_release(
            tables,
            _bio_registry(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={"snapshot_sha256": "other"},
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={"prior_revision": "a" * 40},
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
            enforce_presentation_drift=True,
        )
    assert [event for event, _ in calls] == ["map"]


@pytest.mark.parametrize("bio", [False, True])
def test_atomic_publisher_stages_all_files_and_runs_gate_before_commit(
    monkeypatch, tmp_path, bio
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path, bio=bio)
    revision, receipt = publisher._publish_atomic_release(
        tables,
        _bio_registry(),
        source_frame=pandas.DataFrame({"source": [1]}),
        source_manifest={
            "snapshot_sha256": publisher.BIO_SOURCE_SHA256 if bio else "other"
        },
        hf_token="token",
        prior_revision="a" * 40,
        atomic_intent_sink=lambda intent: None,
        generated_at_utc="2026-10-25T00:00:00+00:00",
        continuity_receipt={"prior_revision": "a" * 40},
        repair_manifest_path=manifest,
        provisional_holds={},
        prior_profiles=[],
        prior_people=_prior_bio_people(),
    )
    assert revision == "b" * 40
    assert receipt["mode"] == "single_parent_commit"
    assert calls[-1][0] == "commit"
    assert calls[-1][1]["prewrite_gate"]["status"] == "pass"
    if bio:
        assert [call[0] for call in calls] == ["preserve", "map", "gate", "commit"]
    else:
        assert [call[0] for call in calls] == ["map", "commit"]


def test_atomic_bio_staging_uses_immutable_parent_pending_labels(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path, bio=True, pending=True)
    manifest.unlink()
    revision, _ = publisher._publish_atomic_release(
        tables,
        _bio_registry(),
        source_frame=pandas.DataFrame({"source": [1]}),
        source_manifest={"snapshot_sha256": publisher.BIO_SOURCE_SHA256},
        hf_token="token",
        prior_revision="a" * 40,
        atomic_intent_sink=lambda intent: None,
        generated_at_utc="2026-10-25T00:00:00+00:00",
        continuity_receipt={"prior_revision": "a" * 40},
        repair_manifest_path=manifest,
        provisional_holds={},
        prior_profiles=[],
        prior_people=_prior_bio_people(),
    )
    assert revision == "b" * 40
    assert calls[0] == ("preserve", {"restore_eligible_work_ids": ["work"]})


def test_atomic_bio_staging_ignores_historical_pending_manifest(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path, bio=True, pending=True)
    manifest.write_text(json.dumps({"restore_eligible_work_ids": ["other"]}))
    revision, _ = publisher._publish_atomic_release(
        tables,
        _bio_registry(),
        source_frame=pandas.DataFrame({"source": [1]}),
        source_manifest={"snapshot_sha256": publisher.BIO_SOURCE_SHA256},
        hf_token="token",
        prior_revision="a" * 40,
        atomic_intent_sink=lambda intent: None,
        generated_at_utc="2026-10-25T00:00:00+00:00",
        continuity_receipt={"prior_revision": "a" * 40},
        repair_manifest_path=manifest,
        provisional_holds={},
        prior_profiles=[],
        prior_people=_prior_bio_people(),
    )
    assert revision == "b" * 40
    assert calls[0] == ("preserve", {"restore_eligible_work_ids": ["work"]})


def test_atomic_generic_staging_rejects_unclean_region_gate_before_commit(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="clean region-continuity"):
        publisher._publish_atomic_release(
            tables,
            _bio_registry(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={"snapshot_sha256": "other"},
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={
                "prior_revision": "a" * 40,
                "unexplained_gains": ["bad"],
            },
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
        )
    assert all(call[0] != "commit" for call in calls)


@pytest.mark.parametrize(
    ("require_bio_release", "bio_source_changed"),
    [(True, False), (False, True)],
)
def test_bio_intent_or_changed_cohort_rejects_drift_before_hub_stage(
    monkeypatch, tmp_path, require_bio_release, bio_source_changed
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path, bio=False)
    memberships = (
        [SimpleNamespace(person_id="bio-person", included=True, department_id="bio")]
        if bio_source_changed
        else []
    )
    source = pandas.DataFrame(
        [
            {
                "person_id": "bio-person",
                "scholar_id": "bio-scholar",
                "author_pub_id": "bio-scholar:one",
                "source_type": "scholar",
                "source_record_json": '{"title":"one"}',
            }
        ]
    )
    with pytest.raises(ValueError, match="Bio release source differs"):
        publisher._publish_atomic_release(
            tables,
            SimpleNamespace(memberships=memberships),
            source_frame=source,
            source_manifest={"snapshot_sha256": "drifted-source"},
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={"prior_revision": "a" * 40},
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
            require_bio_release=require_bio_release,
        )
    assert calls == []


@pytest.mark.parametrize(
    (
        "source_hash",
        "prior_revision",
        "atomic_release",
        "require_bio_release",
        "message",
    ),
    [
        ("changed", "a" * 40, None, True, "Bio release source differs"),
        (
            publisher.BIO_SOURCE_SHA256,
            None,
            None,
            True,
            "immutable prior Hub revision",
        ),
        (publisher.BIO_SOURCE_SHA256, "a" * 40, False, True, "one atomic Hub commit"),
        (publisher.BIO_SOURCE_SHA256, "a" * 40, False, False, "one atomic Hub commit"),
    ],
)
def test_publish_bio_intent_fails_before_enrichment_or_hub_write(
    monkeypatch,
    tmp_path,
    source_hash,
    prior_revision,
    atomic_release,
    require_bio_release,
    message,
) -> None:
    calls = []
    monkeypatch.setattr(
        publisher, "_require_current_dataset_revision", lambda **kwargs: None
    )
    monkeypatch.setattr(publisher, "load_registry", lambda *args: SimpleNamespace())
    monkeypatch.setattr(
        publisher, "implicit_primary_baseline_for_people", lambda *args: {}
    )
    monkeypatch.setattr(
        publisher,
        "validate_included_scholar_primary_homes",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (
            pandas.DataFrame({"source": [1]}),
            {"snapshot_sha256": source_hash},
        ),
    )
    monkeypatch.setattr(
        publisher, "add_embeddings", lambda *args, **kwargs: calls.append("enrich")
    )
    monkeypatch.setattr(
        publisher, "commit_atomic_release", lambda *args, **kwargs: calls.append("hub")
    )
    with pytest.raises(ValueError, match=message):
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
            prior_revision=prior_revision,
            atomic_release=atomic_release,
            atomic_intent_sink=lambda intent: None,
            require_bio_release=require_bio_release,
        )
    assert calls == []
