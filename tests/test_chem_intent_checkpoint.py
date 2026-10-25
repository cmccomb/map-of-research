"""The Chemistry Hub write requires a remotely retained Git intent first."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from map_of_research import chem_delta_release as delta
from map_of_research import registry as registry_module
from map_of_research import snapshot as snapshot_module
from map_of_research.atomic_release import (
    RELEASE_PATHS,
    AtomicReleaseReconciliationRequired,
    commit_atomic_release,
)

PARENT = "a" * 40
PRODUCER = "b" * 40
REVISION = "c" * 40


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _snapshot_repo(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    repo = tmp_path / "checkout"
    remote.mkdir()
    repo.mkdir()
    _git(remote, "init", "--bare")
    _git(repo, "init", "--initial-branch", delta.INTENT_BRANCH)
    _git(repo, "remote", "add", "origin", str(remote))
    status = repo / "status"
    status.mkdir()
    receipt = {"status": "success", "artifact_commit": PARENT}
    for name in ("last-upload.json", "last-successful-upload.json"):
        (status / name).write_text(json.dumps(receipt) + "\n")
    _git(repo, "add", "status")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-m",
        "Prior release",
    )
    _git(repo, "push", "origin", f"HEAD:{delta.INTENT_BRANCH}")
    return repo, remote


def _checkpoint(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> delta.RetainedGitIntent:
    # The production wrapper rejects this local test remote. The checkpoint
    # itself still performs a real push and remote blob readback.
    monkeypatch.setattr(delta, "_require_intent_remote", lambda _repo: None)
    return delta.RetainedGitIntent(
        intent_path=tmp_path / "local-intent.json",
        snapshot_checkout=repo,
        parent_revision=PARENT,
        producer_commit=PRODUCER,
    )


def _files(tmp_path: Path) -> dict[str, Path]:
    files = {}
    for name in RELEASE_PATHS:
        path = tmp_path / "stage" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = path
    return files


def _hub(monkeypatch: pytest.MonkeyPatch, remote: Path) -> list[str]:
    calls: list[str] = []

    class HfApi:
        def __init__(self, token: str) -> None:
            assert token == "token"

        def repo_info(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(sha=PARENT)

        def list_repo_files(self, *_args: object, **_kwargs: object) -> list[str]:
            return sorted(RELEASE_PATHS)

        def create_commit(self, **_kwargs: object) -> None:
            calls.append("create_commit")
            retained = json.loads(
                _git(remote, "show", f"{delta.INTENT_BRANCH}:status/last-upload.json")
            )
            assert retained["atomic_release_intent"]["parent_revision"] == PARENT
            assert len(retained["atomic_release_intent"]["commit_nonce"]) == 32
            assert (
                set(retained["atomic_release_intent"]["sha256_by_path"])
                == RELEASE_PATHS
            )
            assert retained["status"] == "failure"
            raise RuntimeError("first Hub write intercepted")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=HfApi,
            CommitOperationAdd=lambda **kwargs: SimpleNamespace(**kwargs),
            hf_hub_download=lambda *_args, **_kwargs: None,
        ),
    )
    return calls


def _attempt(files: dict[str, Path], checkpoint: delta.RetainedGitIntent) -> None:
    commit_atomic_release(
        files,
        prewrite_gate={"status": "pass"},
        repo_id=delta.REPO_ID,
        hf_token="token",
        prior_revision=PARENT,
        commit_message="Chemistry test release",
        intent_sink=checkpoint,
        intent_metadata={
            "producer_commit": PRODUCER,
            "snapshot_commit": delta.SOURCE_COMMIT,
        },
    )


def test_canonical_remote_is_required_before_any_intent(tmp_path: Path) -> None:
    repo, _ = _snapshot_repo(tmp_path)
    with pytest.raises(ValueError, match="canonical GitHub"):
        delta.RetainedGitIntent(
            intent_path=tmp_path / "intent.json",
            snapshot_checkout=repo,
            parent_revision=PARENT,
            producer_commit=PRODUCER,
        )


def test_missing_prior_status_aborts_before_hub_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = _snapshot_repo(tmp_path)
    monkeypatch.setattr(delta, "_require_intent_remote", lambda _repo: None)
    (repo / "status/last-successful-upload.json").unlink()
    with pytest.raises(ValueError, match="both prior upload receipts"):
        delta.RetainedGitIntent(
            intent_path=tmp_path / "intent.json",
            snapshot_checkout=repo,
            parent_revision=PARENT,
            producer_commit=PRODUCER,
        )


def test_remote_intent_is_read_back_before_first_hub_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, remote = _snapshot_repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, repo, monkeypatch)
    calls = _hub(monkeypatch, remote)
    with pytest.raises(AtomicReleaseReconciliationRequired, match="create_commit"):
        _attempt(_files(tmp_path), checkpoint)
    assert calls == ["create_commit"]
    assert (tmp_path / "local-intent.json").is_file()
    assert (
        json.loads((repo / "status/last-successful-upload.json").read_text())[
            "artifact_commit"
        ]
        == PARENT
    )
    with pytest.raises(ValueError, match="awaits exact reviewed reconciliation"):
        delta.RetainedGitIntent(
            intent_path=tmp_path / "next-intent.json",
            snapshot_checkout=repo,
            parent_revision=PARENT,
            producer_commit=PRODUCER,
        )


def test_stale_checkout_aborts_before_hub_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, remote = _snapshot_repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, repo, monkeypatch)
    other = tmp_path / "other"
    _git(tmp_path, "clone", str(remote), str(other))
    _git(other, "checkout", delta.INTENT_BRANCH)
    (other / "status/other.txt").write_text("advance branch\n")
    _git(other, "add", "status/other.txt")
    _git(
        other,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-m",
        "Advance",
    )
    _git(other, "push", "origin", f"HEAD:{delta.INTENT_BRANCH}")
    calls = _hub(monkeypatch, remote)
    with pytest.raises(RuntimeError, match="branch moved"):
        _attempt(_files(tmp_path), checkpoint)
    assert calls == []


def test_rejected_push_aborts_before_hub_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, remote = _snapshot_repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, repo, monkeypatch)
    original = delta._git

    def reject_push(path: Path, *args: str) -> str:
        if args and args[0] == "push":
            raise subprocess.CalledProcessError(1, ["git", "push"])
        return original(path, *args)

    monkeypatch.setattr(delta, "_git", reject_push)
    calls = _hub(monkeypatch, remote)
    with pytest.raises(subprocess.CalledProcessError):
        _attempt(_files(tmp_path), checkpoint)
    assert calls == []
    assert (
        json.loads(
            _git(remote, "show", f"{delta.INTENT_BRANCH}:status/last-upload.json")
        )["status"]
        == "success"
    )


def test_changed_remote_readback_aborts_before_hub_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, remote = _snapshot_repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, repo, monkeypatch)
    original_run = subprocess.run

    def changed_readback(command: list[str], **kwargs: object) -> object:
        if len(command) >= 4 and command[3] == "show":
            return SimpleNamespace(stdout=b"different remote status")
        return original_run(command, **kwargs)

    monkeypatch.setattr(delta.subprocess, "run", changed_readback)
    calls = _hub(monkeypatch, remote)
    with pytest.raises(RuntimeError, match="remote intent readback differs"):
        _attempt(_files(tmp_path), checkpoint)
    assert calls == []


def test_status_mutation_after_local_readback_aborts_before_hub_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, remote = _snapshot_repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, repo, monkeypatch)
    original = delta.load_json

    def mutate_after_readback(path: Path) -> object:
        loaded = original(path)
        if path == checkpoint.status_path and loaded.get("atomic_release_intent"):
            path.write_text('{"status":"changed after local readback"}\n')
        return loaded

    monkeypatch.setattr(delta, "load_json", mutate_after_readback)
    calls = _hub(monkeypatch, remote)
    with pytest.raises(RuntimeError, match="status changed before remote checkpoint"):
        _attempt(_files(tmp_path), checkpoint)
    assert calls == []


def test_prior_receipt_must_match_live_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = _snapshot_repo(tmp_path)
    monkeypatch.setattr(delta, "_require_intent_remote", lambda _repo: None)
    with pytest.raises(ValueError, match="successful receipt differs"):
        delta.RetainedGitIntent(
            intent_path=tmp_path / "intent.json",
            snapshot_checkout=repo,
            parent_revision=REVISION,
            producer_commit=PRODUCER,
        )


def test_success_finalization_advances_remote_receipt_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "cmccomb/map-of-research")
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    repo, remote = _snapshot_repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, repo, monkeypatch)
    intent = {
        "parent_revision": PARENT,
        "commit_nonce": "d" * 32,
        "sha256_by_path": dict.fromkeys(RELEASE_PATHS, "e" * 64),
        "size_by_path": dict.fromkeys(RELEASE_PATHS, 1),
        "release_context": {
            "producer_commit": PRODUCER,
            "snapshot_commit": delta.SOURCE_COMMIT,
        },
    }
    checkpoint(intent)
    assert (
        json.loads((repo / "status/last-successful-upload.json").read_text())[
            "artifact_commit"
        ]
        == PARENT
    )
    receipt = {
        **intent,
        "artifact_revision": REVISION,
        "hosted_file_bytes_match_prewrite": True,
    }
    checkpoint.finalize_success(REVISION, receipt)
    retained = json.loads(
        _git(remote, "show", f"{delta.INTENT_BRANCH}:status/last-upload.json")
    )
    assert retained["status"] == "success"
    assert retained["artifact_commit"] == REVISION
    assert (
        retained["workflow_url"]
        == "https://github.com/cmccomb/map-of-research/actions/runs/12345"
    )
    assert retained["started_at_utc"] and retained["finished_at_utc"]
    assert retained["dataset_configs"] == list(delta.PARENT_TABLE_ROWS)
    assert retained == json.loads(
        _git(
            remote, "show", f"{delta.INTENT_BRANCH}:status/last-successful-upload.json"
        )
    )
    assert delta.pin_prior_release(repo / "status") == REVISION


def test_failed_success_checkpoint_keeps_remote_prior_success_and_pending_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, remote = _snapshot_repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, repo, monkeypatch)
    intent = {
        "parent_revision": PARENT,
        "commit_nonce": "d" * 32,
        "sha256_by_path": dict.fromkeys(RELEASE_PATHS, "e" * 64),
        "size_by_path": dict.fromkeys(RELEASE_PATHS, 1),
        "release_context": {
            "producer_commit": PRODUCER,
            "snapshot_commit": delta.SOURCE_COMMIT,
        },
    }
    checkpoint(intent)
    original = delta._git

    def reject_push(path: Path, *args: str) -> str:
        if args and args[0] == "push":
            raise subprocess.CalledProcessError(1, ["git", "push"])
        return original(path, *args)

    monkeypatch.setattr(delta, "_git", reject_push)
    receipt = {
        **intent,
        "artifact_revision": REVISION,
        "hosted_file_bytes_match_prewrite": True,
    }
    with pytest.raises(subprocess.CalledProcessError):
        checkpoint.finalize_success(REVISION, receipt)
    pending = json.loads(
        _git(remote, "show", f"{delta.INTENT_BRANCH}:status/last-upload.json")
    )
    successful = json.loads(
        _git(
            remote, "show", f"{delta.INTENT_BRANCH}:status/last-successful-upload.json"
        )
    )
    assert pending["status"] == "failure"
    assert pending["atomic_release_intent"]["commit_nonce"] == "d" * 32
    assert successful["artifact_commit"] == PARENT


def test_success_receipt_must_match_retained_nonce_before_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, remote = _snapshot_repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, repo, monkeypatch)
    intent = {
        "parent_revision": PARENT,
        "commit_nonce": "d" * 32,
        "sha256_by_path": dict.fromkeys(RELEASE_PATHS, "e" * 64),
        "size_by_path": dict.fromkeys(RELEASE_PATHS, 1),
        "release_context": {
            "producer_commit": PRODUCER,
            "snapshot_commit": delta.SOURCE_COMMIT,
        },
    }
    checkpoint(intent)
    with pytest.raises(ValueError, match="differs from retained intent"):
        checkpoint.finalize_success(
            REVISION,
            {
                **intent,
                "commit_nonce": "f" * 32,
                "artifact_revision": REVISION,
                "hosted_file_bytes_match_prewrite": True,
            },
        )
    assert (
        json.loads(
            _git(
                remote,
                "show",
                f"{delta.INTENT_BRANCH}:status/last-successful-upload.json",
            )
        )["artifact_commit"]
        == PARENT
    )


def test_success_receipt_must_match_retained_sizes_before_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, remote = _snapshot_repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, repo, monkeypatch)
    intent = {
        "parent_revision": PARENT,
        "commit_nonce": "d" * 32,
        "sha256_by_path": dict.fromkeys(RELEASE_PATHS, "e" * 64),
        "size_by_path": dict.fromkeys(RELEASE_PATHS, 1),
        "release_context": {
            "producer_commit": PRODUCER,
            "snapshot_commit": delta.SOURCE_COMMIT,
        },
    }
    checkpoint(intent)
    with pytest.raises(ValueError, match="differs from retained intent"):
        checkpoint.finalize_success(
            REVISION,
            {
                **intent,
                "size_by_path": dict.fromkeys(RELEASE_PATHS, 2),
                "artifact_revision": REVISION,
                "hosted_file_bytes_match_prewrite": True,
            },
        )
    assert (
        json.loads(
            _git(remote, "show", f"{delta.INTENT_BRANCH}:status/last-upload.json")
        )["status"]
        == "failure"
    )


def test_finalizer_rejects_changed_checkpoint_tip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, remote = _snapshot_repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, repo, monkeypatch)
    intent = {
        "parent_revision": PARENT,
        "commit_nonce": "d" * 32,
        "sha256_by_path": dict.fromkeys(RELEASE_PATHS, "e" * 64),
        "size_by_path": dict.fromkeys(RELEASE_PATHS, 1),
        "release_context": {
            "producer_commit": PRODUCER,
            "snapshot_commit": delta.SOURCE_COMMIT,
        },
    }
    checkpoint(intent)
    (repo / "status/other.txt").write_text("concurrent status update\n")
    _git(repo, "add", "status/other.txt")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-m",
        "Concurrent update",
    )
    _git(repo, "push", "origin", f"HEAD:{delta.INTENT_BRANCH}")
    with pytest.raises(RuntimeError, match="branch moved"):
        checkpoint.finalize_success(
            REVISION,
            {
                **intent,
                "artifact_revision": REVISION,
                "hosted_file_bytes_match_prewrite": True,
            },
        )
    assert (
        json.loads(
            _git(remote, "show", f"{delta.INTENT_BRANCH}:status/last-upload.json")
        )["status"]
        == "failure"
    )


def test_publish_entry_point_requires_retained_store_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(delta, "_require_producer_checkout", lambda *_args: None)
    monkeypatch.setattr(
        delta,
        "require_live_337_parent",
        lambda *_args, **_kwargs: delta.ExactParent(PARENT, {}),
    )

    def reject_store(*_args: object, **_kwargs: object) -> None:
        calls.append("store")
        raise RuntimeError("off-runner intent unavailable")

    monkeypatch.setattr(delta, "retained_git_intent_sink", reject_store)
    monkeypatch.setattr(
        delta,
        "stage_chem_delta",
        lambda *_args, **_kwargs: calls.append("stage"),
    )
    monkeypatch.setattr(
        delta,
        "commit_atomic_release",
        lambda *_args, **_kwargs: calls.append("commit"),
    )
    with pytest.raises(RuntimeError, match="off-runner intent unavailable"):
        delta.publish_chem_delta(
            expected_parent_revision=PARENT,
            hf_token="token",
            snapshot_path=tmp_path / "snapshot.parquet",
            snapshot_manifest_path=tmp_path / "snapshot.manifest.json",
            source_cache_dir=tmp_path / "authors",
            people_path=tmp_path / "people.csv",
            memberships_path=tmp_path / "memberships.csv",
            departments_path=tmp_path / "departments.csv",
            chem_review_path=tmp_path / "review.json",
            stage_dir=tmp_path / "stage",
            intent_path=tmp_path / "intent.json",
            snapshot_checkout=tmp_path / "snapshot-checkout",
            producer_commit=PRODUCER,
        )
    assert calls == ["store"]


@pytest.mark.parametrize("token", ["", "  "])
def test_direct_publish_requires_explicit_token_before_hub_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token: str
) -> None:
    monkeypatch.setattr(
        delta,
        "require_live_337_parent",
        lambda *_args, **_kwargs: pytest.fail("Hub read reached without token"),
    )
    with pytest.raises(ValueError, match="explicit HF_TOKEN"):
        delta.publish_chem_delta(
            expected_parent_revision=PARENT,
            hf_token=token,
            snapshot_path=tmp_path / "snapshot.parquet",
            snapshot_manifest_path=tmp_path / "snapshot.manifest.json",
            source_cache_dir=tmp_path / "authors",
            people_path=tmp_path / "people.csv",
            memberships_path=tmp_path / "memberships.csv",
            departments_path=tmp_path / "departments.csv",
            chem_review_path=tmp_path / "review.json",
            stage_dir=tmp_path / "stage",
            intent_path=tmp_path / "intent.json",
            snapshot_checkout=tmp_path / "snapshot-checkout",
            producer_commit=PRODUCER,
        )


def test_direct_publish_requires_reviewed_code_before_hub_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_producer(*_args: object) -> None:
        raise ValueError("wrong reviewed code")

    monkeypatch.setattr(
        delta,
        "_require_producer_checkout",
        reject_producer,
    )
    monkeypatch.setattr(
        delta,
        "require_live_337_parent",
        lambda *_args, **_kwargs: pytest.fail("Hub read reached with wrong producer"),
    )
    with pytest.raises(ValueError, match="wrong reviewed code"):
        delta.publish_chem_delta(
            expected_parent_revision=PARENT,
            hf_token="token",
            snapshot_path=tmp_path / "snapshot.parquet",
            snapshot_manifest_path=tmp_path / "snapshot.manifest.json",
            source_cache_dir=tmp_path / "authors",
            people_path=tmp_path / "people.csv",
            memberships_path=tmp_path / "memberships.csv",
            departments_path=tmp_path / "departments.csv",
            chem_review_path=tmp_path / "review.json",
            stage_dir=tmp_path / "stage",
            intent_path=tmp_path / "intent.json",
            snapshot_checkout=tmp_path / "snapshot-checkout",
            producer_commit=PRODUCER,
        )


def test_publish_raw_source_mismatch_aborts_before_stage_or_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(delta, "_require_producer_checkout", lambda *_args: None)
    monkeypatch.setattr(
        delta,
        "require_live_337_parent",
        lambda *_args, **_kwargs: delta.ExactParent(PARENT, {}),
    )
    monkeypatch.setattr(
        delta,
        "retained_git_intent_sink",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(registry_module, "load_registry", lambda *_args: object())
    monkeypatch.setattr(
        snapshot_module,
        "validate_snapshot",
        lambda *_args, **_kwargs: (object(), {"snapshot_sha256": delta.SOURCE_SHA256}),
    )
    monkeypatch.setattr(delta, "sha256_file", lambda _path: "0" * 64)
    monkeypatch.setattr(
        delta,
        "stage_chem_delta",
        lambda *_args, **_kwargs: calls.append("stage"),
    )
    monkeypatch.setattr(
        delta,
        "commit_atomic_release",
        lambda *_args, **_kwargs: calls.append("commit"),
    )
    with pytest.raises(ValueError, match="raw source bytes changed"):
        delta.publish_chem_delta(
            expected_parent_revision=PARENT,
            hf_token="token",
            snapshot_path=tmp_path / "snapshot.parquet",
            snapshot_manifest_path=tmp_path / "snapshot.manifest.json",
            source_cache_dir=tmp_path / "authors",
            people_path=tmp_path / "people.csv",
            memberships_path=tmp_path / "memberships.csv",
            departments_path=tmp_path / "departments.csv",
            chem_review_path=tmp_path / "review.json",
            stage_dir=tmp_path / "stage",
            intent_path=tmp_path / "intent.json",
            snapshot_checkout=tmp_path / "snapshot-checkout",
            producer_commit=PRODUCER,
        )
    assert calls == []


@pytest.mark.parametrize("finalize_failure", [False, True])
def test_publish_entry_point_retains_intent_and_reports_finalization_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finalize_failure: bool
) -> None:
    calls = []
    monkeypatch.setattr(delta, "_require_producer_checkout", lambda *_args: None)
    reviewed = object()
    registry = object()

    class Sink:
        def __call__(self, intent: dict) -> None:
            calls.append(("intent", intent["parent_revision"]))

        def finalize_success(self, revision: str, receipt: dict) -> None:
            calls.append(("success", revision, receipt["commit_nonce"]))
            if finalize_failure:
                raise OSError("remote success checkpoint unavailable")

    sink = Sink()
    monkeypatch.setattr(
        delta,
        "require_live_337_parent",
        lambda *_args, **_kwargs: delta.ExactParent(PARENT, {}),
    )
    monkeypatch.setattr(
        delta,
        "retained_git_intent_sink",
        lambda *_args, **_kwargs: sink,
    )
    monkeypatch.setattr(registry_module, "load_registry", lambda *_args: registry)

    def validate_frozen_snapshot(*_args: object, **kwargs: object) -> tuple:
        assert kwargs["people_path"] == tmp_path / "source-data/registry/people.csv"
        assert kwargs["memberships_path"] == (
            tmp_path / "source-data/registry/memberships.csv"
        )
        assert kwargs["departments_path"] == (
            tmp_path / "source-data/registry/departments.csv"
        )
        return object(), {
            "snapshot_sha256": delta.SOURCE_SHA256,
            "registry_sha256": "frozen-registry",
        }

    monkeypatch.setattr(snapshot_module, "validate_snapshot", validate_frozen_snapshot)
    monkeypatch.setattr(delta, "sha256_file", lambda _path: delta.SOURCE_SHA256)
    monkeypatch.setattr(
        delta,
        "require_frozen_registry_transition",
        lambda *_args, **_kwargs: {"status": "pass"},
    )
    monkeypatch.setattr(
        delta,
        "load_chem_region_review",
        lambda *_args, **_kwargs: reviewed,
    )
    files = _files(tmp_path)
    monkeypatch.setattr(
        delta,
        "stage_chem_delta",
        lambda _parent, decisions, loaded_registry, _stage: (
            files,
            {
                "status": "pass",
                "source": decisions is reviewed and loaded_registry is registry,
            },
        ),
    )

    def atomic(_files: dict, **kwargs: object) -> tuple[str, dict]:
        assert kwargs["intent_sink"] is sink
        assert kwargs["intent_metadata"]["producer_commit"] == PRODUCER
        assert kwargs["prewrite_gate"]["source"] is True
        assert kwargs["prewrite_gate"]["registry_transition"] == {"status": "pass"}
        sink({"parent_revision": PARENT})
        return REVISION, {"commit_nonce": "d" * 32}

    monkeypatch.setattr(delta, "commit_atomic_release", atomic)

    def publish():
        return delta.publish_chem_delta(
            expected_parent_revision=PARENT,
            hf_token="token",
            snapshot_path=tmp_path / "source-data/snapshots/snapshot.parquet",
            snapshot_manifest_path=tmp_path
            / "source-data/snapshots/snapshot.manifest.json",
            source_cache_dir=tmp_path / "authors",
            people_path=tmp_path / "people.csv",
            memberships_path=tmp_path / "memberships.csv",
            departments_path=tmp_path / "departments.csv",
            chem_review_path=tmp_path / "review.json",
            stage_dir=tmp_path / "stage",
            intent_path=tmp_path / "intent.json",
            snapshot_checkout=tmp_path / "snapshot-checkout",
            producer_commit=PRODUCER,
        )

    if finalize_failure:
        with pytest.raises(AtomicReleaseReconciliationRequired) as caught:
            publish()
        assert caught.value.receipt["verification_stage"] == "persist_success_receipt"
        assert caught.value.receipt["artifact_revision"] == REVISION
        assert caught.value.receipt["write_outcome"] == "verified_hub_status_pending"
    else:
        assert publish() == (REVISION, {"commit_nonce": "d" * 32})
    assert calls == [("intent", PARENT), ("success", REVISION, "d" * 32)]


def test_production_cli_uses_frozen_data_and_current_status_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    frozen = tmp_path / "source-data"
    status = tmp_path / "snapshot"
    for path in (source, frozen, status):
        path.mkdir()
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(delta, "_require_producer_checkout", lambda *_args: None)
    monkeypatch.setattr(
        delta,
        "_git",
        lambda _repo, *args: (
            delta.SOURCE_COMMIT if args == ("rev-parse", "HEAD") else ""
        ),
    )
    captured = {}

    def fake_publish(**kwargs: object) -> tuple[str, dict]:
        captured.update(kwargs)
        return REVISION, {"artifact_revision": REVISION}

    monkeypatch.setattr(delta, "publish_chem_delta", fake_publish)
    assert (
        delta.main(
            [
                "--expected-parent-revision",
                PARENT,
                "--expected-producer-commit",
                PRODUCER,
                "--producer-checkout",
                str(source),
                "--source-data-checkout",
                str(frozen),
                "--snapshot-checkout",
                str(status),
                "--stage-dir",
                str(tmp_path / "stage"),
                "--intent-path",
                str(tmp_path / "intent.json"),
            ]
        )
        == 0
    )
    assert (
        captured["snapshot_path"]
        == frozen / "snapshots/cmu-engineering-publications.parquet"
    )
    assert captured["source_cache_dir"] == frozen / "data/authors"
    assert captured["snapshot_checkout"] == status


def test_production_cli_rejects_missing_token_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        delta,
        "publish_chem_delta",
        lambda **_kwargs: pytest.fail("Hub path reached without token"),
    )
    with pytest.raises(ValueError, match="explicit HF_TOKEN"):
        delta.main(
            [
                "--expected-parent-revision",
                PARENT,
                "--expected-producer-commit",
                PRODUCER,
                "--producer-checkout",
                str(tmp_path),
                "--source-data-checkout",
                str(tmp_path),
                "--snapshot-checkout",
                str(tmp_path),
                "--stage-dir",
                str(tmp_path / "stage"),
                "--intent-path",
                str(tmp_path / "intent.json"),
            ]
        )


def test_production_cli_rejects_wrong_frozen_source_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(delta, "_require_producer_checkout", lambda *_args: None)
    monkeypatch.setattr(delta, "_git", lambda _repo, *_args: "0" * 40)
    monkeypatch.setattr(
        delta,
        "publish_chem_delta",
        lambda **_kwargs: pytest.fail("Hub path reached with wrong source"),
    )
    with pytest.raises(ValueError, match="frozen source commit"):
        delta.main(
            [
                "--expected-parent-revision",
                PARENT,
                "--expected-producer-commit",
                PRODUCER,
                "--producer-checkout",
                str(tmp_path),
                "--source-data-checkout",
                str(tmp_path),
                "--snapshot-checkout",
                str(tmp_path),
                "--stage-dir",
                str(tmp_path / "stage"),
                "--intent-path",
                str(tmp_path / "intent.json"),
            ]
        )


def test_production_requires_exact_clean_code_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    module = source / "map_of_research/chem_delta_release.py"
    module.parent.mkdir(parents=True)
    module.write_text("# reviewed source\n")
    _git(source, "init", "--initial-branch", "main")
    _git(source, "add", "map_of_research/chem_delta_release.py")
    _git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-m",
        "Reviewed producer",
    )
    reviewed = _git(source, "rev-parse", "HEAD")
    monkeypatch.setattr(delta, "__file__", str(module))
    delta._require_producer_checkout(source, reviewed)
    with pytest.raises(ValueError, match="reviewed SHA"):
        delta._require_producer_checkout(source, "0" * 40)
    module.write_text("# changed after review\n")
    with pytest.raises(ValueError, match="must be clean"):
        delta._require_producer_checkout(source, reviewed)
