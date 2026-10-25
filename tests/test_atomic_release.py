"""A release must be one parent-pinned Hub write after its gate passes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from map_of_research.atomic_release import (
    RELEASE_PATHS,
    AtomicReleaseReconciliationRequired,
    commit_atomic_release,
    require_single_shard_parent,
)
from map_of_research.io import sha256_file


def _files(tmp_path):
    files = {}
    for name in RELEASE_PATHS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = path
    return files


def _hub(
    monkeypatch,
    tmp_path,
    *,
    head="a" * 40,
    corrupt=False,
    moved_after=False,
    accepted_then_raise=False,
    download_fail=False,
    bad_oid=False,
    head_probe_fail=False,
):
    calls = []
    hosted = tmp_path / "hosted"
    hosted.mkdir(parents=True)

    class HfApi:
        def __init__(self, token):
            assert token == "token"

        def repo_info(self, repo_id, **kwargs):
            calls.append(("head", repo_id, kwargs))
            committed = any(item[0] == "commit" for item in calls)
            if committed and head_probe_fail:
                raise ConnectionError("head probe unavailable")
            return SimpleNamespace(
                sha=("c" * 40 if moved_after else "b" * 40) if committed else head
            )

        def list_repo_files(self, repo_id, **kwargs):
            calls.append(("list", repo_id, kwargs))
            return [*RELEASE_PATHS]

        def create_commit(self, **kwargs):
            calls.append(("commit", kwargs))
            for operation in kwargs["operations"]:
                path = hosted / operation.path_in_repo
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(Path(operation.path_or_fileobj).read_bytes())
            if corrupt:
                (hosted / "README.md").write_text("changed")
            if accepted_then_raise:
                raise ConnectionError("response lost after server accepted commit")
            return SimpleNamespace(oid="invalid" if bad_oid else "b" * 40)

    def add_operation(**kwargs):
        return SimpleNamespace(**kwargs)

    def download(repo_id, filename, **kwargs):
        calls.append(("download", filename, kwargs))
        if download_fail:
            raise ConnectionError("immutable download unavailable")
        return str(hosted / filename)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=HfApi,
            CommitOperationAdd=add_operation,
            hf_hub_download=download,
        ),
    )
    return calls


def _commit(files, *, intent_sink=None):
    if intent_sink is None:

        def intent_sink(intent):
            pass

    return commit_atomic_release(
        files,
        prewrite_gate={"status": "pass", "source_snapshot_sha256": "frozen"},
        repo_id="test/dataset",
        hf_token="token",
        prior_revision="a" * 40,
        commit_message="Publish full candidate",
        intent_sink=intent_sink,
        intent_metadata={
            "producer_commit": "c" * 40,
            "snapshot_commit": "d" * 40,
        },
    )


def test_atomic_release_uses_one_parent_commit_and_verifies_every_file(
    monkeypatch, tmp_path
) -> None:
    files = _files(tmp_path)
    calls = _hub(monkeypatch, tmp_path)
    intents = []
    revision, receipt = _commit(files, intent_sink=intents.append)

    assert revision == "b" * 40
    commits = [entry[1] for entry in calls if entry[0] == "commit"]
    assert len(commits) == 1
    assert commits[0]["parent_commit"] == "a" * 40
    assert {op.path_in_repo for op in commits[0]["operations"]} == RELEASE_PATHS
    assert receipt["committed_files"] == sorted(RELEASE_PATHS)
    assert receipt["hosted_file_bytes_match_prewrite"] is True
    assert {entry[1] for entry in calls if entry[0] == "download"} == RELEASE_PATHS
    assert [intent["intent_status"] for intent in intents] == [
        "before_write",
        "commit_returned_unverified",
    ]
    assert intents[0]["artifact_revision"] is None
    assert intents[1]["artifact_revision"] == revision
    assert intents[0]["size_by_path"] == {
        name: path.stat().st_size for name, path in files.items()
    }
    assert intents[0]["sha256_by_path"] == receipt["sha256_by_path"]
    assert intents[0]["release_context"]["producer_commit"] == "c" * 40
    assert intents[0]["release_context"]["snapshot_commit"] == "d" * 40
    assert (
        f"Release-Intent-Nonce: {intents[0]['commit_nonce']}"
        in commits[0]["commit_message"]
    )


def test_atomic_release_stops_before_any_write_on_gate_or_parent_failure(
    monkeypatch, tmp_path
) -> None:
    files = _files(tmp_path)
    calls = _hub(monkeypatch, tmp_path, head="c" * 40)
    with pytest.raises(RuntimeError, match="moved after prewrite gate"):
        _commit(files)
    assert not any(entry[0] == "commit" for entry in calls)

    with pytest.raises(ValueError, match="passing prewrite gate"):
        commit_atomic_release(
            files,
            prewrite_gate={"status": "review_required"},
            repo_id="test/dataset",
            hf_token="token",
            prior_revision="a" * 40,
            commit_message="Blocked",
            intent_sink=lambda intent: None,
        )
    assert not any(entry[0] == "commit" for entry in calls)


def test_atomic_release_requires_gated_stage_hashes_before_hub_call(
    monkeypatch, tmp_path
) -> None:
    files = _files(tmp_path)
    calls = _hub(monkeypatch, tmp_path)
    expected = {name: sha256_file(path) for name, path in files.items()}
    expected["README.md"] = "0" * 64
    with pytest.raises(ValueError, match="Staged release bytes changed"):
        commit_atomic_release(
            files,
            prewrite_gate={"status": "pass", "stage_sha256_by_path": expected},
            repo_id="test/dataset",
            hf_token="token",
            prior_revision="a" * 40,
            commit_message="Blocked staged-byte mutation",
            intent_sink=lambda intent: None,
        )
    assert calls == []
    expected["README.md"] = sha256_file(files["README.md"])
    revision, _ = commit_atomic_release(
        files,
        prewrite_gate={"status": "pass", "stage_sha256_by_path": expected},
        repo_id="test/dataset",
        hf_token="token",
        prior_revision="a" * 40,
        commit_message="Matched staged bytes",
        intent_sink=lambda intent: None,
    )
    assert revision == "b" * 40


def test_atomic_release_rejects_missing_intent_sink_before_hub_call(
    monkeypatch, tmp_path
) -> None:
    files = _files(tmp_path)
    calls = _hub(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="durable prewrite intent sink"):
        commit_atomic_release(
            files,
            prewrite_gate={"status": "pass"},
            repo_id="test/dataset",
            hf_token="token",
            prior_revision="a" * 40,
            commit_message="Unsafe direct call",
        )
    assert calls == []


def test_atomic_release_stops_if_prewrite_intent_cannot_persist(
    monkeypatch, tmp_path
) -> None:
    files = _files(tmp_path)
    calls = _hub(monkeypatch, tmp_path)

    def failed_checkpoint(intent):
        assert intent["intent_status"] == "before_write"
        raise OSError("intent store unavailable")

    with pytest.raises(OSError, match="intent store unavailable"):
        _commit(files, intent_sink=failed_checkpoint)
    assert not any(entry[0] == "commit" for entry in calls)


def test_atomic_release_records_failure_to_checkpoint_returned_oid(
    monkeypatch, tmp_path
) -> None:
    files = _files(tmp_path)
    calls = _hub(monkeypatch, tmp_path)
    stages = []

    def checkpoint(intent):
        stages.append(intent["intent_status"])
        if intent["intent_status"] == "commit_returned_unverified":
            raise OSError("second checkpoint unavailable")

    with pytest.raises(AtomicReleaseReconciliationRequired) as caught:
        _commit(files, intent_sink=checkpoint)
    assert stages == ["before_write", "commit_returned_unverified"]
    assert len([entry for entry in calls if entry[0] == "commit"]) == 1
    assert caught.value.receipt["verification_stage"] == "persist_commit_oid"
    assert caught.value.receipt["artifact_revision"] == "b" * 40


def test_atomic_release_detects_hosted_byte_mismatch(monkeypatch, tmp_path) -> None:
    files = _files(tmp_path)
    calls = _hub(monkeypatch, tmp_path, corrupt=True)
    with pytest.raises(
        AtomicReleaseReconciliationRequired, match="README.md"
    ) as caught:
        _commit(files)
    assert len([entry for entry in calls if entry[0] == "commit"]) == 1
    receipt = caught.value.receipt
    assert receipt["artifact_revision"] == "b" * 40
    assert receipt["parent_revision"] == "a" * 40
    assert receipt["sha256_by_path"] == {
        name: sha256_file(path) for name, path in files.items()
    }
    assert (
        receipt["hosted_sha256_by_path"]["README.md"]
        != receipt["sha256_by_path"]["README.md"]
    )
    assert receipt["verification_status"] == "reconciliation_required"


def test_atomic_release_rejects_extra_parent_shards() -> None:
    with pytest.raises(ValueError, match="parent shards"):
        require_single_shard_parent(
            [*RELEASE_PATHS, "works/train-00001-of-00002.parquet"]
        )
    with pytest.raises(ValueError, match="missing README.md"):
        require_single_shard_parent(list(RELEASE_PATHS - {"README.md"}))


def test_atomic_release_rejects_incomplete_or_empty_stage(
    monkeypatch, tmp_path
) -> None:
    files = _files(tmp_path)
    calls = _hub(monkeypatch, tmp_path)
    incomplete = dict(files)
    incomplete.pop("README.md")
    with pytest.raises(ValueError, match="exactly four configs"):
        _commit(incomplete)
    assert not any(item[0] == "commit" for item in calls)

    files["README.md"].write_bytes(b"")
    with pytest.raises(ValueError, match="empty staged"):
        _commit(files)
    assert not any(item[0] == "commit" for item in calls)


def test_atomic_release_detects_newer_head_after_commit(monkeypatch, tmp_path) -> None:
    files = _files(tmp_path)
    _hub(monkeypatch, tmp_path, moved_after=True)
    with pytest.raises(RuntimeError, match="no longer Hub main"):
        _commit(files)


@pytest.mark.parametrize(
    ("hub_options", "stage", "write_outcome", "artifact_revision"),
    [
        ({"accepted_then_raise": True}, "create_commit", "unknown", None),
        (
            {"download_fail": True},
            "verify_immutable_commit",
            "accepted_unverified",
            "b" * 40,
        ),
    ],
)
def test_atomic_release_records_ambiguous_write_for_reconciliation(
    monkeypatch, tmp_path, hub_options, stage, write_outcome, artifact_revision
) -> None:
    files = _files(tmp_path)
    calls = _hub(monkeypatch, tmp_path, **hub_options)
    intents = []
    with pytest.raises(AtomicReleaseReconciliationRequired) as caught:
        _commit(files, intent_sink=intents.append)
    assert len([entry for entry in calls if entry[0] == "commit"]) == 1
    receipt = caught.value.receipt
    assert receipt["verification_stage"] == stage
    assert receipt["write_outcome"] == write_outcome
    assert receipt["artifact_revision"] == artifact_revision
    assert receipt["parent_revision"] == "a" * 40
    assert receipt["observed_head_revision"] == "b" * 40
    assert receipt["committed_files"] == sorted(RELEASE_PATHS)
    assert receipt["sha256_by_path"] == {
        name: sha256_file(path) for name, path in files.items()
    }
    assert receipt["hosted_file_bytes_match_prewrite"] is None
    assert intents[0]["commit_nonce"] == receipt["commit_nonce"]
    assert intents[0]["intent_status"] == "before_write"
    assert len(intents) == (1 if stage == "create_commit" else 2)


def test_atomic_release_preserves_unknown_oid_and_failed_head_probe(
    monkeypatch, tmp_path
) -> None:
    files = _files(tmp_path)
    _hub(monkeypatch, tmp_path, bad_oid=True)
    with pytest.raises(AtomicReleaseReconciliationRequired) as caught:
        _commit(files)
    assert caught.value.receipt["verification_stage"] == "parse_commit_oid"
    assert caught.value.receipt["returned_commit_oid"] == "invalid"
    assert caught.value.receipt["observed_head_revision"] == "b" * 40

    _hub(
        monkeypatch, tmp_path / "second", accepted_then_raise=True, head_probe_fail=True
    )
    with pytest.raises(AtomicReleaseReconciliationRequired) as caught:
        _commit(files)
    assert caught.value.receipt["verification_stage"] == "create_commit"
    assert "head probe unavailable" in caught.value.receipt["head_probe_error"]
