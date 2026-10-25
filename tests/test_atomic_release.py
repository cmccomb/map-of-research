"""A release must be one parent-pinned Hub write after its gate passes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from map_of_research.atomic_release import (
    RELEASE_PATHS,
    commit_atomic_release,
    require_single_shard_parent,
)


def _files(tmp_path):
    files = {}
    for name in RELEASE_PATHS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = path
    return files


def _hub(monkeypatch, tmp_path, *, head="a" * 40, corrupt=False, moved_after=False):
    calls = []
    hosted = tmp_path / "hosted"
    hosted.mkdir()

    class HfApi:
        def __init__(self, token):
            assert token == "token"

        def repo_info(self, repo_id, **kwargs):
            calls.append(("head", repo_id, kwargs))
            committed = any(item[0] == "commit" for item in calls)
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
            return SimpleNamespace(oid="b" * 40)

    def add_operation(**kwargs):
        return SimpleNamespace(**kwargs)

    def download(repo_id, filename, **kwargs):
        calls.append(("download", filename, kwargs))
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


def _commit(files):
    return commit_atomic_release(
        files,
        prewrite_gate={"status": "pass", "source_snapshot_sha256": "frozen"},
        repo_id="test/dataset",
        hf_token="token",
        prior_revision="a" * 40,
        commit_message="Publish full candidate",
    )


def test_atomic_release_uses_one_parent_commit_and_verifies_every_file(
    monkeypatch, tmp_path
) -> None:
    files = _files(tmp_path)
    calls = _hub(monkeypatch, tmp_path)
    revision, receipt = _commit(files)

    assert revision == "b" * 40
    commits = [entry[1] for entry in calls if entry[0] == "commit"]
    assert len(commits) == 1
    assert commits[0]["parent_commit"] == "a" * 40
    assert {op.path_in_repo for op in commits[0]["operations"]} == RELEASE_PATHS
    assert receipt["committed_files"] == sorted(RELEASE_PATHS)
    assert receipt["hosted_file_bytes_match_prewrite"] is True
    assert {entry[1] for entry in calls if entry[0] == "download"} == RELEASE_PATHS


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
        )
    assert not any(entry[0] == "commit" for entry in calls)


def test_atomic_release_detects_hosted_byte_mismatch(monkeypatch, tmp_path) -> None:
    files = _files(tmp_path)
    calls = _hub(monkeypatch, tmp_path, corrupt=True)
    with pytest.raises(RuntimeError, match="README.md"):
        _commit(files)
    assert len([entry for entry in calls if entry[0] == "commit"]) == 1


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
