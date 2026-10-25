"""Commit a fully validated dataset release to one immutable Hub revision."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import sha256_file
from .region_continuity import require_revision

CONFIG_NAMES = ("people", "works", "authorships", "profile_publications")
RELEASE_PATHS = frozenset(
    {
        *(f"{name}/train-00000-of-00001.parquet" for name in CONFIG_NAMES),
        "maps/publications.json",
        "maps/manifest.json",
        "README.md",
    }
)


def require_single_shard_parent(
    repo_files: list[str], *, config_names: tuple[str, ...] = CONFIG_NAMES
) -> None:
    """Do not leave a previous shard alongside a replacement one."""

    existing = set(repo_files)
    for name in config_names:
        shards = {
            path
            for path in existing
            if path.startswith(f"{name}/")
            and path.rsplit("/", 1)[-1].startswith("train-")
            and path.endswith(".parquet")
        }
        expected = {f"{name}/train-00000-of-00001.parquet"}
        if shards != expected:
            raise ValueError(
                f"Cannot replace {name} atomically: parent shards are "
                f"{sorted(shards)}, expected {sorted(expected)}"
            )
    for name in ("maps/publications.json", "maps/manifest.json", "README.md"):
        if name not in existing:
            raise ValueError(f"Parent dataset is missing {name}")


def commit_atomic_release(
    files: Mapping[str, Path],
    *,
    prewrite_gate: Mapping[str, Any],
    repo_id: str,
    hf_token: str,
    prior_revision: str,
    commit_message: str,
) -> tuple[str, dict[str, Any]]:
    """Write seven co-resident files only after all prewrite gates pass.

    The Hub parent condition prevents a stale preview from overwriting a newer
    release. The returned receipt includes byte hashes from the immutable
    committed revision, so callers can record it in GitHub only afterward.
    """

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    parent = require_revision(prior_revision)
    if prewrite_gate.get("status") != "pass":
        raise ValueError("Atomic release requires a passing prewrite gate")
    if set(files) != RELEASE_PATHS:
        raise ValueError(
            "Atomic release must stage exactly four configs, both map files, "
            "and README.md"
        )
    for name, path in files.items():
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty staged release file: {name}")
    local_hashes = {name: sha256_file(path) for name, path in sorted(files.items())}

    api = HfApi(token=hf_token)
    current = api.repo_info(repo_id, repo_type="dataset").sha
    if current != parent:
        raise RuntimeError(
            f"Hosted dataset moved after prewrite gate: expected {parent}, "
            f"found {current}"
        )
    require_single_shard_parent(
        api.list_repo_files(repo_id, repo_type="dataset", revision=parent)
    )
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(files[name]))
            for name in sorted(files)
        ],
        commit_message=commit_message,
        parent_commit=parent,
    )
    revision = require_revision(str(getattr(commit, "oid", "")))
    hosted_hashes = {
        name: sha256_file(
            Path(
                hf_hub_download(
                    repo_id,
                    name,
                    repo_type="dataset",
                    revision=revision,
                    token=hf_token,
                )
            )
        )
        for name in sorted(files)
    }
    if hosted_hashes != local_hashes:
        changed = sorted(
            name for name in files if hosted_hashes[name] != local_hashes[name]
        )
        raise RuntimeError(f"Hosted release differs from prewrite files: {changed}")
    hosted_head = api.repo_info(repo_id, repo_type="dataset").sha
    if hosted_head != revision:
        raise RuntimeError(
            f"Committed revision {revision} is no longer Hub main: {hosted_head}"
        )
    return revision, {
        "mode": "single_parent_commit",
        "parent_revision": parent,
        "artifact_revision": revision,
        "committed_files": sorted(files),
        "sha256_by_path": local_hashes,
        "hosted_file_bytes_match_prewrite": True,
        "prewrite_gate": dict(prewrite_gate),
    }
