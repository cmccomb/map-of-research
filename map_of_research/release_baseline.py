"""Pin the last successful hosted release before a new upload starts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from .atomic_release import RELEASE_PATHS
from .region_continuity import require_revision


def _require_resolved_atomic_intent(latest: Mapping, successful: Mapping) -> None:
    """A prior ambiguous write needs an exact reviewed outcome before retry."""

    if not any(
        key in latest
        for key in ("atomic_release_intent", "atomic_release_reconciliation")
    ):
        return
    intent = latest.get("atomic_release_intent") or latest.get(
        "atomic_release_reconciliation"
    )
    if not isinstance(intent, Mapping):
        raise ValueError("Prior atomic release intent is malformed")
    hashes = intent.get("sha256_by_path")
    sizes = intent.get("size_by_path")
    if (
        not isinstance(hashes, Mapping)
        or set(hashes) != RELEASE_PATHS
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in hashes.values()
        )
        or not isinstance(sizes, Mapping)
        or set(sizes) != RELEASE_PATHS
        or any(not isinstance(value, int) or value <= 0 for value in sizes.values())
    ):
        raise ValueError("Prior atomic release intent lacks seven staged files")
    nonce = str(intent.get("commit_nonce") or "")
    parent = require_revision(str(intent.get("parent_revision") or ""))
    successful_revision = require_revision(str(successful.get("artifact_commit") or ""))
    if latest.get("status") == "success":
        verified = latest.get("atomic_release")
        if (
            isinstance(verified, Mapping)
            and verified.get("commit_nonce") == nonce
            and verified.get("parent_revision") == parent
            and verified.get("artifact_revision") == successful_revision
            and verified.get("sha256_by_path") == hashes
            and verified.get("size_by_path") == sizes
            and verified.get("hosted_file_bytes_match_prewrite") is True
        ):
            return
        raise ValueError("Successful atomic receipt differs from staged intent")
    resolution = latest.get("atomic_release_resolution")
    if (
        not re.fullmatch(r"[0-9a-f]{32}", nonce)
        or not isinstance(resolution, Mapping)
        or resolution.get("commit_nonce") != nonce
        or resolution.get("parent_revision") != parent
        or not str(resolution.get("reviewer") or "").strip()
        or not str(resolution.get("reviewed_at_utc") or "").strip()
        or not str(resolution.get("evidence_url") or "").startswith("https://")
    ):
        raise ValueError("Prior atomic write awaits exact reviewed reconciliation")
    outcome = resolution.get("outcome")
    if outcome == "reconciled_no_write":
        if (
            resolution.get("observed_head_revision") != parent
            or resolution.get("matching_commit_ids") != []
            or successful_revision != parent
        ):
            raise ValueError("No-write reconciliation differs from the pinned parent")
    elif outcome == "verified_success":
        candidate = require_revision(str(resolution.get("artifact_revision") or ""))
        if (
            successful_revision != candidate
            or resolution.get("verified_sha256_by_path") != intent.get("sha256_by_path")
            or resolution.get("verified_size_by_path") != intent.get("size_by_path")
        ):
            raise ValueError("Verified release reconciliation differs from intent")
    else:
        raise ValueError("Prior atomic write awaits exact reviewed reconciliation")


def pin_prior_release(status_dir: Path) -> str:
    """Return the immutable prior SHA and save the first validated receipt."""

    successful = status_dir / "last-successful-upload.json"
    source = successful if successful.exists() else status_dir / "last-upload.json"
    receipt = json.loads(source.read_text(encoding="utf-8"))
    if receipt.get("status") != "success":
        raise ValueError(f"Prior hosted release receipt is not successful: {source}")
    revision = require_revision(str(receipt.get("artifact_commit") or ""))
    latest = status_dir / "last-upload.json"
    if latest.exists():
        _require_resolved_atomic_intent(
            json.loads(latest.read_text(encoding="utf-8")), receipt
        )
    if not successful.exists():
        shutil.copyfile(source, successful)
    return revision


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(f"revision={pin_prior_release(args.status_dir)}")


if __name__ == "__main__":
    main()
