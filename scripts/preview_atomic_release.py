"""Exercise the real atomic publisher through its first Hub write, without writing.

This script is intended for a read-only Linux Actions job. It calls the same
publisher as upload.yml, retains all seven staged files, and replaces the only
Hub mutation with an exception raised before any request is sent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any


class FirstHubWriteIntercepted(Exception):
    """The seven-file commit reached its first network mutation boundary."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _require_clean_tracked_files(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD"], check=True)


def _committed_file_bytes(checkout: Path, commit: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(checkout), "show", f"{commit}:{relative_path}"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            f"{relative_path} is missing from commit {commit} in {checkout}"
        )
    return result.stdout


def _require_receipt_lineage(
    snapshot: Path,
    receipt_checkout: Path,
    *,
    snapshot_commit: str,
    status_commit: str,
) -> bytes:
    relative_path = "status/last-successful-upload.json"
    receipt_bytes = _committed_file_bytes(
        receipt_checkout, status_commit, relative_path
    )
    snapshot_bytes = _committed_file_bytes(snapshot, snapshot_commit, relative_path)
    if (receipt_checkout / relative_path).read_bytes() != receipt_bytes:
        raise ValueError(
            "Receipt checkout differs from its committed successful receipt"
        )
    if (snapshot / relative_path).read_bytes() != snapshot_bytes:
        raise ValueError(
            "Snapshot checkout differs from its committed successful receipt"
        )
    if snapshot_bytes != receipt_bytes:
        raise ValueError(
            "Snapshot successful receipt differs from the receipt checkout"
        )
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(snapshot),
            "merge-base",
            "--is-ancestor",
            status_commit,
            snapshot_commit,
        ],
        capture_output=True,
        check=False,
    )
    if ancestry.returncode == 1:
        raise ValueError("Receipt commit is not an ancestor of the snapshot commit")
    if ancestry.returncode:
        raise RuntimeError(
            "Cannot verify receipt ancestry in snapshot checkout: "
            + ancestry.stderr.decode("utf-8", errors="replace").strip()
        )
    return receipt_bytes


def _save_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stage_files(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _require_sha(value: str, *, label: str, length: int) -> str:
    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"{label} must be a {length}-character lowercase hex digest")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--snapshot-commit", required=True)
    parser.add_argument("--status-commit", required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--prior-revision", required=True)
    parser.add_argument("--region-comparison-revision", required=True)
    parser.add_argument("--require-bio-release", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    producer = args.producer_dir.resolve()
    snapshot = args.snapshot_dir.resolve()
    receipt_checkout = args.receipt_dir.resolve()
    output = args.output_dir.resolve()
    receipt_path = output / "receipt.json"
    stage = output / "stage"
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "mode": "read_only_first_write_intercept",
        "producer_commit": args.producer_commit,
        "snapshot_commit": args.snapshot_commit,
        "status_commit": args.status_commit,
        "snapshot_sha256": args.snapshot_sha256,
        "prior_revision": args.prior_revision,
        "region_comparison_revision": args.region_comparison_revision,
        "require_bio_release": args.require_bio_release,
        "workflow_url": (
            f"{os.environ['GITHUB_SERVER_URL']}/"
            f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
            f"{os.environ['GITHUB_RUN_ID']}"
            if all(
                os.environ.get(name)
                for name in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
            )
            else None
        ),
        "hub_writes_completed": 0,
        "blocked_hub_write_calls": 0,
    }
    try:
        _save_receipt(receipt_path, receipt)
        from datasets import Dataset
        from huggingface_hub import HfApi

        from map_of_research import publisher
        from map_of_research.atomic_release import (
            RELEASE_PATHS,
            AtomicReleaseReconciliationRequired,
        )

        for label, value, length in (
            ("producer commit", args.producer_commit, 40),
            ("snapshot commit", args.snapshot_commit, 40),
            ("status commit", args.status_commit, 40),
            ("snapshot SHA-256", args.snapshot_sha256, 64),
            ("prior revision", args.prior_revision, 40),
            ("region comparison revision", args.region_comparison_revision, 40),
        ):
            _require_sha(value, label=label, length=length)
        if args.region_comparison_revision != args.prior_revision:
            raise ValueError("Read-only preview requires the current Hub comparison")
        if any(
            output.is_relative_to(checkout)
            for checkout in (producer, snapshot, receipt_checkout)
        ):
            raise ValueError("Preview output must be outside all Git checkouts")
        if stage.exists():
            raise FileExistsError(f"Preview stage already exists: {stage}")
        if _git_head(producer) != args.producer_commit:
            raise ValueError("Producer checkout differs from the requested commit")
        if _git_head(snapshot) != args.snapshot_commit:
            raise ValueError("Snapshot checkout differs from the requested commit")
        if _git_head(receipt_checkout) != args.status_commit:
            raise ValueError("Receipt checkout differs from the requested commit")
        _require_clean_tracked_files(producer)
        _require_clean_tracked_files(snapshot)
        _require_clean_tracked_files(receipt_checkout)
        baseline_bytes = _require_receipt_lineage(
            snapshot,
            receipt_checkout,
            snapshot_commit=args.snapshot_commit,
            status_commit=args.status_commit,
        )
        raw = snapshot / "snapshots/cmu-engineering-publications.parquet"
        manifest = (
            snapshot / "snapshots/cmu-engineering-publications.parquet.manifest.json"
        )
        if _sha256(raw) != args.snapshot_sha256:
            raise ValueError("Snapshot bytes differ from the requested SHA-256")
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        if manifest_data["snapshot_sha256"] != args.snapshot_sha256:
            raise ValueError("Snapshot manifest differs from the requested SHA-256")
        baseline = json.loads(baseline_bytes)
        if (
            baseline.get("status") != "success"
            or baseline.get("artifact_commit") != args.prior_revision
        ):
            raise ValueError("Successful source receipt differs from the Hub parent")
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            os.environ.pop(name, None)
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        os.environ.pop("MAP_DURABLE_INTENT_REPO", None)
        os.environ.pop("MAP_DURABLE_INTENT_BRANCH", None)
        os.environ["MAP_PRODUCER_COMMIT"] = args.producer_commit
        os.environ["MAP_SOURCE_COMMIT"] = args.snapshot_commit
        api = HfApi(token=None)
        before = api.repo_info(publisher.REPO_ID, repo_type="dataset").sha
        receipt["hub_head_before"] = before
        if before != args.prior_revision:
            raise RuntimeError(f"Hub head differs from the pinned parent: {before}")
        _save_receipt(receipt_path, receipt)

        def intent(value: dict[str, Any]) -> None:
            receipt["local_atomic_intent"] = dict(value)
            _save_receipt(receipt_path, receipt)

        def first_write(*_args: Any, **kwargs: Any) -> None:
            receipt["blocked_hub_write_calls"] += 1
            operations = kwargs.get("operations")
            paths = (
                [operation.path_in_repo for operation in operations]
                if isinstance(operations, list)
                else []
            )
            if (
                kwargs.get("repo_id") != publisher.REPO_ID
                or kwargs.get("repo_type") != "dataset"
                or kwargs.get("parent_commit") != args.prior_revision
                or len(paths) != len(RELEASE_PATHS)
                or set(paths) != RELEASE_PATHS
            ):
                raise AssertionError("First write is not the pinned seven-file commit")
            stage.mkdir(parents=True)
            for operation in operations:
                source = Path(operation.path_or_fileobj)
                destination = stage / operation.path_in_repo
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            staged = _stage_files(stage)
            intent_value = receipt.get("local_atomic_intent")
            if (
                set(staged) != RELEASE_PATHS
                or not isinstance(intent_value, dict)
                or intent_value.get("parent_revision") != args.prior_revision
                or intent_value.get("prewrite_gate", {}).get("status") != "pass"
                or intent_value.get("sha256_by_path")
                != {name: item["sha256"] for name, item in staged.items()}
                or intent_value.get("size_by_path")
                != {name: item["bytes"] for name, item in staged.items()}
            ):
                raise AssertionError("Staged files differ from passing prewrite intent")
            receipt["stage_files"] = staged
            receipt["first_write_intercepted"] = {
                "operation": "HfApi.create_commit",
                "parent_commit": args.prior_revision,
                "files": sorted(staged),
            }
            _save_receipt(receipt_path, receipt)
            raise FirstHubWriteIntercepted("Stopped before HfApi.create_commit")

        def forbidden_write(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("Unexpected Hub write path during preview")

        HfApi.create_commit = first_write
        HfApi.upload_file = forbidden_write
        HfApi.upload_folder = forbidden_write
        Dataset.push_to_hub = forbidden_write
        curated = producer / "data/curated"
        reviews = producer / "registry/reviews"
        try:
            publisher.publish_snapshot(
                raw,
                manifest,
                hf_token=None,
                people_path=producer / "registry/people.csv",
                memberships_path=producer / "registry/memberships.csv",
                departments_path=producer / "registry/departments.csv",
                exclusions_path=curated / "scholar-exclusions.json",
                corrections_path=curated / "scholar-metadata-corrections.json",
                doi_aliases_path=curated / "scholar-doi-aliases.json",
                attribution_reviews_path=curated / "scholar-attribution-reviews.json",
                contexts_path=curated / "scholar-contexts.json",
                profile_releases_path=curated / "scholar-profile-releases.json",
                provisional_holds_path=curated / "scholar-provisional-holds.json",
                source_cache_dir=snapshot / "data/authors",
                primary_home_baseline_path=(
                    reviews / "2026-10-25-implicit-scholar-primary-baseline.json"
                ),
                prior_revision=args.prior_revision,
                region_comparison_revision=args.region_comparison_revision,
                atomic_intent_sink=intent,
                require_bio_release=args.require_bio_release,
                repair_manifest_path=curated / "region-remediation-bernhard.json",
                source_commit=args.snapshot_commit,
                sullivan_loss_approval_path=curated / "sullivan-exact-losses-bb88.json",
                sullivan_continuity_addendum_path=(
                    curated / "sullivan-continuity-addendum-bb88.json"
                ),
                sullivan_crosswalk_path=(
                    reviews / "2026-10-25-chem-sullivan-prior-id-crosswalk.json"
                ),
                curated_works_path=curated / "works.json",
                math_region_review_path=curated / "scholar-region-review.json",
                chem_region_review_path=curated / "scholar-chem-region-review.json",
            )
        except AtomicReleaseReconciliationRequired as error:
            if not isinstance(error.__cause__, FirstHubWriteIntercepted):
                raise
        else:
            raise AssertionError("Publisher returned without an intercepted write")
        after = HfApi(token=None).repo_info(publisher.REPO_ID, repo_type="dataset").sha
        receipt["hub_head_after"] = after
        if after != before or receipt["blocked_hub_write_calls"] != 1:
            raise RuntimeError("Hub moved or preview did not block exactly one write")
        receipt["status"] = "passed_no_upload"
        return 0
    except BaseException as error:
        receipt["status"] = "failed_before_hub_write"
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)[:1000]
        receipt["traceback"] = traceback.format_exc(limit=12)
        return 1
    finally:
        _save_receipt(receipt_path, receipt)
        print(
            json.dumps(
                {
                    key: receipt.get(key)
                    for key in (
                        "status",
                        "error_type",
                        "error",
                        "hub_head_before",
                        "hub_head_after",
                        "blocked_hub_write_calls",
                        "hub_writes_completed",
                    )
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
