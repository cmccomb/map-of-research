"""One-time recovery of the reviewed weak-region decisions before Bernhard.

The current hosted revision remains the upload concurrency guard. The older
revision supplies only the region comparison for the exact Bernhard snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy

from .region_continuity import require_revision

CURRENT_HUB_REVISION = "bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd"
REGION_COMPARISON_REVISION = "15498d5f78e2e269f4b11c630bc9b4ef56efb628"
SNAPSHOT_COMMIT = "f7a0eee8a1829e9c2317ce32f189381c469c6ff3"
HISTORICAL_RECEIPT_COMMIT = "3b3ba167947dc399edc08f17b9985a53b5260e4f"
SNAPSHOT_SHA256 = "5b940614b2238c124bc194e944da277ee2221df27a11eedca6693af020ee86fb"
HISTORICAL_SNAPSHOT_SHA256 = (
    "b72df04cf28e5af1450b4010ac05a91c5f68b1b671fa19d2a2e7a93a22d46e58"
)
SNAPSHOT_BRANCH = "automation/map-snapshot"
CURRENT_UPLOAD_RUN = (
    "https://github.com/cmccomb/map-of-research/actions/runs/40169148827"
)
HISTORICAL_UPLOAD_RUN = (
    "https://github.com/cmccomb/map-of-research/actions/runs/40167632965"
)
HISTORICAL_WORK_COUNT = 47495
REGION_DECISION_FIELDS = (
    "map_eligible",
    "map_region_outlier",
    "map_exclusion_reasons",
)


def require_approved_comparison_revision(
    *, current_revision: str, comparison_revision: str, snapshot_sha256: str
) -> str:
    """Reject every region baseline override except the exact recovery pair."""

    current = require_revision(current_revision)
    comparison = require_revision(comparison_revision)
    if comparison != current and (
        current != CURRENT_HUB_REVISION
        or comparison != REGION_COMPARISON_REVISION
        or snapshot_sha256 != SNAPSHOT_SHA256
    ):
        raise ValueError("Unapproved region comparison revision or snapshot")
    return comparison


def require_frozen_source_tip(snapshot_root: Path) -> None:
    """Reject a repair if the source branch moved after checkout."""

    branch_ref = f"refs/heads/{SNAPSHOT_BRANCH}"
    response = subprocess.run(
        ["git", "-C", str(snapshot_root), "ls-remote", "origin", branch_ref],
        capture_output=True,
        check=True,
        text=True,
    )
    if response.stdout.strip() != f"{SNAPSHOT_COMMIT}\t{branch_ref}":
        raise RuntimeError("Frozen snapshot branch moved before upload")


def require_frozen_producer_tip(snapshot_root: Path, expected_commit: str) -> None:
    """Reject a repair if producer main moved after dispatch."""

    expected = require_revision(expected_commit)
    branch_ref = "refs/heads/main"
    response = subprocess.run(
        ["git", "-C", str(snapshot_root), "ls-remote", "origin", branch_ref],
        capture_output=True,
        check=True,
        text=True,
    )
    if response.stdout.strip() != f"{expected}\t{branch_ref}":
        raise RuntimeError("Producer main moved before upload")


def require_expected_repair_preview(
    *,
    work_count: int,
    map_point_count: int,
    new_embedding_count: int,
    layout_reused: bool,
    continuity_receipt: dict[str, Any],
) -> None:
    """Stop if the exact replay differs from the reviewed no-upload result."""

    if (
        work_count != 47693
        or map_point_count != 42869
        or new_embedding_count != 0
        or not layout_reused
    ):
        raise RuntimeError("Bernhard repair changed map points, embeddings, or layout")
    expected = {
        "prior_revision": REGION_COMPARISON_REVISION,
        "current_hub_revision": CURRENT_HUB_REVISION,
        "prior_visible_observations": 48953,
        "current_visible_observations": 49136,
        "stable_works": 45437,
        "reviewed_losses": [],
        "verified_gains": [],
        "new_source_gains": 183,
        "unexplained_losses": [],
        "unexplained_gains": [],
    }
    if any(continuity_receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Bernhard repair differs from reviewed region replay")


def require_historical_region_decisions(
    historical_works: Sequence[Mapping[str, Any]], final_works: Any
) -> None:
    """Require every pre-Bernhard work to retain its exact region decision."""

    if len(historical_works) != HISTORICAL_WORK_COUNT:
        raise RuntimeError("Bernhard repair lacks the complete historical work set")
    required = {"work_id", *REGION_DECISION_FIELDS}
    if not required.issubset(final_works.columns):
        raise RuntimeError("Bernhard repair lacks final region decision fields")

    def decision(row: Mapping[str, Any]) -> tuple[bool, bool, tuple[str, ...]]:
        if (
            not required.issubset(row)
            or not isinstance(row["map_exclusion_reasons"], (list, tuple))
            or not all(
                isinstance(row[field], (bool, numpy.bool_))
                for field in ("map_eligible", "map_region_outlier")
            )
        ):
            raise RuntimeError("Bernhard repair lacks historical region decisions")
        return (
            bool(row["map_eligible"]),
            bool(row["map_region_outlier"]),
            tuple(row["map_exclusion_reasons"]),
        )

    prior: dict[str, tuple[bool, bool, tuple[str, ...]]] = {}
    for row in historical_works:
        work_id = str(row.get("work_id") or "")
        if not work_id or work_id in prior:
            raise RuntimeError("Bernhard repair has duplicate historical work IDs")
        prior[work_id] = decision(row)

    final: dict[str, tuple[bool, bool, tuple[str, ...]]] = {}
    for row in final_works.loc[:, list(required)].to_dict("records"):
        work_id = str(row.get("work_id") or "")
        if not work_id or work_id in final:
            raise RuntimeError("Bernhard repair has duplicate final work IDs")
        final[work_id] = decision(row)

    changed = sorted(
        work_id
        for work_id, prior_decision in prior.items()
        if final.get(work_id) != prior_decision
    )
    if changed:
        raise RuntimeError(
            "Bernhard repair changed historical region decisions: "
            f"{len(changed)} work IDs; first {changed[:3]}"
        )


def select_region_comparison_revision(
    *,
    requested: bool,
    event_name: str,
    run_ref: str,
    run_commit: str,
    producer_commit: str,
    snapshot_ref: str,
    snapshot_commit: str,
    historical_commit: str,
    snapshot_path: Path,
    manifest_path: Path,
    current_receipt_path: Path,
    historical_receipt_path: Path,
    current_revision: str,
) -> str:
    """Allow the recovery only for one trusted dispatch of the frozen source."""

    current = require_revision(current_revision)
    if not requested:
        return current
    if (
        event_name != "workflow_dispatch"
        or run_ref != "refs/heads/main"
        or producer_commit != require_revision(run_commit)
        or snapshot_ref != SNAPSHOT_BRANCH
        or snapshot_commit != SNAPSHOT_COMMIT
        or historical_commit != HISTORICAL_RECEIPT_COMMIT
        or current != CURRENT_HUB_REVISION
    ):
        raise ValueError("Region recovery requires the frozen dispatch and Hub head")
    current_receipt = json.loads(current_receipt_path.read_text(encoding="utf-8"))
    historical_receipt = json.loads(historical_receipt_path.read_text(encoding="utf-8"))
    expected_receipts = (
        (
            current_receipt,
            {
                "status": "success",
                "artifact_commit": CURRENT_HUB_REVISION,
                "raw_snapshot_sha256": SNAPSHOT_SHA256,
                "workflow_url": CURRENT_UPLOAD_RUN,
                "raw_snapshot_observations": 51823,
                "raw_snapshot_profiles": 310,
            },
        ),
        (
            historical_receipt,
            {
                "status": "success",
                "artifact_commit": REGION_COMPARISON_REVISION,
                "raw_snapshot_sha256": HISTORICAL_SNAPSHOT_SHA256,
                "workflow_url": HISTORICAL_UPLOAD_RUN,
                "raw_snapshot_observations": 51595,
                "raw_snapshot_profiles": 309,
            },
        ),
    )
    for receipt, expected in expected_receipts:
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("Region recovery receipts do not match reviewed uploads")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if digest != SNAPSHOT_SHA256 or manifest.get("snapshot_sha256") != digest:
        raise ValueError("Region recovery snapshot does not match the frozen Parquet")
    return require_approved_comparison_revision(
        current_revision=current,
        comparison_revision=REGION_COMPARISON_REVISION,
        snapshot_sha256=digest,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore-prior-regions", choices=("true", "false"), required=True
    )
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--run-ref", required=True)
    parser.add_argument("--run-commit", required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--snapshot-ref", required=True)
    parser.add_argument("--snapshot-commit", required=True)
    parser.add_argument("--historical-commit", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--current-receipt", type=Path, required=True)
    parser.add_argument("--historical-receipt", type=Path, required=True)
    parser.add_argument("--current-revision", type=require_revision, required=True)
    args = parser.parse_args(argv)
    revision = select_region_comparison_revision(
        requested=args.restore_prior_regions == "true",
        event_name=args.event_name,
        run_ref=args.run_ref,
        run_commit=args.run_commit,
        producer_commit=args.producer_commit,
        snapshot_ref=args.snapshot_ref,
        snapshot_commit=args.snapshot_commit,
        historical_commit=args.historical_commit,
        snapshot_path=args.snapshot,
        manifest_path=args.manifest,
        current_receipt_path=args.current_receipt,
        historical_receipt_path=args.historical_receipt,
        current_revision=args.current_revision,
    )
    print(f"revision={revision}")


if __name__ == "__main__":
    main()
