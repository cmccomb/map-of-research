"""Stage a bounded collection only after its source and snapshot checks pass."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def stage_result(collector_exit_code: str, snapshot_outcome: str) -> None:
    """Keep unvalidated caches local while retaining a collection receipt."""

    validated = collector_exit_code == "0" and snapshot_outcome == "success"
    noop = collector_exit_code == "0" and snapshot_outcome == "skipped"
    if validated:
        receipt_status = "success"
        reason = None
    elif noop:
        receipt_status = "skipped"
        reason = None
    else:
        receipt_status = "failure"
        reason = (
            "collector_failed"
            if collector_exit_code != "0"
            else "source_or_snapshot_validation_failed"
        )

    status_path = Path("status/last-collection.json")
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else {
            "schema_version": 1,
            "status": "failure",
        }
    )
    if not isinstance(status, dict):
        raise ValueError("Collection status receipt must be a JSON object")
    status["snapshot_validation"] = {"status": receipt_status}
    if reason is not None:
        status["snapshot_validation"]["reason"] = reason
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = status_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(status_path)

    paths = (
        ("registry", "data", "status", "snapshots")
        if validated
        else ("status/last-collection.json",)
    )
    subprocess.run(["git", "add", "--", *paths], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-exit-code", required=True)
    parser.add_argument("--snapshot-outcome", required=True)
    args = parser.parse_args()
    stage_result(args.collector_exit_code, args.snapshot_outcome)


if __name__ == "__main__":
    main()
