"""Accept an upload receipt only when this workflow run produced it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_current_run_receipt(
    receipt_path: Path, workflow_url: str, *, repair_mode: bool
) -> bool:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(receipt, dict)
        and receipt.get("workflow_url") == workflow_url
        and receipt.get("status")
        in ({"success"} if repair_mode else {"success", "failure"})
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workflow-url", required=True)
    parser.add_argument("--repair-mode", choices=("true", "false"), required=True)
    args = parser.parse_args()
    return (
        0
        if is_current_run_receipt(
            args.receipt, args.workflow_url, repair_mode=args.repair_mode == "true"
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
