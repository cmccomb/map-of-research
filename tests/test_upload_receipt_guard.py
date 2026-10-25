"""A failed pre-upload gate must not recommit the previous success receipt."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/require_current_upload_receipt.py"
CURRENT_URL = "https://github.com/cmccomb/map-of-research/actions/runs/123"


def accepted(path: Path, *, repair_mode: bool = False) -> bool:
    return (
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--receipt",
                str(path),
                "--workflow-url",
                CURRENT_URL,
                "--repair-mode",
                "true" if repair_mode else "false",
            ],
            capture_output=True,
            check=False,
            text=True,
        ).returncode
        == 0
    )


def test_current_run_success_or_failure_can_be_recorded(tmp_path):
    receipt = tmp_path / "last-upload.json"
    for status in ("success", "failure"):
        receipt.write_text(json.dumps({"status": status, "workflow_url": CURRENT_URL}))
        assert accepted(receipt)
        assert accepted(receipt, repair_mode=True) == (status == "success")


def test_stale_or_malformed_receipt_cannot_advance_source_branch(tmp_path):
    receipt = tmp_path / "last-upload.json"
    assert not accepted(receipt)
    for document in (
        {"status": "success", "workflow_url": "https://old.example/run"},
        {"status": "running", "workflow_url": CURRENT_URL},
        ["not an upload receipt"],
    ):
        receipt.write_text(json.dumps(document))
        assert not accepted(receipt)
    receipt.write_text("{not JSON}")
    assert not accepted(receipt)
