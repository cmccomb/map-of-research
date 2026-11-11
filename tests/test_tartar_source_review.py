"""Keep Tartar's held ledger tied to the complete visible CMU source rows."""

import json
import subprocess
import sys
from pathlib import Path


def test_tartar_source_review_preserves_bounded_sources_without_release() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts/validate_tartar_source_review.py"
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    counts = json.loads(result.stdout)
    assert counts == {
        "cmu_selected_rows": 6,
        "kilthub_rows": 10,
        "cna_notes": 5,
        "source_rows": 21,
        "internal_groups_not_lifetime_works": 19,
    }
