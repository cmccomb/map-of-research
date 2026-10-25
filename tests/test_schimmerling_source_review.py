"""Keep the held bibliography tied to both complete captured source pages."""

import json
import subprocess
import sys
from pathlib import Path


def test_schimmerling_source_crosswalk_preserves_all_observations() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/validate_schimmerling_source_review.py"
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "elements_observations": 45,
        "author_observations": 34,
        "author_matched_elements": 41,
        "elements_only_observations": 4,
        "author_only_observations": 1,
    }
