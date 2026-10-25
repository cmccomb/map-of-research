"""Keep the held Offner bibliography tied to its raw source observations."""

import json
import subprocess
import sys
from pathlib import Path


def test_offner_source_crosswalk_preserves_all_observations_and_genres() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts/validate_offner_source_review.py"
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    counts = json.loads(result.stdout)
    assert counts["elements_observations"] == 29
    assert counts["preprint_journal_pairs"] == 10
    assert counts["distinct_research_works"] == 19
    assert counts["source_union_distinct_outputs_excluding_thesis"] == 21
