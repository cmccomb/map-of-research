"""Keep the held bibliography tied to both complete captured source pages."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


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
        "orcid_thesis_observations": 1,
        "union_source_manifestations": 47,
    }


def test_orcid_book_label_cannot_override_reviewed_thesis_genre(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    shutil.copytree(root / "data/source_reviews/ernest-schimmerling-2026-10-25", source)
    decision_path = source / "orcid-thesis-decision.json"
    decision = json.loads(decision_path.read_text())
    decision["publication_type"] = "book"
    decision_path.write_text(json.dumps(decision))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][decision_path.name] = hashlib.sha256(
        decision_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    validator_path = root / "scripts/validate_schimmerling_source_review.py"
    spec = importlib.util.spec_from_file_location(
        "schimmerling_validator", validator_path
    )
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    with pytest.raises(AssertionError):
        validator.validate(source)
