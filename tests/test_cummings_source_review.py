"""Evidence gate for the held James Cummings bibliography review."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

REVIEW = (
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/james-cummings-2026-10-25"
)
SPEC = importlib.util.spec_from_file_location(
    "cummings_capture", REVIEW / "validate_capture.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_complete_held_three_source_crosswalk():
    assert MODULE.validate_capture(REVIEW) == {
        "cmu_elements_records": 108,
        "researcher_site_items": 51,
        "orcid_work_groups": 51,
        "orcid_exact_doi_elements_matches": 49,
        "publisher_missing_publication_manifestations": 4,
        "candidate_duplicate_pairs": 13,
    }


def test_modified_elements_record_cannot_pass_review(tmp_path):
    copy = tmp_path / "capture"
    shutil.copytree(REVIEW, copy)
    record = copy / "elements-108.csv"
    original = record.read_text(encoding="utf-8")
    record.write_text(
        original.replace("Squares, scales and lines", "Invented title", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pinned source changed"):
        MODULE.validate_capture(copy)
