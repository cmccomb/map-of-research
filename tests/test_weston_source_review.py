from pathlib import Path
from shutil import copytree

import pytest

from scripts.validate_weston_source_review import REVIEW, ROOT, validate


def test_weston_source_review_is_complete_and_held():
    assert validate() == {
        "cmu_scholars": 29,
        "orcid_groups": 24,
        "exact_doi_overlaps": 23,
        "cmu_dois_absent_orcid": 3,
        "cmu_non_doi_not_in_orcid": 2,
    }


def test_weston_source_review_rejects_changed_raw_capture(tmp_path: Path):
    review = tmp_path / "review"
    copytree(REVIEW, review)
    source = review / "orcid-works.json"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source digest changed"):
        validate(review, ROOT / "registry")
