"""Guard held Gheorghiciuc evidence against source loss or release promotion."""

import csv
from pathlib import Path

from scripts.validate_gheorghiciuc_source_review import validate


def test_gheorghiciuc_bounded_primary_sources_are_preserved():
    assert validate() == {
        "cmu_selected": 3,
        "arxiv_exact_author": 4,
        "attributed_source_observations_not_works": 10,
        "provisional_work_groups_not_lifetime_count": 6,
        "excluded_other_author_theses": 1,
    }


def test_gheorghiciuc_registry_identity_remains_unassigned_to_scholar():
    root = Path(__file__).resolve().parents[1]
    with (root / "registry/people.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    person = next(
        row for row in rows if row["person_id"] == "person-irinagheorghiciuc-c53bff46"
    )
    assert person["scholar_id"] == ""
