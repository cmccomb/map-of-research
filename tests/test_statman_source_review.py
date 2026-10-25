"""Guard the held Statman review against source loss or accidental promotion."""

import csv
from pathlib import Path

from scripts.validate_statman_source_review import validate


def test_statman_source_review_is_complete_for_its_bounded_sources():
    assert validate() == {
        "cmu_selected": 16,
        "kilthub_hits": 19,
        "kilthub_attributed": 17,
        "excluded_other_author_theses": 2,
        "held_source_observations_not_works": 33,
    }


def test_statman_registry_identity_remains_unassigned_to_scholar():
    root = Path(__file__).resolve().parents[1]
    with (root / "registry/people.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    person = next(
        row for row in rows if row["person_id"] == "person-richardstatman-2e46bf59"
    )
    assert person["scholar_id"] == ""
