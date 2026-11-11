"""Guard held Handron sources against loss or accidental graph promotion."""

import csv
from pathlib import Path

from scripts.validate_handron_source_review import validate


def test_handron_bounded_sources_and_genres_are_preserved():
    assert validate() == {
        "cmu_selected": 1,
        "author_listed": 2,
        "arxiv_exact_author": 1,
        "attributed_source_observations_not_works": 6,
        "provisional_work_groups_not_lifetime_count": 3,
        "uncaptured_conference_leads": 3,
    }


def test_handron_registry_identity_remains_unassigned_to_scholar():
    root = Path(__file__).resolve().parents[1]
    with (root / "registry/people.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    person = next(
        row for row in rows if row["person_id"] == "person-davidhandron-66faa357"
    )
    assert person["scholar_id"] == ""
