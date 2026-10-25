"""Held Gregory Johnson source-review preservation and registry gate."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.validate_gregory_johnson_source_review import validate

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_gregory_johnson_sources_and_holds() -> None:
    result = validate()
    assert result == {
        "attributed_source_observations_not_works": 8,
        "provisional_manifestations_not_lifetime_count": 6,
        "research_lineages_not_lifetime_count": 2,
        "uncaptured_meeting_leads": 1,
        "excluded_sponsor_only_records": 1,
    }


def test_gregory_johnson_registry_scholar_id_remains_unassigned() -> None:
    with (ROOT / "registry/people.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    person = next(
        row for row in rows if row["person_id"] == "person-gregoryjohnson-26c9645f"
    )
    assert person["scholar_id"] == ""
    assert person["homepage_url"] == (
        "https://www.cmu.edu/math/people/faculty/johnson.html"
    )
