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
        "publisher_confirmed_conference_papers": 1,
        "publisher_confirmed_monographs": 1,
        "historical_report_observations": 1,
        "arxiv_preprint_observations": 1,
        "third_party_bibliography_candidates": 67,
        "third_party_unmatched_titles_needing_primary_review": 49,
        "third_party_doi_leads_with_crossref_registration": 38,
        "third_party_doi_title_variants_for_review": 11,
        "third_party_non_doi_leads_reviewed": 11,
        "held_source_observations_not_works": 37,
    }


def test_statman_registry_identity_remains_unassigned_to_scholar():
    root = Path(__file__).resolve().parents[1]
    with (root / "registry/people.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    person = next(
        row for row in rows if row["person_id"] == "person-richardstatman-2e46bf59"
    )
    assert person["scholar_id"] == ""
    assert "0000-0002-1825-0097" not in ",".join(person.values())
