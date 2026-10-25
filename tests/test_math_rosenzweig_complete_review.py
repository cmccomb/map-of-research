"""Release regression for all 55 saved Rosenzweig Scholar observations."""

import json
from pathlib import Path

import pandas

from map_of_research.dataset import build_dataset_tables
from map_of_research.doi_aliases import (
    DEFAULT_DOI_ALIASES_PATH,
    load_doi_aliases,
    validate_doi_aliases,
)
from map_of_research.exclusions import DEFAULT_EXCLUSIONS_PATH, load_exclusions
from map_of_research.registry import load_registry


def test_complete_rosenzweig_capture_keeps_versions_and_abstracts_distinct() -> None:
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/math_rosenzweig_complete_rows.json"
        ).read_text(encoding="utf-8")
    )
    scholar_id = "rW9dfuQAAAAJ"
    assert fixture["scholar_id"] == scholar_id
    assert fixture["publication_count"] == len(fixture["publications"]) == 55
    assert {row["capture_row_index"] for row in fixture["publications"]} == set(
        range(1, 56)
    )
    assert len({row["author_pub_id"] for row in fixture["publications"]}) == 55

    registry = load_registry()
    person = next(
        person for person in registry.people if person.scholar_id == scholar_id
    )
    rows = [
        {
            "source_type": "scholar",
            "scholar_id": scholar_id,
            "author_pub_id": source["author_pub_id"],
            "person_id": person.person_id,
            "display_name": person.display_name,
            "faculty": person.display_name,
            "department_ids": ["math"],
            "memberships": [
                {"department_id": "math", "role": "faculty", "included": True}
            ],
            "title": source["title"],
            "authors": source["author"],
            "year": int(source["pub_year"]) if source["pub_year"] else None,
            "venue": source["venue"],
            "citation": "",
            "citation_count": source["num_citations"],
            "source_url": source["pub_url"],
            "source_record_json": json.dumps(source),
            "fetched_at_utc": fixture["fetched_at_utc"],
            "embedding": [0.0, 1.0],
        }
        for source in fixture["publications"]
    ]
    frame = pandas.DataFrame(rows)
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    aliases = {
        key: alias
        for key, alias in load_doi_aliases(DEFAULT_DOI_ALIASES_PATH).items()
        if key[0] == scholar_id
    }
    assert len(aliases) == 9
    validate_doi_aliases(frame, aliases, exclusions)
    tables = build_dataset_tables(frame, registry, exclusions, aliases)
    publications = tables["profile_publications"].set_index("author_pub_id")
    assert len(publications) == 55
    assert int(publications["map_eligible"].sum()) == 48
    assert {
        citation_id
        for citation_id, row in publications.iterrows()
        if not row["map_eligible"]
    } == {
        author_pub_id
        for (entry_scholar_id, author_pub_id) in exclusions
        if entry_scholar_id == scholar_id
    }

    point_vortex_ids = {
        f"{scholar_id}:0EnyYjriUFMC",  # Published article.
        f"{scholar_id}:zYLM7Y9cAGgC",  # Earlier preprint.
        f"{scholar_id}:LkGwnXOMwfcC",  # Author's final manuscript.
    }
    point_vortex = publications.loc[list(point_vortex_ids)]
    assert point_vortex["map_eligible"].all()
    assert point_vortex["work_id"].nunique() == 1
    assert set(point_vortex["doi"]) == {"10.1007/s00205-021-01735-3"}

    meeting_ids = {
        f"{scholar_id}:qUcmZB5y_30C",  # AMS Spring 2026.
        f"{scholar_id}:QIV2ME_5wuYC",  # AMS Fall 2025.
        f"{scholar_id}:M3ejUd6NZC8C",  # AMS Spring 2025.
    }
    meeting_abstracts = publications.loc[list(meeting_ids)]
    assert meeting_abstracts["map_eligible"].all()
    assert meeting_abstracts["work_id"].nunique() == 3
    assert set(meeting_abstracts["work_id"]).isdisjoint(point_vortex["work_id"])
