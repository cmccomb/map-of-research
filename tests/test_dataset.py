import pandas

from map_of_research.dataset import build_dataset_tables
from map_of_research.registry import load_registry
from tests.registry_helpers import write_registry


def test_dataset_retains_observations_while_deduplicating_exact_work(tmp_path) -> None:
    people_path, memberships_path, maps_path = write_registry(
        tmp_path / "registry",
        people=[
            {
                "person_id": "person-one",
                "display_name": "One Person",
                "scholar_id": "oneAAAAJ",
                "orcid": "",
                "homepage_url": "",
                "notes": "",
            },
            {
                "person_id": "person-two",
                "display_name": "Two Person",
                "scholar_id": "twoAAAAJ",
                "orcid": "",
                "homepage_url": "",
                "notes": "",
            },
        ],
        memberships=[
            {
                "person_id": person_id,
                "map_slug": "map-of-ece",
                "role": "faculty",
                "included": "true",
                "legacy_label": person_id,
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            }
            for person_id in ("person-one", "person-two")
        ],
    )
    registry = load_registry(people_path, memberships_path, maps_path)
    memberships = [
        {
            "map_slug": "map-of-ece",
            "role": "faculty",
            "included": True,
            "legacy_label": "Person",
            "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
            "verified_at": "2026-07-17",
        }
    ]
    rows = []
    for person_id, scholar_id, display_name in (
        ("person-one", "oneAAAAJ", "One Person"),
        ("person-two", "twoAAAAJ", "Two Person"),
    ):
        rows.append(
            {
                "scholar_id": scholar_id,
                "person_id": person_id,
                "display_name": display_name,
                "faculty": display_name,
                "map_slugs": ["map-of-ece"],
                "memberships": memberships,
                "author_pub_id": f"{scholar_id}:paper",
                "title": "The Same Paper!",
                "authors": "One Person and Two Person",
                "year": 2025,
                "venue": "Journal",
                "citation": "Journal, 2025",
                "citation_count": 4,
                "source_url": "",
                "source_record_json": '{"pages":"1-10"}',
                "fetched_at_utc": "2026-01-01T00:00:00+00:00",
                "embedding": [0.0] * 768,
            }
        )

    tables = build_dataset_tables(pandas.DataFrame(rows), registry)

    assert len(tables["profile_publications"]) == 2
    assert len(tables["authorships"]) == 2
    assert len(tables["works"]) == 1
    work = tables["works"].iloc[0]
    assert work["observation_count"] == 2
    assert work["faculty"] == ["One Person", "Two Person"]
    assert set(work["author_pub_ids"]) == {
        "oneAAAAJ:paper",
        "twoAAAAJ:paper",
    }
