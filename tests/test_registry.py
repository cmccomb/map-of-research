from pathlib import Path

import pytest

from map_of_research.registry import load_registry, map_catalog, unique_profiles
from tests.registry_helpers import write_registry


def test_registry_collapses_cross_appointments_with_stable_identity(
    tmp_path: Path,
) -> None:
    people_path, memberships_path, maps_path = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-byron-yu",
                "display_name": "Byron Yu",
                "scholar_id": "Fz3_tukAAAAJ",
                "orcid": "0000-0001-2345-678X",
                "homepage_url": "https://example.test/byron-yu",
                "notes": "",
            },
            {
                "person_id": "person-missing-id",
                "display_name": "Missing ID",
                "scholar_id": "",
                "orcid": "",
                "homepage_url": "",
                "notes": "",
            },
        ],
        memberships=[
            {
                "person_id": "person-byron-yu",
                "map_slug": "map-of-bme",
                "role": "faculty",
                "included": "true",
                "legacy_label": "BYu",
                "source_url": "https://www.cmu.edu/bme/People/Faculty/",
                "verified_at": "2026-07-17",
            },
            {
                "person_id": "person-byron-yu",
                "map_slug": "map-of-ece",
                "role": "faculty",
                "included": "true",
                "legacy_label": "Yu",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            },
            {
                "person_id": "person-missing-id",
                "map_slug": "map-of-ece",
                "role": "teaching",
                "included": "true",
                "legacy_label": "Missing ID",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            },
        ],
    )

    registry = load_registry(people_path, memberships_path, maps_path)
    profiles = unique_profiles(registry)

    assert len(registry.memberships) == 3
    assert len(profiles) == 1
    assert profiles[0].display_name == "Byron Yu"
    assert profiles[0].map_slugs == ("map-of-bme", "map-of-ece")
    assert map_catalog(registry)["map-of-eng"] == "Engineering"


def test_registry_retains_but_excludes_affiliates(tmp_path: Path) -> None:
    people_path, memberships_path, maps_path = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-affiliate",
                "display_name": "Affiliate Person",
                "scholar_id": "affiliateAAAAJ",
                "orcid": "",
                "homepage_url": "",
                "notes": "",
            }
        ],
        memberships=[
            {
                "person_id": "person-affiliate",
                "map_slug": "map-of-ece",
                "role": "affiliate",
                "included": "false",
                "legacy_label": "Affiliate",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            }
        ],
    )

    registry = load_registry(people_path, memberships_path, maps_path)

    assert registry.memberships[0].included is False
    assert registry.unique_profiles()[0].map_slugs == ()
    assert registry.unique_profiles(included_only=True) == []


def test_registry_rejects_role_inclusion_mismatch(tmp_path: Path) -> None:
    people_path, memberships_path, maps_path = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-affiliate",
                "display_name": "Affiliate Person",
                "scholar_id": "affiliateAAAAJ",
                "orcid": "",
                "homepage_url": "",
                "notes": "",
            }
        ],
        memberships=[
            {
                "person_id": "person-affiliate",
                "map_slug": "map-of-ece",
                "role": "affiliate",
                "included": "true",
                "legacy_label": "Affiliate",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            }
        ],
    )

    with pytest.raises(ValueError, match="requires included=False"):
        load_registry(people_path, memberships_path, maps_path)
