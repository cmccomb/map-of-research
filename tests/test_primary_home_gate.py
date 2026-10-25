"""Publication gate for included Scholar primary academic homes."""

from pathlib import Path

import pytest

from map_of_research.registry import (
    load_registry,
    validate_included_scholar_primary_homes,
)
from tests.registry_helpers import write_registry


def test_current_registry_has_evidenced_included_scholar_homes() -> None:
    registry = load_registry()
    count = validate_included_scholar_primary_homes(registry)
    assert count == len(registry.unique_profiles(included_only=True))
    assert count > 0
    assert {
        registry.primary_affiliation(profile.person_id).home_type
        for profile in registry.unique_profiles(included_only=True)
    } == {"department", "institute", "constituent_school", "academic_area", "school"}


def test_unresolved_joint_scholar_primary_fails_closed(tmp_path: Path) -> None:
    paths = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-joint",
                "display_name": "Joint Faculty",
                "scholar_id": "jointAAAAJ",
            }
        ],
        memberships=[
            {
                "person_id": "person-joint",
                "department_id": department_id,
                "role": "faculty",
                "included": "true",
                "source_url": f"https://example.test/{department_id}",
                "verified_at": "2026-10-25",
            }
            for department_id in ("ece", "bme")
        ],
    )
    registry = load_registry(*paths)
    with pytest.raises(ValueError, match=r"person-joint \(Joint Faculty\)"):
        validate_included_scholar_primary_homes(registry)


def test_single_department_needs_dated_evidence(tmp_path: Path) -> None:
    paths = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-single",
                "display_name": "Single Faculty",
                "scholar_id": "singleAAAAJ",
            }
        ],
        memberships=[
            {
                "person_id": "person-single",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/ece",
                "verified_at": "",
            }
        ],
    )
    registry = load_registry(*paths)
    assert registry.primary_affiliation("person-single").department_id == "ece"
    with pytest.raises(ValueError, match="person-single"):
        validate_included_scholar_primary_homes(registry)


def test_formal_types_and_cmu_africa_preserve_department_semantics(
    tmp_path: Path,
) -> None:
    paths = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-campus",
                "display_name": "Campus Faculty",
                "scholar_id": "campusAAAAJ",
                "primary_academic_home_type": "campus",
                "primary_academic_home_id": "cmu-silicon-valley",
                "primary_academic_home_title": "CMU Silicon Valley",
                "primary_academic_home_source_url": "https://example.test/campus",
                "primary_academic_home_verified_at": "2026-10-25",
                "primary_academic_home_basis": "reviewed",
            },
            {
                "person_id": "person-area",
                "display_name": "Area Faculty",
                "scholar_id": "areaAAAAJ",
                "primary_academic_home_type": "academic_area",
                "primary_academic_home_id": "tepper:finance",
                "primary_academic_home_title": "Finance",
                "primary_academic_home_source_url": "https://example.test/finance",
                "primary_academic_home_verified_at": "2026-10-25",
                "primary_academic_home_basis": "reviewed",
            },
            {
                "person_id": "person-africa",
                "display_name": "Africa Faculty",
                "scholar_id": "africaAAAAJ",
                "primary_department_id": "cmu-africa",
                "primary_department_source_url": "https://example.test/africa",
                "primary_department_verified_at": "2026-10-25",
                "primary_department_basis": "reviewed",
            },
        ],
        memberships=[
            {
                "person_id": person_id,
                "department_id": department_id,
                "role": "faculty",
                "included": "true",
                "source_url": f"https://example.test/{department_id}",
                "verified_at": "2026-10-25",
            }
            for person_id, department_id in (
                ("person-campus", "cmu-silicon-valley"),
                ("person-area", "tepper-faculty"),
                ("person-africa", "cmu-africa"),
            )
        ],
        departments=[
            {
                "department_id": "cmu-silicon-valley",
                "title": "CMU Silicon Valley",
                "unit_type": "campus",
                "directory_url": "https://example.test/campus",
            },
            {
                "department_id": "tepper-faculty",
                "school_id": "tepper",
                "title": "Tepper faculty roster",
                "unit_type": "roster_container",
                "directory_url": "https://example.test/tepper",
            },
            {
                "department_id": "cmu-africa",
                "school_id": "engineering",
                "title": "CMU-Africa",
                "unit_type": "department",
                "directory_url": "https://example.test/africa",
            },
        ],
        schools=[
            {
                "school_id": "engineering",
                "title": "College of Engineering",
                "directory_url": "https://example.test/engineering",
            },
            {
                "school_id": "tepper",
                "title": "Tepper School of Business",
                "directory_url": "https://example.test/tepper",
            },
        ],
    )
    registry = load_registry(*paths)
    assert validate_included_scholar_primary_homes(registry) == 3
    assert registry.primary_affiliation("person-campus").department_id == ""
    assert registry.primary_affiliation("person-area").department_id == ""
    assert registry.primary_affiliation("person-africa").department_id == "cmu-africa"


def test_excluded_scholar_profile_does_not_need_a_primary(tmp_path: Path) -> None:
    paths = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-affiliate",
                "display_name": "Affiliate",
                "scholar_id": "affiliateAAAAJ",
            }
        ],
        memberships=[
            {
                "person_id": "person-affiliate",
                "department_id": "ece",
                "role": "affiliate",
                "included": "false",
                "source_url": "https://example.test/ece",
                "verified_at": "2026-10-25",
            }
        ],
    )
    assert validate_included_scholar_primary_homes(load_registry(*paths)) == 0
