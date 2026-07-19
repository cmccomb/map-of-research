import csv
from copy import deepcopy
from pathlib import Path

import pytest

import map_of_research.registry as registry_module
from map_of_research.registry import department_catalog, load_registry, unique_profiles
from tests.registry_helpers import write_registry


def test_registry_collapses_cross_appointments_with_stable_identity(
    tmp_path: Path,
) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-byron-yu",
                "display_name": "Byron Yu",
                "scholar_id": "Fz3_tukAAAAJ",
                "scholar_id_source_url": "https://example.test/byron-yu",
                "scholar_id_verified_at": "2026-07-17",
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
                "department_id": "bme",
                "role": "faculty",
                "included": "true",
                "legacy_label": "BYu",
                "source_url": "https://www.cmu.edu/bme/People/Faculty/",
                "verified_at": "2026-07-17",
            },
            {
                "person_id": "person-byron-yu",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "legacy_label": "Yu",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            },
            {
                "person_id": "person-missing-id",
                "department_id": "ece",
                "role": "teaching",
                "included": "true",
                "legacy_label": "Missing ID",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            },
        ],
    )

    registry = load_registry(people_path, memberships_path, departments_path)
    profiles = unique_profiles(registry)

    assert len(registry.memberships) == 3
    assert len(profiles) == 1
    assert profiles[0].display_name == "Byron Yu"
    assert profiles[0].person.scholar_id_source_url == "https://example.test/byron-yu"
    assert profiles[0].department_ids == ("bme", "ece")
    assert department_catalog(registry)["ece"] == "Electrical & Computer Engineering"


def test_registry_retains_but_excludes_affiliates(tmp_path: Path) -> None:
    people_path, memberships_path, departments_path = write_registry(
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
                "department_id": "ece",
                "role": "affiliate",
                "included": "false",
                "legacy_label": "Affiliate",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            }
        ],
    )

    registry = load_registry(people_path, memberships_path, departments_path)

    assert registry.memberships[0].included is False
    assert registry.unique_profiles()[0].department_ids == ()
    assert registry.unique_profiles(included_only=True) == []


def test_registry_rejects_role_inclusion_mismatch(tmp_path: Path) -> None:
    people_path, memberships_path, departments_path = write_registry(
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
                "department_id": "ece",
                "role": "affiliate",
                "included": "true",
                "legacy_label": "Affiliate",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            }
        ],
    )

    with pytest.raises(ValueError, match="requires included=False"):
        load_registry(people_path, memberships_path, departments_path)


def test_registry_rejects_scholar_provenance_without_id(tmp_path: Path) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-missing-id",
                "display_name": "Missing ID",
                "scholar_id": "",
                "scholar_id_source_url": "https://example.test/profile",
                "scholar_id_verified_at": "2026-07-17",
                "orcid": "",
                "homepage_url": "",
                "notes": "",
            }
        ],
        memberships=[
            {
                "person_id": "person-missing-id",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "legacy_label": "Missing ID",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            }
        ],
    )

    with pytest.raises(ValueError, match="provenance requires a Scholar ID"):
        load_registry(people_path, memberships_path, departments_path)


def valid_registry_rows():
    return (
        [
            {
                "person_id": "person-one",
                "display_name": "One Person",
                "scholar_id": "oneAAAAJ",
                "scholar_id_source_url": "https://example.test/one",
                "scholar_id_verified_at": "2026-07-17",
                "orcid": "0000-0001-2345-678X",
                "homepage_url": "https://example.test/one",
                "notes": "",
            }
        ],
        [
            {
                "person_id": "person-one",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "legacy_label": "One",
                "source_url": "https://example.test/ece",
                "verified_at": "2026-07-17",
            }
        ],
        [
            {
                "department_id": "ece",
                "title": "Electrical & Computer Engineering",
                "directory_url": "https://example.test/ece",
                "reviewed_at": "2026-07-17",
                "review_notes": "Test",
            }
        ],
    )


def duplicate_row(rows, table):
    rows[table].append(deepcopy(rows[table][0]))


def change(rows, table, field, value):
    rows[table][0][field] = value


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: change(rows, 0, "person_id", "one"), "Invalid person_id"),
        (lambda rows: duplicate_row(rows, 0), "Duplicate person_id"),
        (lambda rows: change(rows, 0, "display_name", ""), "Blank display_name"),
        (lambda rows: change(rows, 0, "scholar_id", "bad id"), "Invalid Scholar ID"),
        (
            lambda rows: rows[0].append({**rows[0][0], "person_id": "person-two"}),
            "belongs to multiple people",
        ),
        (
            lambda rows: change(rows, 0, "scholar_id_source_url", "http://bad.test"),
            "absolute HTTPS URL",
        ),
        (
            lambda rows: change(rows, 0, "scholar_id_verified_at", "July 17"),
            "YYYY-MM-DD",
        ),
        (
            lambda rows: change(rows, 0, "scholar_id_verified_at", ""),
            "requires both",
        ),
        (lambda rows: change(rows, 0, "orcid", "not-an-orcid"), "Invalid ORCID"),
        (
            lambda rows: change(rows, 0, "homepage_url", "relative/path"),
            "absolute HTTPS URL",
        ),
        (
            lambda rows: change(rows, 2, "department_id", "Bad ID"),
            "Invalid department_id",
        ),
        (lambda rows: duplicate_row(rows, 2), "Duplicate department_id"),
        (lambda rows: change(rows, 2, "title", ""), "Blank department title"),
        (
            lambda rows: change(rows, 2, "directory_url", ""),
            "directory_url cannot be blank",
        ),
        (
            lambda rows: change(rows, 2, "reviewed_at", "bad"),
            "reviewed_at must use YYYY-MM-DD",
        ),
        (
            lambda rows: change(rows, 1, "included", "sometimes"),
            "included must be true or false",
        ),
        (
            lambda rows: change(rows, 1, "person_id", "person-missing"),
            "Unknown person_id",
        ),
        (
            lambda rows: change(rows, 1, "department_id", "missing"),
            "Unknown department_id",
        ),
        (lambda rows: change(rows, 1, "role", "wizard"), "Unknown registry role"),
        (lambda rows: duplicate_row(rows, 1), "Duplicate membership"),
        (
            lambda rows: change(rows, 1, "source_url", ""),
            "source_url cannot be blank",
        ),
        (
            lambda rows: change(rows, 1, "verified_at", "bad"),
            "verified_at must use YYYY-MM-DD",
        ),
    ],
)
def test_registry_rejects_each_invalid_field(tmp_path: Path, mutate, message) -> None:
    rows = [deepcopy(items) for items in valid_registry_rows()]
    mutate(rows)
    paths = write_registry(
        tmp_path,
        people=rows[0],
        memberships=rows[1],
        departments=rows[2],
    )
    with pytest.raises(ValueError, match=message):
        load_registry(*paths)


def test_registry_reader_requires_a_complete_header(tmp_path: Path) -> None:
    missing_header = tmp_path / "missing-header.csv"
    missing_header.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="missing a header"):
        registry_module._read_rows(missing_header, ("one",))

    missing_column = tmp_path / "missing-column.csv"
    missing_column.write_text("one\nvalue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        registry_module._read_rows(missing_column, ("one", "two"))


@pytest.mark.parametrize(
    ("empty_table", "message"),
    [(0, "Unknown person_id"), (1, "cannot be empty"), (2, "Unknown department_id")],
)
def test_registry_rejects_empty_normalized_tables(
    tmp_path: Path,
    empty_table,
    message,
) -> None:
    rows = [deepcopy(items) for items in valid_registry_rows()]
    paths = write_registry(
        tmp_path,
        people=rows[0],
        memberships=rows[1],
        departments=rows[2],
    )
    columns = (
        registry_module.PEOPLE_COLUMNS,
        registry_module.MEMBERSHIP_COLUMNS,
        registry_module.DEPARTMENT_COLUMNS,
    )[empty_table]
    with paths[empty_table].open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=columns).writeheader()
    with pytest.raises(ValueError, match=message):
        load_registry(*paths)


def test_required_date_validator_rejects_blank_values() -> None:
    with pytest.raises(ValueError, match="reviewed_at cannot be blank"):
        registry_module._validate_date("", field="reviewed_at", required=True)


def test_registry_indexes_and_profile_properties(tmp_path: Path) -> None:
    people, memberships, departments = valid_registry_rows()
    paths = write_registry(
        tmp_path,
        people=people,
        memberships=memberships,
        departments=departments,
    )
    registry = load_registry(*paths)
    profile = registry.unique_profiles()[0]
    assert registry.people_by_id[profile.person_id] is profile.person
    assert registry.departments_by_id["ece"].title.startswith("Electrical")
    assert profile.scholar_id == "oneAAAAJ"
    assert unique_profiles(registry) == [profile]
