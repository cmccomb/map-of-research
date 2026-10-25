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
    assert registry.schools_by_id["engineering"].title == "College of Engineering"
    assert registry.school_catalog()["engineering"] == "College of Engineering"


def test_registry_retains_one_profile_across_schools(tmp_path: Path) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-cross-school",
                "display_name": "Cross School",
                "scholar_id": "crossAAAAJ",
            }
        ],
        memberships=[
            {
                "person_id": "person-cross-school",
                "department_id": department_id,
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/roster",
                "verified_at": "2026-10-24",
            }
            for department_id in ("ece", "s3d")
        ],
        departments=[
            {
                "department_id": "ece",
                "school_id": "engineering",
                "title": "Electrical & Computer Engineering",
                "directory_url": "https://example.test/ece",
            },
            {
                "department_id": "s3d",
                "school_id": "scs",
                "title": "Software and Societal Systems Department",
                "directory_url": "https://example.test/s3d",
            },
        ],
    )

    registry = load_registry(
        people_path,
        memberships_path,
        departments_path,
        schools_path=tmp_path / "schools.csv",
    )
    profiles = registry.unique_profiles(included_only=True)
    assert len(profiles) == 1
    assert profiles[0].department_ids == ("ece", "s3d")
    assert registry.departments_by_id["s3d"].school_id == "scs"
    assert registry.primary_affiliation("person-cross-school").basis == "unresolved"


def test_single_department_with_other_school_roster_needs_review(
    tmp_path: Path,
) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path,
        people=[{"person_id": "person-joint", "display_name": "Joint Faculty"}],
        memberships=[
            {
                "person_id": "person-joint",
                "department_id": unit_id,
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/roster",
                "verified_at": "2026-10-24",
            }
            for unit_id in ("ece", "tepper-faculty")
        ],
        departments=[
            {
                "department_id": "ece",
                "school_id": "engineering",
                "title": "Electrical & Computer Engineering",
                "directory_url": "https://example.test/ece",
            },
            {
                "department_id": "tepper-faculty",
                "school_id": "tepper",
                "title": "Tepper faculty roster",
                "unit_type": "roster_container",
                "directory_url": "https://example.test/tepper",
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
    registry = load_registry(people_path, memberships_path, departments_path)
    assert registry.primary_affiliation("person-joint").basis == "unresolved"


@pytest.mark.parametrize(
    "basis",
    [
        "reviewed",
        "inferred_advisee_count",
        "inferred_profile_title",
        "display_tie_break_first_listed",
    ],
)
def test_primary_department_requires_reviewed_evidence_for_multiple_memberships(
    tmp_path: Path,
    basis: str,
) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-joint",
                "display_name": "Joint Faculty",
                "scholar_id": "jointAAAAJ",
                "primary_department_id": "s3d",
                "primary_department_source_url": "https://example.test/joint-profile",
                "primary_department_verified_at": "2026-10-24",
                "primary_department_basis": basis,
            },
            {"person_id": "person-single", "display_name": "Single Faculty"},
        ],
        memberships=[
            {
                "person_id": "person-joint",
                "department_id": department_id,
                "role": "faculty",
                "included": "true",
                "source_url": f"https://example.test/{department_id}",
                "verified_at": "2026-10-24",
            }
            for department_id in ("ece", "s3d")
        ]
        + [
            {
                "person_id": "person-single",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/ece",
                "verified_at": "2026-10-24",
            }
        ],
        departments=[
            {
                "department_id": "ece",
                "title": "Electrical and Computer Engineering",
                "directory_url": "https://example.test/ece",
            },
            {
                "department_id": "s3d",
                "school_id": "scs",
                "title": "Software and Societal Systems Department",
                "directory_url": "https://example.test/s3d",
            },
        ],
    )
    registry = load_registry(people_path, memberships_path, departments_path)
    joint = registry.primary_affiliation("person-joint")
    assert (joint.department_id, joint.basis) == ("s3d", basis)
    assert joint.source_url == "https://example.test/joint-profile"
    single = registry.primary_affiliation("person-single")
    assert (single.department_id, single.basis) == ("ece", "single_included_membership")
    assert single.source_url == "https://example.test/ece"


@pytest.mark.parametrize(
    ("person_id", "department_id", "school_id", "title", "source_url", "unit_ids"),
    [
        (
            "person-jeremyavigad-0d62ce13",
            "philosophy",
            "dietrich",
            "Department of Philosophy",
            "https://www.cmu.edu/dietrich/philosophy/people/faculty/jeremy-avigad.html",
            {"math", "philosophy"},
        ),
        (
            "person-robertfsekerka-8dacf315",
            "physics",
            "mcs",
            "Physics",
            "https://www.cmu.edu/leadership/the-provost/university-professors",
            {"math", "physics"},
        ),
    ],
)
def test_joint_math_display_tie_break_preserves_both_memberships(
    person_id: str,
    department_id: str,
    school_id: str,
    title: str,
    source_url: str,
    unit_ids: set[str],
) -> None:
    registry = load_registry()
    person = registry.people_by_id[person_id]
    assert person.primary_academic_home_type == "department"
    assert person.primary_department_id == ""
    home = registry.primary_affiliation(person_id)
    assert (home.home_type, home.home_id, home.title, home.school_id) == (
        "department",
        department_id,
        title,
        school_id,
    )
    assert (home.source_url, home.verified_at, home.basis) == (
        source_url,
        "2026-10-24",
        "display_tie_break_first_listed",
    )
    assert home.published_fields()["primary_department_id"] == department_id
    assert {
        membership.department_id
        for membership in registry.memberships
        if membership.person_id == person_id and membership.included
    } == unit_ids


@pytest.mark.parametrize(
    ("person_id", "home_id", "included_ids"),
    [
        ("person-edwardrubin-5cdd285c", "epp", {"epp", "mech"}),
        (
            "person-elibenmichael-e8530c7b",
            "stats",
            {"stats", "heinz-faculty"},
        ),
        ("person-larryawasserman-299c0ed3", "stats", {"stats", "mld"}),
        ("person-rayidghani-0861bb94", "mld", {"mld", "heinz-faculty"}),
        ("person-robertekass-a29bd712", "stats", {"stats", "mld", "ni"}),
        ("person-teddyseidenfeld-57226b15", "philosophy", {"philosophy", "stats"}),
        ("person-towe-a7ced21f", "ece", {"ece", "mse"}),
    ],
)
def test_coequal_display_tiebreaks_keep_memberships_and_publish_basis(
    person_id: str, home_id: str, included_ids: set[str]
) -> None:
    registry = load_registry()
    person = registry.people_by_id[person_id]
    department = registry.departments_by_id[home_id]
    home = registry.primary_affiliation(person_id)
    assert (home.home_type, home.home_id, home.title, home.school_id) == (
        "department",
        home_id,
        department.title,
        department.school_id,
    )
    assert (home.verified_at, home.basis) == (
        "2026-10-24",
        "display_tie_break_first_listed",
    )
    assert home.source_url.startswith("https://")
    assert person.primary_department_id == ""
    assert {
        membership.department_id
        for membership in registry.memberships
        if membership.person_id == person_id and membership.included
    } == included_ids

    with Path("registry/reviews/2026-10-24-primary-departments.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        review = next(
            row for row in csv.DictReader(handle) if row["person_id"] == person_id
        )
    assert review["review_status"] == "display_tie_break_first_listed"
    assert review["primary_department_id"] == home_id
    assert home.source_url in review["review_source_urls"].split("|")
    assert "Visual display tie-break only" in review["review_note"]
    assert "does not rank his institutional appointments" in review["review_note"]
    published = home.published_fields()
    assert published["primary_department_id"] == home_id
    assert published["primary_department_basis"] == "display_tie_break_first_listed"
    assert published["primary_academic_home_type"] == "department"
    assert published["primary_academic_home_id"] == home_id
    assert published["primary_academic_home_title"] == department.title
    assert published["primary_academic_home_school_id"] == department.school_id
    assert published["primary_academic_home_source_url"] == home.source_url
    assert published["primary_academic_home_verified_at"] == "2026-10-24"
    assert published["primary_academic_home_basis"] == (
        "display_tie_break_first_listed"
    )


@pytest.mark.parametrize(
    ("person_id", "home_id", "source_url", "basis", "included_units"),
    [
        (
            "person-fischbeck-f33a9da9",
            "epp",
            "https://engineering.cmu.edu/directory/bios/fischbeck-paul.html",
            "reviewed",
            {"epp", "sds"},
        ),
        (
            "person-jenellefeather-10f57adf",
            "psychology",
            "https://www.cmu.edu/dietrich/psychology/directory/core-training-faculty/feather-jenelle.html",
            "display_tie_break_first_listed",
            {"ni", "psychology"},
        ),
        (
            "person-markkamlet-eb7946b0",
            "sds",
            "https://www.cmu.edu/dietrich/sds/people/faculty/mark-kamlet.html",
            "display_tie_break_first_listed",
            {"cmist", "heinz-faculty", "sds"},
        ),
    ],
)
def test_three_residual_joint_display_homes_preserve_memberships(
    person_id: str,
    home_id: str,
    source_url: str,
    basis: str,
    included_units: set[str],
) -> None:
    registry = load_registry()
    person = registry.people_by_id[person_id]
    assert person.primary_department_id == ""
    home = registry.primary_affiliation(person_id)
    assert (home.home_type, home.home_id, home.school_id) == (
        "department",
        home_id,
        registry.departments_by_id[home_id].school_id,
    )
    assert (home.source_url, home.verified_at, home.basis) == (
        source_url,
        "2026-10-24",
        basis,
    )
    assert home.published_fields()["primary_department_id"] == home_id
    assert {
        membership.department_id
        for membership in registry.memberships
        if membership.person_id == person_id and membership.included
    } == included_units


def test_campus_membership_is_not_an_academic_department(tmp_path: Path) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path,
        people=[
            {"person_id": "person-campus", "display_name": "Campus Only"},
            {"person_id": "person-academic", "display_name": "Campus and Academic"},
            {"person_id": "person-africa", "display_name": "CMU-Africa Faculty"},
            {
                "person_id": "person-reviewed",
                "display_name": "Reviewed Academic",
                "primary_academic_home_type": "institute",
                "primary_academic_home_id": "iii",
                "primary_academic_home_title": "Integrated Innovation Institute",
                "primary_academic_home_source_url": "https://example.test/iii-profile",
                "primary_academic_home_verified_at": "2026-10-24",
                "primary_academic_home_basis": "reviewed",
            },
        ],
        memberships=[
            {
                "person_id": person_id,
                "department_id": "cmu-silicon-valley",
                "role": "teaching",
                "included": "true",
                "source_url": "https://example.test/campus",
                "verified_at": "2026-10-24",
            }
            for person_id in ("person-campus", "person-academic", "person-reviewed")
        ]
        + [
            {
                "person_id": "person-academic",
                "department_id": "iii",
                "role": "teaching",
                "included": "true",
                "source_url": "https://example.test/iii",
                "verified_at": "2026-10-24",
            },
            {
                "person_id": "person-reviewed",
                "department_id": "iii",
                "role": "teaching",
                "included": "true",
                "source_url": "https://example.test/iii",
                "verified_at": "2026-10-24",
            },
            {
                "person_id": "person-africa",
                "department_id": "cmu-africa",
                "role": "teaching",
                "included": "true",
                "source_url": "https://example.test/africa",
                "verified_at": "2026-10-24",
            },
            {
                "person_id": "person-africa",
                "department_id": "ece",
                "role": "courtesy",
                "included": "false",
                "source_url": "https://example.test/ece",
                "verified_at": "2026-10-24",
            },
        ],
        departments=[
            {
                "department_id": "cmu-silicon-valley",
                "title": "CMU Silicon Valley",
                "directory_url": "https://example.test/campus",
                "unit_type": "campus",
            },
            {
                "department_id": "iii",
                "title": "Integrated Innovation Institute",
                "directory_url": "https://example.test/iii",
                "unit_type": "institute",
            },
            {
                "department_id": "cmu-africa",
                "title": "CMU-Africa",
                "directory_url": "https://example.test/africa",
            },
            {
                "department_id": "ece",
                "title": "Electrical and Computer Engineering",
                "directory_url": "https://example.test/ece",
            },
        ],
    )
    registry = load_registry(people_path, memberships_path, departments_path)
    assert registry.primary_affiliation("person-campus").basis == "unresolved"
    assert registry.primary_affiliation("person-academic").home_id == "iii"
    assert registry.primary_affiliation("person-academic").home_type == "institute"
    assert registry.primary_affiliation("person-reviewed").home_id == "iii"
    assert registry.primary_affiliation("person-africa").department_id == "cmu-africa"


def test_primary_department_rejects_campus(tmp_path: Path) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-campus",
                "display_name": "Campus Only",
                "primary_department_id": "cmu-silicon-valley",
                "primary_department_source_url": "https://example.test/campus",
                "primary_department_verified_at": "2026-10-24",
                "primary_department_basis": "reviewed",
            }
        ],
        memberships=[
            {
                "person_id": "person-campus",
                "department_id": "cmu-silicon-valley",
                "role": "teaching",
                "included": "true",
                "source_url": "https://example.test/campus",
                "verified_at": "2026-10-24",
            }
        ],
        departments=[
            {
                "department_id": "cmu-silicon-valley",
                "title": "CMU Silicon Valley",
                "directory_url": "https://example.test/campus",
            }
        ],
    )
    with pytest.raises(ValueError, match="Primary department cannot be a campus"):
        load_registry(people_path, memberships_path, departments_path)


@pytest.mark.parametrize(
    ("person_id", "source_url"),
    [
        (
            "person-shaikh-2e0f9c29",
            "https://sv.cmu.edu/directory/faculty-and-researchers/bios/shaikh-karimulla.html",
        ),
        (
            "person-thomas-f531a8ff",
            "https://sv.cmu.edu/directory/faculty-and-researchers/bios/thomas-ravi.html",
        ),
    ],
)
def test_reviewed_silicon_valley_home_is_a_campus_not_a_department(
    person_id: str, source_url: str
) -> None:
    registry = load_registry()
    person = registry.people_by_id[person_id]
    home = registry.primary_affiliation(person_id)
    assert person.primary_department_id == ""
    assert (home.home_type, home.home_id, home.title, home.school_id) == (
        "campus",
        "cmu-silicon-valley",
        "CMU Silicon Valley",
        "engineering",
    )
    assert (home.source_url, home.verified_at, home.basis) == (
        source_url,
        "2026-10-24",
        "reviewed",
    )
    published = home.published_fields()
    assert published["primary_department_id"] == ""
    assert published["primary_academic_home_type"] == "campus"
    assert published["primary_academic_home_id"] == "cmu-silicon-valley"
    memberships = {
        item.department_id: item.included
        for item in registry.memberships
        if item.person_id == person_id
    }
    assert memberships == {"cmu-silicon-valley": True, "iii": False}


@pytest.mark.parametrize(
    ("campus_included", "academic_included", "message"),
    [
        (False, False, "not an eligible typed unit"),
        (True, True, "Campus display home requires no included academic unit"),
    ],
)
def test_campus_home_requires_only_an_included_campus_membership(
    tmp_path: Path, campus_included: bool, academic_included: bool, message: str
) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-campus-review",
                "display_name": "Campus Review",
                "primary_academic_home_type": "campus",
                "primary_academic_home_id": "cmu-silicon-valley",
                "primary_academic_home_title": "CMU Silicon Valley",
                "primary_academic_home_source_url": "https://example.test/campus-profile",
                "primary_academic_home_verified_at": "2026-10-24",
                "primary_academic_home_basis": "reviewed",
            }
        ],
        memberships=[
            {
                "person_id": "person-campus-review",
                "department_id": department_id,
                "role": "teaching" if included else "adjunct",
                "included": "true" if included else "false",
                "source_url": f"https://example.test/{department_id}",
                "verified_at": "2026-10-24",
            }
            for department_id, included in (
                ("cmu-silicon-valley", campus_included),
                ("iii", academic_included),
            )
        ],
        departments=[
            {
                "department_id": "cmu-silicon-valley",
                "school_id": "engineering",
                "title": "CMU Silicon Valley",
                "directory_url": "https://example.test/campus",
                "unit_type": "campus",
            },
            {
                "department_id": "iii",
                "school_id": "engineering",
                "title": "Integrated Innovation Institute",
                "directory_url": "https://example.test/iii",
                "unit_type": "institute",
            },
        ],
    )
    with pytest.raises(ValueError, match=message):
        load_registry(people_path, memberships_path, departments_path)


def test_primary_department_rejects_unsupported_membership(tmp_path: Path) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-test",
                "display_name": "Test",
                "primary_department_id": "bme",
                "primary_department_source_url": "https://example.test/profile",
                "primary_department_verified_at": "2026-10-24",
                "primary_department_basis": "reviewed",
            }
        ],
        memberships=[
            {
                "person_id": "person-test",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/ece",
                "verified_at": "2026-10-24",
            },
            {
                "person_id": "person-test",
                "department_id": "bme",
                "role": "courtesy",
                "included": "false",
                "source_url": "https://example.test/bme",
                "verified_at": "2026-10-24",
            },
        ],
    )
    with pytest.raises(ValueError, match="not an eligible membership"):
        load_registry(people_path, memberships_path, departments_path)


def test_primary_department_rejects_unspecified_decision_basis(tmp_path: Path) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-test",
                "display_name": "Test",
                "primary_department_id": "ece",
                "primary_department_source_url": "https://example.test/profile",
                "primary_department_verified_at": "2026-10-24",
                "primary_department_basis": "alphabetical",
            }
        ],
        memberships=[
            {
                "person_id": "person-test",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/ece",
                "verified_at": "2026-10-24",
            }
        ],
    )
    with pytest.raises(ValueError, match="valid basis"):
        load_registry(people_path, memberships_path, departments_path)


@pytest.mark.parametrize(
    ("primary_id", "source_url", "verified_at", "message"),
    [
        ("ece", "", "2026-10-24", "needs source and date"),
        ("", "https://example.test/profile", "", "provenance needs an ID"),
    ],
)
def test_primary_department_provenance_is_complete(
    tmp_path: Path,
    primary_id: str,
    source_url: str,
    verified_at: str,
    message: str,
) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path,
        people=[
            {
                "person_id": "person-test",
                "display_name": "Test",
                "primary_department_id": primary_id,
                "primary_department_source_url": source_url,
                "primary_department_verified_at": verified_at,
                "primary_department_basis": "reviewed" if primary_id else "",
            }
        ],
        memberships=[
            {
                "person_id": "person-test",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/ece",
                "verified_at": "2026-10-24",
            }
        ],
    )
    with pytest.raises(ValueError, match=message):
        load_registry(people_path, memberships_path, departments_path)


def test_primary_review_ledger_covers_every_ambiguous_registry_identity() -> None:
    registry = load_registry()
    ledger_path = Path("registry/reviews/2026-10-24-primary-departments.csv")
    with ledger_path.open(newline="", encoding="utf-8") as handle:
        review_rows = list(csv.DictReader(handle))
    reviews = {row["person_id"]: row for row in review_rows}
    assert len(reviews) == len(review_rows)
    needs_review = set()
    for person in registry.people:
        memberships = [
            item for item in registry.memberships if item.person_id == person.person_id
        ]
        included = {item.department_id for item in memberships if item.included}
        candidates = included or {item.department_id for item in memberships}
        primary = registry.primary_affiliation(person.person_id)
        if len(candidates) > 1 or (
            included and primary.home_type in {"unresolved", "campus"}
        ):
            needs_review.add(person.person_id)
            review = reviews[person.person_id]
            assert review["review_source_urls"].startswith("https://")
            assert review["primary_department_id"] == primary.department_id
            assert review["primary_academic_home_type"] == primary.home_type
            assert review["primary_academic_home_id"] == primary.home_id
            assert review["review_status"] == (
                person.primary_department_basis or person.primary_academic_home_basis
                if person.primary_department_id or person.primary_academic_home_id
                else "derived"
                if primary.home_id
                else "unresolved"
            )
        else:
            included_academic = [
                item
                for item in memberships
                if item.included
                and registry.departments_by_id[item.department_id].unit_type
                not in {"campus"}
            ]
            if not included_academic:
                assert primary.basis == "unresolved"
            else:
                assert primary.home_id
    assert set(reviews) == needs_review


@pytest.mark.parametrize(
    ("person_id", "home_id", "basis", "included_units"),
    [
        (
            "person-andrewilyas-9f6dac37",
            "s3d",
            "inferred_advisee_count",
            {"ece", "s3d"},
        ),
        (
            "person-chrishendrickson-c5d421c4",
            "civil",
            "inferred_profile_title",
            {"civil", "epp", "heinz-faculty"},
        ),
    ],
)
def test_cross_appointed_scholars_have_evidence_bounded_display_homes(
    person_id: str, home_id: str, basis: str, included_units: set[str]
) -> None:
    registry = load_registry()
    home = registry.primary_affiliation(person_id)
    assert (home.home_type, home.home_id, home.department_id, home.basis) == (
        "department",
        home_id,
        home_id,
        basis,
    )
    assert home.source_url.startswith("https://")
    assert {
        membership.department_id
        for membership in registry.memberships
        if membership.person_id == person_id and membership.included
    } == included_units


@pytest.mark.parametrize(
    ("schools", "department_school_id", "message"),
    [
        (
            [
                {
                    "school_id": "Bad ID",
                    "title": "Bad",
                    "directory_url": "https://example.test",
                }
            ],
            "engineering",
            "Invalid school_id",
        ),
        (
            [
                {
                    "school_id": "engineering",
                    "title": "Eng",
                    "directory_url": "https://example.test",
                },
                {
                    "school_id": "engineering",
                    "title": "Eng",
                    "directory_url": "https://example.test",
                },
            ],
            "engineering",
            "Duplicate school_id",
        ),
        (
            [
                {
                    "school_id": "engineering",
                    "title": "",
                    "directory_url": "https://example.test",
                }
            ],
            "engineering",
            "Blank school title",
        ),
        (
            [
                {
                    "school_id": "engineering",
                    "title": "Eng",
                    "directory_url": "http://example.test",
                }
            ],
            "engineering",
            "absolute HTTPS URL",
        ),
        (
            [
                {
                    "school_id": "engineering",
                    "title": "Eng",
                    "directory_url": "https://example.test",
                }
            ],
            "scs",
            "Unknown school_id",
        ),
    ],
)
def test_registry_rejects_invalid_school_hierarchy(
    tmp_path: Path,
    schools: list[dict[str, str]],
    department_school_id: str,
    message: str,
) -> None:
    paths = write_registry(
        tmp_path,
        people=[{"person_id": "person-one", "display_name": "One"}],
        memberships=[
            {
                "person_id": "person-one",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/ece",
            }
        ],
        departments=[
            {
                "department_id": "ece",
                "school_id": department_school_id,
                "title": "ECE",
                "directory_url": "https://example.test/ece",
            }
        ],
        schools=schools,
    )
    with pytest.raises(ValueError, match=message):
        load_registry(*paths)


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


def test_nadtochiy_reviewed_scholar_id_and_math_home() -> None:
    registry = load_registry()
    person = registry.people_by_id["person-sergeynadtochiy-03531018"]
    assert person.scholar_id == "cAiedtsAAAAJ"
    assert person.scholar_id_source_url == (
        "https://scholar.google.com/citations?user=cAiedtsAAAAJ"
    )
    assert person.scholar_id_verified_at == "2026-10-24"
    home = registry.primary_affiliation(person.person_id)
    assert (home.school_id, home.department_id, home.basis) == (
        "mcs",
        "math",
        "single_included_membership",
    )


def test_reviewed_heinz_tepper_scholar_homes_are_typed() -> None:
    registry = load_registry()
    with Path("registry/reviews/2026-10-24-heinz-tepper-primary-units.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 29
    for row in rows:
        person = registry.people_by_id[row["person_id"]]
        assert person.scholar_id
        home = registry.primary_affiliation(person.person_id)
        assert home.department_id == ""
        if row["display_name"] == "Pedro Ferreira":
            assert (home.home_type, home.home_id, home.basis) == (
                "constituent_school",
                "heinz-ism",
                "inferred_advisee_count",
            )
        elif row["one_or_multiple"] == "one":
            assert (home.home_type, home.home_id) == (
                "academic_area",
                row["proposed_primary_academic_home"],
            )
            assert home.title == row["official_unit_names"]
            if person.display_name not in {"Gérard P. Cornuéjols", "Nicholas Muller"}:
                assert home.source_url == row["official_unit_source_urls"]
        else:
            assert (home.home_type, home.home_id, home.basis) == (
                "school",
                "tepper",
                "school_fallback",
            )
        published = home.published_fields()
        assert published["primary_academic_home_type"] == home.home_type
        assert published["primary_academic_home_id"] == home.home_id
        assert published["primary_academic_home_title"] == home.title
        assert published["primary_academic_home_school_id"] == home.school_id
        assert published["primary_academic_home_source_url"] == home.source_url
        assert published["primary_academic_home_basis"] == home.basis
        assert published["primary_department_id"] == ""
        assert published["primary_department_basis"] == "unresolved"
    assert registry.departments_by_id["cmu-africa"].unit_type == "department"
    assert registry.departments_by_id["cmu-silicon-valley"].unit_type == "campus"


@pytest.mark.parametrize(
    ("name", "home_type", "home_id", "basis", "included_ids"),
    [
        (
            "Barbara Johnstone",
            "department",
            "english",
            "inferred_profile_title",
            {"english", "lcal"},
        ),
        (
            "Beibei Li",
            "school",
            "heinz",
            "reviewed",
            {"heinz-faculty", "tepper-faculty"},
        ),
        (
            "Brad Mahon",
            "department",
            "psychology",
            "inferred_profile_title",
            {"ni", "psychology"},
        ),
        ("Dana Scott", "department", "csd", "reviewed", {"csd", "philosophy"}),
        (
            "Daniel Nagin",
            "school",
            "heinz",
            "inferred_profile_title",
            {"heinz-faculty", "stats"},
        ),
        (
            "David Hounshell",
            "department",
            "sds",
            "inferred_profile_title",
            {"epp", "sds"},
        ),
        ("David Touretzky", "department", "csd", "reviewed", {"csd", "ni"}),
        (
            "Jamie Callan",
            "institute",
            "lti",
            "inferred_profile_title",
            {"heinz-faculty", "heinz-ism", "lti"},
        ),
        ("Jessica Cantlon", "institute", "ni", "reviewed", {"ni", "psychology"}),
        (
            "Kim J. Hyatt",
            "school",
            "heinz",
            "school_fallback",
            {"heinz-faculty", "heinz-ism", "heinz-ppm"},
        ),
        (
            "David Lamont",
            "academic_area",
            "tepper:strategy",
            "reviewed",
            {"iii", "tepper-faculty"},
        ),
        (
            "Marcel Just",
            "department",
            "psychology",
            "inferred_profile_title",
            {"bme", "psychology"},
        ),
        (
            "Marlene Behrmann",
            "department",
            "psychology",
            "inferred_profile_title",
            {"ni", "psychology"},
        ),
        ("Paul Hopper", "department", "english", "reviewed", {"english", "lcal"}),
        (
            "Stacy Rosenberg",
            "school",
            "heinz",
            "school_fallback",
            {"heinz-faculty", "heinz-ism", "heinz-ppm"},
        ),
        ("Tai-Sing Lee", "department", "csd", "reviewed", {"csd", "ni"}),
    ],
)
def test_source_backed_display_homes_preserve_memberships_and_published_types(
    name: str,
    home_type: str,
    home_id: str,
    basis: str,
    included_ids: set[str],
) -> None:
    registry = load_registry()
    person = next(person for person in registry.people if person.display_name == name)
    home = registry.primary_affiliation(person.person_id)
    assert (home.home_type, home.home_id, home.basis) == (home_type, home_id, basis)
    assert home.source_url.startswith("https://")
    assert home.verified_at == "2026-10-24"
    assert {
        item.department_id
        for item in registry.memberships
        if item.person_id == person.person_id and item.included
    } == included_ids

    published = home.published_fields()
    assert published["primary_academic_home_type"] == home_type
    assert published["primary_academic_home_id"] == home_id
    assert published["primary_academic_home_title"] == home.title
    assert published["primary_academic_home_school_id"] == home.school_id
    assert published["primary_academic_home_source_url"] == home.source_url
    assert published["primary_academic_home_verified_at"] == "2026-10-24"
    assert published["primary_academic_home_basis"] == basis
    assert published["primary_department_id"] == (
        home_id if home_type == "department" else ""
    )
    assert published["primary_department_basis"] == (
        basis if home_type == "department" else "unresolved"
    )


@pytest.mark.parametrize(
    ("name", "home_type", "home_id", "excluded_id", "role_source"),
    [
        (
            "Amelia M. Haviland",
            "school",
            "heinz",
            "stats",
            "https://www.cmu.edu/dietrich/statistics-datascience/resources/docs/history21.pdf",
        ),
        (
            "Barbara Shinn-Cunningham",
            "institute",
            "ni",
            "psychology",
            "https://www.cmu.edu/ni/people/faculty/barbara-shinn-cunningham",
        ),
        (
            "David Choi",
            "school",
            "heinz",
            "stats",
            "https://www.andrew.cmu.edu/user/davidch/choi_CV.pdf",
        ),
    ],
)
def test_courtesy_corrections_keep_recorded_memberships_and_typed_homes(
    name: str,
    home_type: str,
    home_id: str,
    excluded_id: str,
    role_source: str,
) -> None:
    registry = load_registry()
    person = next(person for person in registry.people if person.display_name == name)
    memberships = [
        item for item in registry.memberships if item.person_id == person.person_id
    ]
    corrected = next(item for item in memberships if item.department_id == excluded_id)
    assert (corrected.role, corrected.included) == ("courtesy", False)
    assert corrected.source_url == role_source
    assert corrected.verified_at == "2026-10-24"
    assert any(item.included for item in memberships)

    home = registry.primary_affiliation(person.person_id)
    assert (home.home_type, home.home_id, home.basis) == (
        home_type,
        home_id,
        "reviewed",
    )
    published = home.published_fields()
    assert published["primary_academic_home_type"] == home_type
    assert published["primary_academic_home_id"] == home_id
    assert published["primary_academic_home_title"] == home.title
    assert published["primary_academic_home_school_id"] == home.school_id
    assert published["primary_academic_home_source_url"] == home.source_url
    assert published["primary_academic_home_verified_at"] == "2026-10-24"
    assert published["primary_academic_home_basis"] == "reviewed"
    assert published["primary_department_id"] == ""
    assert published["primary_department_basis"] == "unresolved"

    with Path("registry/reviews/2026-10-24-courtesy-home-corrections.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reviews = list(csv.DictReader(handle))
    assert len(reviews) == 3
    review = next(row for row in reviews if row["person_id"] == person.person_id)
    assert review["corrected_membership_id"] == excluded_id
    assert review["role_source_url"] == role_source
    assert review["verified_at"] == "2026-10-24"


@pytest.mark.parametrize(
    ("person_id", "home_id", "included_ids", "counts"),
    [
        (
            "person-annlee-3bb27a6a",
            "stats",
            {"stats", "mld"},
            "stats=11;mld=2",
        ),
        (
            "person-davidplaut-2f20021e",
            "psychology",
            {"psychology", "ni"},
            "psychology=7;computer_science=1;neural_computation=2",
        ),
        (
            "person-sivaramanbalakrishnan-911f98b0",
            "stats",
            {"stats", "mld"},
            "stats=4;mld=2",
        ),
    ],
)
def test_advisee_count_homes_keep_memberships_and_publish_evidence(
    person_id: str, home_id: str, included_ids: set[str], counts: str
) -> None:
    registry = load_registry()
    person = registry.people_by_id[person_id]
    home = registry.primary_affiliation(person_id)
    assert (home.home_type, home.home_id, home.basis) == (
        "department",
        home_id,
        "inferred_advisee_count",
    )
    assert home.source_url.startswith("https://")
    assert home.verified_at == "2026-10-24"
    assert {
        membership.department_id
        for membership in registry.memberships
        if membership.person_id == person_id and membership.included
    } == included_ids
    with Path("registry/reviews/2026-10-24-primary-departments.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        review = next(
            row for row in csv.DictReader(handle) if row["person_id"] == person_id
        )
    assert review["phd_advisee_counts"] == counts
    assert review["review_status"] == "inferred_advisee_count"
    assert review["primary_department_id"] == home_id
    assert review["review_source_urls"] == home.source_url
    assert review["review_note"]
    published = home.published_fields()
    assert published["primary_department_id"] == home_id
    assert published["primary_department_basis"] == "inferred_advisee_count"
    assert published["primary_academic_home_type"] == "department"
    assert published["primary_academic_home_id"] == home_id
    assert published["primary_academic_home_school_id"] == "dietrich"
    assert published["primary_academic_home_source_url"] == home.source_url
    assert published["primary_academic_home_verified_at"] == "2026-10-24"
    assert published["primary_academic_home_basis"] == "inferred_advisee_count"
    assert person.primary_department_id == home_id


@pytest.mark.parametrize(
    ("person_id", "source_url", "included_ids", "recorded_ids"),
    [
        (
            "person-farnamjahanian-ec16c348",
            "https://www.csd.cmu.edu/people/faculty/farnam-jahanian",
            {"csd", "heinz-faculty", "heinz-ism"},
            {"csd", "ece", "heinz-faculty", "heinz-ism"},
        ),
        (
            "person-rogerdannenberg-73c4f6bd",
            "https://www.csd.cmu.edu/people/faculty/roger-dannenberg",
            {"csd", "music"},
            {"csd", "music"},
        ),
    ],
)
def test_csd_department_field_homes_preserve_verified_memberships(
    person_id: str,
    source_url: str,
    included_ids: set[str],
    recorded_ids: set[str],
) -> None:
    registry = load_registry()
    home = registry.primary_affiliation(person_id)
    assert (home.home_type, home.home_id, home.title, home.school_id) == (
        "department",
        "csd",
        "Computer Science Department",
        "scs",
    )
    assert (home.source_url, home.verified_at, home.basis) == (
        source_url,
        "2026-10-24",
        "reviewed",
    )
    memberships = [
        membership
        for membership in registry.memberships
        if membership.person_id == person_id
    ]
    assert {membership.department_id for membership in memberships} == recorded_ids
    assert {
        membership.department_id for membership in memberships if membership.included
    } == included_ids
    assert "art" not in recorded_ids
    if person_id == "person-farnamjahanian-ec16c348":
        ece = next(
            membership
            for membership in memberships
            if membership.department_id == "ece"
        )
        assert (ece.role, ece.included, ece.source_url, ece.verified_at) == (
            "courtesy",
            False,
            "https://www.ece.cmu.edu/directory/courtesy-faculty.html",
            "2026-10-24",
        )

    with Path("registry/reviews/2026-10-24-primary-departments.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        review = next(
            row for row in csv.DictReader(handle) if row["person_id"] == person_id
        )
    assert review["review_status"] == "reviewed"
    assert review["primary_department_id"] == "csd"
    assert review["primary_academic_home_type"] == "department"
    assert review["primary_academic_home_id"] == "csd"
    assert source_url in review["review_source_urls"].split("|")
    assert review["review_note"]
    published = home.published_fields()
    assert published["primary_department_id"] == "csd"
    assert published["primary_department_basis"] == "reviewed"
    assert published["primary_academic_home_type"] == "department"
    assert published["primary_academic_home_id"] == "csd"
    assert published["primary_academic_home_title"] == "Computer Science Department"
    assert published["primary_academic_home_school_id"] == "scs"
    assert published["primary_academic_home_source_url"] == source_url
    assert published["primary_academic_home_verified_at"] == "2026-10-24"
    assert published["primary_academic_home_basis"] == "reviewed"


@pytest.mark.parametrize(
    ("home", "unit_ids", "invalid_unit_type", "message"),
    [
        (
            {"primary_department_id": "ece"},
            ("ece", "tepper-faculty"),
            False,
            "Two explicit primary homes",
        ),
        (
            {"primary_academic_home_type": "laboratory"},
            ("tepper-faculty",),
            False,
            "Invalid primary academic home type",
        ),
        (
            {"primary_academic_home_title": ""},
            ("tepper-faculty",),
            False,
            "needs title, source and date",
        ),
        (
            {"primary_academic_home_basis": "alphabetical"},
            ("tepper-faculty",),
            False,
            "needs a valid basis",
        ),
        (
            {"primary_academic_home_id": ""},
            ("tepper-faculty",),
            False,
            "provenance needs an ID",
        ),
        (
            {
                "primary_academic_home_type": "school",
                "primary_academic_home_id": "unknown",
                "primary_academic_home_title": "Unknown",
            },
            ("ece",),
            False,
            "Unknown primary school home",
        ),
        (
            {
                "primary_academic_home_type": "school",
                "primary_academic_home_id": "tepper",
                "primary_academic_home_title": "Tepper School of Business",
            },
            ("ece",),
            False,
            "not an included membership",
        ),
        (
            {"primary_academic_home_id": "tepper:Invalid"},
            ("tepper-faculty",),
            False,
            "Invalid primary academic area",
        ),
        (
            {},
            ("ece",),
            False,
            "needs Tepper membership",
        ),
        (
            {
                "primary_academic_home_type": "institute",
                "primary_academic_home_id": "ece",
                "primary_academic_home_title": "ECE",
            },
            ("ece",),
            False,
            "not an eligible typed unit",
        ),
        (
            {
                "primary_academic_home_id": "",
                "primary_academic_home_type": "",
                "primary_academic_home_title": "",
                "primary_academic_home_source_url": "",
                "primary_academic_home_verified_at": "",
                "primary_academic_home_basis": "",
                "primary_department_id": "iii",
                "primary_department_source_url": "https://example.test/iii",
                "primary_department_verified_at": "2026-10-24",
                "primary_department_basis": "reviewed",
            },
            ("iii",),
            False,
            "must be an actual department",
        ),
        (
            {},
            ("tepper-faculty",),
            True,
            "Invalid registry unit type",
        ),
    ],
)
def test_primary_home_rejects_invalid_evidence_or_type(
    tmp_path: Path,
    home: dict[str, str],
    unit_ids: tuple[str, ...],
    invalid_unit_type: bool,
    message: str,
) -> None:
    person = {
        "person_id": "person-test",
        "display_name": "Test",
        "primary_academic_home_type": "academic_area",
        "primary_academic_home_id": "tepper:finance",
        "primary_academic_home_title": "Finance",
        "primary_academic_home_source_url": "https://example.test/area",
        "primary_academic_home_verified_at": "2026-10-24",
        "primary_academic_home_basis": "reviewed",
        **home,
    }
    departments = [
        {
            "department_id": "ece",
            "school_id": "engineering",
            "title": "ECE",
            "directory_url": "https://example.test/ece",
        },
        {
            "department_id": "iii",
            "school_id": "engineering",
            "title": "III",
            "unit_type": "institute",
            "directory_url": "https://example.test/iii",
        },
        {
            "department_id": "tepper-faculty",
            "school_id": "tepper",
            "title": "Tepper roster",
            "unit_type": "invalid" if invalid_unit_type else "roster_container",
            "directory_url": "https://example.test/tepper",
        },
    ]
    paths = write_registry(
        tmp_path,
        people=[person],
        memberships=[
            {
                "person_id": "person-test",
                "department_id": unit_id,
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/roster",
            }
            for unit_id in unit_ids
        ],
        departments=departments,
        schools=[
            {
                "school_id": "engineering",
                "title": "Engineering",
                "directory_url": "https://example.test/engineering",
            },
            {
                "school_id": "tepper",
                "title": "Tepper School of Business",
                "directory_url": "https://example.test/tepper",
            },
        ],
    )
    with pytest.raises(ValueError, match=message):
        load_registry(*paths)


@pytest.mark.parametrize(
    "person_id",
    [
        "person-deborahbrandon-e4ba69a1",
        "person-dmitrykramkov-b6fe0656",
        "person-layanelhajj-2e531a3b",
        "person-matthewrosenzweig-94bdde39",
        "person-zelealemyilma-fb547edb",
    ],
)
def test_five_verified_math_scholars_have_roster_backed_primary_home(
    person_id: str,
) -> None:
    registry = load_registry()
    person = registry.people_by_id[person_id]
    roster_url = "https://www.cmu.edu/math/people/faculty/index.html"
    assert person.primary_department_id == "math"
    assert person.primary_department_source_url == roster_url
    assert person.primary_department_verified_at == "2026-10-25"
    assert person.primary_department_basis == "reviewed"
    assert {
        membership.department_id
        for membership in registry.memberships
        if membership.person_id == person_id and membership.included
    } == {"math"}
    home = registry.primary_affiliation(person_id)
    assert (home.home_type, home.home_id, home.school_id) == (
        "department",
        "math",
        "mcs",
    )
    assert home.published_fields()["primary_department_id"] == "math"
