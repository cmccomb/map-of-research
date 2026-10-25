"""Normalized people, schools, departments, and membership registry parsing."""

from __future__ import annotations

import csv
import datetime as dt
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .io import load_json, sha256_file

PEOPLE_COLUMNS = (
    "person_id",
    "display_name",
    "scholar_id",
    "scholar_id_source_url",
    "scholar_id_verified_at",
    "orcid",
    "homepage_url",
    "notes",
    "primary_department_id",
    "primary_department_source_url",
    "primary_department_verified_at",
    "primary_department_basis",
    "primary_academic_home_type",
    "primary_academic_home_id",
    "primary_academic_home_title",
    "primary_academic_home_source_url",
    "primary_academic_home_verified_at",
    "primary_academic_home_basis",
)
MEMBERSHIP_COLUMNS = (
    "person_id",
    "department_id",
    "role",
    "included",
    "legacy_label",
    "source_url",
    "verified_at",
)
DEPARTMENT_COLUMNS = (
    "department_id",
    "school_id",
    "title",
    "directory_url",
    "reviewed_at",
    "review_notes",
    "unit_type",
)
SCHOOL_COLUMNS = (
    "school_id",
    "title",
    "directory_url",
)
INCLUDED_ROLES = frozenset({"faculty", "teaching", "emeritus"})
CAMPUS_ONLY_IDS = frozenset({"cmu-silicon-valley"})
ACADEMIC_UNIT_TYPES = frozenset({"department", "constituent_school", "institute"})
UNIT_TYPES = ACADEMIC_UNIT_TYPES | {"campus", "roster_container"}
PRIMARY_HOME_TYPES = ACADEMIC_UNIT_TYPES | {"academic_area", "campus", "school"}
EXPLICIT_PRIMARY_BASES = frozenset(
    {
        "reviewed",
        "inferred_advisee_count",
        "inferred_profile_title",
        "display_tie_break_first_listed",
        "school_fallback",
    }
)
KNOWN_ROLES = INCLUDED_ROLES | {
    "adjunct",
    "affiliate",
    "courtesy",
    "former",
    "special",
    "visiting",
    "unknown",
}
DEPARTMENT_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
SCHOOL_ID_PATTERN = DEPARTMENT_ID_PATTERN
PERSON_ID_PATTERN = re.compile(r"person-[a-z0-9]+(?:-[a-z0-9]+)*\Z")
SCHOLAR_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+\Z")
ORCID_PATTERN = re.compile(r"(?:https://orcid\.org/)?(\d{4}-\d{4}-\d{4}-[\dX]{4})\Z")

DEFAULT_PEOPLE_PATH = Path("registry/people.csv")
DEFAULT_MEMBERSHIPS_PATH = Path("registry/memberships.csv")
DEFAULT_DEPARTMENTS_PATH = Path("registry/departments.csv")
DEFAULT_SCHOOLS_PATH = Path("registry/schools.csv")
DEFAULT_IMPLICIT_PRIMARY_BASELINE_PATH = Path(
    "registry/reviews/2026-10-25-implicit-scholar-primary-baseline.json"
)
IMPLICIT_PRIMARY_BASELINE_SHA256 = (
    "6bb1ac4532a27e6b107385554aa7f70606643c76994412511828359db93bd2dd"
)


@dataclass(frozen=True, order=True)
class Person:
    """One stable person identity, whether or not a Scholar profile exists."""

    person_id: str
    display_name: str
    scholar_id: str
    scholar_id_source_url: str
    scholar_id_verified_at: str
    orcid: str
    homepage_url: str
    notes: str
    primary_department_id: str = ""
    primary_department_source_url: str = ""
    primary_department_verified_at: str = ""
    primary_department_basis: str = ""
    primary_academic_home_type: str = ""
    primary_academic_home_id: str = ""
    primary_academic_home_title: str = ""
    primary_academic_home_source_url: str = ""
    primary_academic_home_verified_at: str = ""
    primary_academic_home_basis: str = ""


@dataclass(frozen=True)
class PrimaryAffiliation:
    """A typed map display home, with its public evidence and decision basis."""

    home_type: str
    home_id: str
    title: str
    school_id: str
    source_url: str
    verified_at: str
    basis: str

    @property
    def department_id(self) -> str:
        """Legacy compatibility field, populated only for actual departments."""

        return self.home_id if self.home_type == "department" else ""

    def published_fields(self) -> dict[str, str]:
        """Expose the typed contract and department-only legacy fields."""

        return {
            "primary_academic_home_type": self.home_type,
            "primary_academic_home_id": self.home_id,
            "primary_academic_home_title": self.title,
            "primary_academic_home_school_id": self.school_id,
            "primary_academic_home_source_url": self.source_url,
            "primary_academic_home_verified_at": self.verified_at,
            "primary_academic_home_basis": self.basis,
            "primary_department_id": self.department_id,
            "primary_department_source_url": self.source_url
            if self.department_id
            else "",
            "primary_department_verified_at": self.verified_at
            if self.department_id
            else "",
            "primary_department_basis": self.basis
            if self.department_id
            else "unresolved",
        }


@dataclass(frozen=True, order=True)
class Department:
    """One registry unit and its authoritative annual-review source."""

    department_id: str
    title: str
    directory_url: str
    reviewed_at: str
    review_notes: str
    school_id: str
    unit_type: str = "department"


@dataclass(frozen=True, order=True)
class School:
    """One CMU school or college and its official public directory."""

    school_id: str
    title: str
    directory_url: str


@dataclass(frozen=True, order=True)
class Membership:
    """One person's department role, including excluded historical roles."""

    person_id: str
    department_id: str
    role: str
    included: bool
    legacy_label: str
    source_url: str
    verified_at: str


@dataclass(frozen=True)
class AuthorProfile:
    """One registered Scholar profile and all retained department memberships."""

    person: Person
    memberships: tuple[Membership, ...]

    @property
    def person_id(self) -> str:
        return self.person.person_id

    @property
    def scholar_id(self) -> str:
        return self.person.scholar_id

    @property
    def display_name(self) -> str:
        return self.person.display_name

    @property
    def department_ids(self) -> tuple[str, ...]:
        return tuple(
            membership.department_id
            for membership in self.memberships
            if membership.included
        )


@dataclass(frozen=True)
class Registry:
    """Fully validated normalized registry."""

    people: tuple[Person, ...]
    memberships: tuple[Membership, ...]
    departments: tuple[Department, ...]
    schools: tuple[School, ...]

    @property
    def people_by_id(self) -> dict[str, Person]:
        return {person.person_id: person for person in self.people}

    @property
    def departments_by_id(self) -> dict[str, Department]:
        return {department.department_id: department for department in self.departments}

    @property
    def schools_by_id(self) -> dict[str, School]:
        return {school.school_id: school for school in self.schools}

    def unique_profiles(self, *, included_only: bool = False) -> list[AuthorProfile]:
        """Return registered Scholar profiles, optionally limiting to active scope."""

        memberships_by_person: dict[str, list[Membership]] = {}
        for membership in self.memberships:
            memberships_by_person.setdefault(membership.person_id, []).append(
                membership
            )
        profiles = []
        for person in self.people:
            if not person.scholar_id:
                continue
            memberships = tuple(
                sorted(
                    memberships_by_person.get(person.person_id, []),
                    key=lambda item: (item.department_id, item.role),
                )
            )
            if included_only and not any(item.included for item in memberships):
                continue
            profiles.append(
                AuthorProfile(
                    person=person,
                    memberships=memberships,
                )
            )
        return sorted(profiles, key=lambda profile: profile.scholar_id)

    def department_catalog(self) -> dict[str, str]:
        return {
            department.department_id: department.title
            for department in sorted(
                self.departments,
                key=lambda item: item.department_id,
            )
        }

    def school_catalog(self) -> dict[str, str]:
        return {
            school.school_id: school.title
            for school in sorted(self.schools, key=lambda item: item.school_id)
        }

    def primary_affiliation(self, person_id: str) -> PrimaryAffiliation:
        """Prefer an evidence-reviewed choice; derive only unambiguous cases."""

        person = self.people_by_id[person_id]
        if person.primary_academic_home_id:
            school_id = (
                person.primary_academic_home_id
                if person.primary_academic_home_type == "school"
                else "tepper"
                if person.primary_academic_home_type == "academic_area"
                else self.departments_by_id[person.primary_academic_home_id].school_id
            )
            return PrimaryAffiliation(
                person.primary_academic_home_type,
                person.primary_academic_home_id,
                person.primary_academic_home_title,
                school_id,
                person.primary_academic_home_source_url,
                person.primary_academic_home_verified_at,
                person.primary_academic_home_basis,
            )
        if person.primary_department_id:
            department = self.departments_by_id[person.primary_department_id]
            return PrimaryAffiliation(
                department.unit_type,
                department.department_id,
                department.title,
                department.school_id,
                person.primary_department_source_url,
                person.primary_department_verified_at,
                person.primary_department_basis,
            )
        memberships = [item for item in self.memberships if item.person_id == person_id]
        included = [item for item in memberships if item.included]
        academic = {
            item.department_id
            for item in included
            if self.departments_by_id[item.department_id].unit_type
            in ACADEMIC_UNIT_TYPES
        }
        container_schools = {
            self.departments_by_id[item.department_id].school_id
            for item in included
            if self.departments_by_id[item.department_id].unit_type
            == "roster_container"
        }
        if len(academic) == 1:
            department = self.departments_by_id[next(iter(academic))]
            if container_schools - {department.school_id}:
                return PrimaryAffiliation(
                    "unresolved", "", "", "", "", "", "unresolved"
                )
            membership = max(
                (
                    item
                    for item in included
                    if item.department_id == department.department_id
                ),
                key=lambda item: (item.verified_at, item.source_url),
            )
            return PrimaryAffiliation(
                department.unit_type,
                department.department_id,
                department.title,
                department.school_id,
                membership.source_url,
                membership.verified_at,
                "single_included_membership",
            )
        if academic or len(container_schools) != 1:
            return PrimaryAffiliation("unresolved", "", "", "", "", "", "unresolved")
        school_id = next(iter(container_schools))
        membership = max(
            (
                item
                for item in included
                if self.departments_by_id[item.department_id].school_id == school_id
                and self.departments_by_id[item.department_id].unit_type
                == "roster_container"
            ),
            key=lambda item: (item.verified_at, item.source_url),
        )
        return PrimaryAffiliation(
            "school",
            school_id,
            self.schools_by_id[school_id].title,
            school_id,
            membership.source_url,
            membership.verified_at,
            "school_fallback",
        )


def _read_rows(path: Path, expected_columns: Iterable[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Registry file is missing a header: {path}")
        missing = set(expected_columns).difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return [
            {column: (row.get(column) or "").strip() for column in expected_columns}
            for row in reader
        ]


def _validate_https_url(value: str, *, field: str, required: bool = False) -> None:
    if not value:
        if required:
            raise ValueError(f"{field} cannot be blank")
        return
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{field} must be an absolute HTTPS URL: {value!r}")


def _validate_date(value: str, *, field: str, required: bool = False) -> None:
    if not value:
        if required:
            raise ValueError(f"{field} cannot be blank")
        return
    try:
        dt.date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must use YYYY-MM-DD: {value!r}") from error


def load_registry(
    people_path: Path = DEFAULT_PEOPLE_PATH,
    memberships_path: Path = DEFAULT_MEMBERSHIPS_PATH,
    departments_path: Path = DEFAULT_DEPARTMENTS_PATH,
    schools_path: Path | None = None,
) -> Registry:
    """Load and cross-validate all normalized registry files."""

    schools_path = schools_path or departments_path.with_name(DEFAULT_SCHOOLS_PATH.name)

    people: list[Person] = []
    seen_person_ids: set[str] = set()
    seen_scholar_ids: set[str] = set()
    for line_number, row in enumerate(
        _read_rows(people_path, PEOPLE_COLUMNS),
        start=2,
    ):
        person = Person(**row)
        if not PERSON_ID_PATTERN.fullmatch(person.person_id):
            raise ValueError(
                f"Invalid person_id on {people_path}:{line_number}: "
                f"{person.person_id!r}"
            )
        if person.person_id in seen_person_ids:
            raise ValueError(f"Duplicate person_id: {person.person_id}")
        if not person.display_name:
            raise ValueError(f"Blank display_name on {people_path}:{line_number}")
        if person.scholar_id:
            if not SCHOLAR_ID_PATTERN.fullmatch(person.scholar_id):
                raise ValueError(f"Invalid Scholar ID: {person.scholar_id!r}")
            if person.scholar_id in seen_scholar_ids:
                raise ValueError(
                    f"Scholar ID belongs to multiple people: {person.scholar_id}"
                )
            seen_scholar_ids.add(person.scholar_id)
        elif person.scholar_id_source_url or person.scholar_id_verified_at:
            raise ValueError(
                f"Scholar ID provenance requires a Scholar ID: {person.person_id}"
            )
        _validate_https_url(
            person.scholar_id_source_url,
            field="scholar_id_source_url",
        )
        _validate_date(
            person.scholar_id_verified_at,
            field="scholar_id_verified_at",
        )
        if bool(person.scholar_id_source_url) != bool(person.scholar_id_verified_at):
            raise ValueError(
                "Scholar ID provenance requires both a source URL and verification "
                f"date: {person.person_id}"
            )
        if person.orcid and not ORCID_PATTERN.fullmatch(person.orcid):
            raise ValueError(f"Invalid ORCID for {person.person_id}: {person.orcid!r}")
        _validate_https_url(person.homepage_url, field="homepage_url")
        _validate_https_url(
            person.primary_department_source_url,
            field="primary_department_source_url",
        )
        _validate_date(
            person.primary_department_verified_at,
            field="primary_department_verified_at",
        )
        _validate_https_url(
            person.primary_academic_home_source_url,
            field="primary_academic_home_source_url",
        )
        _validate_date(
            person.primary_academic_home_verified_at,
            field="primary_academic_home_verified_at",
        )
        if person.primary_academic_home_id:
            if person.primary_department_id:
                raise ValueError(f"Two explicit primary homes for {person.person_id}")
            if person.primary_academic_home_type not in PRIMARY_HOME_TYPES:
                raise ValueError(
                    f"Invalid primary academic home type for {person.person_id}"
                )
            if not (
                person.primary_academic_home_title
                and person.primary_academic_home_source_url
                and person.primary_academic_home_verified_at
            ):
                raise ValueError(
                    "Primary academic home needs title, source and date: "
                    f"{person.person_id}"
                )
            if person.primary_academic_home_basis not in EXPLICIT_PRIMARY_BASES:
                raise ValueError(
                    f"Primary academic home needs a valid basis: {person.person_id}"
                )
        elif any(
            (
                person.primary_academic_home_type,
                person.primary_academic_home_title,
                person.primary_academic_home_source_url,
                person.primary_academic_home_verified_at,
                person.primary_academic_home_basis,
            )
        ):
            raise ValueError(
                f"Primary academic home provenance needs an ID: {person.person_id}"
            )
        if person.primary_department_id:
            if person.primary_department_id in CAMPUS_ONLY_IDS:
                raise ValueError(
                    f"Primary department cannot be a campus: {person.person_id}"
                )
            if not (
                person.primary_department_source_url
                and person.primary_department_verified_at
            ):
                raise ValueError(
                    f"Primary department needs source and date: {person.person_id}"
                )
            if person.primary_department_basis not in EXPLICIT_PRIMARY_BASES:
                raise ValueError(
                    f"Primary department needs a valid basis: {person.person_id}"
                )
        elif (
            person.primary_department_source_url
            or person.primary_department_verified_at
            or person.primary_department_basis
        ):
            raise ValueError(
                f"Primary department provenance needs an ID: {person.person_id}"
            )
        seen_person_ids.add(person.person_id)
        people.append(person)

    schools: list[School] = []
    seen_school_ids: set[str] = set()
    for line_number, row in enumerate(
        _read_rows(schools_path, SCHOOL_COLUMNS),
        start=2,
    ):
        school = School(**row)
        if not SCHOOL_ID_PATTERN.fullmatch(school.school_id):
            raise ValueError(
                f"Invalid school_id on {schools_path}:{line_number}: "
                f"{school.school_id!r}"
            )
        if school.school_id in seen_school_ids:
            raise ValueError(f"Duplicate school_id: {school.school_id}")
        if not school.title:
            raise ValueError(f"Blank school title on {schools_path}:{line_number}")
        _validate_https_url(
            school.directory_url,
            field="directory_url",
            required=True,
        )
        seen_school_ids.add(school.school_id)
        schools.append(school)

    departments: list[Department] = []
    seen_department_ids: set[str] = set()
    for line_number, row in enumerate(
        _read_rows(departments_path, DEPARTMENT_COLUMNS),
        start=2,
    ):
        department = Department(**row)
        if not DEPARTMENT_ID_PATTERN.fullmatch(department.department_id):
            raise ValueError(
                f"Invalid department_id on {departments_path}:{line_number}: "
                f"{department.department_id!r}"
            )
        if department.department_id in seen_department_ids:
            raise ValueError(f"Duplicate department_id: {department.department_id}")
        if department.school_id not in seen_school_ids:
            raise ValueError(
                f"Unknown school_id for {department.department_id}: "
                f"{department.school_id}"
            )
        if not department.title:
            raise ValueError(
                f"Blank department title on {departments_path}:{line_number}"
            )
        if department.unit_type not in UNIT_TYPES:
            raise ValueError(
                f"Invalid registry unit type for {department.department_id}: "
                f"{department.unit_type!r}"
            )
        _validate_https_url(
            department.directory_url,
            field="directory_url",
            required=True,
        )
        _validate_date(department.reviewed_at, field="reviewed_at")
        seen_department_ids.add(department.department_id)
        departments.append(department)

    memberships: list[Membership] = []
    seen_memberships: set[tuple[str, str, str]] = set()
    for line_number, row in enumerate(
        _read_rows(memberships_path, MEMBERSHIP_COLUMNS),
        start=2,
    ):
        included_text = row.pop("included").casefold()
        if included_text not in {"true", "false"}:
            raise ValueError(
                f"included must be true or false on {memberships_path}:{line_number}"
            )
        membership = Membership(included=included_text == "true", **row)
        if membership.person_id not in seen_person_ids:
            raise ValueError(f"Unknown person_id in membership: {membership.person_id}")
        if membership.department_id not in seen_department_ids:
            raise ValueError(
                f"Unknown department_id in membership: {membership.department_id}"
            )
        if membership.role not in KNOWN_ROLES:
            raise ValueError(f"Unknown registry role: {membership.role!r}")
        expected_included = membership.role in INCLUDED_ROLES
        if membership.included != expected_included:
            raise ValueError(
                f"Role {membership.role!r} requires included={expected_included}"
            )
        identity = (
            membership.person_id,
            membership.department_id,
            membership.role,
        )
        if identity in seen_memberships:
            raise ValueError(f"Duplicate membership: {identity}")
        _validate_https_url(membership.source_url, field="source_url", required=True)
        _validate_date(membership.verified_at, field="verified_at")
        seen_memberships.add(identity)
        memberships.append(membership)

    if not people or not memberships or not departments:
        raise ValueError("Normalized registry files cannot be empty")
    memberships_by_person: dict[str, list[Membership]] = {}
    departments_by_id = {item.department_id: item for item in departments}
    schools_by_id = {item.school_id: item for item in schools}
    for membership in memberships:
        memberships_by_person.setdefault(membership.person_id, []).append(membership)
    for person in people:
        options = memberships_by_person.get(person.person_id, [])
        valid = {
            item.department_id
            for item in options
            if item.included
            and departments_by_id[item.department_id].unit_type
            in ACADEMIC_UNIT_TYPES | {"campus"}
        }
        if person.primary_department_id:
            department = departments_by_id.get(person.primary_department_id)
            if department is None or department.unit_type != "department":
                raise ValueError(
                    "Primary department must be an actual department: "
                    f"{person.person_id}"
                )
            if person.primary_department_id not in valid:
                raise ValueError(
                    "Primary department is not an eligible membership: "
                    f"{person.person_id}"
                )
        if not person.primary_academic_home_id:
            continue
        home_type = person.primary_academic_home_type
        home_id = person.primary_academic_home_id
        if home_type == "school":
            school = schools_by_id.get(home_id)
            if school is None or school.title != person.primary_academic_home_title:
                raise ValueError(f"Unknown primary school home: {person.person_id}")
            if not any(
                item.included
                and departments_by_id[item.department_id].school_id == home_id
                and departments_by_id[item.department_id].unit_type
                == "roster_container"
                for item in options
            ):
                raise ValueError(
                    f"Primary school is not an included membership: {person.person_id}"
                )
        elif home_type == "academic_area":
            if not home_id.startswith("tepper:") or not DEPARTMENT_ID_PATTERN.fullmatch(
                home_id.removeprefix("tepper:")
            ):
                raise ValueError(f"Invalid primary academic area: {person.person_id}")
            if not any(
                item.included and item.department_id == "tepper-faculty"
                for item in options
            ):
                raise ValueError(
                    f"Primary academic area needs Tepper membership: {person.person_id}"
                )
        else:
            department = departments_by_id.get(home_id)
            if (
                department is None
                or department.unit_type != home_type
                or department.title != person.primary_academic_home_title
                or home_id not in valid
            ):
                raise ValueError(
                    "Primary academic home is not an eligible typed unit: "
                    f"{person.person_id}"
                )
            if home_type == "campus" and any(
                item.included
                and departments_by_id[item.department_id].unit_type
                in ACADEMIC_UNIT_TYPES
                for item in options
            ):
                raise ValueError(
                    "Campus display home requires no included academic unit: "
                    f"{person.person_id}"
                )
    return Registry(
        people=tuple(sorted(people)),
        memberships=tuple(sorted(memberships)),
        departments=tuple(sorted(departments)),
        schools=tuple(sorted(schools)),
    )


def unique_profiles(
    registry: Registry, *, included_only: bool = False
) -> list[AuthorProfile]:
    """Compatibility helper for callers that operate on Scholar profiles."""

    return registry.unique_profiles(included_only=included_only)


def load_implicit_primary_baseline(path: Path) -> dict[str, str]:
    """Load the immutable exception list for profiles existing before this gate."""

    if sha256_file(path) != IMPLICIT_PRIMARY_BASELINE_SHA256:
        raise ValueError("Implicit Scholar primary-home baseline was changed")
    return load_json(path)["profiles"]


def implicit_primary_baseline_for_people(
    people_path: Path, baseline_path: Path | None = None
) -> dict[str, str]:
    """Use a colocated baseline when present; custom registries start empty."""

    path = (
        baseline_path
        if baseline_path is not None
        else people_path.parent
        / "reviews"
        / DEFAULT_IMPLICIT_PRIMARY_BASELINE_PATH.name
    )
    if baseline_path is None and not path.is_file():
        return {}
    return load_implicit_primary_baseline(path)


def validate_included_scholar_primary_homes(
    registry: Registry, *, legacy_implicit_primary: Mapping[str, str]
) -> int:
    """Require one evidenced display home for every included Scholar profile.

    The department compatibility field is required only when the selected home
    is an actual department. Other formal CMU home types retain their typed ID.
    """

    profiles = registry.unique_profiles(included_only=True)
    invalid: list[str] = []
    for profile in profiles:
        home = registry.primary_affiliation(profile.person_id)
        published = home.published_fields()
        included_departments = {
            membership.department_id
            for membership in profile.memberships
            if membership.included
            and registry.departments_by_id[membership.department_id].unit_type
            == "department"
        }
        if (
            home.home_type not in PRIMARY_HOME_TYPES
            or not all(
                (
                    home.home_id,
                    home.title,
                    home.source_url,
                    home.verified_at,
                    home.basis,
                )
            )
            or home.basis == "unresolved"
            or (
                home.home_type == "department"
                and published["primary_department_id"] not in included_departments
            )
            or (home.home_type == "school" and included_departments)
            or (home.home_type != "department" and published["primary_department_id"])
        ):
            invalid.append(f"{profile.person_id} ({profile.display_name})")
        elif (
            not profile.person.primary_department_id
            and not profile.person.primary_academic_home_id
            and legacy_implicit_primary.get(profile.person_id) != profile.scholar_id
        ):
            invalid.append(
                f"{profile.person_id} ({profile.display_name}; "
                "new Scholar ID needs an explicit primary home)"
            )
    if invalid:
        raise ValueError(
            "Included Scholar profiles need one evidenced primary academic home: "
            + ", ".join(invalid)
        )
    return len(profiles)


def department_catalog(registry: Registry) -> dict[str, str]:
    """Return stable department IDs and display titles."""

    return registry.department_catalog()
