"""Normalized people, map, and membership registry parsing."""

from __future__ import annotations

import csv
import datetime as dt
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

PEOPLE_COLUMNS = (
    "person_id",
    "display_name",
    "scholar_id",
    "scholar_id_source_url",
    "scholar_id_verified_at",
    "orcid",
    "homepage_url",
    "notes",
)
MEMBERSHIP_COLUMNS = (
    "person_id",
    "map_slug",
    "role",
    "included",
    "legacy_label",
    "source_url",
    "verified_at",
)
MAP_COLUMNS = (
    "map_slug",
    "title",
    "directory_url",
    "reviewed_at",
    "review_notes",
)
INCLUDED_ROLES = frozenset({"faculty", "teaching", "emeritus"})
KNOWN_ROLES = INCLUDED_ROLES | {
    "adjunct",
    "affiliate",
    "courtesy",
    "former",
    "special",
    "visiting",
    "unknown",
}
MAP_SLUG_PATTERN = re.compile(r"map-of-[a-z0-9]+(?:-[a-z0-9]+)*\Z")
PERSON_ID_PATTERN = re.compile(r"person-[a-z0-9]+(?:-[a-z0-9]+)*\Z")
SCHOLAR_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+\Z")
ORCID_PATTERN = re.compile(r"(?:https://orcid\.org/)?(\d{4}-\d{4}-\d{4}-[\dX]{4})\Z")

DEFAULT_PEOPLE_PATH = Path("registry/people.csv")
DEFAULT_MEMBERSHIPS_PATH = Path("registry/memberships.csv")
DEFAULT_MAPS_PATH = Path("registry/maps.csv")


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


@dataclass(frozen=True, order=True)
class MapDefinition:
    """One published map and its authoritative annual-review source."""

    map_slug: str
    title: str
    directory_url: str
    reviewed_at: str
    review_notes: str


@dataclass(frozen=True, order=True)
class Membership:
    """One person's role in one map, including excluded historical roles."""

    person_id: str
    map_slug: str
    role: str
    included: bool
    legacy_label: str
    source_url: str
    verified_at: str


@dataclass(frozen=True)
class AuthorProfile:
    """One registered Scholar profile and all retained map memberships."""

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
    def map_slugs(self) -> tuple[str, ...]:
        return tuple(
            membership.map_slug
            for membership in self.memberships
            if membership.included
        )


@dataclass(frozen=True)
class Registry:
    """Fully validated normalized registry."""

    people: tuple[Person, ...]
    memberships: tuple[Membership, ...]
    maps: tuple[MapDefinition, ...]

    @property
    def people_by_id(self) -> dict[str, Person]:
        return {person.person_id: person for person in self.people}

    @property
    def maps_by_slug(self) -> dict[str, MapDefinition]:
        return {map_definition.map_slug: map_definition for map_definition in self.maps}

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
                    key=lambda item: (item.map_slug, item.role),
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

    def map_catalog(self) -> dict[str, str]:
        return {
            definition.map_slug: definition.title
            for definition in sorted(self.maps, key=lambda item: item.map_slug)
        }


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
    maps_path: Path = DEFAULT_MAPS_PATH,
) -> Registry:
    """Load and cross-validate all normalized registry files."""

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
        if bool(person.scholar_id_source_url) != bool(
            person.scholar_id_verified_at
        ):
            raise ValueError(
                "Scholar ID provenance requires both a source URL and verification "
                f"date: {person.person_id}"
            )
        if person.orcid and not ORCID_PATTERN.fullmatch(person.orcid):
            raise ValueError(f"Invalid ORCID for {person.person_id}: {person.orcid!r}")
        _validate_https_url(person.homepage_url, field="homepage_url")
        seen_person_ids.add(person.person_id)
        people.append(person)

    maps: list[MapDefinition] = []
    seen_map_slugs: set[str] = set()
    for line_number, row in enumerate(_read_rows(maps_path, MAP_COLUMNS), start=2):
        definition = MapDefinition(**row)
        if not MAP_SLUG_PATTERN.fullmatch(definition.map_slug):
            raise ValueError(
                f"Invalid map_slug on {maps_path}:{line_number}: "
                f"{definition.map_slug!r}"
            )
        if definition.map_slug in seen_map_slugs:
            raise ValueError(f"Duplicate map_slug: {definition.map_slug}")
        if not definition.title:
            raise ValueError(f"Blank map title on {maps_path}:{line_number}")
        _validate_https_url(
            definition.directory_url,
            field="directory_url",
            required=definition.map_slug != "map-of-eng",
        )
        _validate_date(definition.reviewed_at, field="reviewed_at")
        seen_map_slugs.add(definition.map_slug)
        maps.append(definition)

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
        if membership.map_slug not in seen_map_slugs:
            raise ValueError(f"Unknown map_slug in membership: {membership.map_slug}")
        if membership.map_slug == "map-of-eng":
            raise ValueError("map-of-eng is aggregate-only and cannot have memberships")
        if membership.role not in KNOWN_ROLES:
            raise ValueError(f"Unknown registry role: {membership.role!r}")
        expected_included = membership.role in INCLUDED_ROLES
        if membership.included != expected_included:
            raise ValueError(
                f"Role {membership.role!r} requires included={expected_included}"
            )
        identity = (membership.person_id, membership.map_slug, membership.role)
        if identity in seen_memberships:
            raise ValueError(f"Duplicate membership: {identity}")
        _validate_https_url(membership.source_url, field="source_url", required=True)
        _validate_date(membership.verified_at, field="verified_at")
        seen_memberships.add(identity)
        memberships.append(membership)

    if not people or not memberships or not maps:
        raise ValueError("Normalized registry files cannot be empty")
    return Registry(
        people=tuple(sorted(people)),
        memberships=tuple(sorted(memberships)),
        maps=tuple(sorted(maps)),
    )


def unique_profiles(
    registry: Registry, *, included_only: bool = False
) -> list[AuthorProfile]:
    """Compatibility helper for callers that operate on Scholar profiles."""

    return registry.unique_profiles(included_only=included_only)


def map_catalog(registry: Registry) -> dict[str, str]:
    """Compatibility helper returning map slugs and titles."""

    return registry.map_catalog()
