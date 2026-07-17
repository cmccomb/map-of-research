"""Canonical faculty registry parsing and validation."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REGISTRY_COLUMNS = ("map_slug", "department", "faculty", "scholar_id")
MAP_SLUG_PATTERN = re.compile(r"map-of-[a-z0-9]+(?:-[a-z0-9]+)*\Z")
SCHOLAR_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+\Z")


@dataclass(frozen=True, order=True)
class Membership:
    """One person's label and membership in a rendered map."""

    map_slug: str
    department: str
    faculty: str
    scholar_id: str


@dataclass(frozen=True)
class AuthorProfile:
    """One unique Scholar profile and all of its map memberships."""

    scholar_id: str
    display_name: str
    memberships: tuple[Membership, ...]

    @property
    def map_slugs(self) -> tuple[str, ...]:
        return tuple(membership.map_slug for membership in self.memberships)


def _display_name(names: Iterable[str]) -> str:
    """Choose a stable readable label from site-specific aliases."""

    counts = Counter(names)
    return min(
        counts,
        key=lambda name: (
            -counts[name],
            -name.count(" "),
            len(name),
            name.casefold(),
        ),
    )


def load_registry(path: Path) -> list[Membership]:
    """Load and strictly validate the canonical membership registry."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Registry is missing a header")
        missing_columns = set(REGISTRY_COLUMNS).difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"Registry is missing columns: {sorted(missing_columns)}")

        memberships: list[Membership] = []
        seen_rows: set[tuple[str, str, str]] = set()
        map_departments: dict[str, str] = {}
        for line_number, row in enumerate(reader, start=2):
            membership = Membership(
                map_slug=(row.get("map_slug") or "").strip(),
                department=(row.get("department") or "").strip(),
                faculty=(row.get("faculty") or "").strip(),
                scholar_id=(row.get("scholar_id") or "").strip(),
            )
            if not MAP_SLUG_PATTERN.fullmatch(membership.map_slug):
                raise ValueError(
                    f"Invalid map_slug on registry line {line_number}: "
                    f"{membership.map_slug!r}"
                )
            if not membership.department or not membership.faculty:
                raise ValueError(
                    "Department and faculty are required on registry line "
                    f"{line_number}"
                )
            if any(separator in membership.faculty for separator in ("/", "\\", "\0")):
                raise ValueError(f"Unsafe faculty label on registry line {line_number}")
            if membership.scholar_id and not SCHOLAR_ID_PATTERN.fullmatch(
                membership.scholar_id
            ):
                raise ValueError(f"Invalid Scholar ID on registry line {line_number}")

            prior_department = map_departments.setdefault(
                membership.map_slug,
                membership.department,
            )
            if prior_department != membership.department:
                raise ValueError(
                    f"Map {membership.map_slug} has conflicting department labels"
                )

            identity = (
                membership.map_slug,
                membership.scholar_id,
                membership.faculty.casefold(),
            )
            if identity in seen_rows:
                raise ValueError(
                    f"Duplicate registry membership on line {line_number}: {identity}"
                )
            seen_rows.add(identity)
            memberships.append(membership)

    if not memberships:
        raise ValueError("Registry contains no memberships")
    return memberships


def unique_profiles(memberships: Iterable[Membership]) -> list[AuthorProfile]:
    """Collapse memberships to one stable record per non-empty Scholar ID."""

    grouped: dict[str, list[Membership]] = {}
    for membership in memberships:
        if membership.scholar_id:
            grouped.setdefault(membership.scholar_id, []).append(membership)

    profiles = []
    for scholar_id, profile_memberships in grouped.items():
        ordered_memberships = tuple(
            sorted(
                profile_memberships,
                key=lambda item: (
                    item.map_slug,
                    item.faculty.casefold(),
                ),
            )
        )
        profiles.append(
            AuthorProfile(
                scholar_id=scholar_id,
                display_name=_display_name(
                    membership.faculty for membership in ordered_memberships
                ),
                memberships=ordered_memberships,
            )
        )
    return sorted(profiles, key=lambda profile: profile.scholar_id)


def map_catalog(memberships: Iterable[Membership]) -> dict[str, str]:
    """Return the stable map-slug to display-name mapping."""

    return dict(
        sorted(
            {
                membership.map_slug: membership.department for membership in memberships
            }.items()
        )
    )
