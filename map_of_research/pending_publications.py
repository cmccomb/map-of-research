"""Publish reviewed works without pretending to know their map placement.

The list is intentionally nonspatial. A pending work is a selected public
observation with a separately reviewed placement hold, not an unselected
Scholar row or a point waiting at an invented coordinate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

import pandas

from .public_subsets import ids_sha256
from .registry import PRIMARY_HOME_TYPES, Registry

PENDING_MAP_SCHEMA_VERSION = 8
PENDING_REASON = "pending_placement_review"


def build_pending_publications(
    works: Any,
    profiles: Any,
    release_receipt: Sequence[Mapping[str, Any]],
    registry: Registry,
    held_work_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Return exact selected pending rows or fail before a Hub write.

    ``held_work_ids`` comes from the exact-parent placement review. Every
    resulting row must be represented by selected public profile rows and by
    an included faculty member with a source-backed typed primary home.
    """

    held = [str(value) for value in held_work_ids]
    if len(held) != len(set(held)) or any(not value for value in held):
        raise ValueError("Pending work IDs are missing or duplicated")
    if not held:
        return []
    required_works = {
        "work_id",
        "map_eligible",
        "map_exclusion_reasons",
        "collaboration_eligible",
        "memberships",
        "author_pub_ids",
        "source_urls",
        "title",
        "year",
        "x",
        "y",
        "tsne_x",
        "tsne_y",
    }
    required_profiles = {
        "work_id",
        "person_id",
        "scholar_id",
        "author_pub_id",
        "source_url",
        "collaboration_eligible",
        "map_eligible",
        "map_exclusion_reasons",
    }
    if not required_works <= set(works.columns) or not required_profiles <= set(
        profiles.columns
    ):
        raise ValueError("Pending publications lack required public table columns")
    work_by_id = {str(row.work_id): row for row in works.itertuples(index=False)}
    if len(work_by_id) != len(works) or not set(held) <= work_by_id.keys():
        raise ValueError("Pending review refers to missing or duplicate works")

    selected_owner: dict[str, tuple[str, str]] = {}
    for decision in release_receipt:
        if decision.get("cache_scope") != "reviewed_public_subset":
            continue
        ids = decision.get("selected_author_pub_ids")
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError("Pending subset receipt has invalid selected IDs")
        if ids_sha256(set(ids)) != decision.get("selected_ids_sha256"):
            raise ValueError("Pending subset receipt selected digest changed")
        owner = (
            str(decision.get("person_id") or ""),
            str(decision.get("scholar_id") or ""),
        )
        if not all(owner):
            raise ValueError("Pending subset receipt lacks owner identity")
        for author_pub_id in ids:
            if author_pub_id in selected_owner:
                raise ValueError("Pending selected citation repeats across receipts")
            selected_owner[author_pub_id] = owner
    if not selected_owner:
        raise ValueError("Pending publications need reviewed selected subsets")

    profile_by_work: dict[str, list[Any]] = {}
    for row in profiles.itertuples(index=False):
        work_id = str(row.work_id)
        if work_id in held:
            profile_by_work.setdefault(work_id, []).append(row)
    included_memberships = {
        (item.person_id, item.department_id)
        for item in registry.memberships
        if item.included
    }
    rows: list[dict[str, Any]] = []
    for work_id in sorted(held):
        work = work_by_id[work_id]
        reasons = work.map_exclusion_reasons
        if (
            type(work.map_eligible) is not bool
            or work.map_eligible
            or not isinstance(reasons, list)
            or reasons != [PENDING_REASON]
        ):
            raise ValueError(
                f"Pending work lacks an exclusive placement hold: {work_id}"
            )
        if hasattr(work, "has_included_source_profile") and not bool(
            work.has_included_source_profile
        ):
            raise ValueError(
                f"Pending work lacks an included source profile: {work_id}"
            )
        if any(
            not bool(pandas.isna(getattr(work, field)))
            for field in ("x", "y", "tsne_x", "tsne_y")
        ):
            raise ValueError(f"Pending work has unreviewed coordinates: {work_id}")
        memberships = work.memberships
        if not isinstance(memberships, list) or not memberships:
            raise ValueError(f"Pending work lacks included faculty: {work_id}")
        faculty_ids: set[str] = set()
        for membership in memberships:
            if not isinstance(membership, Mapping):
                raise ValueError(
                    f"Pending work has invalid faculty membership: {work_id}"
                )
            person_id = str(membership.get("person_id") or "")
            department_id = str(membership.get("department_id") or "")
            if (person_id, department_id) not in included_memberships:
                raise ValueError(
                    f"Pending work has excluded faculty membership: {work_id}"
                )
            home = registry.primary_affiliation(person_id)
            if (
                home.home_type not in PRIMARY_HOME_TYPES
                or not home.home_id
                or not home.source_url
                or not home.verified_at
                or not home.basis
            ):
                raise ValueError(
                    f"Pending faculty lacks evidenced typed home: {person_id}"
                )
            faculty_ids.add(person_id)
        public_rows = profile_by_work.get(work_id, [])
        collaboration_eligible = work.collaboration_eligible
        if type(collaboration_eligible) is not bool or any(
            type(row.collaboration_eligible) is not bool
            or row.collaboration_eligible is not collaboration_eligible
            for row in public_rows
        ):
            raise ValueError(
                "Pending work collaboration eligibility disagrees with public "
                f"rows: {work_id}"
            )
        public_ids = [str(row.author_pub_id) for row in public_rows]
        source_ids = work.author_pub_ids
        if (
            not public_ids
            or len(public_ids) != len(set(public_ids))
            or not isinstance(source_ids, list)
            or len(source_ids) != len(set(source_ids))
            or set(source_ids) != set(public_ids)
        ):
            raise ValueError(f"Pending work lacks exact public source rows: {work_id}")
        for row in public_rows:
            if (
                type(row.map_eligible) is not bool
                or row.map_eligible is not True
                or not isinstance(row.map_exclusion_reasons, list)
                or row.map_exclusion_reasons
            ):
                raise ValueError(
                    f"Pending work has source-ineligible public row: {work_id}"
                )
            expected = selected_owner.get(str(row.author_pub_id))
            if expected != (str(row.person_id), str(row.scholar_id)):
                raise ValueError(
                    f"Pending work contains an unselected source row: {work_id}"
                )
            if str(row.person_id) not in faculty_ids:
                raise ValueError(
                    f"Pending source owner lacks included membership: {work_id}"
                )
        source_urls = work.source_urls
        if not isinstance(source_urls, list) or not source_urls:
            raise ValueError(f"Pending work lacks a public source URL: {work_id}")
        source_url = str(source_urls[0])
        parsed = urlparse(source_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or source_url not in {str(row.source_url) for row in public_rows}
        ):
            raise ValueError(
                f"Pending work source URL is not selected HTTPS: {work_id}"
            )
        title = str(work.title or "").strip()
        if not title:
            raise ValueError(f"Pending work lacks title: {work_id}")
        if pandas.isna(work.year):
            year = None
        else:
            try:
                numeric_year = float(work.year)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Pending work has invalid year: {work_id}") from error
            if (
                not math.isfinite(numeric_year)
                or not numeric_year.is_integer()
                or not 1800 <= numeric_year <= 2200
            ):
                raise ValueError(f"Pending work has invalid year: {work_id}")
            year = int(numeric_year)
        rows.append(
            {
                "work_id": work_id,
                "title": title,
                "year": year,
                "faculty_ids": sorted(faculty_ids),
                "source_url": source_url,
                "release_decision": "selected",
                "map_hold_reason": "pending_topic_review",
                "collaboration_eligible": collaboration_eligible,
            }
        )
    return rows
