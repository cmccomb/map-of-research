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

from .composite_region import BLANK_TOPICS, COORDINATE_FIELDS
from .public_subsets import ids_sha256
from .region_continuity import WORK_FINGERPRINT_FIELDS
from .registry import PRIMARY_HOME_TYPES, Registry

PENDING_MAP_SCHEMA_VERSION = 8
PENDING_REASON = "pending_placement_review"


def carry_forward_parent_pending_works(
    works: Any,
    profiles: Any,
    prior_works: Sequence[Mapping[str, Any]],
    prior_profiles: Sequence[Mapping[str, Any]],
) -> tuple[Any, list[str]]:
    """Keep exact parent pending works nonspatial before the region diff gate.

    A later additive release may add different selected works. Rebuilding the
    corpus must not silently promote an earlier pending work into the map.
    This applies only to the parent's explicit placement holds, and rejects
    changed source bindings or bibliographic identity instead of hiding them.
    """

    prior_by_id: dict[str, Mapping[str, Any]] = {}
    pending: dict[str, Mapping[str, Any]] = {}
    for row in prior_works:
        work_id = str(row.get("work_id") or "")
        if not work_id or work_id in prior_by_id:
            raise ValueError("Parent pending carryforward has duplicate work IDs")
        prior_by_id[work_id] = row
        if "map_exclusion_reasons" not in row:
            raise ValueError("Parent pending carryforward lacks exclusion reasons")
        reasons = row["map_exclusion_reasons"]
        if not isinstance(reasons, list):
            raise ValueError("Parent pending exclusion reasons are malformed")
        if PENDING_REASON in reasons:
            if reasons != [PENDING_REASON] or row.get("map_eligible") is not False:
                raise ValueError(
                    f"Parent pending work has mixed exclusion state: {work_id}"
                )
            if row.get("map_region_outlier") is not False:
                raise ValueError(
                    f"Parent pending work has a region-outlier state: {work_id}"
                )
            if any(
                not bool(pandas.isna(row.get(field))) for field in COORDINATE_FIELDS
            ) or any(row.get(field) != blank for field, blank in BLANK_TOPICS.items()):
                raise ValueError(
                    f"Parent pending work has spatial presentation: {work_id}"
                )
            pending[work_id] = row
    if not pending:
        return works, []
    required = {
        "work_id",
        "map_eligible",
        "map_region_outlier",
        "map_exclusion_reasons",
        *COORDINATE_FIELDS,
        *BLANK_TOPICS,
    }
    if not required <= set(works.columns) or "work_id" not in profiles.columns:
        raise ValueError("Parent pending carryforward lacks public table fields")
    current_ids = works["work_id"].astype(str)
    if current_ids.duplicated().any() or not pending.keys() <= set(current_ids):
        raise ValueError("Additive candidate drops or duplicates parent pending work")

    def bindings(
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, set[tuple[str, str, str]]]:
        result: dict[str, set[tuple[str, str, str]]] = {}
        for row in rows:
            work_id = str(row.get("work_id") or "")
            if work_id in pending:
                identity = tuple(
                    str(row.get(field) or "")
                    for field in ("person_id", "scholar_id", "author_pub_id")
                )
                if not all(identity):
                    raise ValueError(f"Parent pending binding is incomplete: {work_id}")
                bucket = result.setdefault(work_id, set())
                if identity in bucket:
                    raise ValueError(f"Parent pending binding repeats: {work_id}")
                bucket.add(identity)
        return result

    old_bindings = bindings(prior_profiles)
    new_bindings = bindings(profiles.to_dict("records"))
    if old_bindings.keys() != pending.keys() or new_bindings != old_bindings:
        raise ValueError("Parent pending source bindings changed")

    output = works.copy()
    fingerprint = (
        *WORK_FINGERPRINT_FIELDS,
        "department_ids",
        "quality_assessment_version",
    )
    for index, work_id in current_ids.items():
        if work_id not in pending:
            continue
        prior = pending[work_id]
        current = output.loc[index]
        if any(
            field not in prior or field not in output.columns for field in fingerprint
        ):
            raise ValueError(f"Parent pending fingerprint is incomplete: {work_id}")
        for field in fingerprint:
            old = prior[field]
            new = current[field]
            if hasattr(new, "tolist") and isinstance(new.tolist(), list):
                new = new.tolist()
            if isinstance(old, list) and isinstance(new, list):
                same = old == new
            elif pandas.isna(old):
                same = bool(pandas.isna(new))
            else:
                same = old == new
            if not same:
                raise ValueError(
                    f"Parent pending fingerprint changed: {work_id}:{field}"
                )
        current_reasons = current["map_exclusion_reasons"]
        if not isinstance(current_reasons, list) or not (
            (bool(current["map_eligible"]) and current_reasons == [])
            or (
                not bool(current["map_eligible"])
                and current_reasons == [PENDING_REASON]
            )
        ):
            raise ValueError(
                f"Parent pending candidate has a separate exclusion: {work_id}"
            )
        output.at[index, "map_eligible"] = False
        output.at[index, "map_region_outlier"] = False
        output.at[index, "map_exclusion_reasons"] = [PENDING_REASON]
        for field in COORDINATE_FIELDS:
            output.at[index, field] = None
        for field, blank in BLANK_TOPICS.items():
            output.at[index, field] = blank
    return output, sorted(pending)


def require_parent_pending_list_preserved(
    parent: Mapping[str, Any],
    current: Sequence[Mapping[str, Any]],
    parent_pending_ids: Sequence[str],
) -> None:
    """Reject a changed or missing parent schema8 pending-list row."""

    schema = parent.get("schema_version", 7)
    if type(schema) is not int or schema < 7:
        raise ValueError("Parent pending map schema is invalid")
    if schema >= PENDING_MAP_SCHEMA_VERSION and not {
        "pending_publications",
        "pending_publication_count",
    } <= set(parent):
        raise ValueError("Parent pending list is missing")
    prior = parent.get("pending_publications", [])
    count = parent.get(
        "pending_publication_count", len(prior) if isinstance(prior, list) else None
    )
    if not isinstance(prior, list) or type(count) is not int or count != len(prior):
        raise ValueError("Parent pending list is missing")
    previous_by_id = {str(row.get("work_id") or ""): row for row in prior}
    current_by_id = {str(row.get("work_id") or ""): row for row in current}
    if (
        len(previous_by_id) != len(prior)
        or len(current_by_id) != len(current)
        or set(previous_by_id) != set(parent_pending_ids)
        or any(
            current_by_id.get(work_id) != previous_by_id[work_id]
            for work_id in previous_by_id
        )
    ):
        raise ValueError("Additive release changes a parent pending-list row")


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
