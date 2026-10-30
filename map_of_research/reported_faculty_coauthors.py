"""Validate one-sided, source-bound faculty coauthor reports for map edges."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .io import load_json, sha256_file
from .registry import Registry

FIELDS = frozenset(
    {
        "person_id",
        "source_person_id",
        "source_scholar_id",
        "source_author_pub_id",
        "source_observation_id",
        "source_work_id",
        "source_title",
        "source_authors",
        "source_url",
        "verified_at",
    }
)


def require_reported_source_sync(producer_path: Path, frozen_path: Path) -> bool:
    """Require the producer's report manifest to match the frozen source copy."""

    producer_exists = producer_path.is_file()
    frozen_exists = frozen_path.is_file()
    if producer_exists != frozen_exists or (
        producer_exists and sha256_file(producer_path) != sha256_file(frozen_path)
    ):
        raise ValueError("Reported coauthor producer and frozen source differ")
    return frozen_exists


def _source_observation_id(scholar_id: str, author_pub_id: str) -> str:
    digest = hashlib.sha256(f"{scholar_id}\0{author_pub_id}".encode()).hexdigest()
    return f"observation-{digest}"


def apply_reported_faculty_coauthors(
    works: Any,
    profile_publications: Any,
    authorships: Any,
    registry: Registry,
    *,
    source_path: Path,
    source_sha256: str,
    held_person_ids: frozenset[str] = frozenset(),
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    """Validate edge reports without changing publication or authorship tables.

    The assertion file is copied to the frozen snapshot and SHA-bound in
    its manifest. Every assertion must resolve to one released, eligible Scholar
    observation on its expected work. An unused or changed assertion is fatal.
    """

    if sha256_file(source_path) != source_sha256:
        raise ValueError("Reported coauthor source differs from snapshot manifest")
    document = load_json(source_path)
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "assertions"}
        or document.get("schema_version") != 1
    ):
        raise ValueError("Reported coauthor source is invalid")
    assertions = document["assertions"]
    if not isinstance(assertions, list):
        raise ValueError("Reported faculty coauthors must be a list")
    if not assertions:
        return {}, []

    people = registry.people_by_id
    included = {
        membership.person_id
        for membership in registry.memberships
        if membership.included
    }
    profiles: dict[tuple[str, str], dict[str, Any]] = {}
    observed_person_works: set[tuple[str, str]] = set()
    for row in profile_publications.to_dict("records"):
        key = (str(row.get("scholar_id") or ""), str(row.get("author_pub_id") or ""))
        if key in profiles:
            raise ValueError("Reported coauthor source observation is duplicated")
        profiles[key] = row
        observed_person_works.add(
            (str(row.get("work_id") or ""), str(row.get("person_id") or ""))
        )
    work_rows = {str(row["work_id"]): row for row in works.to_dict("records")}
    if len(work_rows) != len(works):
        raise ValueError("Reported coauthor work identity is duplicated")
    own_authorships = {
        (str(row["work_id"]), str(row["person_id"]))
        for row in authorships.to_dict("records")
    }
    by_work: dict[str, list[dict[str, str]]] = {}
    seen: set[tuple[str, str]] = set()
    receipt: list[dict[str, str]] = []
    for entry in assertions:
        if not isinstance(entry, dict) or set(entry) != FIELDS:
            raise ValueError("Reported faculty coauthor assertion has invalid fields")
        values = {key: entry[key] for key in FIELDS}
        if any(
            not isinstance(value, str) or not value.strip() for value in values.values()
        ):
            raise ValueError("Reported faculty coauthor assertion needs exact text")
        target_id = entry["person_id"]
        reporter_id = entry["source_person_id"]
        scholar_id = entry["source_scholar_id"]
        author_pub_id = entry["source_author_pub_id"]
        observation_id = entry["source_observation_id"]
        work_id = entry["source_work_id"]
        source_url = entry["source_url"]
        parsed = urlparse(source_url)
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry["verified_at"]) is None:
                raise ValueError("Date must be YYYY-MM-DD")
            dt.date.fromisoformat(entry["verified_at"])
        except ValueError as error:
            raise ValueError(
                "Reported faculty coauthor review date is invalid"
            ) from error
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or re.fullmatch(r"work-[0-9a-f]{64}", work_id) is None
            or author_pub_id.split(":", 1)[0] != scholar_id
            or observation_id != _source_observation_id(scholar_id, author_pub_id)
            or reporter_id == target_id
            or reporter_id not in included
            or target_id not in included
            or reporter_id not in people
            or target_id not in people
            or people[reporter_id].scholar_id != scholar_id
        ):
            raise ValueError("Reported faculty coauthor identity is invalid")
        if target_id in held_person_ids:
            raise ValueError("Reported faculty coauthor target has a publication hold")
        profile = profiles.get((scholar_id, author_pub_id))
        work = work_rows.get(work_id)
        if profile is None or work is None:
            raise ValueError("Reported faculty coauthor assertion is unused")
        try:
            raw = json.loads(str(profile["source_record_json"]))
        except (ValueError, TypeError) as error:
            raise ValueError(
                "Reported faculty coauthor raw source is invalid"
            ) from error
        if (
            profile.get("source_type") != "scholar"
            or profile.get("person_id") != reporter_id
            or profile.get("observation_id") != observation_id
            or profile.get("work_id") != work_id
            or not bool(profile.get("map_eligible"))
            or not bool(profile.get("collaboration_eligible"))
            or not bool(work.get("map_eligible"))
            or not bool(work.get("collaboration_eligible"))
            or raw.get("title") != entry["source_title"]
            or raw.get("author") != entry["source_authors"]
            or (work_id, reporter_id) not in own_authorships
        ):
            raise ValueError("Reported faculty coauthor source binding differs")
        if (work_id, target_id) in own_authorships:
            raise ValueError(
                "Reported faculty coauthor assertion is superseded by target authorship"
            )
        if (work_id, target_id) in observed_person_works:
            raise ValueError(
                "Reported faculty coauthor target already has an observation "
                "on this work; "
                "review the exclusion or remove the assertion"
            )
        surname = people[target_id].display_name.rsplit(" ", 1)[-1]
        if re.search(rf"(?i)\b{re.escape(surname)}\b", entry["source_authors"]) is None:
            raise ValueError("Reported faculty coauthor is absent from source byline")
        pair = (work_id, target_id)
        if pair in seen:
            raise ValueError("Reported faculty coauthor assertion is duplicated")
        seen.add(pair)
        detail = {
            "person_id": target_id,
            "source_person_id": reporter_id,
            "source_observation_id": observation_id,
            "source_url": source_url,
            "verified_at": entry["verified_at"],
        }
        by_work.setdefault(work_id, []).append(detail)
        receipt.append({"source_work_id": work_id, **detail})
    return {
        work_id: sorted(rows, key=lambda row: row["person_id"])
        for work_id, rows in by_work.items()
    }, sorted(receipt, key=lambda row: (row["source_work_id"], row["person_id"]))
