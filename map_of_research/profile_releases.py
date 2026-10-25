"""Keep complete but unreleased Scholar observations out of public dataset rows."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .attribution import AttributionReviews
from .io import load_json, sha256_file
from .registry import Registry

DEFAULT_PROFILE_RELEASES_PATH = Path("data/curated/scholar-profile-releases.json")
PROFILE_RELEASE_SCHEMA_VERSION = 1
RELEASE_STATES = frozenset({"held", "released", "reviewed_subset"})


@dataclass(frozen=True)
class ProfileRelease:
    person_id: str
    scholar_id: str
    state: str
    expected_source_rows: int
    cache_sha256: str
    reason: str
    reviewed_at: str
    released_author_pub_ids: tuple[str, ...] = ()
    excluded_author_pub_ids: tuple[str, ...] = ()
    held_author_pub_ids: tuple[str, ...] = ()


def _row_ids(entry: dict[str, Any], field: str, scholar_id: str) -> tuple[str, ...]:
    value = entry.get(field)
    if (
        not isinstance(value, list)
        or any(
            not isinstance(item, str) or not item.startswith(f"{scholar_id}:")
            for item in value
        )
        or len(value) != len(set(value))
    ):
        raise ValueError(f"Profile-release {field} must contain unique citation IDs")
    return tuple(value)


def load_profile_releases(path: Path) -> tuple[ProfileRelease, ...]:
    """Read explicit, dated decisions for profiles needing a release boundary."""

    source = load_json(path)
    if not isinstance(source, dict) or source.get("schema_version") != (
        PROFILE_RELEASE_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported profile-release schema: {path}")
    entries = source.get("profiles")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Profile-release manifest needs profile decisions")
    decisions: list[ProfileRelease] = []
    seen_people: set[str] = set()
    seen_scholars: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Profile-release decision must be an object")
        person_id = entry.get("person_id")
        scholar_id = entry.get("scholar_id")
        state = entry.get("state")
        count = entry.get("expected_source_rows")
        digest = entry.get("cache_sha256")
        reason = entry.get("reason")
        reviewed_at = entry.get("reviewed_at")
        if (
            not isinstance(person_id, str)
            or not person_id.startswith("person-")
            or person_id in seen_people
            or not isinstance(scholar_id, str)
            or not scholar_id
            or scholar_id in seen_scholars
            or state not in RELEASE_STATES
            or type(count) is not int
            or count <= 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("Profile-release decision has invalid fields")
        try:
            dt.date.fromisoformat(str(reviewed_at))
        except ValueError as error:
            raise ValueError("Profile-release decision needs a date") from error
        row_fields = (
            "released_author_pub_ids",
            "excluded_author_pub_ids",
            "held_author_pub_ids",
        )
        if state == "reviewed_subset":
            released_ids, excluded_ids, held_ids = (
                _row_ids(entry, field, scholar_id) for field in row_fields
            )
            sets = tuple(map(set, (released_ids, excluded_ids, held_ids)))
            if (
                not released_ids
                or any(
                    sets[left].intersection(sets[right])
                    for left, right in ((0, 1), (0, 2), (1, 2))
                )
                or sum(map(len, sets)) != count
            ):
                raise ValueError(
                    "Profile-release reviewed subset needs an exact row partition"
                )
        else:
            if any(field in entry for field in row_fields):
                raise ValueError(
                    "Profile-release row partition requires reviewed_subset"
                )
            released_ids = excluded_ids = held_ids = ()
        decisions.append(
            ProfileRelease(
                person_id=person_id,
                scholar_id=scholar_id,
                state=state,
                expected_source_rows=count,
                cache_sha256=digest,
                reason=reason.strip(),
                reviewed_at=str(reviewed_at),
                released_author_pub_ids=released_ids,
                excluded_author_pub_ids=excluded_ids,
                held_author_pub_ids=held_ids,
            )
        )
        seen_people.add(person_id)
        seen_scholars.add(scholar_id)
    return tuple(decisions)


def select_released_observations(
    frame: Any,
    registry: Registry,
    decisions: tuple[ProfileRelease, ...],
    *,
    cache_dir: Path,
) -> tuple[Any, list[dict[str, Any]], set[str]]:
    """Verify complete raw sources, then omit held owners or exact held rows."""

    held_mask = frame.index.isin([])
    receipt: list[dict[str, Any]] = []
    held_observation_ids: set[str] = set()
    for decision in decisions:
        person = registry.people_by_id.get(decision.person_id)
        if person is None or person.scholar_id != decision.scholar_id:
            raise ValueError(
                f"Profile-release identity differs from registry: {decision.scholar_id}"
            )
        cache_path = cache_dir / f"{decision.scholar_id}.json"
        if not cache_path.is_file() or sha256_file(cache_path) != decision.cache_sha256:
            raise ValueError(
                "Profile-release source cache is missing or changed: "
                f"{decision.scholar_id}"
            )
        cache = load_json(cache_path)
        publications = cache.get("publications") if isinstance(cache, dict) else None
        capture = cache.get("browser_capture") if isinstance(cache, dict) else None
        if (
            not isinstance(cache, dict)
            or cache.get("scholar_id") != decision.scholar_id
            or not isinstance(publications, list)
            or len(publications) != decision.expected_source_rows
            or cache.get("publication_count") != decision.expected_source_rows
            or not isinstance(capture, dict)
            or capture.get("coverage") != "complete_visible_profile"
            or capture.get("more_remain") is not False
        ):
            raise ValueError(
                f"Profile-release source cache is incomplete: {decision.scholar_id}"
            )
        cached_ids = [
            row.get("author_pub_id") if isinstance(row, dict) else None
            for row in publications
        ]
        if (
            any(
                not isinstance(row_id, str)
                or not row_id.startswith(f"{decision.scholar_id}:")
                for row_id in cached_ids
            )
            or len(set(cached_ids)) != decision.expected_source_rows
        ):
            raise ValueError(
                f"Profile-release source IDs are invalid: {decision.scholar_id}"
            )
        if decision.state == "reviewed_subset" and (
            set(cached_ids)
            != set(decision.released_author_pub_ids)
            | set(decision.excluded_author_pub_ids)
            | set(decision.held_author_pub_ids)
        ):
            raise ValueError(
                "Profile-release row partition differs from source: "
                f"{decision.scholar_id}"
            )
        source = frame.loc[
            (frame["source_type"] == "scholar")
            & (frame["scholar_id"] == decision.scholar_id)
        ]
        scope = frame.loc[
            (frame["person_id"] == decision.person_id)
            | (frame["scholar_id"] == decision.scholar_id)
        ]
        if (
            len(source) != decision.expected_source_rows
            or source["person_id"].ne(decision.person_id).any()
            or set(source["author_pub_id"]) != set(cached_ids)
            or scope.loc[scope["scholar_id"] == decision.scholar_id, "person_id"]
            .ne(decision.person_id)
            .any()
        ):
            raise ValueError(
                f"Profile-release snapshot differs from source: {decision.scholar_id}"
            )
        if decision.state == "held":
            removed = scope
        elif decision.state == "reviewed_subset":
            allowed = (
                scope["source_type"].eq("scholar")
                & scope["scholar_id"].eq(decision.scholar_id)
                & scope["author_pub_id"].isin(decision.released_author_pub_ids)
            )
            removed = scope.loc[~allowed]
        else:
            removed = scope.iloc[0:0]
        receipt.append(
            {
                "person_id": decision.person_id,
                "scholar_id": decision.scholar_id,
                "state": decision.state,
                "source_observation_rows": len(source),
                "person_observation_rows": len(scope),
                "released_source_observation_rows": len(source)
                - int(source.index.isin(removed.index).sum()),
                "held_source_observation_rows": int(
                    source.index.isin(removed.index).sum()
                ),
                "held_person_observation_rows": len(removed),
                "cache_sha256": decision.cache_sha256,
                "reason": decision.reason,
                "reviewed_at": decision.reviewed_at,
            }
        )
        if not removed.empty:
            held_mask |= frame.index.isin(removed.index)
            held_observation_ids.update(
                "observation-"
                + hashlib.sha256(
                    f"{row.scholar_id}\0{row.author_pub_id}".encode()
                ).hexdigest()
                for row in removed.itertuples(index=False)
            )
    return frame.loc[~held_mask].copy(), receipt, held_observation_ids


def validate_reviewed_subset_decisions(
    decisions: tuple[ProfileRelease, ...],
    attribution_reviews: AttributionReviews,
    exclusions: dict[tuple[str, str], Any],
) -> None:
    """Bind each public subset to exact positive and exclusion reviews."""

    for decision in decisions:
        if decision.state != "reviewed_subset":
            continue
        scholar_id = decision.scholar_id
        confirmed_ids = {
            author_pub_id
            for profile_id, author_pub_id in attribution_reviews.confirmed
            if profile_id == scholar_id
        }
        excluded_ids = {
            author_pub_id
            for profile_id, author_pub_id in exclusions
            if profile_id == scholar_id
        }
        if (
            scholar_id not in attribution_reviews.held_profiles
            or confirmed_ids != set(decision.released_author_pub_ids)
            or excluded_ids != set(decision.excluded_author_pub_ids)
        ):
            raise ValueError(
                f"Profile-release subset differs from exact reviews: {scholar_id}"
            )


def require_no_held_contributions(
    tables: dict[str, Any],
    decisions: tuple[ProfileRelease, ...],
    held_observation_ids: set[str],
) -> None:
    """Reject any held-profile contribution before the first external upload."""

    held = [decision for decision in decisions if decision.state == "held"]
    if not held and not held_observation_ids:
        return
    held_people = {decision.person_id for decision in held}
    held_scholars = {decision.scholar_id for decision in held}
    has_subset = any(decision.state == "reviewed_subset" for decision in decisions)
    for name in ("profile_publications", "authorships"):
        frame = tables[name]
        required = {"person_id", "scholar_id"}
        if has_subset:
            required.add("observation_id")
        if not required <= set(frame.columns):
            raise ValueError(f"Published {name} lacks profile identity columns")
        if (
            frame["person_id"].isin(held_people).any()
            or frame["scholar_id"].isin(held_scholars).any()
            or (has_subset and frame["observation_id"].isin(held_observation_ids).any())
        ):
            raise ValueError(f"Held profile contributed to published {name}")
    works = tables["works"]
    if "observation_ids" not in works:
        raise ValueError("Published works lack source observation IDs")
    if any(
        held_observation_ids.intersection(observation_ids)
        for observation_ids in works["observation_ids"]
    ):
        raise ValueError("Held profile contributed to published works")
    people = tables["people"]
    counts = (
        "publication_observation_count",
        "scholar_observation_count",
        "curated_observation_count",
        "unique_work_count",
        "mapped_work_count",
    )
    if not {"person_id", *counts} <= set(people.columns):
        raise ValueError("Published people lack held-profile count columns")
    for person_id in held_people:
        row = people.loc[people["person_id"] == person_id]
        if len(row) != 1 or row[list(counts)].ne(0).any().any():
            raise ValueError("Held profile's roster row must have zero own works")
