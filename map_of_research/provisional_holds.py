"""Source-bound holds for individual, still-unattributed Scholar observations."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .io import load_json, sha256_file
from .profile_releases import ProfileRelease

DEFAULT_PROVISIONAL_HOLDS_PATH = Path("data/curated/scholar-provisional-holds.json")


@dataclass(frozen=True)
class ProvisionalHold:
    scholar_id: str
    author_pub_id: str
    capture_row_index: int
    reason: str
    evidence_url: str
    reviewed_at: str
    primary_evidence_url: str = ""


@dataclass(frozen=True)
class ProvisionalHoldProfile:
    person_id: str
    scholar_id: str
    expected_source_rows: int
    cache_sha256: str
    held_observations: tuple[ProvisionalHold, ...]
    expected_reviewed_exclusion_rows: int = 0
    reviewed_exclusion_ids_sha256: str = ""


def _dated_https(url: object, date: object) -> bool:
    parsed = urlparse(str(url or "").strip())
    try:
        dt.date.fromisoformat(str(date or ""))
    except ValueError:
        return False
    return parsed.scheme == "https" and bool(parsed.netloc)


def hold_ids_sha256(author_pub_ids: set[str]) -> str:
    """Bind a release decision to the exact set of held Scholar IDs."""

    return hashlib.sha256("\n".join(sorted(author_pub_ids)).encode()).hexdigest()


def load_provisional_holds(path: Path) -> tuple[ProvisionalHoldProfile, ...]:
    """Load explicit, dated provisional decisions separate from exclusions."""

    source = load_json(path)
    if not isinstance(source, dict) or source.get("schema_version") != 1:
        raise ValueError(f"Unsupported provisional-hold schema: {path}")
    entries = source.get("profiles")
    if not isinstance(entries, list):
        raise ValueError("Provisional holds need a profile list")
    profiles: list[ProvisionalHoldProfile] = []
    seen_people: set[str] = set()
    seen_scholars: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Provisional-hold profile must be an object")
        person_id = entry.get("person_id")
        scholar_id = entry.get("scholar_id")
        count = entry.get("expected_source_rows")
        digest = entry.get("cache_sha256")
        held = entry.get("held_observations")
        exclusion_count = entry.get("expected_reviewed_exclusion_rows")
        exclusion_digest = entry.get("reviewed_exclusion_ids_sha256")
        if (
            not isinstance(person_id, str)
            or not person_id.startswith("person-")
            or person_id in seen_people
            or not isinstance(scholar_id, str)
            or not scholar_id
            or scholar_id in seen_scholars
            or type(count) is not int
            or count <= 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(held, list)
            or not held
            or type(exclusion_count) is not int
            or exclusion_count < 0
            or exclusion_count >= count
            or not isinstance(exclusion_digest, str)
            or (
                re.fullmatch(r"[0-9a-f]{64}", exclusion_digest) is None
                if exclusion_count
                else bool(exclusion_digest)
            )
        ):
            raise ValueError("Provisional-hold profile has invalid source fields")
        rows: list[ProvisionalHold] = []
        seen_ids: set[str] = set()
        seen_indices: set[int] = set()
        for item in held:
            if not isinstance(item, dict):
                raise ValueError("Provisional-hold observation must be an object")
            author_pub_id = item.get("author_pub_id")
            index = item.get("capture_row_index")
            reason = item.get("reason")
            if (
                not isinstance(author_pub_id, str)
                or not author_pub_id.startswith(f"{scholar_id}:")
                or author_pub_id == f"{scholar_id}:"
                or author_pub_id in seen_ids
                or type(index) is not int
                or index <= 0
                or index > count
                or index in seen_indices
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise ValueError("Provisional-hold observation has invalid identity")
            if not _dated_https(item.get("evidence_url"), item.get("reviewed_at")):
                raise ValueError(
                    "Provisional-hold observation needs dated HTTPS evidence"
                )
            primary_evidence_url = str(item.get("primary_evidence_url") or "").strip()
            if primary_evidence_url and not _dated_https(
                primary_evidence_url, item.get("reviewed_at")
            ):
                raise ValueError(
                    "Provisional-hold primary evidence needs dated HTTPS URL"
                )
            rows.append(
                ProvisionalHold(
                    scholar_id=scholar_id,
                    author_pub_id=author_pub_id,
                    capture_row_index=index,
                    reason=reason.strip(),
                    evidence_url=str(item["evidence_url"]).strip(),
                    reviewed_at=str(item["reviewed_at"]).strip(),
                    primary_evidence_url=primary_evidence_url,
                )
            )
            seen_ids.add(author_pub_id)
            seen_indices.add(index)
        profiles.append(
            ProvisionalHoldProfile(
                person_id=person_id,
                scholar_id=scholar_id,
                expected_source_rows=count,
                cache_sha256=digest,
                held_observations=tuple(rows),
                expected_reviewed_exclusion_rows=exclusion_count,
                reviewed_exclusion_ids_sha256=exclusion_digest,
            )
        )
        seen_people.add(person_id)
        seen_scholars.add(scholar_id)
    return tuple(profiles)


def validate_provisional_holds(
    profiles: tuple[ProvisionalHoldProfile, ...],
    decisions: tuple[ProfileRelease, ...],
    release_receipt: list[dict[str, object]],
    *,
    cache_dir: Path,
    excluded_ids: set[tuple[str, str]],
    confirmed_ids: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], ProvisionalHold], list[dict[str, object]]]:
    """Require a complete released source and exact reviewed hold IDs."""

    expected = {
        item.scholar_id: item
        for item in decisions
        if item.expected_provisional_hold_rows
    }
    if {item.scholar_id for item in profiles} != set(expected):
        raise ValueError("Provisional-hold profiles differ from release decisions")
    receipts = {item["scholar_id"]: item for item in release_receipt}
    holds: dict[tuple[str, str], ProvisionalHold] = {}
    result: list[dict[str, object]] = []
    for profile in profiles:
        decision = expected[profile.scholar_id]
        receipt = receipts.get(profile.scholar_id)
        ids = {item.author_pub_id for item in profile.held_observations}
        exclusion_ids = {
            author_pub_id
            for scholar_id, author_pub_id in excluded_ids
            if scholar_id == profile.scholar_id
        }
        if (
            len(exclusion_ids) != profile.expected_reviewed_exclusion_rows
            or (hold_ids_sha256(exclusion_ids) if exclusion_ids else "")
            != profile.reviewed_exclusion_ids_sha256
        ):
            raise ValueError(
                f"Provisional-hold reviewed exclusions differ: {profile.scholar_id}"
            )
        if (
            decision.state != "released"
            or decision.person_id != profile.person_id
            or decision.expected_source_rows != profile.expected_source_rows
            or decision.cache_sha256 != profile.cache_sha256
            or decision.expected_provisional_hold_rows != len(ids)
            or decision.provisional_hold_ids_sha256 != hold_ids_sha256(ids)
            or receipt is None
            or receipt["state"] != "released"
            or receipt["source_observation_rows"] != profile.expected_source_rows
            or receipt["cache_sha256"] != profile.cache_sha256
        ):
            raise ValueError(
                f"Provisional-hold release source differs: {profile.scholar_id}"
            )
        cache_path = cache_dir / f"{profile.scholar_id}.json"
        if not cache_path.is_file() or sha256_file(cache_path) != profile.cache_sha256:
            raise ValueError(
                f"Provisional-hold cache is missing or changed: {profile.scholar_id}"
            )
        cache = load_json(cache_path)
        publications = cache.get("publications") if isinstance(cache, dict) else None
        if not isinstance(publications, list) or len(publications) != (
            profile.expected_source_rows
        ):
            raise ValueError(
                f"Provisional-hold cache row count differs: {profile.scholar_id}"
            )
        indexed = {
            item.get("capture_row_index"): item
            for item in publications
            if isinstance(item, dict)
        }
        if len(indexed) != profile.expected_source_rows:
            raise ValueError(
                f"Provisional-hold cache indices differ: {profile.scholar_id}"
            )
        if not exclusion_ids <= {
            item.get("author_pub_id") for item in publications if isinstance(item, dict)
        }:
            raise ValueError(
                "Provisional-hold reviewed exclusions lack source IDs: "
                f"{profile.scholar_id}"
            )
        for item in profile.held_observations:
            row = indexed.get(item.capture_row_index)
            key = (profile.scholar_id, item.author_pub_id)
            if (
                row is None
                or row.get("author_pub_id") != item.author_pub_id
                or row.get("pub_url") != item.evidence_url
                or key in excluded_ids
                or key in confirmed_ids
            ):
                raise ValueError(
                    f"Provisional-hold ID, evidence, or review conflicts: {key}"
                )
            holds[key] = item
        result.append(
            {
                "person_id": profile.person_id,
                "scholar_id": profile.scholar_id,
                "source_observation_rows": profile.expected_source_rows,
                "held_observation_rows": len(ids),
                "cache_sha256": profile.cache_sha256,
                "held_ids_sha256": hold_ids_sha256(ids),
                "reviewed_exclusion_rows": len(exclusion_ids),
                "reviewed_exclusion_ids_sha256": profile.reviewed_exclusion_ids_sha256,
            }
        )
    return holds, result


def require_no_provisional_hold_contributions(
    tables: dict[str, object], holds: dict[tuple[str, str], ProvisionalHold]
) -> None:
    """Reject held authorships or mapped-work provenance before uploading."""

    if not holds:
        return
    raw = tables["profile_publications"]
    authorships = tables["authorships"]
    works = tables["works"]
    if not {
        "scholar_id",
        "author_pub_id",
        "observation_id",
        "map_eligible",
        "map_exclusion_reasons",
        "review_attribution_status",
    } <= set(raw.columns):
        raise ValueError("Published profile observations lack hold audit columns")
    if not {"scholar_id", "author_pub_id", "observation_id"} <= set(
        authorships.columns
    ) or not {"map_eligible", "observation_ids", "author_pub_ids"} <= set(
        works.columns
    ):
        raise ValueError("Published relationships lack hold provenance columns")
    for key in holds:
        matching = raw.loc[
            (raw["scholar_id"] == key[0]) & (raw["author_pub_id"] == key[1])
        ]
        if (
            len(matching) != 1
            or bool(matching.iloc[0]["map_eligible"])
            or "attribution_provisional_hold"
            not in matching.iloc[0]["map_exclusion_reasons"]
            or matching.iloc[0]["review_attribution_status"] != "held"
        ):
            raise ValueError(f"Provisional hold escaped raw map gate: {key}")
        observation_id = matching.iloc[0]["observation_id"]
        if (
            (
                (authorships["scholar_id"] == key[0])
                & (authorships["author_pub_id"] == key[1])
            ).any()
            or authorships["observation_id"].eq(observation_id).any()
            or any(
                bool(work.map_eligible)
                and (
                    observation_id in work.observation_ids
                    or key[1] in work.author_pub_ids
                )
                for work in works.itertuples(index=False)
            )
        ):
            raise ValueError(f"Provisional hold contributed to a map link: {key}")
