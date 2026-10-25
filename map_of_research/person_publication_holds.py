"""Keep incomplete curated-only person records in raw source, not public tables."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .io import load_json
from .registry import Registry

DEFAULT_PERSON_PUBLICATION_HOLDS_PATH = Path(
    "data/curated/person-publication-holds.json"
)
_DOI = re.compile(r"10\.\d{4,9}/\S+")


@dataclass(frozen=True)
class PersonPublicationHold:
    person_id: str
    source_snapshot_sha256: str
    expected_source_rows: int
    source_observations: tuple[tuple[str, str, str], ...]
    display_reason: str
    reason: str
    evidence_url: str
    reviewed_at: str


def load_person_publication_holds(path: Path) -> tuple[PersonPublicationHold, ...]:
    """Read reviewed, DOI-bound holds for people without full source coverage."""

    source = load_json(path)
    if not isinstance(source, dict) or source.get("schema_version") != 1:
        raise ValueError(f"Unsupported person-publication hold schema: {path}")
    entries = source.get("holds")
    if not isinstance(entries, list):
        raise ValueError("Person-publication holds need a list")
    holds = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Person-publication hold must be an object")
        person_id = entry.get("person_id")
        snapshot_sha = entry.get("source_snapshot_sha256")
        count = entry.get("expected_source_rows")
        observations = entry.get("source_observations")
        display_reason = entry.get("display_reason")
        reason = entry.get("reason")
        evidence_url = entry.get("evidence_url")
        reviewed_at = entry.get("reviewed_at")
        parsed_url = urlparse(str(evidence_url or ""))
        try:
            dt.date.fromisoformat(str(reviewed_at))
        except ValueError as error:
            raise ValueError("Person-publication hold needs a review date") from error
        rows: list[tuple[str, str, str]] = []
        if not isinstance(observations, list):
            raise ValueError("Person-publication hold needs exact observations")
        for observation in observations:
            if not isinstance(observation, dict):
                raise ValueError(
                    "Person-publication hold observation must be an object"
                )
            doi = observation.get("doi")
            author_pub_id = observation.get("author_pub_id")
            observation_id = observation.get("observation_id")
            if (
                not isinstance(doi, str)
                or _DOI.fullmatch(doi) is None
                or doi != doi.casefold()
                or author_pub_id != f"curated-doi:{person_id}:{doi}"
                or observation_id
                != "observation-"
                + hashlib.sha256(f"\0{author_pub_id}".encode()).hexdigest()
            ):
                raise ValueError("Person-publication hold observation identity differs")
            rows.append((doi, author_pub_id, observation_id))
        if (
            not isinstance(person_id, str)
            or not person_id.startswith("person-")
            or person_id in seen
            or not isinstance(snapshot_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", snapshot_sha) is None
            or type(count) is not int
            or count <= 0
            or len(rows) != count
            or len(set(rows)) != count
            or len({row[0] for row in rows}) != count
            or display_reason != "Publication history pending full capture"
            or not isinstance(reason, str)
            or not reason.strip()
            or parsed_url.scheme != "https"
            or not parsed_url.netloc
        ):
            raise ValueError("Person-publication hold has invalid fields")
        holds.append(
            PersonPublicationHold(
                person_id=person_id,
                source_snapshot_sha256=snapshot_sha,
                expected_source_rows=count,
                source_observations=tuple(rows),
                display_reason=display_reason,
                reason=reason.strip(),
                evidence_url=str(evidence_url),
                reviewed_at=str(reviewed_at),
            )
        )
        seen.add(person_id)
    return tuple(holds)


def select_released_person_observations(
    frame: Any,
    registry: Registry,
    holds: tuple[PersonPublicationHold, ...],
    *,
    source_snapshot_sha256: str,
) -> tuple[Any, list[dict[str, Any]], set[str]]:
    """Remove only reviewed held owners after exact raw-row validation."""

    held_indices: set[Any] = set()
    held_observation_ids: set[str] = set()
    receipt = []
    if not holds:
        return frame.copy(), receipt, held_observation_ids
    if "person_id" not in frame:
        raise ValueError("Held person source lacks person IDs")
    for hold in holds:
        person = registry.people_by_id.get(hold.person_id)
        source = frame.loc[frame["person_id"] == hold.person_id]
        if person is None or person.scholar_id:
            raise ValueError(
                "Person-publication hold identity differs from registry: "
                f"{hold.person_id}"
            )
        actual_observations = sorted(
            (
                str(row.source_doi),
                str(row.author_pub_id),
                "observation-"
                + hashlib.sha256(
                    f"{row.scholar_id}\0{row.author_pub_id}".encode()
                ).hexdigest(),
            )
            for row in source.itertuples(index=False)
        )
        if (
            hold.source_snapshot_sha256 != source_snapshot_sha256
            or len(source) != hold.expected_source_rows
            or source["source_type"].ne("curated_verified").any()
            or actual_observations != sorted(hold.source_observations)
        ):
            raise ValueError(
                f"Person-publication hold source rows differ: {hold.person_id}"
            )
        held_indices.update(source.index)
        held_observation_ids.update(item[2] for item in actual_observations)
        receipt.append(
            {
                "person_id": hold.person_id,
                "held_source_rows": len(source),
                "source_observations": [
                    {
                        "doi": doi,
                        "author_pub_id": author_pub_id,
                        "observation_id": observation_id,
                    }
                    for doi, author_pub_id, observation_id in hold.source_observations
                ],
                "display_reason": hold.display_reason,
                "reason": hold.reason,
                "evidence_url": hold.evidence_url,
                "reviewed_at": hold.reviewed_at,
            }
        )
    return (
        frame.loc[~frame.index.isin(held_indices)].copy(),
        receipt,
        held_observation_ids,
    )


def require_no_held_person_contributions(
    tables: Mapping[str, Any],
    held_people: set[str],
    held_observation_ids: set[str],
) -> None:
    """Reject a held person-to-work link before staging any public file."""

    if not held_people:
        return
    for name in ("profile_publications", "authorships"):
        if tables[name]["person_id"].isin(held_people).any():
            raise ValueError(f"Held person contributed to published {name}")
    if any(
        held_observation_ids.intersection(ids)
        for ids in tables["works"]["observation_ids"]
    ):
        raise ValueError("Held person contributed to published works")
    if "person_ids" in tables["works"] and any(
        held_people.intersection(ids) for ids in tables["works"]["person_ids"]
    ):
        raise ValueError("Held person assigned to published work")
    counts = (
        "publication_observation_count",
        "scholar_observation_count",
        "curated_observation_count",
        "unique_work_count",
        "mapped_work_count",
    )
    for person_id in held_people:
        person = tables["people"].loc[tables["people"]["person_id"] == person_id]
        if len(person) != 1 or person[list(counts)].ne(0).any().any():
            raise ValueError("Held person's roster row must have zero own works")
