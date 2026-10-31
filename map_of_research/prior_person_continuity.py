"""Fail-closed continuity for every previously published person row.

An additive snapshot may add selected observations and author bindings. It may
not silently rewrite a published identity, home, membership, or source record.
The one Hoda provenance correction needs a source-bound, exact-parent review.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from .io import sha256_file

HODA_PERSON_ID = "person-hodaheidari-477d966d"
HODA_PROVENANCE_CHANGES = {
    "primary_academic_home_source_url": (
        "https://scholars.cmu.edu/10167-hoda-heidari",
        "https://www.cs.cmu.edu/~hheidari/CV_HodaHeidari.pdf",
    ),
    "primary_academic_home_verified_at": ("2026-10-24", "2026-10-31"),
    "primary_department_source_url": (
        "https://scholars.cmu.edu/10167-hoda-heidari",
        "https://www.cs.cmu.edu/~hheidari/CV_HodaHeidari.pdf",
    ),
    "primary_department_verified_at": ("2026-10-24", "2026-10-31"),
}
COUNT_FIELDS = (
    "publication_observation_count",
    "scholar_observation_count",
    "curated_observation_count",
    "unique_work_count",
    "mapped_work_count",
)
REQUIRED_PEOPLE_FIELDS = {
    "person_id",
    "display_name",
    "scholar_id",
    "scholar_id_source_url",
    "scholar_id_verified_at",
    "orcid",
    "homepage_url",
    "notes",
    "primary_academic_home_type",
    "primary_academic_home_id",
    "primary_academic_home_title",
    "primary_academic_home_school_id",
    "primary_academic_home_source_url",
    "primary_academic_home_verified_at",
    "primary_academic_home_basis",
    "primary_department_id",
    "primary_department_source_url",
    "primary_department_verified_at",
    "primary_department_basis",
    "memberships",
    "included_department_ids",
    "included_school_ids",
    "publication_hold_reason",
    *COUNT_FIELDS,
}
_REVIEW_FIELDS = {
    "schema_version",
    "decision",
    "prior_revision",
    "parent_people_sha256",
    "source_snapshot_sha256",
    "producer_commit",
    "producer_people_registry_sha256",
    "person_id",
    "reviewed_at",
    "rationale",
    "changes",
}
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA64 = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_FIELDS = (
    "person_id",
    "scholar_id",
    "author_pub_id",
    "source_type",
    "work_id",
)


def _normal(value: Any) -> Any:
    """Normalize only Arrow/Pandas physical wrappers, not logical field values."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _normal(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normal(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _same(left: Any, right: Any) -> bool:
    return json.dumps(_normal(left), sort_keys=True, allow_nan=False) == json.dumps(
        _normal(right), sort_keys=True, allow_nan=False
    )


def _indexed_people(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> tuple[dict[str, Mapping[str, Any]], set[str]]:
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{label} people rows are missing or invalid")
    fields = set(rows[0])
    if not fields >= REQUIRED_PEOPLE_FIELDS or any(set(row) != fields for row in rows):
        raise ValueError(f"{label} people schema is incomplete")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        person_id = row["person_id"]
        if not isinstance(person_id, str) or not person_id or person_id in by_id:
            raise ValueError(f"{label} people IDs are missing or duplicated")
        by_id[person_id] = row
    return by_id, fields


def read_pinned_prior_people(
    *, repo_id: str, prior_revision: str, hf_token: str
) -> tuple[list[dict[str, Any]], str]:
    """Read every column of the immutable parent's public people Parquet."""

    if not _SHA40.fullmatch(prior_revision):
        raise ValueError("Prior people need an immutable Hub revision")
    import pyarrow.parquet as parquet
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=prior_revision,
            allow_patterns="people/train-*.parquet",
            token=hf_token or None,
        )
    )
    shards = sorted((root / "people").glob("train-*.parquet"))
    if len(shards) != 1 or shards[0].name != "train-00000-of-00001.parquet":
        raise ValueError("Pinned parent people need exactly one reviewed shard")
    path = shards[0]
    table = parquet.read_table(path)
    rows = table.to_pylist()
    _, fields = _indexed_people(rows, label="Pinned parent")
    if set(table.schema.names) != fields or len(table.schema.names) != len(fields):
        raise ValueError("Pinned parent people Parquet schema is incomplete")
    return rows, sha256_file(path)


def _new_selected_profiles(
    prior_profiles: Sequence[Mapping[str, Any]],
    current_profiles: Sequence[Mapping[str, Any]],
    release_receipt: Sequence[Mapping[str, Any]] | None,
) -> dict[str, list[Mapping[str, Any]]]:
    if any(not set(_PROFILE_FIELDS) <= set(row) for row in prior_profiles):
        raise ValueError("Prior profile rows lack person/source identity fields")
    if any(not set(_PROFILE_FIELDS) <= set(row) for row in current_profiles):
        raise ValueError("Candidate profile rows lack person/source identity fields")
    selected: dict[tuple[str, str], set[str]] = {}
    for item in release_receipt or ():
        if item.get("cache_scope") != "reviewed_public_subset":
            continue
        person_id, scholar_id = item.get("person_id"), item.get("scholar_id")
        ids = item.get("selected_author_pub_ids")
        if (
            not isinstance(person_id, str)
            or not isinstance(scholar_id, str)
            or not isinstance(ids, list)
            or any(not isinstance(value, str) for value in ids)
            or len(ids) != len(set(ids))
            or (person_id, scholar_id) in selected
        ):
            raise ValueError("Selected profile receipt is incomplete")
        selected[(person_id, scholar_id)] = set(ids)

    def identity(row: Mapping[str, Any]) -> tuple[str | None, ...]:
        values = tuple(_normal(row[field]) for field in _PROFILE_FIELDS)
        if any(
            not isinstance(value, str) or not value
            for value in (values[0], *values[2:])
        ) or (values[1] is not None and not isinstance(values[1], str)):
            raise ValueError("Profile source identity has invalid field types")
        return values

    previous = Counter(identity(row) for row in prior_profiles)
    current = Counter(identity(row) for row in current_profiles)
    if previous - current:
        raise ValueError("Prior person continuity loses published profiles")
    if any(current[key] > max(1, previous[key]) for key in current):
        raise ValueError("Prior person continuity adds duplicate source identities")
    by_identity = {identity(row): row for row in current_profiles}
    additions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for key, count in (current - previous).items():
        row = by_identity[key]
        if row["source_type"] != "scholar" or row["author_pub_id"] not in selected.get(
            (row["person_id"], row["scholar_id"]), set()
        ):
            raise ValueError("New person count lacks selected-profile receipt")
        additions[row["person_id"]].extend([row] * count)
    return additions


def _reviewed_changes(
    review_path: Path | None,
    *,
    prior_revision: str,
    parent_people_sha256: str,
    source_snapshot_sha256: str,
    producer_commit: str,
    producer_people_registry_sha256: str,
) -> tuple[dict[str, tuple[Any, Any]], str | None]:
    if review_path is None:
        return {}, None
    try:
        raw = review_path.read_bytes()
        review = json.loads(raw)
    except (OSError, ValueError) as error:
        raise ValueError("Prior person review cannot be read") from error
    expected = {
        "schema_version": 1,
        "decision": "allow_exact_hoda_provenance",
        "prior_revision": prior_revision,
        "parent_people_sha256": parent_people_sha256,
        "source_snapshot_sha256": source_snapshot_sha256,
        "producer_commit": producer_commit,
        "producer_people_registry_sha256": producer_people_registry_sha256,
        "person_id": HODA_PERSON_ID,
    }
    if (
        not isinstance(review, dict)
        or set(review) != _REVIEW_FIELDS
        or any(review.get(field) != value for field, value in expected.items())
        or not _SHA40.fullmatch(producer_commit)
        or not all(
            isinstance(value, str) and _SHA64.fullmatch(value)
            for value in (
                parent_people_sha256,
                source_snapshot_sha256,
                producer_people_registry_sha256,
            )
        )
    ):
        raise ValueError("Prior person review is stale or has invalid binding")
    try:
        date.fromisoformat(review["reviewed_at"])
    except (TypeError, ValueError) as error:
        raise ValueError("Prior person review date is invalid") from error
    if (
        not isinstance(review["rationale"], str)
        or len(review["rationale"].strip()) < 20
    ):
        raise ValueError("Prior person review rationale is incomplete")
    changes = review["changes"]
    if not isinstance(changes, dict) or set(changes) != set(HODA_PROVENANCE_CHANGES):
        raise ValueError("Prior person review does not cover four Hoda fields")
    for field, (before, after) in HODA_PROVENANCE_CHANGES.items():
        if changes[field] != {"before": before, "after": after}:
            raise ValueError("Prior person review changes Hoda evidence values")
    return HODA_PROVENANCE_CHANGES, hashlib.sha256(raw).hexdigest()


def require_prior_person_continuity(
    prior_people: Sequence[Mapping[str, Any]],
    current_people: Sequence[Mapping[str, Any]],
    prior_profiles: Sequence[Mapping[str, Any]],
    current_profiles: Sequence[Mapping[str, Any]],
    current_authorships: Sequence[Mapping[str, Any]],
    current_works: Sequence[Mapping[str, Any]],
    release_receipt: Sequence[Mapping[str, Any]] | None,
    *,
    prior_revision: str,
    parent_people_sha256: str,
    source_snapshot_sha256: str,
    producer_commit: str,
    producer_people_registry_sha256: str,
    review_path: Path | None,
) -> dict[str, Any]:
    """Allow only selected count growth and the exact reviewed Hoda URL/date delta."""

    prior, prior_fields = _indexed_people(prior_people, label="Pinned parent")
    current, current_fields = _indexed_people(current_people, label="Candidate")
    if prior_fields != current_fields or not prior.keys() <= current.keys():
        raise ValueError("Prior people columns or identities changed")
    additions = _new_selected_profiles(
        prior_profiles, current_profiles, release_receipt
    )
    allowed_hoda, review_sha256 = _reviewed_changes(
        review_path,
        prior_revision=prior_revision,
        parent_people_sha256=parent_people_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
        producer_commit=producer_commit,
        producer_people_registry_sha256=producer_people_registry_sha256,
    )
    authorship_work_ids: dict[str, set[str]] = defaultdict(set)
    for row in current_authorships:
        if not {"person_id", "work_id"} <= set(row):
            raise ValueError("Candidate authorships lack person/work IDs")
        authorship_work_ids[str(row["person_id"])].add(str(row["work_id"]))
    mapped_work_ids: dict[str, set[str]] = defaultdict(set)
    for row in current_works:
        if not {"work_id", "map_eligible", "memberships"} <= set(row):
            raise ValueError("Candidate works lack mapped person memberships")
        if _normal(row["map_eligible"]) is True:
            for membership in _normal(row["memberships"]):
                if not isinstance(membership, dict) or "person_id" not in membership:
                    raise ValueError("Candidate mapped work has invalid membership")
                mapped_work_ids[str(membership["person_id"])].add(str(row["work_id"]))

    changed_people: dict[str, list[str]] = {}
    for person_id, before in prior.items():
        after = current[person_id]
        new_rows = additions.get(person_id, [])
        new_work_ids = {str(row["work_id"]) for row in new_rows}
        changed_fields = [
            field
            for field in sorted(prior_fields)
            if not _same(before[field], after[field])
        ]
        for field in COUNT_FIELDS:
            if not new_rows and field not in changed_fields:
                continue
            if not new_rows:
                raise ValueError("Prior person count growth lacks selected profile")
            old, new = before[field], after[field]
            if type(old) is not int or type(new) is not int or new < old:
                raise ValueError("Prior person counts decrease or change type")
            if field == "publication_observation_count":
                expected_growth = len(new_rows)
            elif field == "scholar_observation_count":
                expected_growth = sum(
                    row["source_type"] == "scholar" for row in new_rows
                )
            elif field == "curated_observation_count":
                expected_growth = sum(
                    row["source_type"] == "curated_verified" for row in new_rows
                )
            elif field == "unique_work_count":
                expected_growth = len(authorship_work_ids[person_id]) - old
                if expected_growth > len(new_work_ids):
                    raise ValueError("Prior person work growth lacks selected rows")
            else:
                expected_growth = len(mapped_work_ids[person_id]) - old
                if expected_growth > len(new_work_ids & mapped_work_ids[person_id]):
                    raise ValueError("Prior person mapped growth lacks selected rows")
            if new - old != expected_growth:
                raise ValueError("Prior person count growth lacks selected profile")
        for field in changed_fields:
            if field in COUNT_FIELDS:
                continue
            if (
                person_id == HODA_PERSON_ID
                and field in allowed_hoda
                and (before[field], after[field]) == allowed_hoda[field]
            ):
                continue
            else:
                raise ValueError(
                    f"Prior person field changed without review: {person_id}/{field}"
                )
        if changed_fields:
            changed_people[person_id] = changed_fields
    actual_hoda = set(changed_people.get(HODA_PERSON_ID, [])) & set(
        HODA_PROVENANCE_CHANGES
    )
    if allowed_hoda and actual_hoda != set(HODA_PROVENANCE_CHANGES):
        raise ValueError("Prior person Hoda review is unused or incomplete")
    return {
        "status": "pass",
        "gate": "exact-prior-person-continuity-v1",
        "prior_revision": prior_revision,
        "parent_people_sha256": parent_people_sha256,
        "source_snapshot_sha256": source_snapshot_sha256,
        "producer_people_registry_sha256": producer_people_registry_sha256,
        "review_sha256": review_sha256,
        "prior_person_count": len(prior),
        "changed_people": changed_people,
    }


def require_staged_people_equal(
    staged_path: Path, current_people: Sequence[Mapping[str, Any]]
) -> None:
    """Catch any prewrite-to-Parquet person drift before the atomic Hub call."""

    import pyarrow.parquet as parquet

    expected, fields = _indexed_people(current_people, label="Prewrite")
    table = parquet.read_table(staged_path)
    actual, staged_fields = _indexed_people(table.to_pylist(), label="Staged")
    if (
        set(table.schema.names) != fields
        or len(table.schema.names) != len(fields)
        or staged_fields != fields
        or actual.keys() != expected.keys()
    ):
        raise ValueError("Staged people schema or IDs differ from prewrite")
    for person_id, before in expected.items():
        for field in fields:
            if not _same(before[field], actual[person_id][field]):
                raise ValueError(
                    f"Staged people differ from prewrite: {person_id}/{field}"
                )
