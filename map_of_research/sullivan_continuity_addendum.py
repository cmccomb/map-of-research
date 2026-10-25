"""Source-bound approval for eight additional Sullivan bb88 continuity deltas.

The original 22-loss decision remains immutable. This addendum names five
reviewed losses and three work rekeys found by replaying that exact release.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file
from .sullivan_loss_approval import (
    PRIOR_REVISION,
    SOURCE_COMMIT,
    SOURCE_SHA256,
    SULLIVAN_ID,
    _identity,
    _normalized,
    _source_record_sha256,
    _values,
)

DEFAULT_ADDENDUM_PATH = Path("data/curated/sullivan-continuity-addendum-bb88.json")
ADDENDUM_SHA256 = "1821b82135f810722379c612f0b7a58a2f82a5a0e2088a7b3392454e64c4a19c"
LOSS_IDS_SHA256 = "1f49bec1a3446d634c5011f188964511e56b612c062690d8c3dfe3b9889e7a9f"
BINDINGS_SHA256 = "0f1c4b74d4a23ad4dedf03e471d6dbc9040bb5a2f125f9cf96368d2fed8f8d3e"
TARGETS_SHA256 = "bf0b99af0a095678e885372cec0c5dca296231ed01d75a6eada9624be35dd529"
PERSON_ID = "person-sullivan-589c9545"
IDENTITY_FIELDS = ("person_id", "scholar_id", "author_pub_id", "source_type")
LOSS_FIELDS = {
    *IDENTITY_FIELDS,
    "prior_work_id",
    "prior_title",
    "prior_year",
    "prior_source_record_sha256",
    "current_work_id",
    "current_title",
    "current_year",
    "current_source_record_sha256",
    "current_review_exclusion_code",
    "current_review_evidence_url",
}
BINDING_FIELDS = {
    *IDENTITY_FIELDS,
    "prior_work_id",
    "prior_title",
    "prior_year",
    "prior_source_record_sha256",
    "current_work_id",
    "current_title",
    "current_year",
    "current_source_record_sha256",
    "publisher_doi",
    "evidence_url",
}
TARGET_FIELDS = {
    "source_native_id",
    "author_pub_id",
    "work_id",
    "title",
    "year",
    "source_url",
    "source_record_sha256",
}
DOCUMENT_FIELDS = {
    "schema_version",
    "reviewed_at",
    "prior_revision",
    "source_snapshot_commit",
    "source_snapshot_sha256",
    "additional_approved_losses",
    "approved_binding_changes",
    "required_curated_targets",
}
ObservationKey = tuple[str, str, str, str]
BindingChange = tuple[ObservationKey, str, str, bool, bool]


@dataclass(frozen=True)
class ExactSullivanContinuityAddendum:
    loss_identities: frozenset[ObservationKey]
    binding_changes: frozenset[BindingChange]
    required_curated_work_ids: frozenset[str]
    manifest_sha256: str


def decision_set_digests(
    addendum: ExactSullivanContinuityAddendum,
) -> tuple[str, str, str]:
    """Bind the reviewed identities and work IDs independently of object type."""

    losses = sorted("/".join(identity) for identity in addendum.loss_identities)
    bindings = sorted(
        "/".join(identity)
        + f"|{prior_work_id}|{current_work_id}|"
        + str(prior_collaboration_eligible).lower()
        + "|"
        + str(current_collaboration_eligible).lower()
        for (
            identity,
            prior_work_id,
            current_work_id,
            prior_collaboration_eligible,
            current_collaboration_eligible,
        ) in addendum.binding_changes
    )
    targets = sorted(addendum.required_curated_work_ids)
    return tuple(
        hashlib.sha256("\n".join(lines).encode()).hexdigest()
        for lines in (losses, bindings, targets)
    )


def _index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[ObservationKey, Mapping[str, Any]]:
    records = list(rows)
    indexed = {_identity(row): row for row in records}
    if len(indexed) != len(records):
        raise ValueError("Sullivan addendum observations contain duplicate identities")
    return indexed


def _year(value: Any) -> int | None:
    return int(value) if value is not None and str(value) != "nan" else None


def _same_row(
    row: Mapping[str, Any] | None, entry: Mapping[str, Any], prefix: str
) -> bool:
    return bool(
        row is not None
        and str(row.get("work_id") or "") == entry[f"{prefix}_work_id"]
        and str(row.get("title") or "") == entry[f"{prefix}_title"]
        and _year(row.get("year")) == entry[f"{prefix}_year"]
        and _source_record_sha256(row) == entry[f"{prefix}_source_record_sha256"]
    )


def validate_sullivan_continuity_addendum(
    *,
    addendum_path: Path,
    prior_revision: str,
    source_snapshot_commit: str,
    source_snapshot_sha256: str,
    prior_works: Iterable[Mapping[str, Any]],
    prior_profiles: Iterable[Mapping[str, Any]],
    source_rows: Iterable[Mapping[str, Any]],
    current_works: Iterable[Mapping[str, Any]],
    current_profiles: Iterable[Mapping[str, Any]],
) -> ExactSullivanContinuityAddendum:
    """Validate exact prior/current states and the pinned source for each delta."""

    if (
        prior_revision != PRIOR_REVISION
        or source_snapshot_commit != SOURCE_COMMIT
        or source_snapshot_sha256 != SOURCE_SHA256
    ):
        raise ValueError("Sullivan addendum is bound to a different release")
    if sha256_file(addendum_path) != ADDENDUM_SHA256:
        raise ValueError("Sullivan continuity addendum manifest changed")
    document = load_json(addendum_path)
    if not isinstance(document, dict) or set(document) != DOCUMENT_FIELDS:
        raise ValueError("Sullivan continuity addendum schema changed")
    if (
        document["schema_version"] != 1
        or document["reviewed_at"] != "2026-10-25"
        or document["prior_revision"] != PRIOR_REVISION
        or document["source_snapshot_commit"] != SOURCE_COMMIT
        or document["source_snapshot_sha256"] != SOURCE_SHA256
    ):
        raise ValueError("Sullivan continuity addendum release binding changed")
    losses = document["additional_approved_losses"]
    bindings = document["approved_binding_changes"]
    targets = document["required_curated_targets"]
    if (
        not isinstance(losses, list)
        or len(losses) != 5
        or any(not isinstance(row, dict) or set(row) != LOSS_FIELDS for row in losses)
        or not isinstance(bindings, list)
        or len(bindings) != 3
        or any(
            not isinstance(row, dict) or set(row) != BINDING_FIELDS for row in bindings
        )
        or not isinstance(targets, list)
        or len(targets) != 2
        or any(
            not isinstance(row, dict) or set(row) != TARGET_FIELDS for row in targets
        )
    ):
        raise ValueError("Sullivan continuity addendum decision set changed")
    loss_ids = [_identity(row) for row in losses]
    binding_ids = [_identity(row) for row in bindings]
    if (
        loss_ids != sorted(set(loss_ids))
        or binding_ids != sorted(set(binding_ids))
        or set(loss_ids) & set(binding_ids)
        or any(
            identity[0] != PERSON_ID
            or identity[1] != SULLIVAN_ID
            or identity[3] != "scholar"
            for identity in [*loss_ids, *binding_ids]
        )
        or [row["source_native_id"] for row in targets]
        != ["agu2017:a13j-05", "escholarship:qt89c072pv"]
    ):
        raise ValueError("Sullivan continuity addendum identities changed")

    previous = _index(prior_profiles)
    source = _index(source_rows)
    current = _index(current_profiles)
    prior_work_rows = list(prior_works)
    current_work_rows = list(current_works)
    prior_work = {str(row["work_id"]): row for row in prior_work_rows}
    current_work = {str(row["work_id"]): row for row in current_work_rows}
    if len(prior_work) != len(prior_work_rows) or len(current_work) != len(
        current_work_rows
    ):
        raise ValueError("Sullivan continuity addendum work IDs are duplicated")

    for entry, identity in zip(losses, loss_ids, strict=True):
        old = previous.get(identity)
        now = current.get(identity)
        raw = source.get(identity)
        old_work = prior_work.get(entry["prior_work_id"])
        if (
            not _same_row(old, entry, "prior")
            or old_work is None
            or not bool(old.get("map_eligible"))
            or not bool(old.get("collaboration_eligible"))
            or not bool(old_work.get("map_eligible"))
            or not _same_row(now, entry, "current")
            or raw is None
            or _source_record_sha256(raw) != entry["current_source_record_sha256"]
            or not bool(entry["current_review_exclusion_code"])
            or not str(entry["current_review_evidence_url"]).startswith("https://")
            or bool(now.get("map_eligible"))
            or bool(now.get("collaboration_eligible"))
            or now.get("review_exclusion_code")
            != entry["current_review_exclusion_code"]
            or now.get("review_evidence_url") != entry["current_review_evidence_url"]
        ):
            raise ValueError(
                f"Sullivan added loss differs from exact review: {identity}"
            )

    approved_bindings: set[BindingChange] = set()
    for entry, identity in zip(bindings, binding_ids, strict=True):
        old = previous.get(identity)
        now = current.get(identity)
        raw = source.get(identity)
        old_work = prior_work.get(entry["prior_work_id"])
        new_work = current_work.get(entry["current_work_id"])
        if (
            not _same_row(old, entry, "prior")
            or not _same_row(now, entry, "current")
            or raw is None
            or _source_record_sha256(raw) != entry["current_source_record_sha256"]
            or old_work is None
            or new_work is None
            or not bool(old.get("map_eligible"))
            or not bool(now.get("map_eligible"))
            or not bool(old.get("collaboration_eligible"))
            or not bool(now.get("collaboration_eligible"))
            or not bool(old_work.get("map_eligible"))
            or not bool(new_work.get("map_eligible"))
            or not bool(new_work.get("collaboration_eligible"))
            or entry["prior_work_id"] == entry["current_work_id"]
            or entry["evidence_url"] != f"https://doi.org/{entry['publisher_doi']}"
        ):
            raise ValueError(
                f"Sullivan work binding differs from exact review: {identity}"
            )
        approved_bindings.add(
            (identity, entry["prior_work_id"], entry["current_work_id"], True, True)
        )

    required_work_ids: set[str] = set()
    for entry in targets:
        matching = [
            row
            for row in current.values()
            if row.get("person_id") == PERSON_ID
            and row.get("source_type") == "curated_verified"
            and row.get("source_native_id") == entry["source_native_id"]
        ]
        raw = [
            row
            for row in source.values()
            if row.get("person_id") == PERSON_ID
            and row.get("source_type") == "curated_verified"
            and row.get("source_native_id") == entry["source_native_id"]
        ]
        target_work = current_work.get(entry["work_id"])
        same_source = all(
            row.get("author_pub_id") == entry["author_pub_id"]
            and _normalized(row.get("title")) == _normalized(entry["title"])
            and _year(row.get("year")) == entry["year"]
            and row.get("source_url") == entry["source_url"]
            and _source_record_sha256(row) == entry["source_record_sha256"]
            for row in [*matching, *raw]
        )
        if (
            len(matching) != 1
            or len(raw) != 1
            or target_work is None
            or not same_source
            or matching[0].get("work_id") != entry["work_id"]
            or not bool(matching[0].get("map_eligible"))
            or not bool(matching[0].get("collaboration_eligible"))
            or not bool(target_work.get("map_eligible"))
            or not bool(target_work.get("collaboration_eligible"))
            or PERSON_ID not in _values(target_work.get("person_ids"))
        ):
            raise ValueError(
                "Sullivan curated target differs from exact review: "
                f"{entry['source_native_id']}"
            )
        required_work_ids.add(entry["work_id"])

    result = ExactSullivanContinuityAddendum(
        loss_identities=frozenset(loss_ids),
        binding_changes=frozenset(approved_bindings),
        required_curated_work_ids=frozenset(required_work_ids),
        manifest_sha256=ADDENDUM_SHA256,
    )
    if decision_set_digests(result) != (
        LOSS_IDS_SHA256,
        BINDINGS_SHA256,
        TARGETS_SHA256,
    ):
        raise ValueError("Sullivan continuity addendum exact decision set changed")
    return result
