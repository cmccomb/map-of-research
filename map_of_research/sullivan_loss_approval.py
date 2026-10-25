"""Exact, source-bound approval for Sullivan's bb88 visible-row transition.

This one-time decision is separate from the Bio poster decision. It permits
only the 18 source-validated provisional holds and four reviewed Scholar ID
rekeys in the immutable Sullivan source, after checking the hosted prior rows.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file
from .provisional_holds import ProvisionalHold

SULLIVAN_ID = "wXSgFr4AAAAJ"
PRIOR_REVISION = "bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd"
SOURCE_COMMIT = "8ba438215b50ff877f1a4dd0ce21ca2ed5f9b2d0"
SOURCE_SHA256 = "f8d7ffec683eb1830dadc77e612cbad7f3fa18ff199696041a71a9b8edbcefa2"
APPROVAL_SHA256 = "2c2c1ae0cd8fc71a722ad5a4f5bd4100d4cd9112ec9ba1738b325f1205777d2f"
APPROVED_IDS_SHA256 = "30df0017bfcc094334d3bbafbe511d74627d83bd102ccca0e2f5d8a4fcfb4761"
DEFAULT_APPROVAL_PATH = Path("data/curated/sullivan-exact-losses-bb88.json")
DEFAULT_CROSSWALK_PATH = Path(
    "registry/reviews/2026-10-25-chem-sullivan-prior-id-crosswalk.json"
)

ObservationKey = tuple[str, str, str, str]
FIELDS = ("person_id", "scholar_id", "author_pub_id", "source_type")
ENTRY_FIELDS = {
    *FIELDS,
    "decision",
    "prior_title",
    "prior_year",
    "prior_work_id",
    "prior_source_record_sha256",
    "evidence_url",
}
DOCUMENT_FIELDS = {
    "schema_version",
    "reviewed_at",
    "prior_revision",
    "source_snapshot_commit",
    "source_snapshot_sha256",
    "sullivan_cache_sha256",
    "profile_releases_sha256",
    "provisional_holds_sha256",
    "prior_id_crosswalk_sha256",
    "curated_works_sha256",
    "reviewed_exclusions_sha256",
    "approved_losses",
}


@dataclass(frozen=True)
class ExactSullivanLossApproval:
    identities: frozenset[ObservationKey]
    manifest_sha256: str
    held_identities: frozenset[ObservationKey]
    rekey_identities: frozenset[ObservationKey]


def _identity(row: Mapping[str, Any]) -> ObservationKey:
    return tuple(str(row.get(field) or "") for field in FIELDS)  # type: ignore[return-value]


def _source_record_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(str(row.get("source_record_json") or "").encode()).hexdigest()


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def validate_sullivan_exact_losses(
    *,
    approval_path: Path,
    prior_revision: str,
    source_snapshot_commit: str,
    source_snapshot_sha256: str,
    cache_path: Path,
    profile_releases_path: Path,
    provisional_holds_path: Path,
    crosswalk_path: Path,
    curated_works_path: Path,
    exclusions_path: Path,
    prior_works: Iterable[Mapping[str, Any]],
    prior_profiles: Iterable[Mapping[str, Any]],
    source_rows: Iterable[Mapping[str, Any]],
    current_profiles: Iterable[Mapping[str, Any]],
    validated_holds: Mapping[tuple[str, str], ProvisionalHold],
) -> ExactSullivanLossApproval:
    """Verify each proposed loss against bb88 rows and the reviewed source."""

    if (
        prior_revision != PRIOR_REVISION
        or source_snapshot_commit != SOURCE_COMMIT
        or source_snapshot_sha256 != SOURCE_SHA256
    ):
        raise ValueError("Sullivan loss approval is bound to a different release")
    if sha256_file(approval_path) != APPROVAL_SHA256:
        raise ValueError("Sullivan exact-loss approval manifest changed")
    document = load_json(approval_path)
    if not isinstance(document, dict) or set(document) != DOCUMENT_FIELDS:
        raise ValueError("Sullivan exact-loss approval schema changed")
    if (
        document["schema_version"] != 1
        or document["reviewed_at"] != "2026-10-25"
        or document["prior_revision"] != PRIOR_REVISION
        or document["source_snapshot_commit"] != SOURCE_COMMIT
        or document["source_snapshot_sha256"] != SOURCE_SHA256
    ):
        raise ValueError("Sullivan exact-loss approval release binding changed")
    for field, path in (
        ("sullivan_cache_sha256", cache_path),
        ("profile_releases_sha256", profile_releases_path),
        ("provisional_holds_sha256", provisional_holds_path),
        ("prior_id_crosswalk_sha256", crosswalk_path),
        ("curated_works_sha256", curated_works_path),
        ("reviewed_exclusions_sha256", exclusions_path),
    ):
        if sha256_file(path) != document[field]:
            raise ValueError(f"Sullivan exact-loss approval source changed: {field}")

    entries = document["approved_losses"]
    if not isinstance(entries, list) or len(entries) != 22:
        raise ValueError("Sullivan exact-loss approval needs exactly 22 rows")
    if any(not isinstance(row, dict) or set(row) != ENTRY_FIELDS for row in entries):
        raise ValueError("Sullivan exact-loss approval entry schema changed")
    identities = [_identity(row) for row in entries]
    if identities != sorted(identities) or len(set(identities)) != 22:
        raise ValueError("Sullivan exact-loss identities must be sorted and unique")
    actual_digest = hashlib.sha256(
        "\n".join("/".join(identity) for identity in identities).encode()
    ).hexdigest()
    if actual_digest != APPROVED_IDS_SHA256:
        raise ValueError("Sullivan exact-loss identity set changed")
    if any(
        identity[0] != "person-sullivan-589c9545"
        or identity[1] != SULLIVAN_ID
        or identity[3] != "scholar"
        for identity in identities
    ):
        raise ValueError("Sullivan exact-loss approval includes another profile")

    prior_work_rows = list(prior_works)
    prior_profile_rows = list(prior_profiles)
    raw_rows = list(source_rows)
    candidate_rows = list(current_profiles)
    prior_work_by_id = {str(row["work_id"]): row for row in prior_work_rows}
    prior_profile_by_id = {_identity(row): row for row in prior_profile_rows}
    raw_by_id = {_identity(row): row for row in raw_rows}
    current_profile_by_id = {_identity(row): row for row in candidate_rows}
    if (
        len(prior_work_by_id) != len(prior_work_rows)
        or len(prior_profile_by_id) != len(prior_profile_rows)
        or len(raw_by_id) != len(raw_rows)
        or len(current_profile_by_id) != len(candidate_rows)
    ):
        raise ValueError("Prior profile observations contain duplicate identities")
    crosswalk = load_json(crosswalk_path)
    if (
        not isinstance(crosswalk, dict)
        or crosswalk.get("scholar_id") != SULLIVAN_ID
        or crosswalk.get("person_id") != "person-sullivan-589c9545"
        or crosswalk.get("replacement_cache_sha256")
        != document["sullivan_cache_sha256"]
        or not isinstance(crosswalk.get("entries"), list)
    ):
        raise ValueError("Sullivan exact-loss crosswalk differs from reviewed source")
    rekeys = {item["prior_author_pub_id"]: item for item in crosswalk["entries"]}
    if len(rekeys) != 4:
        raise ValueError("Sullivan exact-loss crosswalk must contain four rekeys")
    held: set[ObservationKey] = set()
    rekeyed: set[ObservationKey] = set()
    for entry, identity in zip(entries, identities, strict=True):
        prior = prior_profile_by_id.get(identity)
        if prior is None:
            raise ValueError(
                f"Approved Sullivan loss missing from Hub parent: {identity}"
            )
        prior_work_id = str(prior.get("work_id") or "")
        prior_work = prior_work_by_id.get(prior_work_id)
        if (
            prior_work_id != entry["prior_work_id"]
            or prior_work is None
            or not bool(prior.get("map_eligible"))
            or not bool(prior_work.get("map_eligible"))
            or _source_record_sha256(prior) != entry["prior_source_record_sha256"]
            or str(prior.get("title") or "") != str(entry["prior_title"] or "")
            or prior.get("year") != entry["prior_year"]
        ):
            raise ValueError(
                f"Approved Sullivan loss differs from Hub parent: {identity}"
            )
        current = current_profile_by_id.get(identity)
        raw = raw_by_id.get(identity)
        evidence_url = str(entry["evidence_url"])
        if entry["decision"] == "provisional_hold":
            hold = validated_holds.get((identity[1], identity[2]))
            if (
                current is None
                or raw is None
                or hold is None
                or hold.evidence_url != evidence_url
                or bool(current.get("map_eligible"))
                or current.get("review_attribution_status") != "held"
                or current.get("review_attribution_evidence_url") != evidence_url
                or "attribution_provisional_hold"
                not in _values(current.get("map_exclusion_reasons"))
            ):
                raise ValueError(f"Sullivan held loss lacks exact evidence: {identity}")
            held.add(identity)
        elif entry["decision"] == "source_rekey":
            rekey = rekeys.get(identity[2])
            if (
                current is not None
                or raw is not None
                or rekey is None
                or rekey["legacy_row_sha256"] != entry["prior_source_record_sha256"]
                or rekey["evidence_url"] != evidence_url
            ):
                raise ValueError(
                    f"Sullivan rekey loss lacks exact evidence: {identity}"
                )
            if rekey["target_kind"] == "current_scholar":
                target_identity = (
                    identity[0],
                    identity[1],
                    rekey["target_id"],
                    "scholar",
                )
                target = current_profile_by_id.get(target_identity)
                raw_target = raw_by_id.get(target_identity)
                if (
                    target is None
                    or raw_target is None
                    or _normalized(raw_target.get("title"))
                    != _normalized(rekey["target_title"])
                    or str(raw_target.get("year")) != rekey["target_year"]
                ):
                    raise ValueError(f"Sullivan rekey target absent: {identity}")
                if rekey["classification"] == "false_profile_attribution_rekeyed":
                    if target.get(
                        "review_exclusion_code"
                    ) != "false_profile_attribution" or bool(
                        target.get("map_eligible")
                    ):
                        raise ValueError(
                            f"Sullivan excluded rekey target changed: {identity}"
                        )
                elif rekey["classification"] == "duplicate_rekeyed":
                    if (
                        target.get("review_exclusion_code")
                        or not bool(target.get("map_eligible"))
                        or not bool(target.get("collaboration_eligible"))
                        or str(target.get("work_id") or "") != prior_work_id
                    ):
                        raise ValueError(
                            f"Sullivan duplicate rekey target not mapped: {identity}"
                        )
                else:
                    raise ValueError(
                        f"Sullivan rekey classification changed: {identity}"
                    )
            elif rekey["target_kind"] == "curated_native":
                target = [
                    row
                    for row in current_profile_by_id.values()
                    if row.get("person_id") == identity[0]
                    and row.get("source_type") == "curated_verified"
                    and row.get("source_native_id") == rekey["target_id"]
                ]
                raw_target = [
                    row
                    for row in raw_rows
                    if row.get("person_id") == identity[0]
                    and row.get("source_type") == "curated_verified"
                    and row.get("source_native_id") == rekey["target_id"]
                ]
                if (
                    rekey["classification"] != "historical_conference_abstract"
                    or len(target) != 1
                    or len(raw_target) != 1
                    or not bool(target[0].get("map_eligible"))
                    or _normalized(raw_target[0].get("title"))
                    != _normalized(rekey["target_title"])
                    or str(raw_target[0].get("year")) != rekey["target_year"]
                    or evidence_url
                    not in [
                        raw_target[0].get("source_url"),
                        *_values(raw_target[0].get("source_verification_urls")),
                    ]
                ):
                    raise ValueError(
                        f"Sullivan curated rekey target absent: {identity}"
                    )
            else:
                raise ValueError(f"Sullivan rekey target type changed: {identity}")
            rekeyed.add(identity)
        else:
            raise ValueError(f"Sullivan loss decision changed: {identity}")
    return ExactSullivanLossApproval(
        identities=frozenset(identities),
        manifest_sha256=APPROVAL_SHA256,
        held_identities=frozenset(held),
        rekey_identities=frozenset(rekeyed),
    )
