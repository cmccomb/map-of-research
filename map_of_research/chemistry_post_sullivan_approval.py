"""Exact approval for the first Chemistry metadata release after Sullivan.

The checked-in manifest is deliberately inactive until the Sullivan artifact
exists on the Hub and its immutable revision is reviewed and recorded. A
future activation must update its pinned SHA-256 in this module as well.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file
from .sullivan_loss_approval import _source_record_sha256

DEFAULT_APPROVAL_PATH = Path(
    "data/curated/chemistry-post-sullivan-exact-continuity.json"
)
APPROVAL_SHA256 = "9d81662c4ff1c9b68cc67e3b940f0525bc3d55d4e2c3b7ae29eb11fcefd7506c"
DECISION_SET_SHA256 = "7a762c9eb4c500f9bc8b0f310f7a0190a6740e8e45f6f2532e660a6b9df031cd"
SOURCE_COMMIT = "8be6a2400cf72dd959c874bfcada8624540820ef"
SOURCE_SHA256 = "f8d7ffec683eb1830dadc77e612cbad7f3fa18ff199696041a71a9b8edbcefa2"
SULLIVAN_PARENT_REVISION = "bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd"
REVIEWED_BASE_COMMIT = "44c9d64c4b79ba66b1a6ee151746bc9ed8fd6270"
SCHOLAR_COUNTS = {
    "vF51LxIAAAAJ": 41,
    "zKKBwB4AAAAJ": 108,
    "FRtd8_QAAAAJ": 195,
    "ILambBMAAAAJ": 623,
}
PARENT_PATHS = frozenset(
    {
        "README.md",
        "authorships/train-00000-of-00001.parquet",
        "maps/manifest.json",
        "maps/publications.json",
        "people/train-00000-of-00001.parquet",
        "profile_publications/train-00000-of-00001.parquet",
        "works/train-00000-of-00001.parquet",
    }
)
DOCUMENT_FIELDS = {
    "schema_version",
    "reviewed_at",
    "status",
    "prior_revision",
    "reviewed_producer_base_commit",
    "source_snapshot_commit",
    "source_snapshot_sha256",
    "expected_parent_sha256_by_path",
    "raw_profile_rows_before_after",
    "scholar_cache_sha256",
    "affected_observations",
    "expected_counts",
}
ENTRY_FIELDS = {
    "author_pub_id",
    "person_id",
    "scholar_id",
    "prior_work_id",
    "candidate_work_id",
    "prior_map_eligible",
    "candidate_map_eligible",
    "prior_collaboration_eligible",
    "candidate_collaboration_eligible",
    "source_record_sha256",
    "decision",
    "evidence_urls",
    "review_path",
}
ObservationKey = tuple[str, str, str, str]
BindingChange = tuple[ObservationKey, str, str, bool, bool]


@dataclass(frozen=True)
class ExactChemistryPostSullivanApproval:
    prior_revision: str
    loss_identities: frozenset[ObservationKey]
    gain_identities: frozenset[ObservationKey]
    binding_changes: frozenset[BindingChange]
    manifest_sha256: str


def decision_set_digest(approval: ExactChemistryPostSullivanApproval) -> str:
    """Bind approval identities independently of the manifest object type."""

    lines = ["L|" + "/".join(identity) for identity in approval.loss_identities]
    lines.extend("G|" + "/".join(identity) for identity in approval.gain_identities)
    lines.extend(
        "B|"
        + "/".join(identity)
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
        ) in approval.binding_changes
    )
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def _identity(row: Mapping[str, Any]) -> ObservationKey:
    return tuple(
        str(row.get(field) or "")
        for field in ("person_id", "scholar_id", "author_pub_id", "source_type")
    )  # type: ignore[return-value]


def _index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[ObservationKey, Mapping[str, Any]]:
    records = list(rows)
    result = {_identity(row): row for row in records}
    if len(records) != len(result):
        raise ValueError("Chemistry approval contains duplicate observation identities")
    return result


def _state(row: Mapping[str, Any]) -> tuple[str, bool, bool]:
    return (
        str(row.get("work_id") or ""),
        bool(row.get("map_eligible")),
        bool(row.get("collaboration_eligible")),
    )


def _load_approved_document(
    approval_path: Path,
    *,
    prior_revision: str,
    source_snapshot_commit: str,
    source_snapshot_sha256: str,
) -> dict[str, Any]:
    if sha256_file(approval_path) != APPROVAL_SHA256:
        raise ValueError("Chemistry post-Sullivan approval manifest changed")
    document = load_json(approval_path)
    if not isinstance(document, dict) or set(document) != DOCUMENT_FIELDS:
        raise ValueError("Chemistry post-Sullivan approval schema changed")
    if (
        document["schema_version"] != 1
        or document["reviewed_at"] != "2026-10-25"
        or document["reviewed_producer_base_commit"] != REVIEWED_BASE_COMMIT
        or document["source_snapshot_commit"] != SOURCE_COMMIT
        or document["source_snapshot_sha256"] != SOURCE_SHA256
        or source_snapshot_commit != SOURCE_COMMIT
        or source_snapshot_sha256 != SOURCE_SHA256
    ):
        raise ValueError("Chemistry post-Sullivan approval source binding changed")
    # The draft's null parent and candidate status MUST NOT authorize a write.
    # Activation requires a subsequent reviewed commit with the actual Hub SHA.
    if (
        not isinstance(prior_revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", prior_revision)
        or prior_revision == SULLIVAN_PARENT_REVISION
        or document["status"] != "reviewed_exact_parent"
        or document["prior_revision"] != prior_revision
    ):
        raise ValueError("Chemistry post-Sullivan approval awaits actual Hub parent")
    if (
        not isinstance(document["expected_parent_sha256_by_path"], dict)
        or set(document["expected_parent_sha256_by_path"]) != PARENT_PATHS
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in document["expected_parent_sha256_by_path"].values()
        )
    ):
        raise ValueError("Chemistry post-Sullivan parent file set changed")
    return document


def _hosted_parent_hashes(
    *, prior_revision: str, hf_token: str, repo_id: str
) -> dict[str, str]:
    """Hash every file from the immutable hosted parent, never a local stage."""

    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=prior_revision,
            allow_patterns=sorted(PARENT_PATHS),
            token=hf_token or None,
        )
    )
    result: dict[str, str] = {}
    for relative_path in sorted(PARENT_PATHS):
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"Chemistry parent artifact is missing {relative_path}")
        result[relative_path] = sha256_file(path)
    return result


def validate_chemistry_post_sullivan_approval(
    *,
    approval_path: Path,
    prior_revision: str,
    source_snapshot_commit: str,
    source_snapshot_sha256: str,
    hf_token: str,
    repo_id: str,
    cache_dir: Path,
    prior_works: Iterable[Mapping[str, Any]],
    prior_profiles: Iterable[Mapping[str, Any]],
    source_rows: Iterable[Mapping[str, Any]],
    current_works: Iterable[Mapping[str, Any]],
    current_profiles: Iterable[Mapping[str, Any]],
) -> ExactChemistryPostSullivanApproval:
    """Require one exact loss, one gain, and five work bindings, with no drift."""

    document = _load_approved_document(
        approval_path,
        prior_revision=prior_revision,
        source_snapshot_commit=source_snapshot_commit,
        source_snapshot_sha256=source_snapshot_sha256,
    )
    if (
        _hosted_parent_hashes(
            prior_revision=prior_revision, hf_token=hf_token, repo_id=repo_id
        )
        != document["expected_parent_sha256_by_path"]
    ):
        raise ValueError(
            "Hosted Sullivan parent differs from reviewed seven-file stage"
        )
    if (
        document["raw_profile_rows_before_after"] != [967, 967]
        or document["expected_counts"]
        != {
            "held_losses": 1,
            "gains": 1,
            "still_visible_work_bindings": 5,
            "changed_rows": 7,
        }
        or not isinstance(document["scholar_cache_sha256"], dict)
        or set(document["scholar_cache_sha256"]) != set(SCHOLAR_COUNTS)
    ):
        raise ValueError("Chemistry post-Sullivan approval counts changed")
    for scholar_id, expected_hash in document["scholar_cache_sha256"].items():
        if sha256_file(cache_dir / f"{scholar_id}.json") != expected_hash:
            raise ValueError(f"Chemistry complete Scholar cache changed: {scholar_id}")

    previous = _index(prior_profiles)
    current = _index(current_profiles)
    source = _index(source_rows)
    if previous.keys() != current.keys() or previous.keys() != source.keys():
        raise ValueError("Chemistry release changed the frozen source identity set")
    for label, rows in (
        ("prior", previous),
        ("candidate", current),
        ("source", source),
    ):
        counts = Counter(
            identity[1] for identity in rows if identity[1] in SCHOLAR_COUNTS
        )
        if counts != SCHOLAR_COUNTS:
            raise ValueError(f"Chemistry {label} raw profile counts changed: {counts}")
    for identity in previous:
        if not (
            _source_record_sha256(previous[identity])
            == _source_record_sha256(current[identity])
            == _source_record_sha256(source[identity])
        ):
            raise ValueError(f"Frozen source record changed: {identity}")

    entries = document["affected_observations"]
    if (
        not isinstance(entries, list)
        or len(entries) != 7
        or any(
            not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS
            for entry in entries
        )
    ):
        raise ValueError("Chemistry post-Sullivan decision schema changed")
    ids = [
        (
            entry["person_id"],
            entry["scholar_id"],
            entry["author_pub_id"],
            "scholar",
        )
        for entry in entries
    ]
    if len(ids) != len(set(ids)) or ids != sorted(ids, key=lambda item: item[2]):
        raise ValueError("Chemistry reviewed identities are not sorted and unique")
    changed = {
        identity
        for identity in previous
        if _state(previous[identity]) != _state(current[identity])
    }
    if changed != set(ids):
        raise ValueError("Chemistry release has an unreviewed or unused row transition")

    prior_work_rows = list(prior_works)
    current_work_rows = list(current_works)
    prior_work = {str(row["work_id"]): row for row in prior_work_rows}
    current_work = {str(row["work_id"]): row for row in current_work_rows}
    if len(prior_work) != len(prior_work_rows) or len(current_work) != len(
        current_work_rows
    ):
        raise ValueError("Chemistry release contains duplicate work IDs")
    losses: set[ObservationKey] = set()
    gains: set[ObservationKey] = set()
    bindings: set[BindingChange] = set()
    review_root = approval_path.parent.parent.parent
    for entry, identity in zip(entries, ids, strict=True):
        old = previous[identity]
        now = current[identity]
        old_state = (
            entry["prior_work_id"],
            entry["prior_map_eligible"],
            entry["prior_collaboration_eligible"],
        )
        new_state = (
            entry["candidate_work_id"],
            entry["candidate_map_eligible"],
            entry["candidate_collaboration_eligible"],
        )
        if (
            _state(old) != old_state
            or _state(now) != new_state
            or _source_record_sha256(now) != entry["source_record_sha256"]
            or entry["prior_work_id"] not in prior_work
            or entry["candidate_work_id"] not in current_work
            or not isinstance(entry["evidence_urls"], list)
            or not entry["evidence_urls"]
            or any(
                not str(url).startswith("https://") for url in entry["evidence_urls"]
            )
            or not (review_root / entry["review_path"]).is_file()
        ):
            raise ValueError(f"Chemistry reviewed observation differs: {identity}")
        decision = entry["decision"]
        if decision == "approved_held_loss":
            if not (
                old_state[1:] == (True, True)
                and new_state[1:] == (False, False)
                and bool(prior_work[old_state[0]].get("map_eligible"))
            ):
                raise ValueError(f"Chemistry held loss changed: {identity}")
            losses.add(identity)
        elif decision == "approved_gain_and_binding_change":
            if not (
                old_state[1:] == (False, False)
                and new_state[1:] == (True, True)
                and bool(current_work[new_state[0]].get("map_eligible"))
            ):
                raise ValueError(f"Chemistry gain changed: {identity}")
            gains.add(identity)
        elif decision == "approved_binding_change":
            if not (
                old_state[1:] == (True, True)
                and new_state[1:] == (True, True)
                and old_state[0] != new_state[0]
                and bool(prior_work[old_state[0]].get("map_eligible"))
                and bool(current_work[new_state[0]].get("map_eligible"))
            ):
                raise ValueError(f"Chemistry work binding changed: {identity}")
            bindings.add((identity, old_state[0], new_state[0], True, True))
        else:
            raise ValueError(f"Chemistry decision is unknown: {identity}")
    if (len(losses), len(gains), len(bindings)) != (1, 1, 5):
        raise ValueError("Chemistry exact decision count changed")
    approval = ExactChemistryPostSullivanApproval(
        prior_revision=prior_revision,
        loss_identities=frozenset(losses),
        gain_identities=frozenset(gains),
        binding_changes=frozenset(bindings),
        manifest_sha256=APPROVAL_SHA256,
    )
    if decision_set_digest(approval) != DECISION_SET_SHA256:
        raise ValueError("Chemistry exact decision set changed")
    return approval
