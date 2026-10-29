"""Validate the one reviewed McComb profile rekey and year-correction release."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file

PRIOR_REVISION = "d065c5153de7dea5b2fb7df80461ea70e5b95aad"
SNAPSHOT_COMMIT = "d926f2b593dcd445879f89772c45ba9676c3ae40"
SNAPSHOT_SHA256 = "1987a58644139ef191788d33aa7ddcd37cc7e0d83df967fa8ba2c6e6c27546e3"
CACHE_SHA256 = "8caef8d4f83afbf582281b2572202a23a820e8ab093e6d423dd9fd853287e56f"
CROSSWALK_SHA256 = "8718731034cd88011480387498ca97db7a8b8190ad6ea76b032c77db48084675"
APPROVAL_SHA256 = "3ced6245bd1a53758573488a2d7aefee25412037124db4f8c34c018866e644a0"
PERSON_ID = "person-mccomb-03260969"
SCHOLAR_ID = "0P9w_S0AAAAJ"
LOSS_IDS = frozenset(
    {
        f"{SCHOLAR_ID}:5qfkUJPXOUwC",
        f"{SCHOLAR_ID}:IRz6iEL74y4C",
        f"{SCHOLAR_ID}:QYdC8u9Cj1oC",
    }
)
BINDING_ID = f"{SCHOLAR_ID}:kuK5TVdYjLIC"
DEFAULT_APPROVAL_PATH = Path("data/curated/mccomb-exact-continuity-20261029.json")
DEFAULT_CROSSWALK_PATH = Path(
    "registry/reviews/2026-10-29-mccomb-prior-id-crosswalk.json"
)

ObservationKey = tuple[str, str, str, str]
BindingChange = tuple[ObservationKey, str, str, bool, bool]


@dataclass(frozen=True)
class ExactMcCombContinuity:
    manifest_sha256: str
    loss_identities: frozenset[ObservationKey]
    binding_changes: frozenset[BindingChange]


def _identity(row: Mapping[str, Any]) -> ObservationKey:
    return (
        str(row.get("person_id") or ""),
        str(row.get("scholar_id") or ""),
        str(row.get("author_pub_id") or ""),
        str(row.get("source_type") or ""),
    )


def _indexed(rows: list[Mapping[str, Any]]) -> dict[ObservationKey, Mapping[str, Any]]:
    indexed = {_identity(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("McComb continuity found duplicate observation identities")
    return indexed


def validate_exact_mccomb_continuity(
    approval_path: Path,
    *,
    prior_revision: str,
    source_snapshot_commit: str,
    source_snapshot_sha256: str,
    cache_path: Path,
    crosswalk_path: Path,
    prior_profiles: list[Mapping[str, Any]],
    current_profiles: list[Mapping[str, Any]],
) -> ExactMcCombContinuity:
    """Authorize only the exact reviewed transition, source, and visible rows."""

    if (
        prior_revision != PRIOR_REVISION
        or source_snapshot_commit != SNAPSHOT_COMMIT
        or source_snapshot_sha256 != SNAPSHOT_SHA256
        or sha256_file(cache_path) != CACHE_SHA256
        or sha256_file(crosswalk_path) != CROSSWALK_SHA256
        or sha256_file(approval_path) != APPROVAL_SHA256
    ):
        raise ValueError("McComb continuity approval differs from reviewed source")
    approval = load_json(approval_path)
    crosswalk = load_json(crosswalk_path)
    if (
        not isinstance(approval, dict)
        or approval.get("schema_version") != 1
        or approval.get("person_id") != PERSON_ID
        or approval.get("scholar_id") != SCHOLAR_ID
        or approval.get("prior_revision") != PRIOR_REVISION
        or approval.get("source_snapshot_commit") != SNAPSHOT_COMMIT
        or approval.get("source_snapshot_sha256") != SNAPSHOT_SHA256
        or approval.get("source_cache_sha256") != CACHE_SHA256
        or approval.get("crosswalk_sha256") != CROSSWALK_SHA256
        or not isinstance(approval.get("losses"), list)
        or len(approval["losses"]) != 3
        or not isinstance(approval.get("binding_changes"), list)
        or len(approval["binding_changes"]) != 1
        or not isinstance(crosswalk, dict)
        or not isinstance(crosswalk.get("entries"), list)
    ):
        raise ValueError("McComb continuity approval is invalid")
    crosswalk_targets = {
        item["prior_author_pub_id"]: item["target_id"] for item in crosswalk["entries"]
    }
    old = _indexed(prior_profiles)
    new = _indexed(current_profiles)
    loss_keys: set[ObservationKey] = set()
    for item in approval["losses"]:
        prior_id = item["prior_author_pub_id"]
        target_id = item["target_author_pub_id"]
        prior_key = (PERSON_ID, SCHOLAR_ID, prior_id, "scholar")
        target_key = (PERSON_ID, SCHOLAR_ID, target_id, "scholar")
        before = old.get(prior_key)
        after = new.get(target_key)
        if (
            prior_id not in LOSS_IDS
            or prior_key in loss_keys
            or crosswalk_targets.get(prior_id) != target_id
            or before is None
            or prior_key in new
            or after is None
            or str(before.get("work_id") or "") != item["prior_work_id"]
            or str(after.get("work_id") or "") != item["target_work_id"]
            or not bool(before.get("map_eligible"))
            or not bool(before.get("collaboration_eligible"))
            or not bool(after.get("map_eligible"))
            or not bool(after.get("collaboration_eligible"))
            or sum(
                str(row.get("work_id") or "") == item["prior_work_id"]
                for row in prior_profiles
            )
            != item["prior_work_observation_count"]
        ):
            raise ValueError("McComb rekey loss or target differs from review")
        # Other faculty's observations on a versioned preprint must remain.
        for identity, prior_row in old.items():
            if (
                identity == prior_key
                or prior_row.get("work_id") != item["prior_work_id"]
            ):
                continue
            current_row = new.get(identity)
            if (
                current_row is None
                or current_row.get("work_id") != prior_row.get("work_id")
                or bool(current_row.get("map_eligible"))
                != bool(prior_row.get("map_eligible"))
                or bool(current_row.get("collaboration_eligible"))
                != bool(prior_row.get("collaboration_eligible"))
            ):
                raise ValueError("McComb rekey would change another faculty edge")
        loss_keys.add(prior_key)
    if {key[2] for key in loss_keys} != LOSS_IDS:
        raise ValueError("McComb continuity does not cover the exact three rekeys")

    item = approval["binding_changes"][0]
    key = (PERSON_ID, SCHOLAR_ID, BINDING_ID, "scholar")
    before = old.get(key)
    after = new.get(key)
    if (
        item["author_pub_id"] != BINDING_ID
        or before is None
        or after is None
        or str(before.get("work_id") or "") != item["prior_work_id"]
        or str(after.get("work_id") or "") != item["current_work_id"]
        or int(before.get("year") or 0) != item["prior_year"]
        or int(after.get("year") or 0) != item["current_year"]
        or str(before.get("title") or "") != str(after.get("title") or "")
        or bool(before.get("collaboration_eligible")) is not True
        or bool(after.get("collaboration_eligible")) is not True
        or bool(before.get("map_eligible")) is not True
        or bool(after.get("map_eligible")) is not True
    ):
        raise ValueError("McComb year-correction binding differs from review")
    return ExactMcCombContinuity(
        manifest_sha256=APPROVAL_SHA256,
        loss_identities=frozenset(loss_keys),
        binding_changes=frozenset(
            {(key, item["prior_work_id"], item["current_work_id"], True, True)}
        ),
    )
