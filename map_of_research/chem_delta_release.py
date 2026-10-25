"""Apply the 19 reviewed Chemistry decisions to the exact published #337 parent.

This is a separate release path. The earlier bb88 Sullivan/Math composite gate
stays intact; it cannot be reused after that release becomes the Hub parent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic_release import (
    RELEASE_PATHS,
    AtomicReleaseReconciliationRequired,
    commit_atomic_release,
)
from .chem_region_review import (
    CHEM_REVIEW_SHA256,
    SOURCE_COMMIT,
    SOURCE_SHA256,
    load_chem_region_review,
)
from .composite_region import (
    BLANK_TOPICS,
    HIDDEN_MATH_ATTRIBUTION_FIELDS,
    UNCLASSIFIED_TOPICS,
    _digest_row,
)
from .hub_card import build_updated_dataset_card
from .io import atomic_write_json, load_json, sha256_file
from .publisher import (
    LAYOUTS,
    REPO_ID,
    STATUS_SCHEMA_VERSION,
    _to_hub_dataset,
    _write_map_artifacts,
)
from .region_continuity import (
    WORK_FINGERPRINT_FIELDS,
    _fingerprint_fields,
    require_revision,
)
from .region_review import ReviewedRegionDecision
from .registry import Registry
from .release_baseline import pin_prior_release

OLD_PARENT = "bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd"
REVIEWED_337_PRODUCER = "a68fbf0c4ea1623b84c01497f40f718f5edac7f8"
PARENT_POINT_COUNT = 42909
TARGET_POINT_COUNT = 42907
PARENT_TABLE_ROWS = {
    "people": 2001,
    "works": 47743,
    "authorships": 49629,
    "profile_publications": 51879,
}
TARGET_TABLE_ROWS = {**PARENT_TABLE_ROWS, "authorships": 49620}
INTENT_BRANCH = "automation/map-snapshot"
INTENT_REMOTE_PATH = "cmccomb/map-of-research"
PRODUCER_REGISTRY_SHA256 = (
    "7b5cfc69839a5428d317db768c87994f506b5d833980ba63233799bbbd2e0953"
)
MATH_BACKFILL_FIELDS = frozenset(
    {
        "primary_department_id",
        "primary_department_source_url",
        "primary_department_verified_at",
        "primary_department_basis",
    }
)
MATH_BACKFILL_COUNT = 49

# These are the seven immutable files in the successful #337 Hub release
# 38f119d71692cdb9112210217b532a7dd7c1633a (producer a68fbf0c, frozen
# source 8be6a240). The status/last-upload.json receipt records their hashes.
# A later Hub head qualifies only if *all seven* immutable files match.
PARENT_SHA256_BY_PATH = {
    "README.md": ("69c92647bf73c293e7e372e64b5a7ce0902923dda0e25933b1e957a29c6d7173"),
    "authorships/train-00000-of-00001.parquet": (
        "1398a6d9410945c0eb17601cc42e13efb851ebddb481af6b1725e1e836175a7d"
    ),
    "maps/manifest.json": (
        "2707db369661b39703cbd08b05db324ad624a2d3815a1844a06569d3ca84ae04"
    ),
    "maps/publications.json": (
        "431ca9f1ed98e0a04e025438d16940296528efa145271be987596c664d32eff3"
    ),
    "people/train-00000-of-00001.parquet": (
        "8c369014272b84d44f61621ed0c93077f0195d73156f95065acc3b19d52d57e4"
    ),
    "profile_publications/train-00000-of-00001.parquet": (
        "d9f7202470c927fde0b7e776bbea0b32d9192eae38516c7f187926f4fde3b5ba"
    ),
    "works/train-00000-of-00001.parquet": (
        "ef071bce69745fe9815f0e038267541b6ad9c242d195cc73de3c9573f74a7c91"
    ),
}


@dataclass(frozen=True)
class ExactParent:
    revision: str
    files: Mapping[str, Path]


def require_exact_parent_files(
    files: Mapping[str, Path],
    *,
    expected_hashes: Mapping[str, str] = PARENT_SHA256_BY_PATH,
) -> None:
    """Reject any parent other than the seven-file #337 candidate."""

    if set(files) != RELEASE_PATHS or set(expected_hashes) != RELEASE_PATHS:
        raise ValueError("Chemistry parent must contain the exact seven release files")
    for name in sorted(RELEASE_PATHS):
        if sha256_file(files[name]) != expected_hashes[name]:
            raise ValueError(f"Chemistry parent differs from #337 receipt: {name}")


def require_frozen_registry_transition(
    frozen: Registry,
    current: Registry,
    *,
    people_path: Path,
    memberships_path: Path,
    departments_path: Path,
    source_registry_sha256: str,
) -> dict[str, Any]:
    """Allow only the reviewed 49-row Math home backfill after source capture."""

    from .snapshot import _registry_paths, _registry_sha256

    current_digest = _registry_sha256(
        _registry_paths(people_path, memberships_path, departments_path)
    )
    if current_digest != PRODUCER_REGISTRY_SHA256:
        raise ValueError("Chemistry producer registry differs from reviewed backfill")
    if (
        tuple(person.person_id for person in frozen.people)
        != tuple(person.person_id for person in current.people)
        or frozen.memberships != current.memberships
        or frozen.departments != current.departments
        or frozen.schools != current.schools
    ):
        raise ValueError("Chemistry source and producer registry membership drift")

    changed_people = []
    for before, after in zip(frozen.people, current.people, strict=True):
        before_fields = vars(before)
        after_fields = vars(after)
        changed_fields = {
            key for key in before_fields if before_fields[key] != after_fields[key]
        }
        if frozen.primary_affiliation(before.person_id).published_fields() != (
            current.primary_affiliation(after.person_id).published_fields()
        ):
            raise ValueError("Chemistry effective primary academic home changed")
        if not changed_fields:
            continue
        included = [
            membership
            for membership in frozen.memberships
            if membership.person_id == before.person_id and membership.included
        ]
        if (
            changed_fields != MATH_BACKFILL_FIELDS
            or any(before_fields[key] for key in MATH_BACKFILL_FIELDS)
            or len(included) != 1
            or included[0].department_id != "math"
            or after.primary_department_id != "math"
            or after.primary_department_source_url != included[0].source_url
            or after.primary_department_verified_at != included[0].verified_at
            or after.primary_department_basis != "single_included_membership"
        ):
            raise ValueError(
                "Chemistry producer registry has unreviewed person changes"
            )
        changed_people.append(before.person_id)
    if len(changed_people) != MATH_BACKFILL_COUNT:
        raise ValueError("Chemistry producer registry Math backfill count changed")
    return {
        "status": "pass",
        "source_registry_sha256": source_registry_sha256,
        "producer_registry_sha256": current_digest,
        "people": len(frozen.people),
        "explicit_math_home_rows": len(changed_people),
        "effective_primary_homes_frozen": len(frozen.people),
    }


def require_live_337_parent(
    expected_revision: str,
    *,
    hf_token: str | None,
    api: Any | None = None,
    download: Callable[..., str] | None = None,
) -> ExactParent:
    """Read the current immutable Hub revision and pin every #337 file byte."""

    from huggingface_hub import HfApi, hf_hub_download

    revision = require_revision(expected_revision)
    if revision == OLD_PARENT:
        raise ValueError("#337 is not yet the live Hub parent")
    api = api or HfApi(token=hf_token)
    download = download or hf_hub_download
    if api.repo_info(REPO_ID, repo_type="dataset").sha != revision:
        raise ValueError("Live Hub head is not the requested Chemistry parent")
    paths = {
        name: Path(
            download(
                REPO_ID,
                name,
                repo_type="dataset",
                revision=revision,
                token=hf_token,
            )
        )
        for name in sorted(RELEASE_PATHS)
    }
    require_exact_parent_files(paths)
    if api.repo_info(REPO_ID, repo_type="dataset").sha != revision:
        raise ValueError("Live Hub head moved while reading the #337 parent")
    return ExactParent(revision, paths)


def durable_intent_sink(path: Path) -> Callable[[Mapping[str, Any]], None]:
    """Write and read back each checkpoint on the current filesystem.

    Release orchestration must retain the intent outside an ephemeral runner
    before allowing a real Hub write; local readback alone is not cross-run
    durability.
    """

    if path.exists():
        raise ValueError("Chemistry intent path must be new for this release attempt")

    def persist(intent: Mapping[str, Any]) -> None:
        payload = dict(intent)
        atomic_write_json(path, payload)
        if load_json(path) != payload:
            raise RuntimeError("Chemistry atomic intent did not survive readback")

    return persist


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _serialized_status_bytes(payload: Mapping[str, Any]) -> bytes:
    """Match atomic_write_json without rereading a mutable checkout file."""

    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _remote_intent_tip(repo: Path) -> str:
    fields = _git(repo, "ls-remote", "--heads", "origin", INTENT_BRANCH).split()
    if len(fields) != 2 or fields[1] != f"refs/heads/{INTENT_BRANCH}":
        raise RuntimeError("Snapshot intent branch is unavailable")
    return fields[0]


def _require_intent_remote(repo: Path) -> None:
    remote = _git(repo, "remote", "get-url", "origin")
    if remote not in {
        f"https://github.com/{INTENT_REMOTE_PATH}",
        f"https://github.com/{INTENT_REMOTE_PATH}.git",
        f"git@github.com:{INTENT_REMOTE_PATH}.git",
    }:
        raise ValueError(
            "Chemistry intent requires the canonical GitHub snapshot remote"
        )


def _push_and_readback_intent(
    repo: Path,
    paths: Sequence[Path],
    *,
    expected_previous_tip: str,
    expected_bytes_by_path: Mapping[str, bytes],
) -> str:
    """Commit status, push without force, then compare remote blobs byte for byte."""

    _require_intent_remote(repo)
    if (
        _git(repo, "rev-parse", "HEAD") != expected_previous_tip
        or _remote_intent_tip(repo) != expected_previous_tip
    ):
        raise RuntimeError("Snapshot branch moved before Chemistry intent checkpoint")
    relative = []
    for path in paths:
        try:
            name = path.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(
                "Chemistry intent status is outside snapshot checkout"
            ) from error
        if name not in {
            "status/last-upload.json",
            "status/last-successful-upload.json",
        }:
            raise ValueError("Chemistry intent may checkpoint only upload status")
        relative.append(name)
    if set(relative) != set(expected_bytes_by_path):
        raise ValueError("Chemistry checkpoint expected bytes differ from status paths")
    if any(
        (repo / name).read_bytes() != expected_bytes_by_path[name] for name in relative
    ):
        raise RuntimeError("Chemistry status changed before remote checkpoint")
    _git(
        repo,
        "-c",
        "user.name=github-actions[bot]",
        "-c",
        "user.email=41898282+github-actions[bot]@users.noreply.github.com",
        "commit",
        "--only",
        "-m",
        "Record Chemistry atomic release intent [skip ci]",
        "--",
        *relative,
    )
    local_tip = _git(repo, "rev-parse", "HEAD")
    _require_intent_remote(repo)
    _git(repo, "push", "origin", f"HEAD:{INTENT_BRANCH}")
    if _remote_intent_tip(repo) != local_tip:
        raise RuntimeError("Chemistry intent push did not become remote head")
    remote_ref = f"refs/remotes/origin/{INTENT_BRANCH}"
    _git(
        repo,
        "fetch",
        "--no-tags",
        "origin",
        f"refs/heads/{INTENT_BRANCH}:{remote_ref}",
    )
    if _git(repo, "rev-parse", remote_ref) != local_tip:
        raise RuntimeError("Chemistry fetched intent differs from pushed commit")
    _require_intent_remote(repo)
    for name, expected in expected_bytes_by_path.items():
        remote_bytes = subprocess.run(
            ["git", "-C", str(repo), "show", f"{remote_ref}:{name}"],
            check=True,
            capture_output=True,
        ).stdout
        if remote_bytes != expected:
            raise RuntimeError(f"Chemistry remote intent readback differs: {name}")
    if _remote_intent_tip(repo) != local_tip:
        raise RuntimeError("Snapshot branch moved after Chemistry intent readback")
    return local_tip


class RetainedGitIntent:
    """Keep a pending release intent on the snapshot branch before Hub writes."""

    def __init__(
        self,
        *,
        intent_path: Path,
        snapshot_checkout: Path,
        parent_revision: str,
        producer_commit: str,
    ) -> None:
        self.repo = snapshot_checkout.resolve(strict=True)
        self.status_path = self.repo / "status/last-upload.json"
        self.success_path = self.repo / "status/last-successful-upload.json"
        self.parent_revision = require_revision(parent_revision)
        self.producer_commit = require_revision(producer_commit)
        self.started_at_utc = dt.datetime.now(dt.UTC).isoformat()
        server = os.environ.get("GITHUB_SERVER_URL")
        repository = os.environ.get("GITHUB_REPOSITORY")
        run_id = os.environ.get("GITHUB_RUN_ID")
        self.workflow_url = (
            f"{server}/{repository}/actions/runs/{run_id}"
            if server and repository and run_id
            else None
        )
        self.local_sink = durable_intent_sink(intent_path)
        _require_intent_remote(self.repo)
        if not self.success_path.is_file() or not self.status_path.is_file():
            raise ValueError("Chemistry intent requires both prior upload receipts")
        if _git(self.repo, "status", "--porcelain", "--", "status"):
            raise ValueError("Chemistry snapshot status checkout is dirty")
        self.checkpoint_tip = _git(self.repo, "rev-parse", "HEAD")
        if self.checkpoint_tip != _remote_intent_tip(self.repo):
            raise RuntimeError("Snapshot branch moved before Chemistry release")
        if pin_prior_release(self.repo / "status") != self.parent_revision:
            raise ValueError(
                "Snapshot successful receipt differs from live #337 parent"
            )
        self.last_intent: dict[str, Any] | None = None

    def __call__(self, intent: Mapping[str, Any]) -> None:
        if (
            intent.get("parent_revision") != self.parent_revision
            or intent.get("release_context", {}).get("producer_commit")
            != self.producer_commit
            or intent.get("release_context", {}).get("snapshot_commit") != SOURCE_COMMIT
        ):
            raise ValueError("Chemistry intent context differs from frozen release")
        self.local_sink(intent)
        status = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "status": "failure",
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": None,
            "dataset_id": REPO_ID,
            "dataset_configs": list(PARENT_TABLE_ROWS),
            "workflow_url": self.workflow_url,
            "producer_commit": self.producer_commit,
            "snapshot_commit": SOURCE_COMMIT,
            "prior_revision": self.parent_revision,
            "error_type": "AtomicWriteOutcomeUnknown",
            "error_message": (
                "Inspect the retained Chemistry atomic intent before retrying"
            ),
            "atomic_release_intent": dict(intent),
        }
        atomic_write_json(self.status_path, status)
        if load_json(self.status_path) != status:
            raise RuntimeError("Chemistry status intent failed local readback")
        self.checkpoint_tip = _push_and_readback_intent(
            self.repo,
            [self.status_path],
            expected_previous_tip=self.checkpoint_tip,
            expected_bytes_by_path={
                "status/last-upload.json": _serialized_status_bytes(status)
            },
        )
        self.last_intent = dict(intent)

    def finalize_success(self, revision: str, receipt: Mapping[str, Any]) -> None:
        if self.last_intent is None:
            raise RuntimeError("Chemistry release has no retained prewrite intent")
        revision = require_revision(revision)
        if (
            receipt.get("artifact_revision") != revision
            or receipt.get("parent_revision") != self.parent_revision
            or receipt.get("commit_nonce") != self.last_intent.get("commit_nonce")
            or receipt.get("sha256_by_path") != self.last_intent.get("sha256_by_path")
            or receipt.get("size_by_path") != self.last_intent.get("size_by_path")
            or receipt.get("hosted_file_bytes_match_prewrite") is not True
        ):
            raise ValueError(
                "Chemistry successful receipt differs from retained intent"
            )
        status = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "status": "success",
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "dataset_id": REPO_ID,
            "dataset_configs": list(PARENT_TABLE_ROWS),
            "workflow_url": self.workflow_url,
            "producer_commit": self.producer_commit,
            "snapshot_commit": SOURCE_COMMIT,
            "prior_revision": self.parent_revision,
            "artifact_commit": revision,
            "error_type": None,
            "error_message": None,
            "atomic_release_intent": self.last_intent,
            "atomic_release": dict(receipt),
        }
        atomic_write_json(self.status_path, status)
        atomic_write_json(self.success_path, status)
        if (
            load_json(self.status_path) != status
            or load_json(self.success_path) != status
        ):
            raise RuntimeError("Chemistry success receipt failed local readback")
        self.checkpoint_tip = _push_and_readback_intent(
            self.repo,
            [self.status_path, self.success_path],
            expected_previous_tip=self.checkpoint_tip,
            expected_bytes_by_path={
                "status/last-upload.json": _serialized_status_bytes(status),
                "status/last-successful-upload.json": _serialized_status_bytes(status),
            },
        )


def retained_git_intent_sink(
    intent_path: Path,
    snapshot_checkout: Path,
    *,
    parent_revision: str,
    producer_commit: str,
) -> RetainedGitIntent:
    return RetainedGitIntent(
        intent_path=intent_path,
        snapshot_checkout=snapshot_checkout,
        parent_revision=parent_revision,
        producer_commit=producer_commit,
    )


def _review_sets(
    decisions: Sequence[ReviewedRegionDecision],
) -> tuple[dict[str, ReviewedRegionDecision], dict[str, ReviewedRegionDecision]]:
    by_work = {item.work_id: item for item in decisions}
    by_observation = {item.author_pub_id: item for item in decisions}
    if (
        len(decisions) != 19
        or len(by_work) != 19
        or len(by_observation) != 19
        or sum(item.visible and item.transition == "restore" for item in decisions) != 3
        or sum(
            not item.visible and item.transition == "reexclude" for item in decisions
        )
        != 5
        or sum(not item.visible for item in decisions) != 9
        or sum(item.output_type == "conference_presentation" for item in decisions) != 2
    ):
        raise ValueError("Chemistry release is not the exact 19-decision set")
    return by_work, by_observation


def _target_profile(
    prior: Mapping[str, Any], decision: ReviewedRegionDecision
) -> dict[str, Any]:
    """Derive one target observation while keeping every raw Scholar field."""

    if (
        prior.get("person_id") != decision.person_id
        or prior.get("scholar_id") != decision.scholar_id
        or prior.get("author_pub_id") != decision.author_pub_id
        or prior.get("source_type") != "scholar"
        or prior.get("work_id") != decision.work_id
        or hashlib.sha256(str(prior.get("source_record_json", "")).encode()).hexdigest()
        != decision.source_record_sha256
        or prior.get("map_eligible") is not True
        or prior.get("collaboration_eligible") is not True
        or prior.get("map_exclusion_reasons") != []
        or prior.get("publication_type") != "publication"
        or prior.get("review_exclusion_code")
        or prior.get("review_attribution_status")
    ):
        raise ValueError(
            f"Chemistry parent observation drift: {decision.author_pub_id}"
        )
    target = dict(prior)
    if decision.output_type:
        target.update(
            publication_type=decision.output_type,
            context_review_reason=decision.reason,
            context_evidence_url=decision.evidence_url,
            context_verified_at=decision.reviewed_at,
        )
    if decision.decision == "hold":
        target.update(
            map_eligible=False,
            map_exclusion_reasons=["attribution_provisional_hold"],
            review_attribution_status="held",
            review_attribution_evidence_url=decision.evidence_url,
            collaboration_eligible=False,
        )
    elif decision.decision == "exclude":
        target.update(
            map_eligible=False,
            map_exclusion_reasons=[f"reviewed_{decision.reason_code}"],
            review_exclusion_code=decision.reason_code,
            review_exclusion_reason=decision.reason,
            review_evidence_url=decision.evidence_url,
            review_verified_at=decision.reviewed_at,
            collaboration_eligible=False,
        )
    return target


def _target_work(
    prior: Mapping[str, Any], decision: ReviewedRegionDecision
) -> dict[str, Any]:
    """Apply the exact decision without recomputing another Bernhard row."""

    fingerprint = _fingerprint_fields(
        prior, (*WORK_FINGERPRINT_FIELDS, "quality_assessment_version")
    )
    was_visible = decision.transition == "reexclude"
    if (
        prior.get("work_id") != decision.work_id
        or fingerprint != decision.work_fingerprint_sha256
        or bool(prior.get("map_eligible")) != was_visible
        or bool(prior.get("map_region_outlier")) == was_visible
        or prior.get("map_exclusion_reasons")
        != ([] if was_visible else ["low_information_region"])
    ):
        raise ValueError(f"Chemistry parent work drift: {decision.work_id}")
    target = dict(prior)
    if decision.visible and decision.transition == "restore":
        target.update(
            map_eligible=True,
            map_region_outlier=False,
            map_exclusion_reasons=[],
        )
        target.update(UNCLASSIFIED_TOPICS)
    elif not decision.visible:
        reason = (
            "attribution_provisional_hold"
            if decision.decision == "hold"
            else f"reviewed_{decision.reason_code}"
        )
        target.update(
            map_eligible=False,
            map_region_outlier=False,
            map_exclusion_reasons=[reason],
            collaboration_eligible=False,
        )
        for field in HIDDEN_MATH_ATTRIBUTION_FIELDS - {"collaboration_eligible"}:
            target[field] = []
        target.update(BLANK_TOPICS)
    if decision.output_type:
        target["publication_type"] = decision.output_type
        target["context_evidence_urls"] = [decision.evidence_url]
        if not target["collaboration_eligible"]:
            raise ValueError("Reviewed Donahue presentation lost authored links")
    return target


def _target_person(
    prior: Mapping[str, Any], hidden_count: int, map_delta: int
) -> dict[str, Any]:
    target = dict(prior)
    for field, delta in (
        ("unique_work_count", -hidden_count),
        ("mapped_work_count", map_delta),
    ):
        value = prior[field]
        if type(value) is not int or value + delta < 0:
            raise ValueError(
                f"Chemistry parent person count drift: {prior['person_id']}"
            )
        target[field] = value + delta
    return target


def _row_key(config_name: str, row: Mapping[str, Any]) -> str:
    field = (
        "person_id"
        if config_name == "people"
        else ("work_id" if config_name == "works" else "observation_id")
    )
    return str(row.get(field) or "")


def _rows(path: Path):
    import pyarrow.parquet as parquet

    for batch in parquet.ParquetFile(path).iter_batches(batch_size=128):
        yield from batch.to_pylist()


def _parent_targets(
    parent: ExactParent,
    decisions: Sequence[ReviewedRegionDecision],
) -> tuple[dict[str, dict[str, dict[str, Any]]], set[str]]:
    """Pin all 19 work/profile identities and the exact nine authored removals."""

    by_work, by_observation = _review_sets(decisions)
    targets: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in PARENT_TABLE_ROWS
    }
    observation_ids: dict[str, str] = {}
    observations_by_work: Counter[str] = Counter()
    for row in _rows(parent.files["works/train-00000-of-00001.parquet"]):
        if (decision := by_work.get(row["work_id"])) is not None:
            targets["works"][row["work_id"]] = _target_work(row, decision)
    for row in _rows(parent.files["profile_publications/train-00000-of-00001.parquet"]):
        if row["work_id"] in by_work:
            observations_by_work[row["work_id"]] += 1
        if (decision := by_observation.get(row["author_pub_id"])) is not None:
            if row["author_pub_id"] in observation_ids:
                raise ValueError(
                    "Chemistry parent repeats a reviewed source observation"
                )
            observation_ids[row["author_pub_id"]] = row["observation_id"]
            targets["profile_publications"][row["observation_id"]] = _target_profile(
                row, decision
            )
    if (
        len(targets["works"]) != 19
        or len(observation_ids) != 19
        or set(observations_by_work) != set(by_work)
        or any(value != 1 for value in observations_by_work.values())
    ):
        raise ValueError("Chemistry parent is missing or repeats reviewed works/rows")
    hidden = {
        observation_ids[item.author_pub_id] for item in decisions if not item.visible
    }
    authored: Counter[str] = Counter()
    authorships_by_work: Counter[str] = Counter()
    for row in _rows(parent.files["authorships/train-00000-of-00001.parquet"]):
        if row["work_id"] in by_work:
            authorships_by_work[row["work_id"]] += 1
        if row["observation_id"] in observation_ids.values():
            authored[row["observation_id"]] += 1
    if (
        len(authored) != 19
        or any(value != 1 for value in authored.values())
        or any(value != 1 for value in authorships_by_work.values())
        or set(authorships_by_work) != set(by_work)
        or not hidden <= authored.keys()
    ):
        raise ValueError("Chemistry parent lacks the exact 19 authored observations")
    hidden_by_person = Counter(item.person_id for item in decisions if not item.visible)
    map_delta_by_person: Counter[str] = Counter()
    for item in decisions:
        if item.visible and item.transition == "restore":
            map_delta_by_person[item.person_id] += 1
        elif not item.visible and item.transition == "reexclude":
            map_delta_by_person[item.person_id] -= 1
    affected_people = set(hidden_by_person) | {
        person_id for person_id, delta in map_delta_by_person.items() if delta
    }
    for row in _rows(parent.files["people/train-00000-of-00001.parquet"]):
        if row["person_id"] in affected_people:
            targets["people"][row["person_id"]] = _target_person(
                row,
                hidden_by_person[row["person_id"]],
                map_delta_by_person[row["person_id"]],
            )
    if len(targets["people"]) != 6 or len(affected_people) != 6:
        raise ValueError("Chemistry parent lacks six affected people")
    return targets, hidden


def _replace_rows(
    frame: Any,
    config_name: str,
    targets: Mapping[str, Mapping[str, Any]],
    hidden: set[str],
) -> Any:
    """Retain source order and replace only checked target rows."""

    if config_name == "authorships":
        output = frame.loc[~frame["observation_id"].isin(hidden)].copy()
        return output.reset_index(drop=True)
    output = frame.copy()
    key_field = (
        "person_id"
        if config_name == "people"
        else ("work_id" if config_name == "works" else "observation_id")
    )
    indices = {str(value): index for index, value in output[key_field].items()}
    if not set(targets) <= indices.keys():
        raise ValueError(f"Chemistry {config_name} target is missing from parent")
    for key, target in targets.items():
        index = indices[key]
        for field, value in target.items():
            if field not in output:
                raise ValueError(f"Chemistry {config_name} field disappeared: {field}")
            output.at[index, field] = value
    return output


def validate_exact_tables(
    parent: ExactParent,
    staged: Mapping[str, Path],
    targets: Mapping[str, Mapping[str, Mapping[str, Any]]],
    hidden: set[str],
) -> dict[str, int]:
    """Compare every staged row with its parent or source-bound target."""

    import pyarrow.parquet as parquet

    counts: dict[str, int] = {}
    for name, before_count in PARENT_TABLE_ROWS.items():
        path = f"{name}/train-00000-of-00001.parquet"
        before = parquet.ParquetFile(parent.files[path])
        after = parquet.ParquetFile(staged[path])
        if (
            not before.schema_arrow.equals(after.schema_arrow, check_metadata=True)
            or before.metadata.num_rows != before_count
            or after.metadata.num_rows != TARGET_TABLE_ROWS[name]
        ):
            raise ValueError(f"Chemistry {name} schema or row count changed")
        prior_digest: dict[str, str] = {}
        prior_order: list[str] = []
        for row in _rows(parent.files[path]):
            key = _row_key(name, row)
            if not key or key in prior_digest:
                raise ValueError(f"Chemistry parent {name} repeats an identity")
            prior_order.append(key)
            expected = targets[name].get(key, row)
            prior_digest[key] = _digest_row(expected, exclude=frozenset())
        expected_order = (
            [key for key in prior_order if key not in hidden]
            if name == "authorships"
            else prior_order
        )
        staged_order: list[str] = []
        for row in _rows(staged[path]):
            key = _row_key(name, row)
            staged_order.append(key)
            if key not in prior_digest or key in hidden and name == "authorships":
                raise ValueError(f"Unreviewed Chemistry {name} row appeared: {key}")
            if _digest_row(row, exclude=frozenset()) != prior_digest[key]:
                raise ValueError(f"Unreviewed Chemistry {name} payload drift: {key}")
        if staged_order != expected_order:
            raise ValueError(f"Chemistry {name} row identity or order drift")
        counts[name] = len(staged_order)
    return counts


def validate_exact_map(
    prior: Mapping[str, Any],
    candidate: Mapping[str, Any],
    decisions: Sequence[ReviewedRegionDecision],
) -> dict[str, Any]:
    """Freeze every shared point payload, including all Sullivan coordinates."""

    previous = prior.get("points")
    current = candidate.get("points")
    if not isinstance(previous, list) or not isinstance(current, list):
        raise ValueError("Chemistry map lacks point arrays")
    before = {point["work_id"]: point for point in previous}
    after = {point["work_id"]: point for point in current}
    by_work, _ = _review_sets(decisions)
    gains = {
        item.work_id
        for item in by_work.values()
        if item.visible and item.transition == "restore"
    }
    losses = {
        item.work_id
        for item in by_work.values()
        if not item.visible and item.transition == "reexclude"
    }
    if (
        len(before) != PARENT_POINT_COUNT
        or len(previous) != PARENT_POINT_COUNT
        or len(after) != TARGET_POINT_COUNT
        or len(current) != TARGET_POINT_COUNT
        or candidate.get("point_count") != TARGET_POINT_COUNT
        or after.keys() - before.keys() != gains
        or before.keys() - after.keys() != losses
    ):
        raise ValueError("Chemistry map changed more than exact +3/-5 points")
    shared = before.keys() & after.keys()
    if any(before[key] != after[key] for key in shared):
        raise ValueError("Chemistry map changed an existing point payload")
    if [p["work_id"] for p in previous if p["work_id"] in shared] != [
        p["work_id"] for p in current if p["work_id"] in shared
    ]:
        raise ValueError("Chemistry map reordered shared points")
    if type(prior.get("excluded_work_count")) is not int or candidate.get(
        "excluded_work_count"
    ) != prior["excluded_work_count"] + len(previous) - len(current):
        raise ValueError("Chemistry map excluded-work count is not derived")
    old_keywords = prior.get("keywords")
    new_keywords = candidate.get("keywords")
    if (
        not isinstance(old_keywords, list)
        or not isinstance(new_keywords, list)
        or type(prior.get("keyword_count")) is not int
        or prior["keyword_count"] != len(old_keywords)
        or candidate.get("keyword_count") != len(new_keywords)
        or len(old_keywords) != len(new_keywords)
        or [entry.get("keyword_id") for entry in old_keywords]
        != [entry.get("keyword_id") for entry in new_keywords]
    ):
        raise ValueError("Chemistry map keyword inventory changed")
    members_by_keyword: dict[str, list[dict[str, Any]]] = {}
    for point in current:
        for keyword_id in point.get("keyword_ids", []):
            members_by_keyword.setdefault(str(keyword_id), []).append(point)
    if set(members_by_keyword) != {entry["keyword_id"] for entry in new_keywords}:
        raise ValueError("Chemistry map keyword members changed")

    def static_fields(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in item.items()
            if key not in {"publication_count", "coordinates"}
        }

    for old_keyword, new_keyword in zip(old_keywords, new_keywords, strict=True):
        keyword_id = new_keyword["keyword_id"]
        members = members_by_keyword[keyword_id]
        expected_coordinates = {
            str(layout["layout_id"]): {
                axis: round(
                    sum(float(point[str(layout[field])]) for point in members)
                    / len(members),
                    7,
                )
                for axis, field in (("x", "x_field"), ("y", "y_field"))
            }
            for layout in LAYOUTS
        }
        if (
            static_fields(old_keyword) != static_fields(new_keyword)
            or new_keyword.get("publication_count") != len(members)
            or new_keyword.get("coordinates") != expected_coordinates
        ):
            raise ValueError("Chemistry map keyword metadata is not derived")
    for decision in decisions:
        if decision.output_type != "conference_presentation":
            continue
        point = after.get(decision.work_id)
        if (
            point is None
            or point.get("faculty_ids") != [decision.person_id]
            or point.get("external_coauthor_ids") != []
            or point.get("collaboration_eligible") is not True
            or point.get("publication_type") != "conference_presentation"
        ):
            raise ValueError(
                "Chemistry presentation unexpectedly creates a coauthor edge"
            )
    before_catalogs = prior.get("catalogs", {})
    after_catalogs = candidate.get("catalogs", {})
    if (
        not isinstance(before_catalogs, dict)
        or not isinstance(after_catalogs, dict)
        or before_catalogs.keys() != after_catalogs.keys()
    ):
        raise ValueError("Chemistry map changed unrelated catalogs")
    for catalog, identity, point_field in (
        ("schools", "school_id", "school_ids"),
        ("departments", "department_id", "department_ids"),
        ("faculty", "person_id", "faculty_ids"),
    ):
        if catalog not in before_catalogs:
            continue
        old_entries = before_catalogs[catalog]
        new_entries = after_catalogs[catalog]
        if [entry[identity] for entry in old_entries] != [
            entry[identity] for entry in new_entries
        ]:
            raise ValueError(f"Chemistry map changed {catalog} identities or order")
        old_counts = Counter(
            member for point in previous for member in point.get(point_field, [])
        )
        new_counts = Counter(
            member for point in current for member in point.get(point_field, [])
        )
        for old_entry, new_entry in zip(old_entries, new_entries, strict=True):
            member_id = old_entry[identity]
            if (
                {
                    key: value
                    for key, value in old_entry.items()
                    if key != "publication_count"
                }
                != {
                    key: value
                    for key, value in new_entry.items()
                    if key != "publication_count"
                }
                or old_entry["publication_count"] != old_counts[member_id]
                or new_entry["publication_count"] != new_counts[member_id]
            ):
                raise ValueError(f"Chemistry map changed {catalog} metadata or count")
    if "external_coauthors" in before_catalogs:
        old_external = before_catalogs["external_coauthors"]
        new_external = after_catalogs["external_coauthors"]
        old_ids = {entry["person_id"] for entry in old_external}
        new_ids = {entry["person_id"] for entry in new_external}
        if (
            len(old_ids) != len(old_external)
            or len(new_ids) != len(new_external)
            or old_ids
            != {
                member
                for point in previous
                for member in point.get("external_coauthor_ids", [])
            }
            or new_ids
            != {
                member
                for point in current
                for member in point.get("external_coauthor_ids", [])
            }
            or {
                entry["person_id"]: entry
                for entry in old_external
                if entry["person_id"] in new_ids
            }
            != {
                entry["person_id"]: entry
                for entry in new_external
                if entry["person_id"] in old_ids
            }
        ):
            raise ValueError("Chemistry map changed unrelated external coauthors")
    if prior.keys() != candidate.keys():
        raise ValueError("Chemistry map changed non-derived metadata keys")
    if any(
        field
        not in {
            "point_count",
            "excluded_work_count",
            "keyword_count",
            "keywords",
            "catalogs",
            "points",
        }
        and prior.get(field) != candidate.get(field)
        for field in prior
    ):
        raise ValueError("Chemistry map changed non-derived metadata")
    return {
        "status": "pass",
        "map_points": TARGET_POINT_COUNT,
        "restored_work_ids": sorted(gains),
        "removed_work_ids": sorted(losses),
        "shared_points_frozen": len(shared),
    }


def stage_chem_delta(
    parent: ExactParent,
    decisions: Sequence[ReviewedRegionDecision],
    registry: Registry,
    stage_dir: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Build all seven files and validate the actual serialized stage."""

    import pyarrow.parquet as parquet

    require_revision(parent.revision)
    if parent.revision == OLD_PARENT:
        raise ValueError("Chemistry delta needs a published #337 parent")
    require_exact_parent_files(parent.files)
    stage_dir.mkdir(parents=True, exist_ok=True)
    targets, hidden = _parent_targets(parent, decisions)
    files: dict[str, Path] = {}
    staged_datasets: dict[str, tuple[Any, Path]] = {}
    for name in PARENT_TABLE_ROWS:
        relpath = f"{name}/train-00000-of-00001.parquet"
        frame = parquet.read_table(parent.files[relpath]).to_pandas()
        changed = _replace_rows(frame, name, targets[name], hidden)
        if name == "profile_publications":
            # Arrow int64 with nulls becomes float64 on Pandas conversion. Restore
            # the parent's nullable integer feature before HF serialization.
            changed["year"] = changed["year"].astype("Int64")
        dataset = _to_hub_dataset(changed)
        path = stage_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(path))
        files[relpath] = path
        staged_datasets[name] = dataset, path
    table_counts = validate_exact_tables(parent, files, targets, hidden)
    prior_map = json.loads(parent.files["maps/publications.json"].read_text())
    staged_works = parquet.read_table(
        files["works/train-00000-of-00001.parquet"]
    ).to_pandas()
    continuity_receipt = {
        "version": "exact-chemistry-post-337-delta-v1",
        "prior_revision": parent.revision,
        "source_commit": SOURCE_COMMIT,
        "source_snapshot_sha256": SOURCE_SHA256,
        "chem_region_review_sha256": CHEM_REVIEW_SHA256,
        "reviewed_337_producer": REVIEWED_337_PRODUCER,
        "restored_work_ids": sorted(
            item.work_id
            for item in decisions
            if item.visible and item.transition == "restore"
        ),
        "removed_work_ids": sorted(
            item.work_id
            for item in decisions
            if not item.visible and item.transition == "reexclude"
        ),
    }
    map_path, manifest_path, artifact = _write_map_artifacts(
        staged_works,
        registry,
        maps_dir=stage_dir / "maps",
        generated_at_utc=str(prior_map["generated_at_utc"]),
        dataset_commits={},
        continuity_receipt=continuity_receipt,
    )
    serialized_artifact = json.loads(map_path.read_text(encoding="utf-8"))
    if serialized_artifact != artifact:
        raise ValueError("Chemistry serialized map differs from validated artifact")
    map_receipt = validate_exact_map(prior_map, serialized_artifact, decisions)
    prior_manifest = json.loads(parent.files["maps/manifest.json"].read_text())
    staged_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        **prior_manifest,
        "region_continuity": continuity_receipt,
        "artifact": {
            "title": artifact["title"],
            "point_count": artifact["point_count"],
            "excluded_work_count": artifact["excluded_work_count"],
            "department_count": artifact["department_count"],
            "school_count": artifact["school_count"],
            "faculty_count": artifact["faculty_count"],
            "keyword_count": artifact["keyword_count"],
            "keyword_levels": artifact["keyword_levels"],
            "file": "publications.json",
        },
    }
    if staged_manifest != expected_manifest:
        raise ValueError("Chemistry serialized map manifest changed")
    files["maps/publications.json"] = map_path
    files["maps/manifest.json"] = manifest_path
    prior_readme = parent.files["README.md"].read_text(encoding="utf-8")
    readme = stage_dir / "README.md"
    updated_card = build_updated_dataset_card(prior_readme, staged_datasets)
    readme.write_text(updated_card, encoding="utf-8")
    if readme.read_text(encoding="utf-8") != updated_card:
        raise ValueError("Chemistry serialized README changed")
    files["README.md"] = readme
    if set(files) != RELEASE_PATHS:
        raise ValueError("Chemistry stage does not contain exactly seven files")
    stage_hashes = {name: sha256_file(path) for name, path in sorted(files.items())}
    gate = {
        "status": "pass",
        "gate": "exact-chemistry-post-337-delta-v1",
        "prior_revision": parent.revision,
        "source_commit": SOURCE_COMMIT,
        "source_snapshot_sha256": SOURCE_SHA256,
        "chem_region_review_sha256": CHEM_REVIEW_SHA256,
        "reviewed_337_producer": REVIEWED_337_PRODUCER,
        "parent_sha256_by_path": dict(PARENT_SHA256_BY_PATH),
        "stage_sha256_by_path": stage_hashes,
        "tables": table_counts,
        "authorship_delta": {
            "added": 0,
            "removed": [
                {
                    "observation_id": observation_id,
                    "work_id": profile["work_id"],
                    "person_id": profile["person_id"],
                }
                for observation_id, profile in sorted(
                    targets["profile_publications"].items()
                )
                if observation_id in hidden
            ],
            "shared_payloads_frozen": table_counts["authorships"],
            "donahue_presentation_coauthor_edges_added": 0,
        },
        "map": map_receipt,
    }
    return files, gate


def publish_chem_delta(
    *,
    expected_parent_revision: str,
    hf_token: str,
    snapshot_path: Path,
    snapshot_manifest_path: Path,
    source_cache_dir: Path,
    people_path: Path,
    memberships_path: Path,
    departments_path: Path,
    chem_review_path: Path,
    stage_dir: Path,
    intent_path: Path,
    snapshot_checkout: Path,
    producer_commit: str,
) -> tuple[str, dict[str, Any]]:
    """Release only after a future live #337 parent passes every pinned gate."""

    from .registry import load_registry
    from .snapshot import validate_snapshot

    if not hf_token or not hf_token.strip():
        raise ValueError("Chemistry release requires an explicit HF_TOKEN")
    _require_producer_checkout(Path(__file__).resolve().parents[1], producer_commit)
    parent = require_live_337_parent(expected_parent_revision, hf_token=hf_token)
    intent_sink = retained_git_intent_sink(
        intent_path,
        snapshot_checkout,
        parent_revision=parent.revision,
        producer_commit=producer_commit,
    )
    registry = load_registry(people_path, memberships_path, departments_path)
    frozen_registry_dir = snapshot_path.parent.parent / "registry"
    frozen_people = frozen_registry_dir / "people.csv"
    frozen_memberships = frozen_registry_dir / "memberships.csv"
    frozen_departments = frozen_registry_dir / "departments.csv"
    frame, manifest = validate_snapshot(
        snapshot_path,
        snapshot_manifest_path,
        people_path=frozen_people,
        memberships_path=frozen_memberships,
        departments_path=frozen_departments,
    )
    if sha256_file(snapshot_path) != SOURCE_SHA256:
        raise ValueError("Chemistry release raw source bytes changed")
    frozen_registry = load_registry(
        frozen_people, frozen_memberships, frozen_departments
    )
    registry_transition = require_frozen_registry_transition(
        frozen_registry,
        registry,
        people_path=people_path,
        memberships_path=memberships_path,
        departments_path=departments_path,
        source_registry_sha256=str(manifest["registry_sha256"]),
    )
    decisions = load_chem_region_review(
        chem_review_path,
        snapshot_frame=frame,
        snapshot_sha256=str(manifest["snapshot_sha256"]),
        cache_dir=source_cache_dir,
        registry=registry,
    )
    files, gate = stage_chem_delta(parent, decisions, registry, stage_dir)
    gate["registry_transition"] = registry_transition
    revision, receipt = commit_atomic_release(
        files,
        prewrite_gate=gate,
        repo_id=REPO_ID,
        hf_token=hf_token,
        prior_revision=parent.revision,
        commit_message="Publish exact Chemistry region review atomically",
        intent_sink=intent_sink,
        intent_metadata={
            "source_commit": SOURCE_COMMIT,
            "producer_review_commit": REVIEWED_337_PRODUCER,
            "producer_commit": require_revision(producer_commit),
            "snapshot_commit": SOURCE_COMMIT,
        },
    )
    try:
        intent_sink.finalize_success(revision, receipt)
    except Exception as error:
        raise AtomicReleaseReconciliationRequired(
            {
                **receipt,
                "artifact_revision": revision,
                "verification_status": "reconciliation_required",
                "verification_stage": "persist_success_receipt",
                "write_outcome": "verified_hub_status_pending",
                "verification_error_type": type(error).__name__,
                "verification_error_message": str(error)[:500],
            }
        ) from error
    return revision, receipt


def _require_producer_checkout(checkout: Path, expected_commit: str) -> None:
    """Run the reviewed producer code from an exact clean Git checkout."""

    checkout = checkout.resolve(strict=True)
    expected_commit = require_revision(expected_commit)
    if _git(checkout, "rev-parse", "HEAD") != expected_commit:
        raise ValueError("Chemistry producer checkout differs from reviewed SHA")
    if _git(checkout, "status", "--porcelain"):
        raise ValueError("Chemistry producer checkout must be clean")
    if Path(__file__).resolve() != checkout / "map_of_research/chem_delta_release.py":
        raise ValueError("Chemistry release code is outside reviewed checkout")


def main(argv: Sequence[str] | None = None) -> int:
    """Manual production entrypoint; no release is inferred from a branch name."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-parent-revision", required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--producer-checkout", type=Path, required=True)
    parser.add_argument("--source-data-checkout", type=Path, required=True)
    parser.add_argument("--snapshot-checkout", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--intent-path", type=Path, required=True)
    args = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError("Chemistry release requires an explicit HF_TOKEN")
    parent_revision = require_revision(args.expected_parent_revision)
    if parent_revision == OLD_PARENT:
        raise ValueError("Chemistry release requires the live #337 Hub parent")
    producer_commit = require_revision(args.expected_producer_commit)
    _require_producer_checkout(args.producer_checkout, producer_commit)
    source_data = args.source_data_checkout.resolve(strict=True)
    if _git(source_data, "rev-parse", "HEAD") != SOURCE_COMMIT:
        raise ValueError("Chemistry source data differs from frozen source commit")
    if _git(source_data, "status", "--porcelain"):
        raise ValueError("Chemistry frozen source data checkout must be clean")
    if args.stage_dir.exists():
        raise ValueError("Chemistry stage directory must be new for this attempt")
    if args.intent_path.exists():
        raise ValueError("Chemistry local intent path must be new for this attempt")
    source = args.producer_checkout.resolve(strict=True)
    snapshot = args.snapshot_checkout.resolve(strict=True)
    revision, receipt = publish_chem_delta(
        expected_parent_revision=parent_revision,
        hf_token=token,
        snapshot_path=source_data / "snapshots/cmu-engineering-publications.parquet",
        snapshot_manifest_path=source_data
        / "snapshots/cmu-engineering-publications.parquet.manifest.json",
        source_cache_dir=source_data / "data/authors",
        people_path=source / "registry/people.csv",
        memberships_path=source / "registry/memberships.csv",
        departments_path=source / "registry/departments.csv",
        chem_review_path=source / "data/curated/scholar-chem-region-review.json",
        stage_dir=args.stage_dir,
        intent_path=args.intent_path,
        snapshot_checkout=snapshot,
        producer_commit=producer_commit,
    )
    if receipt.get("artifact_revision") != revision:
        raise RuntimeError("Chemistry final receipt differs from hosted revision")
    print(f"revision={revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
