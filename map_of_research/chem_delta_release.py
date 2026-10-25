"""Apply the 19 reviewed Chemistry decisions to the exact published #337 parent.

This is a separate release path. The earlier bb88 Sullivan/Math composite gate
stays intact; it cannot be reused after that release becomes the Hub parent.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic_release import RELEASE_PATHS, commit_atomic_release
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
from .io import sha256_file
from .publisher import (
    REPO_ID,
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

OLD_PARENT = "bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd"
REVIEWED_337_PRODUCER = "514254a8118c28ea3f145c9a30f6f89b589ec550"
PARENT_POINT_COUNT = 42909
TARGET_POINT_COUNT = 42907
PARENT_TABLE_ROWS = {
    "people": 2001,
    "works": 47743,
    "authorships": 49629,
    "profile_publications": 51879,
}
TARGET_TABLE_ROWS = {**PARENT_TABLE_ROWS, "authorships": 49620}

# These are the seven bytes in the merged-main, first-write-intercepted #337 receipt:
# /Users/work/Documents/Codex/2026-10-24/you-x20/work/
# sullivan-math-merged-main-preview/receipt.json (producer 514254a8, source 8be6a240).
# A later Hub head qualifies only if *all seven* immutable files match.
PARENT_SHA256_BY_PATH = {
    "README.md": ("69c92647bf73c293e7e372e64b5a7ce0902923dda0e25933b1e957a29c6d7173"),
    "authorships/train-00000-of-00001.parquet": (
        "1398a6d9410945c0eb17601cc42e13efb851ebddb481af6b1725e1e836175a7d"
    ),
    "maps/manifest.json": (
        "3a7cac63bbaac51662c5f907d1552d8739f4a36ee73d30f654ad69173101a6ae"
    ),
    "maps/publications.json": (
        "01e3486c026125d6c3d8a943a56d193080bf2e2b642838649bdca88713be1872"
    ),
    "people/train-00000-of-00001.parquet": (
        "8c369014272b84d44f61621ed0c93077f0195d73156f95065acc3b19d52d57e4"
    ),
    "profile_publications/train-00000-of-00001.parquet": (
        "d9f7202470c927fde0b7e776bbea0b32d9192eae38516c7f187926f4fde3b5ba"
    ),
    "works/train-00000-of-00001.parquet": (
        "a655680907be759bbebfab5195f4b6a000f1d9f540911cb72f9525e08e705a71"
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


def _target_person(prior: Mapping[str, Any], hidden_count: int) -> dict[str, Any]:
    target = dict(prior)
    for field in ("unique_work_count", "mapped_work_count"):
        value = prior[field]
        if type(value) is not int or value < hidden_count:
            raise ValueError(
                f"Chemistry parent person count drift: {prior['person_id']}"
            )
        target[field] = value - hidden_count
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
    for row in _rows(parent.files["people/train-00000-of-00001.parquet"]):
        if row["person_id"] in hidden_by_person:
            targets["people"][row["person_id"]] = _target_person(
                row, hidden_by_person[row["person_id"]]
            )
    if len(targets["people"]) != 5 or len(hidden_by_person) != 5:
        raise ValueError("Chemistry parent lacks five affected people")
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
    map_path, manifest_path, artifact = _write_map_artifacts(
        staged_works,
        registry,
        maps_dir=stage_dir / "maps",
        generated_at_utc=str(prior_map["generated_at_utc"]),
        dataset_commits={},
        continuity_receipt={
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
        },
    )
    map_receipt = validate_exact_map(prior_map, artifact, decisions)
    files["maps/publications.json"] = map_path
    files["maps/manifest.json"] = manifest_path
    prior_readme = parent.files["README.md"].read_text(encoding="utf-8")
    readme = stage_dir / "README.md"
    readme.write_text(
        build_updated_dataset_card(prior_readme, staged_datasets), encoding="utf-8"
    )
    files["README.md"] = readme
    if set(files) != RELEASE_PATHS:
        raise ValueError("Chemistry stage does not contain exactly seven files")
    gate = {
        "status": "pass",
        "gate": "exact-chemistry-post-337-delta-v1",
        "prior_revision": parent.revision,
        "source_commit": SOURCE_COMMIT,
        "source_snapshot_sha256": SOURCE_SHA256,
        "chem_region_review_sha256": CHEM_REVIEW_SHA256,
        "reviewed_337_producer": REVIEWED_337_PRODUCER,
        "parent_sha256_by_path": dict(PARENT_SHA256_BY_PATH),
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
    atomic_intent_sink: Callable[[Mapping[str, Any]], None],
) -> tuple[str, dict[str, Any]]:
    """Release only after a future live #337 parent passes every pinned gate."""

    from .registry import load_registry
    from .snapshot import validate_snapshot

    parent = require_live_337_parent(expected_parent_revision, hf_token=hf_token)
    registry = load_registry(people_path, memberships_path, departments_path)
    frame, manifest = validate_snapshot(
        snapshot_path,
        snapshot_manifest_path,
        people_path=people_path,
        memberships_path=memberships_path,
        departments_path=departments_path,
    )
    if sha256_file(snapshot_path) != SOURCE_SHA256:
        raise ValueError("Chemistry release raw source bytes changed")
    decisions = load_chem_region_review(
        chem_review_path,
        snapshot_frame=frame,
        snapshot_sha256=str(manifest["snapshot_sha256"]),
        cache_dir=source_cache_dir,
        registry=registry,
    )
    files, gate = stage_chem_delta(parent, decisions, registry, stage_dir)
    return commit_atomic_release(
        files,
        prewrite_gate=gate,
        repo_id=REPO_ID,
        hf_token=hf_token,
        prior_revision=parent.revision,
        commit_message="Publish exact Chemistry region review atomically",
        intent_sink=atomic_intent_sink,
        intent_metadata={
            "source_commit": SOURCE_COMMIT,
            "producer_review_commit": REVIEWED_337_PRODUCER,
        },
    )
