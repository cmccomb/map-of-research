"""Carry the frozen Chemistry review through later ordinary snapshots.

The one-off Chemistry delta changed nineteen existing observations. A full
snapshot must apply those same source-bound decisions before rebuilding works,
authorships, people, and the map. The immutable Chemistry Hub release is the
root of the parent-state check; later parents may change unrelated works.
"""

from __future__ import annotations

import json
import math
import numbers
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .chem_region_review import (
    CHEM_REVIEW_SHA256,
)
from .chem_region_review import (
    SOURCE_COMMIT as CHEMISTRY_SOURCE_COMMIT,
)
from .chem_region_review import (
    SOURCE_SHA256 as CHEMISTRY_SOURCE_SHA256,
)
from .composite_region import TOPIC_FIELDS
from .contexts import ReviewedContext
from .exclusions import ReviewedExclusion
from .provisional_holds import ProvisionalHold
from .region_continuity import require_revision
from .region_review import ReviewedRegionDecision

CHEMISTRY_HUB_REVISION = "dcbdfacd7e8cfeaefd4c91081c4b347b7c690061"
CHEMISTRY_DELTA_PARENT = "38f119d71692cdb9112210217b532a7dd7c1633a"
PRE_CHEMISTRY_REVISIONS = frozenset(
    {"bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd", CHEMISTRY_DELTA_PARENT}
)
CHEMISTRY_SCHOLAR_IDS = frozenset(
    {
        "ILambBMAAAAJ",
        "ctYLFRIAAAAJ",
        "0e1jwkkAAAAJ",
        "TSkli5EAAAAJ",
        "IggMbacAAAAJ",
        "L0nXO5EAAAAJ",
        "FRtd8_QAAAAJ",
        "BWNUifwAAAAJ",
    }
)
PARENT_FILES = (
    "maps/manifest.json",
    "maps/publications.json",
    "works/train-00000-of-00001.parquet",
    "profile_publications/train-00000-of-00001.parquet",
    "authorships/train-00000-of-00001.parquet",
    "people/train-00000-of-00001.parquet",
)
WORK_RECOMPUTED_FIELDS = frozenset(
    {
        *TOPIC_FIELDS,
        "keyword_model_version",
        "topic_label_review_version",
        "layout_version",
        "x",
        "y",
        "tsne_x",
        "tsne_y",
    }
)
PERSON_COUNT_FIELDS = frozenset(
    {
        "publication_observation_count",
        "scholar_observation_count",
        "curated_observation_count",
        "unique_work_count",
        "mapped_work_count",
    }
)
MAP_POINT_RECOMPUTED_FIELDS = frozenset({"keyword_ids", "x", "y", "tsne_x", "tsne_y"})
POST_BIO_WORK_FIELDS = frozenset(
    {"map_region_prior_eligible_override", "map_region_prior_excluded_override"}
)
POST_BIO_PERSON_FIELDS = frozenset({"publication_hold_reason"})
POST_BIO_REGION_AUDIT_VERSION = "low-information-semantic-region-v3"
# Values are pinned to published Hub revision 211cdbc, not the earlier Bio preview.
# The published 8dcc024f row has this override set to false.
POST_BIO_PRIOR_ELIGIBLE_WORK_IDS = frozenset(
    {
        "work-2b4a1b0c1ba3c97a3a59cdeff547ee405ba08c3eb3801f5a2451931e1f7487f2",
        "work-7cf42c9fd2912068c1a0198c94b9ca19f67cd16f4112f52c44b98a59a2c0662a",
        "work-b29eec36af051204a05cc4120514d040740998193eb641d2bb90fde9fde1b7e8",
        "work-dab872d55c3138e1e51fc1f7d6f959218b6772b130d46e4c3ff2540f9f0bdd60",
        "work-df35fb5e533c0e0d2f295a5e0e4745259fdad2e9a190e78f189fd21ecd1fede9",
    }
)
POST_BIO_DOCUMENTED_OVERRIDE_WORK_ID = (
    "work-f07078afd572186934d04127dcd8526c089e5089ad69b1c4504702ac49c4329b"
)
BIO_REGION_REFIT_PARENT = "211cdbcde850867ef9b5a81f7001f3b5b27425cc"
BIO_REGION_REFIT_SOURCE_SHA256 = (
    "d1ab914ec3fe57e30361fa6b01f30787b985e554e058faa01b32c6496e869ac6"
)
BIO_REGION_REFIT_FIELDS = frozenset(
    {"map_region_documented_override", "map_region_prior_eligible_override"}
)
MCS_PUBLISHED_PARENT = "f194bdab4d9d19b012709f1dde06bcef2bf59fb6"
PHYSICS_FIVE_PUBLISHED_PARENT = "4e43639c091a4b826a4651eddc5aa0fe16129f0f"
MCS_DIAGNOSTIC_PARENT_REVISIONS = frozenset(
    {MCS_PUBLISHED_PARENT, PHYSICS_FIVE_PUBLISHED_PARENT}
)
# The immutable MCS parent retains the 19 reviewed Chemistry decisions. Its
# full refit changed only these three derived diagnostics from the Bio parent.
MCS_PARENT_PRIOR_ELIGIBLE_WORK_IDS = POST_BIO_PRIOR_ELIGIBLE_WORK_IDS | {
    "work-8dcc024f119371868cec839a22cab25ec785f6679b17ee9f1a732ec7b1f03342",
    "work-a0cfe68d0794a4afe20e8a55b401afed795f14a8445cb9dc54275a84acb6b3d4",
}
MCS_PARENT_DOCUMENTED_OVERRIDE_WORK_IDS = {
    POST_BIO_DOCUMENTED_OVERRIDE_WORK_ID,
    "work-4e63e66d343de69522c77ffde7a55853382c39d01a9379ca47f4fa4eb939e802",
}
MCS_PARENT_CANDIDATE_PRIOR_ELIGIBLE_LOSSES = frozenset(
    {
        "work-8dcc024f119371868cec839a22cab25ec785f6679b17ee9f1a732ec7b1f03342",
        "work-a0cfe68d0794a4afe20e8a55b401afed795f14a8445cb9dc54275a84acb6b3d4",
    }
)
MCS_PARENT_CANDIDATE_DOCUMENTED_LOSS_ID = POST_BIO_DOCUMENTED_OVERRIDE_WORK_ID


def allows_bio_region_refit(
    *, prior_revision: str, source_sha256: str, layout_reused: bool
) -> bool:
    """Scope platform-dependent region diagnostics to one reviewed release."""

    return (
        not layout_reused
        and prior_revision == BIO_REGION_REFIT_PARENT
        and source_sha256 == BIO_REGION_REFIT_SOURCE_SHA256
    )


def allows_physics_five_region_refit(
    *, prior_revision: str, source_sha256: str, layout_reused: bool
) -> bool:
    """Allow derived region diagnostics after the exact published Physics five."""

    return (
        not layout_reused
        and prior_revision == PHYSICS_FIVE_PUBLISHED_PARENT
        and source_sha256 == BIO_REGION_REFIT_SOURCE_SHA256
    )


def allows_mcs_parent_diagnostic_refit(
    *, prior_revision: str, source_sha256: str, layout_reused: bool
) -> bool:
    """Recognize the exact published MCS parent and unchanged raw source."""

    return (
        not layout_reused
        and prior_revision == MCS_PUBLISHED_PARENT
        and source_sha256 == BIO_REGION_REFIT_SOURCE_SHA256
    )


def needs_chemistry_continuity(
    scholar_ids: set[str],
    prior_revision: str,
    *,
    composite: bool,
    snapshot_sha256: str = "",
) -> bool:
    """Require Chemistry review for every real post-dcb full snapshot.

    A complete validated snapshot has a 64-hex digest. The Scholar-ID fallback
    also catches relevant synthetic inputs that do not use the snapshot reader.
    """

    require_revision(prior_revision)
    return (
        not composite
        and prior_revision not in PRE_CHEMISTRY_REVISIONS
        and (
            prior_revision == CHEMISTRY_HUB_REVISION
            or bool(scholar_ids & CHEMISTRY_SCHOLAR_IDS)
            or re.fullmatch(r"[0-9a-f]{64}", snapshot_sha256) is not None
        )
    )


def merge_chemistry_review_inputs(
    decisions: Sequence[ReviewedRegionDecision],
    exclusions: Mapping[tuple[str, str], ReviewedExclusion],
    holds: Mapping[tuple[str, str], ProvisionalHold],
    contexts: Mapping[tuple[str, str], ReviewedContext],
) -> tuple[
    dict[tuple[str, str], ReviewedExclusion],
    dict[tuple[str, str], ProvisionalHold],
    dict[tuple[str, str], ReviewedContext],
]:
    """Apply only the four exclusions, five holds, and two output contexts."""

    counts = Counter(item.decision for item in decisions)
    if len(decisions) != 19 or counts != {"include": 10, "hold": 5, "exclude": 4}:
        raise ValueError("Chemistry ordinary review is not the exact 19 decisions")
    merged_exclusions = dict(exclusions)
    merged_holds = dict(holds)
    merged_contexts = dict(contexts)
    output_count = 0
    for item in decisions:
        key = (item.scholar_id, item.author_pub_id)
        expected_exclusion = None
        expected_hold = None
        expected_context = None
        if item.decision == "exclude":
            expected_exclusion = ReviewedExclusion(
                reason_code=item.reason_code,
                reason=item.reason,
                evidence_url=item.evidence_url,
                verified_at=item.reviewed_at,
            )
        elif item.decision == "hold":
            expected_hold = ProvisionalHold(
                scholar_id=item.scholar_id,
                author_pub_id=item.author_pub_id,
                capture_row_index=item.capture_row_index,
                reason=item.reason,
                evidence_url=item.evidence_url,
                reviewed_at=item.reviewed_at,
            )
        if item.output_type:
            output_count += 1
            expected_context = ReviewedContext(
                publication_type=item.output_type,
                document_id="",
                reason=item.reason,
                evidence_url=item.evidence_url,
                verified_at=item.reviewed_at,
            )
        prior_hold = merged_holds.get(key)
        saved_scholar_hold = (
            replace(
                expected_hold,
                evidence_url=item.saved_scholar_url,
                primary_evidence_url=item.evidence_url,
            )
            if expected_hold is not None
            else None
        )
        if (
            merged_exclusions.get(key) not in (None, expected_exclusion)
            or prior_hold not in (None, expected_hold, saved_scholar_hold)
            or merged_contexts.get(key) not in (None, expected_context)
        ):
            raise ValueError(
                f"Chemistry decision conflicts with existing review: {key}"
            )
        if expected_exclusion is not None:
            merged_exclusions[key] = expected_exclusion
        if expected_hold is not None:
            merged_holds[key] = expected_hold
        if expected_context is not None:
            merged_contexts[key] = expected_context
    if output_count != 2:
        raise ValueError("Chemistry ordinary review lacks two presentations")
    return merged_exclusions, merged_holds, merged_contexts


def _parent_files(revision: str, hf_token: str) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    return {
        name: Path(
            hf_hub_download(
                "ccm/cmu-engineering-publications",
                name,
                repo_type="dataset",
                revision=require_revision(revision),
                token=hf_token or None,
            )
        )
        for name in PARENT_FILES
    }


def _selected_rows(
    path: Path, field: str, wanted: set[str]
) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as parquet

    found: dict[str, dict[str, Any]] = {}
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            key = str(row.get(field) or "")
            if key not in wanted:
                continue
            if key in found:
                raise ValueError(f"Chemistry parent repeats {field}: {key}")
            found[key] = row
    if set(found) != wanted:
        raise ValueError(f"Chemistry parent lacks reviewed {field} identities")
    return found


def _authored_rows(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as parquet

    found: dict[str, dict[str, Any]] = {}
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            key = str(row.get("observation_id") or "")
            if key not in wanted:
                continue
            if key in found:
                raise ValueError(f"Chemistry parent repeats authorship: {key}")
            found[key] = row
    return found


def _equal_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Mapping):
        return {key: _equal_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_equal_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _equal_value(value.tolist())
    return value


def _same_rows(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    *,
    excluded: frozenset[str] = frozenset(),
    added: frozenset[str] = frozenset(),
    label: str,
) -> None:
    if before.keys() != after.keys():
        raise ValueError(f"Chemistry {label} identities changed")
    for key, row in before.items():
        current = after[key]
        if set(current) != set(row) | added:
            raise ValueError(f"Chemistry {label} schema changed: {key}")
        for field in row:
            if field not in excluded and _equal_value(row[field]) != _equal_value(
                current[field]
            ):
                raise ValueError(f"Chemistry {label} payload changed: {key}: {field}")


def _reviewed_points(path: Path, work_ids: set[str]) -> dict[str, dict[str, Any]]:
    artifact = json.loads(path.read_text())
    points = artifact.get("points")
    if not isinstance(points, list) or artifact.get("point_count") != len(points):
        raise ValueError("Chemistry parent map points are invalid")
    found: dict[str, dict[str, Any]] = {}
    for point in points:
        work_id = point.get("work_id")
        if work_id in work_ids:
            if work_id in found:
                raise ValueError(f"Chemistry parent repeats map point: {work_id}")
            found[work_id] = point
    return found


def validate_chemistry_parent(
    decisions: Sequence[ReviewedRegionDecision],
    *,
    baseline_files: Mapping[str, Path],
    parent_files: Mapping[str, Path],
    prior_revision: str,
) -> dict[str, Any]:
    """Prove the pinned parent retained the exact dcb Chemistry decisions."""

    require_revision(prior_revision)
    marker = json.loads(baseline_files["maps/manifest.json"].read_text()).get(
        "region_continuity"
    )
    visible = {item.work_id for item in decisions if item.visible}
    hidden = {item.work_id for item in decisions if not item.visible}
    restored = sorted(
        item.work_id
        for item in decisions
        if item.transition == "restore" and item.visible
    )
    removed = sorted(
        item.work_id
        for item in decisions
        if item.transition == "reexclude" and not item.visible
    )
    if (
        not isinstance(marker, dict)
        or marker.get("version") != "exact-chemistry-post-337-delta-v1"
        or marker.get("prior_revision") != CHEMISTRY_DELTA_PARENT
        or marker.get("source_commit") != CHEMISTRY_SOURCE_COMMIT
        or marker.get("source_snapshot_sha256") != CHEMISTRY_SOURCE_SHA256
        or marker.get("chem_region_review_sha256") != CHEM_REVIEW_SHA256
        or marker.get("restored_work_ids") != restored
        or marker.get("removed_work_ids") != removed
        or len(visible) != 10
        or len(hidden) != 9
    ):
        raise ValueError("Pinned dcb Chemistry release marker changed")
    work_ids = visible | hidden
    source_ids = {item.author_pub_id for item in decisions}
    people_ids = {item.person_id for item in decisions}

    def selected(files: Mapping[str, Path]) -> dict[str, Any]:
        profiles = _selected_rows(
            files["profile_publications/train-00000-of-00001.parquet"],
            "author_pub_id",
            source_ids,
        )
        works = _selected_rows(
            files["works/train-00000-of-00001.parquet"], "work_id", work_ids
        )
        people = _selected_rows(
            files["people/train-00000-of-00001.parquet"], "person_id", people_ids
        )
        observed_ids = {row["observation_id"] for row in profiles.values()}
        authored = _authored_rows(
            files["authorships/train-00000-of-00001.parquet"], observed_ids
        )
        points = _reviewed_points(files["maps/publications.json"], work_ids)
        if set(points) != visible:
            raise ValueError("Chemistry parent map visibility changed")
        if {row["work_id"] for row in authored.values()} != visible:
            raise ValueError("Chemistry parent authored visibility changed")
        for item in decisions:
            profile = profiles[item.author_pub_id]
            work = works[item.work_id]
            if (
                profile["person_id"] != item.person_id
                or profile["work_id"] != item.work_id
                or bool(profile["map_eligible"]) != item.visible
                or bool(profile["collaboration_eligible"]) != item.visible
                or bool(work["map_eligible"]) != item.visible
                or bool(work["collaboration_eligible"]) != item.visible
                or bool(work["map_region_outlier"])
                or work["observation_ids"] != [profile["observation_id"]]
            ):
                raise ValueError(f"Chemistry parent decision changed: {item.work_id}")
            if item.visible and points[item.work_id]["keyword_ids"] != [
                work["keyword_id"],
                work["detail_keyword_id"],
            ]:
                raise ValueError(
                    f"Chemistry parent topic point changed: {item.work_id}"
                )
        return {
            "profiles": profiles,
            "works": works,
            "people": people,
            "authored": authored,
            "points": points,
        }

    baseline = selected(baseline_files)
    parent = selected(parent_files)
    current_marker = json.loads(parent_files["maps/manifest.json"].read_text()).get(
        "region_continuity"
    )
    if prior_revision == CHEMISTRY_HUB_REVISION and current_marker != marker:
        raise ValueError("Pinned dcb Chemistry parent marker changed")
    if (
        prior_revision != CHEMISTRY_HUB_REVISION
        and isinstance(current_marker, dict)
        and "chem_region_review_sha256" in current_marker
        and current_marker["chem_region_review_sha256"] != CHEM_REVIEW_SHA256
    ):
        raise ValueError("Later Chemistry parent lineage marker changed")
    post_bio_schema = False
    if prior_revision != CHEMISTRY_HUB_REVISION:
        work_extras = {
            frozenset(set(row) - set(baseline["works"][key]))
            for key, row in parent["works"].items()
        }
        person_extras = {
            frozenset(set(row) - set(baseline["people"][key]))
            for key, row in parent["people"].items()
        }
        if work_extras == {POST_BIO_WORK_FIELDS} and person_extras == {
            POST_BIO_PERSON_FIELDS
        }:
            post_bio_schema = True
        elif work_extras != {frozenset()} or person_extras != {frozenset()}:
            raise ValueError("Chemistry later parent has unreviewed schema fields")
    _same_rows(baseline["profiles"], parent["profiles"], label="profile")
    _same_rows(
        baseline["works"],
        parent["works"],
        excluded=(
            WORK_RECOMPUTED_FIELDS
            | {"map_region_audit_version", "map_region_documented_override"}
            if post_bio_schema
            else WORK_RECOMPUTED_FIELDS
        ),
        added=POST_BIO_WORK_FIELDS if post_bio_schema else frozenset(),
        label="work",
    )
    _same_rows(baseline["authored"], parent["authored"], label="authorship")
    _same_rows(
        baseline["people"],
        parent["people"],
        excluded=PERSON_COUNT_FIELDS,
        added=POST_BIO_PERSON_FIELDS if post_bio_schema else frozenset(),
        label="person",
    )
    if post_bio_schema:
        prior_eligible_ids = (
            MCS_PARENT_PRIOR_ELIGIBLE_WORK_IDS
            if prior_revision in MCS_DIAGNOSTIC_PARENT_REVISIONS
            else POST_BIO_PRIOR_ELIGIBLE_WORK_IDS
        )
        documented_override_ids = (
            MCS_PARENT_DOCUMENTED_OVERRIDE_WORK_IDS
            if prior_revision in MCS_DIAGNOSTIC_PARENT_REVISIONS
            else {POST_BIO_DOCUMENTED_OVERRIDE_WORK_ID}
        )
        for work_id, work in parent["works"].items():
            before = baseline["works"][work_id]
            if (
                work["map_region_prior_eligible_override"]
                is not (work_id in prior_eligible_ids)
                or work["map_region_prior_excluded_override"] is not False
                or work["map_region_audit_version"] != POST_BIO_REGION_AUDIT_VERSION
                or work["map_region_documented_override"]
                is not (
                    True
                    if work_id in documented_override_ids
                    else before["map_region_documented_override"]
                )
            ):
                raise ValueError(
                    f"Chemistry post-Bio work derivation changed: {work_id}"
                )
        if any(
            row["publication_hold_reason"] != "" for row in parent["people"].values()
        ):
            raise ValueError("Chemistry post-Bio person hold reason changed")
    _same_rows(
        baseline["points"],
        parent["points"],
        excluded=MAP_POINT_RECOMPUTED_FIELDS,
        label="map point",
    )
    return {
        "prior_revision": prior_revision,
        "review_sha256": CHEM_REVIEW_SHA256,
        "reviewed_works": len(work_ids),
        "visible_works": len(visible),
        "lineage": (
            "exact_dcb_marker"
            if prior_revision == CHEMISTRY_HUB_REVISION
            else "exact_dcb_reviewed_rows"
        ),
        "parent_rows": parent,
    }


def load_chemistry_parent(
    decisions: Sequence[ReviewedRegionDecision], prior_revision: str, hf_token: str
) -> dict[str, Any]:
    """Read only immutable Hub revisions and verify their reviewed row lineage."""

    baseline_files = _parent_files(CHEMISTRY_HUB_REVISION, hf_token)
    parent_files = (
        baseline_files
        if prior_revision == CHEMISTRY_HUB_REVISION
        else _parent_files(prior_revision, hf_token)
    )
    return validate_chemistry_parent(
        decisions,
        baseline_files=baseline_files,
        parent_files=parent_files,
        prior_revision=prior_revision,
    )


def reconcile_chemistry_person_counts(
    people: Any, decisions: Sequence[ReviewedRegionDecision]
) -> Any:
    """Add the seven source-bound restores omitted by pre-region map counts.

    The ordinary builder counts eligible works before the region gate. The
    Chemistry delta's reviewed restores are applied after that gate, so those
    seven works need a bounded offset. No other person's count is changed.
    """

    offsets = Counter(
        item.person_id for item in decisions if item.transition == "restore"
    )
    if sum(offsets.values()) != 7 or len(offsets) != 5:
        raise ValueError("Chemistry mapped-count restore set changed")
    result = people.copy()
    for person_id, offset in offsets.items():
        matches = result.index[result["person_id"] == person_id]
        if len(matches) != 1:
            raise ValueError(
                f"Chemistry reviewed person is missing or repeated: {person_id}"
            )
        index = matches[0]
        value = result.at[index, "mapped_work_count"]
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or value < 0
        ):
            raise ValueError(f"Chemistry reviewed person count is invalid: {person_id}")
        result.at[index, "mapped_work_count"] = int(value) + offset
    return result


def validate_chemistry_candidate(
    tables: Mapping[str, Any],
    parent_state: Mapping[str, Any],
    *,
    require_count_parity: bool = False,
    allow_region_refit: bool = False,
    allow_mcs_parent_diagnostic_refit: bool = False,
) -> None:
    """Reject changes to reviewed Chemistry source and visibility decisions.

    A full layout refit may change two derived region diagnostics for parent-
    visible reviewed works across platforms. The caller pins the Bio or
    Physics-five parent, unchanged source, and fresh layout. From the
    published MCS parent, only two exact prior-eligible
    diagnostics and one exact documented-region diagnostic may change from true
    to false. Source, visibility, attribution, and authorship stay exact.
    """

    parent = parent_state["parent_rows"]
    for table, key_field, excluded in (
        ("profile_publications", "author_pub_id", frozenset()),
        ("works", "work_id", WORK_RECOMPUTED_FIELDS),
        ("authorships", "observation_id", frozenset()),
        (
            "people",
            "person_id",
            frozenset() if require_count_parity else PERSON_COUNT_FIELDS,
        ),
    ):
        parent_name = {
            "profile_publications": "profiles",
            "works": "works",
            "authorships": "authored",
            "people": "people",
        }[table]
        expected = parent[parent_name]
        rows = tables[table].to_dict("records")
        selected = [row for row in rows if str(row[key_field]) in expected]
        current = {str(row[key_field]): row for row in selected}
        if len(selected) != len(expected) or len(current) != len(expected):
            raise ValueError(f"Chemistry candidate {table} lost reviewed rows")
        for key, prior in expected.items():
            row = current[key]
            if set(prior) - set(row) or (table != "works" and set(prior) != set(row)):
                raise ValueError(f"Chemistry candidate {table} schema changed: {key}")
            for field in prior:
                if field in excluded or _equal_value(prior[field]) == _equal_value(
                    row[field]
                ):
                    continue
                if (
                    table == "works"
                    and allow_region_refit
                    and key in parent["points"]
                    and field in BIO_REGION_REFIT_FIELDS
                    and type(_equal_value(prior[field])) is bool
                    and type(_equal_value(row[field])) is bool
                ):
                    continue
                if (
                    table == "works"
                    and allow_mcs_parent_diagnostic_refit
                    and key in MCS_PARENT_CANDIDATE_PRIOR_ELIGIBLE_LOSSES
                    and key in parent["points"]
                    and field == "map_region_prior_eligible_override"
                    and _equal_value(prior[field]) is True
                    and _equal_value(row[field]) is False
                ):
                    continue
                if (
                    table == "works"
                    and allow_mcs_parent_diagnostic_refit
                    and key == MCS_PARENT_CANDIDATE_DOCUMENTED_LOSS_ID
                    and key in parent["points"]
                    and field == "map_region_documented_override"
                    and _equal_value(prior[field]) is True
                    and _equal_value(row[field]) is False
                ):
                    continue
                raise ValueError(f"Chemistry candidate {table} changed: {key}: {field}")
    reviewed_observation_ids = {
        row["observation_id"] for row in parent["profiles"].values()
    }
    current_authorship_ids = {
        row["observation_id"]
        for row in tables["authorships"].to_dict("records")
        if row["observation_id"] in reviewed_observation_ids
    }
    if current_authorship_ids != set(parent["authored"]):
        raise ValueError("Chemistry candidate authored visibility changed")
