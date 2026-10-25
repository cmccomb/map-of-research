"""One source-bound Bernhard repair with the later exact Math review.

The historical manifest supplies decisions only for the 249 rows outside the
Math review.  The 31 Math decisions take precedence.  Every affected work and
Scholar observation must still be the one present in the immutable bb88 parent.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .region_review import ReviewedRegionDecision

BASELINE_REVISION = "bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd"
HISTORICAL_REVISION = "15498d5f78e2e269f4b11c630bc9b4ef56efb628"
SOURCE_COMMIT = "8be6a2400cf72dd959c874bfcada8624540820ef"
SOURCE_SHA256 = "f8d7ffec683eb1830dadc77e612cbad7f3fa18ff199696041a71a9b8edbcefa2"
REPAIR_MANIFEST_SHA256 = (
    "8679bb3b01483e9e87ba5220df4ef9a4b673264a79cc8cd48392f119dbe1fefc"
)
MATH_REVIEW_MANIFEST_SHA256 = (
    "4ede0738f5f0ec5fff2473262db2a89e9a6c61849007b176534bc85d052d415d"
)
SULLIVAN_WORK_FIXTURE_SHA256 = (
    "358cbad94d8fb82f823e28805e0516972175b0db6b5b28d76588d7da78b1e3b7"
)
SULLIVAN_DELTA_FIXTURE_SHA256 = (
    "ec7ee73e437ae0f3e22611cb107c708dea162cf76afc7c21d73684fd270b62d1"
)
SULLIVAN_NONWORK_FIXTURE_SHA256 = (
    "01bc8d1f0932fcc8a5d1c6ef76fd86404fb1144a0c271ed603b4d4932629fa77"
)
MATH_EXCLUSIONS_SHA256 = (
    "adb6532806737f794d02ea1fed30ad9e03a419abb27b819a7d6a01cbf3122f66"
)
MATH_HOLDS_SHA256 = "d52bbf51d65b8922b2dbde1d51afe4e287ce3a5818f460a96f69181e8c8197b0"
BASELINE_MAP_SHA256 = "358e23af6774ea4fd53bab5ed8b55fdd1891a4e7730c0924d74e428e94cd4a88"
BASELINE_MAP_POINTS = 42925
FINAL_MAP_POINTS = 42850
COORDINATE_FIELDS = ("x", "y", "tsne_x", "tsne_y")
TOPIC_FIELDS = (
    "keyword_id",
    "keyword",
    "keyword_extracted",
    "keyword_label_reviewed",
    "detail_keyword_id",
    "detail_keyword",
    "detail_keyword_extracted",
    "detail_keyword_label_reviewed",
)
PRESENTATION_FIELDS = (*COORDINATE_FIELDS, *TOPIC_FIELDS)
VARIANT_FIELDS = (
    "title_variants",
    "venue_variants",
    "citation_variants",
    "author_variants",
)
UNCLASSIFIED_TOPICS = {
    "keyword_id": "keyword-unclassified",
    "keyword": "Unclassified publications",
    "keyword_extracted": "Unclassified publications",
    "keyword_label_reviewed": False,
    "detail_keyword_id": "keyword-unclassified-pending-review",
    "detail_keyword": "Pending topic review",
    "detail_keyword_extracted": "Pending topic review",
    "detail_keyword_label_reviewed": False,
}
BLANK_TOPICS = {
    field: False if field.endswith("_label_reviewed") else "" for field in TOPIC_FIELDS
}
SULLIVAN_PERSON_ID = "person-sullivan-589c9545"
SULLIVAN_POINT_FIELDS = frozenset(
    {
        "title",
        "authors",
        "source_url",
        "citation_count",
        "venue",
        "faculty_ids",
        "department_ids",
        "school_ids",
        "observation_count",
    }
)
WORK_REGION_FIELDS = frozenset(
    {
        "map_eligible",
        "map_region_outlier",
        "map_exclusion_reasons",
    }
)
HIDDEN_MATH_ATTRIBUTION_FIELDS = frozenset(
    {
        "faculty",
        "person_ids",
        "memberships",
        "department_ids",
        "department_titles",
        "school_ids",
        "school_titles",
        "collaboration_eligible",
    }
)
FULLERENES_WORK_ID = (
    "work-fda500c2821091a0f441404779cf526c144bb34b1feb6c4ee0a40b15b0efea7e"
)
DRUG_REPURPOSING_WORK_ID = (
    "work-3cdf5a11569ffec20aaa9823ad8d98c97dcca69ee3e9ca279217a387ef024ba5"
)
FULLERENES_OBSERVATION_ID = (
    "observation-338edc5769bb3e75b6d64ae0019140fcf065cb662c3df1c31cfb3704a1701ddb"
)
FULLERENES_REVIEW_FIELDS = {
    "faculty": [],
    "person_ids": [],
    "memberships": [],
    "department_ids": [],
    "department_titles": [],
    "school_ids": [],
    "school_titles": [],
    "collaboration_eligible": False,
    "map_region_outlier": False,
    "map_exclusion_reasons": ["reviewed_false_profile_attribution"],
}


def _load_stage_fixture(
    curated_dir: Path, name: str, expected_sha256: str
) -> dict[str, Any]:
    path = curated_dir / name
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Reviewed Sullivan fixture bytes changed: {name}")
    fixture = json.loads(raw)
    if (
        fixture.get("schema_version") != 1
        or fixture.get("hub_parent") != BASELINE_REVISION
        or fixture.get("source_snapshot_sha256") != SOURCE_SHA256
        or fixture.get("source_commit_at_stage")
        != "8ba438215b50ff877f1a4dd0ce21ca2ed5f9b2d0"
        or fixture.get("producer_commit_at_stage")
        != "3630084aafd351726651708d270a87cbb597ad27"
    ):
        raise ValueError(f"Reviewed Sullivan fixture provenance changed: {name}")
    return fixture


@dataclass(frozen=True)
class CompositeRegionPlan:
    restore: frozenset[str]
    reexclude: frozenset[str]
    math: tuple[ReviewedRegionDecision, ...]
    manifest_sha256: str

    @property
    def gains(self) -> frozenset[str]:
        math_ids = {item.work_id for item in self.math}
        return frozenset(
            (self.restore - math_ids)
            | {
                item.work_id
                for item in self.math
                if item.visible and item.transition == "restore"
            }
        )

    @property
    def losses(self) -> frozenset[str]:
        math_ids = {item.work_id for item in self.math}
        return frozenset(
            (self.reexclude - math_ids)
            | {
                item.work_id
                for item in self.math
                if not item.visible and item.transition == "reexclude"
            }
        )


def load_composite_region_plan(
    manifest_path: Path,
    math: tuple[ReviewedRegionDecision, ...],
    *,
    source_commit: str,
    source_sha256: str,
    prior_revision: str,
) -> CompositeRegionPlan:
    """Require the immutable source, repair bytes, and all 31 Math overrides."""

    if (
        source_commit != SOURCE_COMMIT
        or source_sha256 != SOURCE_SHA256
        or prior_revision != BASELINE_REVISION
        or len(math) != 31
    ):
        raise ValueError("Composite repair needs the exact f8/bb88/31 review")
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != REPAIR_MANIFEST_SHA256:
        raise ValueError("Bernhard repair manifest bytes changed")
    manifest = json.loads(raw)
    restore = manifest.get("restore_eligible_work_ids")
    reexclude = manifest.get("restore_outlier_work_ids")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("head_revision") != BASELINE_REVISION
        or manifest.get("decision_revision") != HISTORICAL_REVISION
        or not isinstance(restore, list)
        or not isinstance(reexclude, list)
        or len(restore) != 112
        or len(reexclude) != 168
        or restore != sorted(set(restore))
        or reexclude != sorted(set(reexclude))
        or set(restore) & set(reexclude)
    ):
        raise ValueError("Bernhard repair work set differs from reviewed 280")
    math_ids = {item.work_id for item in math}
    if (
        len(math_ids) != 31
        or math_ids - (set(restore) | set(reexclude))
        or any(
            (item.work_id in restore) != (item.transition == "restore") for item in math
        )
    ):
        raise ValueError("Math review does not exactly override the historical repair")
    plan = CompositeRegionPlan(frozenset(restore), frozenset(reexclude), math, digest)
    if len(plan.gains) != 99 or len(plan.losses) != 156:
        raise ValueError("Composite repair does not have exact +99/-156 scope")
    return plan


def apply_composite_region_decisions(
    works: Any,
    *,
    prior_works: Mapping[str, Mapping[str, Any]],
    historical_works: Mapping[str, Mapping[str, Any]],
    prior_profiles: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    current_profiles: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    plan: CompositeRegionPlan,
) -> Any:
    """Apply only the reviewed work IDs after validating complete source parity."""

    # This import is delayed because region_continuity calls this function.
    from .region_continuity import (
        FINGERPRINT_FIELDS,
        WORK_FINGERPRINT_FIELDS,
        _fingerprint_fields,
    )

    affected = plan.restore | plan.reexclude
    old_non_sullivan = {
        identity: row
        for identity, row in prior_profiles.items()
        if identity[0] != SULLIVAN_PERSON_ID
    }
    new_non_sullivan = {
        identity: row
        for identity, row in current_profiles.items()
        if identity[0] != SULLIVAN_PERSON_ID
    }
    if old_non_sullivan.keys() != new_non_sullivan.keys() or any(
        old_non_sullivan[identity].get("work_id")
        != new_non_sullivan[identity].get("work_id")
        or _fingerprint_fields(old_non_sullivan[identity], FINGERPRINT_FIELDS)
        != _fingerprint_fields(new_non_sullivan[identity], FINGERPRINT_FIELDS)
        for identity in old_non_sullivan
    ):
        raise ValueError("Non-Sullivan Scholar observation source or work ID drifted")
    output = works.copy()
    indices = {str(row.work_id): index for index, row in output.iterrows()}
    if len(indices) != len(output) or not affected.issubset(indices):
        raise ValueError("Composite repair is missing affected work IDs")
    old_members: dict[str, dict[tuple[str, str, str, str], Mapping[str, Any]]] = (
        defaultdict(dict)
    )
    new_members: dict[str, dict[tuple[str, str, str, str], Mapping[str, Any]]] = (
        defaultdict(dict)
    )
    for identity, row in prior_profiles.items():
        if (work_id := str(row.get("work_id") or "")) in affected:
            old_members[work_id][identity] = row
    for identity, row in current_profiles.items():
        if (work_id := str(row.get("work_id") or "")) in affected:
            new_members[work_id][identity] = row
    if (
        sum(len(rows) for rows in old_members.values()) != 284
        or sum(len(rows) for rows in new_members.values()) != 284
    ):
        raise ValueError("Composite repair lacks the exact 284 old work observations")
    reviewed = {item.work_id: item for item in plan.math}
    for work_id in sorted(affected):
        old = prior_works.get(work_id)
        historical = historical_works.get(work_id)
        index = indices[work_id]
        current = output.loc[index].to_dict()
        old_observations = old_members[work_id]
        new_observations = new_members[work_id]
        if (
            old is None
            or historical is None
            or not old_observations
            or old_observations.keys() != new_observations.keys()
            or _fingerprint_fields(
                old, (*WORK_FINGERPRINT_FIELDS, "quality_assessment_version")
            )
            != _fingerprint_fields(
                current, (*WORK_FINGERPRINT_FIELDS, "quality_assessment_version")
            )
            or any(
                _fingerprint_fields(old_observations[identity], FINGERPRINT_FIELDS)
                != _fingerprint_fields(new_observations[identity], FINGERPRINT_FIELDS)
                for identity in old_observations
            )
            or bool(old.get("map_eligible")) != (work_id in plan.reexclude)
            or bool(old.get("map_region_outlier")) != (work_id in plan.restore)
            or bool(historical.get("map_eligible")) != (work_id in plan.restore)
            or bool(historical.get("map_region_outlier")) != (work_id in plan.reexclude)
        ):
            raise ValueError(
                f"Composite repair source or historical decision drift: {work_id}"
            )
        math_decision = reviewed.get(work_id)
        if math_decision is not None:
            identity = math_decision.observation_identity
            if (
                set(old_observations) != {identity}
                or set(new_observations) != {identity}
                or hashlib.sha256(
                    str(old_observations[identity]["source_record_json"]).encode()
                ).hexdigest()
                != math_decision.source_record_sha256
                or bool(new_observations[identity].get("map_eligible"))
                != math_decision.visible
                or _fingerprint_fields(
                    old, (*WORK_FINGERPRINT_FIELDS, "quality_assessment_version")
                )
                != math_decision.work_fingerprint_sha256
            ):
                raise ValueError(
                    f"Math override source or observation drift: {work_id}"
                )
            visible = math_decision.visible
        else:
            visible = work_id in plan.restore
        if visible:
            output.at[index, "map_eligible"] = True
            output.at[index, "map_region_outlier"] = False
            output.at[index, "map_exclusion_reasons"] = [
                reason
                for reason in current["map_exclusion_reasons"]
                if reason != "low_information_region"
            ]
        elif math_decision is None:
            output.at[index, "map_eligible"] = False
            output.at[index, "map_region_outlier"] = True
            output.at[index, "map_exclusion_reasons"] = [
                *[
                    reason
                    for reason in current["map_exclusion_reasons"]
                    if reason != "low_information_region"
                ],
                "low_information_region",
            ]
        else:
            output.at[index, "map_eligible"] = False
        if bool(output.at[index, "map_eligible"]) != visible:
            raise ValueError(f"Composite repair visibility drift: {work_id}")
    old_visible = {item for item in affected if bool(prior_works[item]["map_eligible"])}
    final_visible = {
        item for item in affected if bool(output.at[indices[item], "map_eligible"])
    }
    if (
        final_visible - old_visible != plan.gains
        or old_visible - final_visible != plan.losses
    ):
        raise ValueError("Composite repair work ID delta exceeds +99/-156")
    return output


def require_baseline_map(path: Path) -> dict[str, Any]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASELINE_MAP_SHA256:
        raise ValueError("Composite presentation baseline map bytes changed")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    points = artifact.get("points")
    if (
        not isinstance(points, list)
        or len(points) != BASELINE_MAP_POINTS
        or artifact.get("point_count") != len(points)
        or len({point.get("work_id") for point in points}) != len(points)
    ):
        raise ValueError("Composite presentation baseline point set changed")
    return artifact


def freeze_composite_presentation(
    works: Any,
    prior_works: Sequence[Mapping[str, Any]],
    prior_artifact: Mapping[str, Any],
    plan: CompositeRegionPlan,
    sullivan_work_ids: set[str],
) -> Any:
    """Keep bb88 placement and labels; assign reviewed new points an explicit topic."""

    old = {str(row["work_id"]): row for row in prior_works}
    old_points = {str(point["work_id"]): point for point in prior_artifact["points"]}
    if len(old) != len(prior_works) or not set(old_points).issubset(old):
        raise ValueError("Composite presentation baseline work IDs repeat or disappear")
    transient = (
        "map_region_prior_eligible_override",
        "map_region_prior_excluded_override",
    )
    if any(field not in works.columns for field in transient):
        raise ValueError("Composite presentation lacks continuity diagnostics")
    output = works.drop(columns=list(transient)).copy()
    for index, row in output.iterrows():
        work_id = str(row.work_id)
        prior = old.get(work_id)
        visible = bool(row.map_eligible)
        map_selected = (
            visible and bool(row.has_included_source_profile) and bool(row.memberships)
        )
        if prior is not None:
            if work_id not in sullivan_work_ids:
                for field in VARIANT_FIELDS:
                    before = list(prior[field] or [])
                    after = list(row[field] or [])
                    if Counter(before) != Counter(after):
                        raise ValueError(
                            f"Non-Sullivan source variants changed: {work_id}: {field}"
                        )
                    output.at[index, field] = before
            if work_id == DRUG_REPURPOSING_WORK_ID:
                output.at[index, "map_region_documented_override"] = prior[
                    "map_region_documented_override"
                ]
            for field in COORDINATE_FIELDS:
                try:
                    coordinate = float(prior[field])
                except (TypeError, ValueError):
                    coordinate = math.nan
                if math.isfinite(coordinate):
                    output.at[index, field] = coordinate
                elif work_id in old_points or work_id in plan.gains:
                    raise ValueError(f"Missing frozen map coordinate: {work_id}")
                elif not visible:
                    output.at[index, field] = math.nan
        if map_selected and work_id in old_points:
            if prior is None or any(
                prior[field] != old_points[work_id]["keyword_ids"][position]
                for position, field in enumerate(("keyword_id", "detail_keyword_id"))
            ):
                raise ValueError(f"Frozen work topic differs from map point: {work_id}")
            topics = {field: prior[field] for field in TOPIC_FIELDS}
        elif map_selected:
            topics = UNCLASSIFIED_TOPICS
        elif prior is not None and visible:
            topics = {field: prior[field] for field in TOPIC_FIELDS}
        else:
            topics = BLANK_TOPICS
        for field, value in topics.items():
            output.at[index, field] = value
    return output


def validate_composite_map_points(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    prior_works: Sequence[Mapping[str, Any]],
    prior_profiles: Sequence[Mapping[str, Any]],
    current_profiles: Sequence[Mapping[str, Any]],
    plan: CompositeRegionPlan,
    *,
    curated_dir: Path,
) -> dict[str, Any]:
    """Reject all layout/topic drift and every unreviewed point delta."""

    fixture = _load_stage_fixture(
        curated_dir,
        "composite-sullivan-exact-deltas.json",
        SULLIVAN_DELTA_FIXTURE_SHA256,
    )
    old_points = {str(point["work_id"]): point for point in baseline["points"]}
    new_list = candidate.get("points")
    if not isinstance(new_list, list):
        raise ValueError("Composite map has no points")
    new_points = {str(point["work_id"]): point for point in new_list}
    if len(new_points) != len(new_list) or len(new_points) != FINAL_MAP_POINTS:
        raise ValueError("Composite map lacks exact 42,850 unique points")
    gains = new_points.keys() - old_points.keys()
    losses = old_points.keys() - new_points.keys()
    old_sullivan = {
        str(row["work_id"])
        for row in prior_profiles
        if row.get("person_id") == SULLIVAN_PERSON_ID
    }
    new_sullivan = {
        str(row["work_id"])
        for row in current_profiles
        if row.get("person_id") == SULLIVAN_PERSON_ID
    }
    if old_sullivan | new_sullivan != set(fixture["sullivan_work_id_union"]):
        raise ValueError("Sullivan source work IDs differ from reviewed f8 stage")
    if (
        len(gains) != 110
        or len(losses) != 185
        or not plan.gains.issubset(gains)
        or not plan.losses.issubset(losses)
        or len(gains - plan.gains) != 11
        or len(losses - plan.losses) != 29
        or not gains - plan.gains <= new_sullivan
        or not losses - plan.losses <= old_sullivan
        or gains - plan.gains != set(fixture["new_sullivan_map_work_ids"])
        or losses - plan.losses != set(fixture["removed_sullivan_map_work_ids"])
        or candidate.get("point_count") != FINAL_MAP_POINTS
    ):
        raise ValueError(
            "Composite map point IDs exceed 99/156 repair and 11/29 Sullivan"
        )
    old_works = {str(row["work_id"]): row for row in prior_works}
    digest_exclusions = frozenset(fixture["map_digest_excluded_fields"])
    expected_shared_sullivan = fixture[
        "shared_sullivan_map_nonlayout_sha256_by_work_id"
    ]
    if set(expected_shared_sullivan) != (
        old_points.keys() & new_points.keys() & (old_sullivan | new_sullivan)
    ):
        raise ValueError("Shared Sullivan map point IDs differ from reviewed stage")
    for work_id in sorted(old_points.keys() & new_points.keys()):
        previous = old_points[work_id]
        current = new_points[work_id]
        if any(current[field] != previous[field] for field in COORDINATE_FIELDS) or (
            current["keyword_ids"] != previous["keyword_ids"]
        ):
            raise ValueError(f"Shared map placement or topics drifted: {work_id}")
        changed = {
            field
            for field in previous.keys() | current.keys()
            if previous.get(field) != current.get(field)
        }
        if changed and (
            work_id not in old_sullivan | new_sullivan
            or not changed <= SULLIVAN_POINT_FIELDS
        ):
            raise ValueError(
                f"Unreviewed shared map payload changed: {work_id}: {sorted(changed)}"
            )
        if (
            work_id in expected_shared_sullivan
            and _digest_row(current, exclude=digest_exclusions)
            != expected_shared_sullivan[work_id]
        ):
            raise ValueError(
                f"Sullivan shared map point differs from reviewed stage: {work_id}"
            )
    expected_new_sullivan = fixture["new_sullivan_map_nonlayout_sha256_by_work_id"]
    if set(expected_new_sullivan) != gains - plan.gains:
        raise ValueError("New Sullivan map point IDs differ from reviewed stage")
    for work_id in sorted(gains):
        point = new_points[work_id]
        if (
            work_id in expected_new_sullivan
            and _digest_row(point, exclude=digest_exclusions)
            != expected_new_sullivan[work_id]
        ):
            raise ValueError(
                f"New Sullivan map point differs from reviewed stage: {work_id}"
            )
        if point["keyword_ids"] != [
            UNCLASSIFIED_TOPICS["keyword_id"],
            UNCLASSIFIED_TOPICS["detail_keyword_id"],
        ]:
            raise ValueError(
                f"New map point lacks explicit unclassified topic: {work_id}"
            )
        if work_id in plan.gains:
            prior = old_works[work_id]
            if any(
                point[field] != round(float(prior[field]), 7)
                for field in COORDINATE_FIELDS
            ):
                raise ValueError(
                    f"Restored point moved from frozen bb88 layout: {work_id}"
                )
        elif any(not math.isfinite(float(point[field])) for field in COORDINATE_FIELDS):
            raise ValueError(f"Sullivan point has invalid placement: {work_id}")
    return {
        "status": "pass",
        "map_points": len(new_points),
        "shared_points_frozen": len(old_points.keys() & new_points.keys()),
        "restored_work_ids": sorted(plan.gains),
        "removed_work_ids": sorted(plan.losses),
        "sullivan_new_work_ids": sorted(gains - plan.gains),
        "sullivan_removed_work_ids": sorted(losses - plan.losses),
    }


def _digest_row(row: Mapping[str, Any], *, exclude: frozenset[str]) -> str:
    def canonical(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): canonical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    payload = {
        key: canonical(value) for key, value in row.items() if key not in exclude
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _require_hidden_math_attribution(row: Mapping[str, Any], work_id: str) -> None:
    if (
        any(
            row[field] != []
            for field in HIDDEN_MATH_ATTRIBUTION_FIELDS
            if field != "collaboration_eligible"
        )
        or row["collaboration_eligible"] is not False
    ):
        raise ValueError(f"Hidden Math attribution drift: {work_id}")


def _work_payload_exceptions(
    work_id: str, *, affected: frozenset[str], hidden_math: set[str]
) -> frozenset[str]:
    if work_id in hidden_math:
        return (
            WORK_REGION_FIELDS
            | frozenset(TOPIC_FIELDS)
            | HIDDEN_MATH_ATTRIBUTION_FIELDS
        )
    if work_id in affected:
        return WORK_REGION_FIELDS | frozenset(TOPIC_FIELDS)
    if work_id == FULLERENES_WORK_ID:
        return frozenset(FULLERENES_REVIEW_FIELDS)
    return frozenset()


def validate_composite_staged_works(
    baseline_path: Path,
    staged_path: Path,
    *,
    plan: CompositeRegionPlan,
    sullivan_work_ids: set[str],
    current_sullivan_work_ids: set[str],
    curated_dir: Path,
) -> dict[str, Any]:
    """Compare actual staged Parquet against every frozen non-Sullivan work."""

    import pyarrow.parquet as parquet

    work_fixture = _load_stage_fixture(
        curated_dir,
        "composite-sullivan-work-digests.json",
        SULLIVAN_WORK_FIXTURE_SHA256,
    )
    delta_fixture = _load_stage_fixture(
        curated_dir,
        "composite-sullivan-exact-deltas.json",
        SULLIVAN_DELTA_FIXTURE_SHA256,
    )
    exact_sullivan = work_fixture["work_sha256_by_id"]
    order_exceptions = work_fixture["order_only_variant_exceptions"]
    if (
        len(exact_sullivan) != 251
        or len(order_exceptions) != 2
        or set(exact_sullivan) != current_sullivan_work_ids
        or set(delta_fixture["sullivan_work_id_union"]) != sullivan_work_ids
    ):
        raise ValueError("Sullivan work scope differs from reviewed f8 stage")
    baseline = parquet.ParquetFile(baseline_path)
    staged = parquet.ParquetFile(staged_path)
    if not baseline.schema_arrow.equals(staged.schema_arrow):
        raise ValueError("Composite staged works schema differs from bb88")
    expected: dict[str, str] = {}
    prior_topics: dict[str, dict[str, Any]] = {}
    affected = plan.restore | plan.reexclude
    math = {item.work_id: item for item in plan.math}
    hidden_math = {work_id for work_id, item in math.items() if not item.visible}
    exclusion_path = curated_dir / "scholar-exclusions.json"
    if (
        hashlib.sha256(exclusion_path.read_bytes()).hexdigest()
        != MATH_EXCLUSIONS_SHA256
    ):
        raise ValueError("Math exact exclusions changed before staged works gate")
    exclusions = {
        (item["scholar_id"], item["author_pub_id"]): item
        for item in json.loads(exclusion_path.read_text())["exclusions"]
    }

    for batch in baseline.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            work_id = str(row.get("work_id") or "")
            if not work_id or work_id in expected:
                raise ValueError("bb88 works have missing or duplicate IDs")
            expected[work_id] = (
                "sullivan"
                if work_id in sullivan_work_ids
                else _digest_row(
                    row,
                    exclude=_work_payload_exceptions(
                        work_id, affected=affected, hidden_math=hidden_math
                    ),
                )
            )
            if work_id in affected:
                prior_topics[work_id] = {field: row[field] for field in TOPIC_FIELDS}
    seen: set[str] = set()
    new_ids: set[str] = set()
    for batch in staged.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            work_id = str(row.get("work_id") or "")
            if not work_id or work_id in seen:
                raise ValueError("Staged works have missing or duplicate IDs")
            seen.add(work_id)
            if work_id in exact_sullivan:
                if work_id in order_exceptions:
                    exception = order_exceptions[work_id]
                    field = exception["field"]
                    if (
                        field not in VARIANT_FIELDS
                        or row[field] != exception["bb88_order"]
                        or Counter(row[field]) != Counter(exception["run04_order"])
                        or _digest_row(
                            row,
                            exclude=frozenset([*work_fixture["exclude_fields"], field]),
                        )
                        != exception["sha256_without_field"]
                    ):
                        raise ValueError(
                            f"Sullivan reviewed variant order drift: {work_id}"
                        )
                elif (
                    _digest_row(row, exclude=frozenset(work_fixture["exclude_fields"]))
                    != exact_sullivan[work_id]
                ):
                    raise ValueError(
                        f"Sullivan work differs from reviewed f8 stage: {work_id}"
                    )
            if work_id not in expected:
                new_ids.add(work_id)
                if work_id not in exact_sullivan:
                    raise ValueError(f"Unexpected non-Sullivan new work: {work_id}")
                continue
            if work_id in sullivan_work_ids:
                if work_id not in exact_sullivan:
                    raise ValueError(
                        f"Retained Sullivan work absent from reviewed stage: {work_id}"
                    )
                continue
            if (
                _digest_row(
                    row,
                    exclude=_work_payload_exceptions(
                        work_id, affected=affected, hidden_math=hidden_math
                    ),
                )
                != expected[work_id]
            ):
                raise ValueError(f"Unreviewed staged works payload drift: {work_id}")
            if work_id == FULLERENES_WORK_ID and any(
                row.get(field) != value
                for field, value in FULLERENES_REVIEW_FIELDS.items()
            ):
                raise ValueError("Fullerenes exact attribution review drifted")
            if work_id in affected:
                target = (
                    math[work_id].visible
                    if work_id in math
                    else work_id in plan.restore
                )
                if bool(row.get("map_eligible")) != target:
                    raise ValueError(
                        f"Staged Bernhard/Math visibility drift: {work_id}"
                    )
                if work_id in math:
                    decision = math[work_id]
                    reasons = (
                        []
                        if decision.visible
                        else ["attribution_provisional_hold"]
                        if decision.decision == "hold"
                        else [
                            "reviewed_"
                            + exclusions[(decision.scholar_id, decision.author_pub_id)][
                                "reason_code"
                            ]
                        ]
                    )
                    if (
                        row["map_region_outlier"]
                        or row["map_exclusion_reasons"] != reasons
                    ):
                        raise ValueError(f"Staged Math region reason drift: {work_id}")
                elif bool(row["map_region_outlier"]) != (
                    work_id in plan.reexclude
                ) or row["map_exclusion_reasons"] != (
                    ["low_information_region"] if work_id in plan.reexclude else []
                ):
                    raise ValueError(f"Staged Bernhard region reason drift: {work_id}")
                expected_topics = (
                    UNCLASSIFIED_TOPICS
                    if work_id in plan.gains
                    else prior_topics[work_id]
                    if target
                    else BLANK_TOPICS
                )
                if any(row[field] != expected_topics[field] for field in TOPIC_FIELDS):
                    raise ValueError(f"Staged Bernhard/Math topic drift: {work_id}")
            if work_id in hidden_math:
                _require_hidden_math_attribution(row, work_id)
    removed_ids = expected.keys() - seen
    if (
        removed_ids != set(delta_fixture["removed_old_work_ids"])
        or new_ids != set(delta_fixture["added_new_work_ids"])
        or len(seen) != 47743
    ):
        raise ValueError(
            "Composite staged works differ from exact 13/63 Sullivan rekeys"
        )
    return {
        "status": "pass",
        "baseline_work_rows": len(expected),
        "staged_work_rows": len(seen),
        "sullivan_new_work_ids": sorted(new_ids),
        "sullivan_removed_work_ids": sorted(removed_ids),
        "old_non_sullivan_rows_exact": len(expected)
        - len(expected.keys() & sullivan_work_ids),
    }


def validate_composite_staged_nonwork_tables(
    paths: Mapping[str, tuple[Path, Path]],
    *,
    plan: CompositeRegionPlan,
    curated_dir: Path,
    exclusions_path: Path,
    holds_path: Path,
) -> dict[str, Any]:
    """Compare all four observations classes against exact reviewed values."""

    import pyarrow.parquet as parquet

    if (
        hashlib.sha256(exclusions_path.read_bytes()).hexdigest()
        != MATH_EXCLUSIONS_SHA256
        or hashlib.sha256(holds_path.read_bytes()).hexdigest() != MATH_HOLDS_SHA256
    ):
        raise ValueError("Math and Noonan review manifests changed")
    exclusions = {
        (item["scholar_id"], item["author_pub_id"]): item
        for item in json.loads(exclusions_path.read_text())["exclusions"]
    }
    holds = {
        (profile["scholar_id"], item["author_pub_id"]): item
        for profile in json.loads(holds_path.read_text())["profiles"]
        for item in profile["held_observations"]
    }
    reviewed = {(item.person_id, item.author_pub_id): item for item in plan.math}
    if len(reviewed) != 31:
        raise ValueError("Math review observation identities repeat")
    hidden_math_by_person = Counter(
        item.person_id for item in plan.math if not item.visible
    )
    fixture = _load_stage_fixture(
        curated_dir,
        "composite-sullivan-nonwork-digests.json",
        SULLIVAN_NONWORK_FIXTURE_SHA256,
    )
    receipts: dict[str, Any] = {}
    for config_name in ("people", "authorships", "profile_publications"):
        before_path, after_path = paths[config_name]
        before = parquet.ParquetFile(before_path)
        after = parquet.ParquetFile(after_path)
        if not before.schema_arrow.equals(after.schema_arrow):
            raise ValueError(f"Composite staged {config_name} schema changed")
        key_field = "person_id" if config_name == "people" else "observation_id"
        expected: dict[str, str] = {}
        old_seen: set[str] = set()
        old_sullivan: set[str] = set()
        math_ids: set[str] = set()
        for batch in before.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                key = str(row.get(key_field) or "")
                if not key or key in old_seen:
                    raise ValueError(f"bb88 {config_name} has duplicate IDs")
                old_seen.add(key)
                if config_name == "people":
                    row = dict(row)
                    if key == SULLIVAN_PERSON_ID:
                        row.update(
                            {
                                "publication_observation_count": 252,
                                "scholar_observation_count": 250,
                                "curated_observation_count": 2,
                                "unique_work_count": 160,
                                "mapped_work_count": 160,
                            }
                        )
                    elif key == "person-kevinnoonan-72af9ba9":
                        if (
                            row["unique_work_count"] != 102
                            or row["mapped_work_count"] != 102
                        ):
                            raise ValueError("Noonan bb88 count baseline drifted")
                        row["unique_work_count"] = 101
                        row["mapped_work_count"] = 101
                    if key in hidden_math_by_person:
                        row["unique_work_count"] -= hidden_math_by_person[key]
                        row["mapped_work_count"] -= hidden_math_by_person[key]
                    expected[key] = _digest_row(row, exclude=frozenset())
                    continue
                if config_name != "people" and row["person_id"] == SULLIVAN_PERSON_ID:
                    old_sullivan.add(key)
                    continue
                if config_name != "people":
                    decision = reviewed.get((row["person_id"], row["author_pub_id"]))
                    if decision is not None:
                        math_ids.add(key)
                        if config_name == "authorships" and not decision.visible:
                            continue
                        if config_name == "profile_publications":
                            row = dict(row)
                            row["map_eligible"] = decision.visible
                            if decision.decision == "hold":
                                hold = holds[
                                    (decision.scholar_id, decision.author_pub_id)
                                ]
                                row["map_exclusion_reasons"] = [
                                    "attribution_provisional_hold"
                                ]
                                row["review_attribution_status"] = "held"
                                row["review_attribution_evidence_url"] = hold[
                                    "evidence_url"
                                ]
                                row["collaboration_eligible"] = False
                            elif decision.decision == "exclude":
                                review = exclusions[
                                    (decision.scholar_id, decision.author_pub_id)
                                ]
                                row["map_exclusion_reasons"] = [
                                    f"reviewed_{review['reason_code']}"
                                ]
                                row["review_exclusion_code"] = review["reason_code"]
                                row["review_exclusion_reason"] = review["reason"]
                                row["review_evidence_url"] = review["evidence_url"]
                                row["review_verified_at"] = review["verified_at"]
                                row["collaboration_eligible"] = False
                if key == FULLERENES_OBSERVATION_ID:
                    if config_name == "authorships":
                        continue
                    if config_name == "profile_publications":
                        review = exclusions[(row["scholar_id"], row["author_pub_id"])]
                        if review["reason_code"] != "false_profile_attribution":
                            raise ValueError("Fullerenes exact exclusion changed")
                        row = dict(row)
                        row["map_eligible"] = False
                        row["map_exclusion_reasons"] = [
                            "reviewed_false_profile_attribution"
                        ]
                        row["review_exclusion_code"] = review["reason_code"]
                        row["review_exclusion_reason"] = review["reason"]
                        row["review_evidence_url"] = review["evidence_url"]
                        row["review_verified_at"] = review["verified_at"]
                        row["collaboration_eligible"] = False
                expected[key] = _digest_row(row, exclude=frozenset())
        if config_name != "people" and len(math_ids) != 31:
            raise ValueError(f"bb88 {config_name} lacks all 31 Math observations")
        exact_sullivan = (
            fixture["tables"][config_name]["sha256_by_observation_id"]
            if config_name != "people"
            else {}
        )
        if (
            config_name != "people"
            and len(exact_sullivan)
            != fixture["tables"][config_name]["sullivan_row_count"]
        ):
            raise ValueError(f"Sullivan {config_name} fixture repeats IDs")
        seen: set[str] = set()
        for batch in after.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                key = str(row.get(key_field) or "")
                if not key or key in seen:
                    raise ValueError(f"Staged {config_name} has duplicate IDs")
                seen.add(key)
                if key in expected:
                    digest = expected[key]
                elif key in exact_sullivan and row["person_id"] == SULLIVAN_PERSON_ID:
                    digest = exact_sullivan[key]
                else:
                    raise ValueError(f"Unexpected staged {config_name} row: {key}")
                if _digest_row(row, exclude=frozenset()) != digest:
                    raise ValueError(
                        f"Staged {config_name} row differs from review: {key}"
                    )
        if seen != expected.keys() | exact_sullivan.keys():
            raise ValueError(
                f"Staged {config_name} observation set differs from review"
            )
        receipts[config_name] = {
            "baseline_rows": len(old_seen),
            "staged_rows": len(seen),
            "exact_reviewed_non_sullivan_rows": len(expected),
            "exact_sullivan_rows": len(exact_sullivan),
            "old_sullivan_rows": len(old_sullivan),
        }
    return {"status": "pass", "tables": receipts}
