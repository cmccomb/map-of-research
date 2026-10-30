"""Measure and review presentation changes to retained public map works.

The ordinary publication continuity gate protects work identities and faculty
bindings. A seeded full-corpus refit can still move retained points or change
their semantic topic labels, so that separate surface needs its own review.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from .composite_region import (
    BLANK_TOPICS,
    COORDINATE_FIELDS,
    TOPIC_FIELDS,
    UNCLASSIFIED_TOPICS,
)
from .io import load_json

SCHEMA_VERSION = 1


class PresentationDriftError(ValueError):
    """Material retained-work presentation drift lacks exact review."""

    def __init__(self, receipt: dict[str, Any], reason: str) -> None:
        self.receipt = receipt
        super().__init__(f"Presentation drift gate failed: {reason}")


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _indexed_points(
    artifact: Mapping[str, Any], label: str
) -> dict[str, Mapping[str, Any]]:
    rows = artifact.get("points")
    if not isinstance(rows, list):
        raise ValueError(f"{label} map lacks a point list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} map has a non-object point")
        work_id = str(row.get("work_id") or "")
        if not work_id or work_id in indexed:
            raise ValueError(f"{label} map has a missing or duplicate work ID")
        indexed[work_id] = row
    return indexed


def _keyword_labels(artifact: Mapping[str, Any], label: str) -> dict[str, str]:
    keywords = artifact.get("keywords")
    if not isinstance(keywords, list):
        raise ValueError(f"{label} map lacks a keyword catalog")
    indexed: dict[str, str] = {}
    for row in keywords:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} keyword catalog has a non-object row")
        keyword_id = str(row.get("keyword_id") or "")
        text = str(row.get("label") or "").strip()
        if not keyword_id or not text or keyword_id in indexed:
            raise ValueError(f"{label} map has a missing or duplicate keyword label")
        indexed[keyword_id] = text
    return indexed


def _point_topics(
    point: Mapping[str, Any], labels: Mapping[str, str], label: str
) -> tuple[tuple[str, str], tuple[str, str]]:
    ids = point.get("keyword_ids")
    if not isinstance(ids, list) or len(ids) != 2:
        raise ValueError(f"{label} point lacks exactly two keyword IDs")
    result = []
    for value in ids:
        keyword_id = str(value or "")
        if keyword_id not in labels:
            raise ValueError(f"{label} point references an unknown keyword ID")
        result.append((keyword_id, labels[keyword_id]))
    return result[0], result[1]


def _coordinates(
    point: Mapping[str, Any], label: str
) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(point[field]) for field in ("x", "y", "tsne_x", "tsne_y"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} point has invalid map coordinates") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} point has non-finite map coordinates")
    return values  # type: ignore[return-value]


def _quantile(sorted_values: list[float], proportion: float) -> float:
    return sorted_values[round((len(sorted_values) - 1) * proportion)]


def compare_presentation(
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    prior_revision: str,
    source_snapshot_sha256: str,
    parent_map_sha256: str,
    candidate_map_sha256: str,
) -> dict[str, Any]:
    """Return an exact-content-bound report for retained public work points."""

    old = _indexed_points(parent, "Parent")
    new = _indexed_points(candidate, "Candidate")
    old_labels = _keyword_labels(parent, "Parent")
    new_labels = _keyword_labels(candidate, "Candidate")
    retained = sorted(old.keys() & new.keys())
    if not retained:
        raise ValueError("Presentation drift comparison has no retained works")
    pca_distances: list[float] = []
    tsne_distances: list[float] = []
    coordinates_changed: list[tuple[Any, ...]] = []
    topic_ids_changed: list[tuple[Any, ...]] = []
    broad_changed: list[tuple[Any, ...]] = []
    detail_changed: list[tuple[Any, ...]] = []
    broad_transitions: Counter[tuple[str, str]] = Counter()
    detail_transitions: Counter[tuple[str, str]] = Counter()
    for work_id in retained:
        before, after = old[work_id], new[work_id]
        old_coords = _coordinates(before, "Parent")
        new_coords = _coordinates(after, "Candidate")
        pca_distances.append(math.dist(old_coords[:2], new_coords[:2]))
        tsne_distances.append(math.dist(old_coords[2:], new_coords[2:]))
        if old_coords != new_coords:
            coordinates_changed.append((work_id, old_coords, new_coords))
        old_topics = _point_topics(before, old_labels, "Parent")
        new_topics = _point_topics(after, new_labels, "Candidate")
        if tuple(topic[0] for topic in old_topics) != tuple(
            topic[0] for topic in new_topics
        ):
            topic_ids_changed.append((work_id, old_topics, new_topics))
        if old_topics[0][1] != new_topics[0][1]:
            broad_changed.append((work_id, old_topics[0][1], new_topics[0][1]))
            broad_transitions[(old_topics[0][1], new_topics[0][1])] += 1
        if old_topics[1][1] != new_topics[1][1]:
            detail_changed.append((work_id, old_topics[1][1], new_topics[1][1]))
            detail_transitions[(old_topics[1][1], new_topics[1][1])] += 1
    pca_distances.sort()
    tsne_distances.sort()
    region_before = {
        old_labels[_point_topics(old[work_id], old_labels, "Parent")[0][0]]
        for work_id in retained
    }
    region_after = {
        new_labels[_point_topics(new[work_id], new_labels, "Candidate")[0][0]]
        for work_id in retained
    }
    moved_over_0_1 = sum(distance > 0.1 for distance in tsne_distances)
    pca_moved_over_0_1 = sum(distance > 0.1 for distance in pca_distances)
    material_reasons = []
    # Topic IDs are URL facet identities, and even one incorrect retained label
    # is a visible attribution error. Counts remain in the report for review;
    # they are not safe pass thresholds for these semantic changes.
    if topic_ids_changed:
        material_reasons.append("retained_keyword_id_change")
    if broad_changed:
        material_reasons.append("broad_semantic_topic_churn")
    if detail_changed:
        material_reasons.append("detail_semantic_topic_churn")
    if moved_over_0_1 >= 100 or moved_over_0_1 / len(retained) >= 0.01:
        material_reasons.append("retained_point_movement")
    if pca_moved_over_0_1 >= 100 or pca_moved_over_0_1 / len(retained) >= 0.01:
        material_reasons.append("retained_pca_point_movement")
    if region_before != region_after:
        material_reasons.append("broad_region_name_change")
    report = {
        "schema_version": SCHEMA_VERSION,
        "prior_revision": prior_revision,
        "source_snapshot_sha256": source_snapshot_sha256,
        "parent_map_sha256": parent_map_sha256,
        "candidate_map_sha256": candidate_map_sha256,
        "point_counts": {
            "parent": len(old),
            "candidate": len(new),
            "retained": len(retained),
            "added": len(new.keys() - old.keys()),
            "removed": len(old.keys() - new.keys()),
        },
        "retained": {
            "coordinate_changes": len(coordinates_changed),
            "coordinate_changes_sha256": _digest(coordinates_changed),
            "topic_id_changes": len(topic_ids_changed),
            "topic_id_changes_sha256": _digest(topic_ids_changed),
            "broad_semantic_label_changes": len(broad_changed),
            "broad_semantic_changes_sha256": _digest(broad_changed),
            "detail_semantic_label_changes": len(detail_changed),
            "detail_semantic_changes_sha256": _digest(detail_changed),
            "pca_distance_median": _quantile(pca_distances, 0.5),
            "pca_distance_p95": _quantile(pca_distances, 0.95),
            "pca_distance_over_0_1": pca_moved_over_0_1,
            "tsne_distance_median": _quantile(tsne_distances, 0.5),
            "tsne_distance_p95": _quantile(tsne_distances, 0.95),
            "tsne_distance_p99": _quantile(tsne_distances, 0.99),
            "tsne_distance_max": tsne_distances[-1],
            "tsne_distance_over_0_1": moved_over_0_1,
            "new_broad_region_labels": sorted(region_after - region_before),
            "retired_broad_region_labels": sorted(region_before - region_after),
        },
        "top_broad_label_transitions": [
            {"from": before, "to": after, "works": count}
            for (before, after), count in broad_transitions.most_common(10)
        ],
        "top_detail_label_transitions": [
            {"from": before, "to": after, "works": count}
            for (before, after), count in detail_transitions.most_common(10)
        ],
        "material_reasons": material_reasons,
    }
    report["report_sha256"] = _digest(report)
    return report


def preserve_deletion_only_presentation(
    works: Any,
    prior_works: list[Mapping[str, Any]],
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    """Keep parent placement/topics only for a strict public-point deletion.

    New public points need an explicit placement and topic policy, so this path
    never tries to combine parent coordinates with a newly fitted map.
    """

    old_points = _indexed_points(parent, "Parent")
    new_points = _indexed_points(candidate, "Candidate")
    if not new_points or not new_points.keys() < old_points.keys():
        return works, None
    labels = _keyword_labels(parent, "Parent")
    prior_by_id = {str(row["work_id"]): row for row in prior_works}
    if len(prior_by_id) != len(prior_works):
        raise ValueError("Deletion-only parent works contain duplicate IDs")
    if not old_points.keys() <= prior_by_id.keys():
        raise ValueError("Deletion-only parent map lacks matching works rows")
    if not set((*COORDINATE_FIELDS, *TOPIC_FIELDS)) <= set(works.columns):
        raise ValueError("Deletion-only candidate works lack presentation fields")
    current_ids = works["work_id"].astype(str)
    if current_ids.duplicated().any() or not new_points.keys() <= set(current_ids):
        raise ValueError("Deletion-only candidate works lack unique mapped IDs")
    output = works.copy()
    for index, work_id in current_ids.items():
        if work_id not in new_points:
            continue
        prior = prior_by_id[work_id]
        point = old_points[work_id]
        topics = _point_topics(point, labels, "Parent")
        if any(field not in prior for field in TOPIC_FIELDS):
            raise ValueError(f"Deletion-only parent work lacks topics: {work_id}")
        if (
            str(prior["keyword_id"]) != topics[0][0]
            or str(prior["detail_keyword_id"]) != topics[1][0]
            or str(prior["keyword"]) != topics[0][1]
            or str(prior["detail_keyword"]) != topics[1][1]
        ):
            raise ValueError(f"Deletion-only parent topic differs from map: {work_id}")
        for field, value in zip(
            COORDINATE_FIELDS, _coordinates(point, "Parent"), strict=True
        ):
            output.at[index, field] = value
        for field in TOPIC_FIELDS:
            output.at[index, field] = prior[field]
    return output, {
        "status": "parent_presentation_preserved_for_deletions",
        "parent_points": len(old_points),
        "retained_points": len(new_points),
        "removed_points": len(old_points) - len(new_points),
        "new_point_policy": "no_new_public_points_allowed",
    }


def preserve_reviewed_additive_presentation(
    works: Any,
    prior_works: list[Mapping[str, Any]],
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    prior_revision: str,
    source_snapshot_sha256: str,
    selected_ids_sha256: str,
    parent_map_sha256: str,
    review_path: Path | None,
) -> tuple[Any, dict[str, Any]]:
    """Freeze retained points and require an exact review for every new point.

    The candidate refit supplies only a proposal. New points receive reviewed,
    finite placement or remain in the source dataset with a map hold. They never
    inherit refitted topic labels: mapped additions are explicitly unclassified.
    """

    parent_points = _indexed_points(parent, "Parent")
    candidate_points = _indexed_points(candidate, "Candidate")
    if not parent_points.keys() <= candidate_points.keys():
        raise ValueError("Additive presentation cannot remove parent map points")
    added = sorted(candidate_points.keys() - parent_points.keys())
    if not added:
        raise ValueError("Additive presentation review has no new map points")
    parent_labels = _keyword_labels(parent, "Parent")
    for field in (
        "quality_assessment_version",
        "keyword_model_version",
        "topic_label_review_version",
        "region_audit_version",
        "layout_version",
    ):
        if parent.get(field) != candidate.get(field):
            raise ValueError(f"Additive parent/candidate {field} differs")
    parent_catalog = {str(item["keyword_id"]): item for item in parent["keywords"]}
    for topic_id, expected in (
        (
            UNCLASSIFIED_TOPICS["keyword_id"],
            {
                "label": UNCLASSIFIED_TOPICS["keyword"],
                "extracted_label": UNCLASSIFIED_TOPICS["keyword_extracted"],
                "label_reviewed": UNCLASSIFIED_TOPICS["keyword_label_reviewed"],
                "level": 0,
                "parent_keyword_id": None,
            },
        ),
        (
            UNCLASSIFIED_TOPICS["detail_keyword_id"],
            {
                "label": UNCLASSIFIED_TOPICS["detail_keyword"],
                "extracted_label": UNCLASSIFIED_TOPICS["detail_keyword_extracted"],
                "label_reviewed": UNCLASSIFIED_TOPICS["detail_keyword_label_reviewed"],
                "level": 1,
                "parent_keyword_id": UNCLASSIFIED_TOPICS["keyword_id"],
            },
        ),
    ):
        if topic_id in parent_catalog and any(
            parent_catalog[topic_id].get(field) != value
            for field, value in expected.items()
        ):
            raise ValueError(f"Unclassified topic ID collides with parent: {topic_id}")

    prior_by_id = {str(row["work_id"]): row for row in prior_works}
    if len(prior_by_id) != len(prior_works):
        raise ValueError("Additive parent works contain duplicate IDs")
    if not parent_points.keys() <= prior_by_id.keys():
        raise ValueError("Additive parent map lacks matching works rows")
    if set(added) & prior_by_id.keys():
        raise ValueError(
            "Previously excluded parent work needs separate promotion review"
        )
    required_columns = {
        "work_id",
        "map_eligible",
        "map_exclusion_reasons",
        *COORDINATE_FIELDS,
        *TOPIC_FIELDS,
    }
    if not required_columns <= set(works.columns):
        raise ValueError("Additive candidate works lack placement or topic fields")
    current_ids = works["work_id"].astype(str)
    if current_ids.duplicated().any() or not candidate_points.keys() <= set(
        current_ids
    ):
        raise ValueError("Additive candidate works lack unique mapped IDs")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "new_point_placement_review_required",
        "prior_revision": prior_revision,
        "source_snapshot_sha256": source_snapshot_sha256,
        "selected_ids_sha256": selected_ids_sha256,
        "parent_map_sha256": parent_map_sha256,
        "candidate_map_content_sha256": _digest(
            {
                key: value
                for key, value in candidate.items()
                if key != "generated_at_utc"
            }
        ),
        "new_work_ids_sha256": _digest(added),
        "new_point_count": len(added),
        "retained_point_count": len(parent_points.keys() & candidate_points.keys()),
    }
    if review_path is None:
        raise PresentationDriftError(receipt, "new points lack reviewed placement")
    try:
        review_bytes = review_path.read_bytes()
        review = json.loads(review_bytes.decode("utf-8"))
    except (OSError, ValueError) as error:
        raise PresentationDriftError(
            receipt, "placement review cannot be read"
        ) from error
    binding = {
        "schema_version": SCHEMA_VERSION,
        "decision": "freeze_retained_review_new",
        "prior_revision": prior_revision,
        "source_snapshot_sha256": source_snapshot_sha256,
        "selected_ids_sha256": selected_ids_sha256,
        "parent_map_sha256": parent_map_sha256,
        "candidate_map_content_sha256": receipt["candidate_map_content_sha256"],
        "new_work_ids_sha256": receipt["new_work_ids_sha256"],
    }
    if not isinstance(review, Mapping) or set(review) != {
        *binding,
        "reviewed_at",
        "new_points",
    }:
        raise PresentationDriftError(receipt, "placement review schema is invalid")
    if any(review.get(key) != value for key, value in binding.items()):
        raise PresentationDriftError(receipt, "placement review is stale or mismatched")
    try:
        date.fromisoformat(str(review.get("reviewed_at") or ""))
    except ValueError as error:
        raise PresentationDriftError(
            receipt, "placement review date is invalid"
        ) from error
    decisions = review.get("new_points")
    if not isinstance(decisions, list) or len(decisions) != len(added):
        raise PresentationDriftError(
            receipt, "placement decisions must cover every new point"
        )
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in decisions:
        if not isinstance(item, Mapping) or set(item) != {
            "work_id",
            "decision",
            "coordinates",
            "rationale",
        }:
            raise PresentationDriftError(receipt, "placement row schema is invalid")
        work_id = str(item["work_id"])
        if work_id in by_id or work_id not in added:
            raise PresentationDriftError(
                receipt, "placement work ID repeats or is unexpected"
            )
        decision = item["decision"]
        if decision not in ("place_unclassified", "hold_pending_placement"):
            raise PresentationDriftError(receipt, "placement decision is invalid")
        if len(str(item["rationale"] or "").strip()) < 20:
            raise PresentationDriftError(receipt, "placement rationale is too short")
        if decision == "hold_pending_placement":
            if item["coordinates"] is not None:
                raise PresentationDriftError(receipt, "held placement has coordinates")
        elif not isinstance(item["coordinates"], Mapping) or set(
            item["coordinates"]
        ) != set(COORDINATE_FIELDS):
            raise PresentationDriftError(receipt, "placed point lacks four coordinates")
        else:
            for field in COORDINATE_FIELDS:
                try:
                    value = float(item["coordinates"][field])
                except (TypeError, ValueError) as error:
                    raise PresentationDriftError(
                        receipt, "placement coordinate is invalid"
                    ) from error
                if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                    raise PresentationDriftError(
                        receipt, "placement coordinate is not finite normalized"
                    )
                if round(value, 7) != value:
                    raise PresentationDriftError(
                        receipt, "placement coordinate exceeds map precision"
                    )
        by_id[work_id] = item
    output = works.copy()
    retained = parent_points.keys() & candidate_points.keys()
    for index, work_id in current_ids.items():
        if work_id in retained:
            prior = prior_by_id[work_id]
            point = parent_points[work_id]
            topics = _point_topics(point, parent_labels, "Parent")
            if any(field not in prior for field in TOPIC_FIELDS):
                raise ValueError(f"Additive parent work lacks topic fields: {work_id}")
            for position, prefix in enumerate(("keyword", "detail_keyword")):
                topic_id = topics[position][0]
                metadata = parent_catalog[topic_id]
                if (
                    str(prior[f"{prefix}_id"]) != topic_id
                    or str(prior[prefix]) != metadata["label"]
                    or str(prior[f"{prefix}_extracted"]) != metadata["extracted_label"]
                    or bool(prior[f"{prefix}_label_reviewed"])
                    != bool(metadata["label_reviewed"])
                ):
                    raise ValueError(
                        f"Additive parent topic differs from map: {work_id}"
                    )
            for field, value in zip(
                COORDINATE_FIELDS, _coordinates(point, "Parent"), strict=True
            ):
                output.at[index, field] = value
            for field in TOPIC_FIELDS:
                output.at[index, field] = prior[field]
        elif work_id in by_id:
            decision = by_id[work_id]
            if decision["decision"] == "hold_pending_placement":
                output.at[index, "map_eligible"] = False
                reasons = list(output.at[index, "map_exclusion_reasons"] or [])
                output.at[index, "map_exclusion_reasons"] = list(
                    dict.fromkeys([*reasons, "pending_placement_review"])
                )
                for field, value in BLANK_TOPICS.items():
                    output.at[index, field] = value
            else:
                for field in COORDINATE_FIELDS:
                    output.at[index, field] = float(decision["coordinates"][field])
                for field, value in UNCLASSIFIED_TOPICS.items():
                    output.at[index, field] = value
    receipt["status"] = "parent_frozen_new_points_reviewed"
    receipt["placed_new_points"] = sum(
        decision["decision"] == "place_unclassified" for decision in by_id.values()
    )
    receipt["held_new_points"] = len(by_id) - receipt["placed_new_points"]
    receipt["placed_new_work_ids"] = sorted(
        work_id
        for work_id, item in by_id.items()
        if item["decision"] == "place_unclassified"
    )
    receipt["held_new_work_ids"] = sorted(
        work_id
        for work_id, item in by_id.items()
        if item["decision"] == "hold_pending_placement"
    )
    receipt["placed_new_coordinates_sha256"] = _digest(
        [
            (
                work_id,
                [
                    float(by_id[work_id]["coordinates"][field])
                    for field in COORDINATE_FIELDS
                ],
            )
            for work_id in receipt["placed_new_work_ids"]
        ]
    )
    receipt["review_sha256"] = hashlib.sha256(review_bytes).hexdigest()
    return output, receipt


def require_preserved_additive_artifact(
    parent: Mapping[str, Any],
    staged: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    """Reject any staged map that loses a parent point or changes its presentation."""

    old = _indexed_points(parent, "Parent")
    new = _indexed_points(staged, "Staged")
    if not old.keys() <= new.keys():
        raise ValueError("Additive staged map loses parent points")
    placed = set(receipt["placed_new_work_ids"])
    held = set(receipt["held_new_work_ids"])
    if placed & held or new.keys() - old.keys() != placed or held & new.keys():
        raise ValueError("Additive staged map differs from reviewed new work IDs")
    old_catalog = {str(item["keyword_id"]): item for item in parent["keywords"]}
    new_catalog = {str(item["keyword_id"]): item for item in staged["keywords"]}
    if not old_catalog.keys() <= new_catalog.keys():
        raise ValueError("Additive staged map loses parent keyword IDs")
    for topic_id, old_metadata in old_catalog.items():
        new_metadata = new_catalog[topic_id]
        for field in (
            "label",
            "extracted_label",
            "label_reviewed",
            "level",
            "parent_keyword_id",
        ):
            if old_metadata.get(field) != new_metadata.get(field):
                raise ValueError("Additive staged map changes parent topic metadata")
    for work_id, point in old.items():
        staged_point = new[work_id]
        if _coordinates(point, "Parent") != _coordinates(staged_point, "Staged"):
            raise ValueError("Additive staged map moves a retained point")
        if point["keyword_ids"] != staged_point["keyword_ids"]:
            raise ValueError("Additive staged map changes retained topic IDs")
    for work_id in placed:
        point = new[work_id]
        if point["keyword_ids"] != [
            UNCLASSIFIED_TOPICS["keyword_id"],
            UNCLASSIFIED_TOPICS["detail_keyword_id"],
        ]:
            raise ValueError("Additive staged map labels new work as reviewed")
        coordinates = _coordinates(point, "Staged")
        if any(not -1.0 <= value <= 1.0 for value in coordinates):
            raise ValueError("Additive staged map has unnormalized placement")
    staged_placements = [
        (work_id, list(_coordinates(new[work_id], "Staged")))
        for work_id in sorted(placed)
    ]
    if _digest(staged_placements) != receipt.get("placed_new_coordinates_sha256"):
        raise ValueError("Additive staged map differs from reviewed work coordinates")


def require_presentation_review(
    report: Mapping[str, Any], approval_path: Path | None = None
) -> dict[str, Any]:
    """Fail before a Hub write if material drift lacks an exact reviewed approval."""

    receipt = dict(report)
    reasons = report.get("material_reasons")
    if not isinstance(reasons, list):
        raise ValueError("Presentation drift report lacks material reasons")
    if not reasons:
        if approval_path is not None:
            raise PresentationDriftError(
                receipt, "unused approval on an unchanged presentation"
            )
        receipt["status"] = "pass_no_material_drift"
        return receipt
    if approval_path is None:
        receipt["status"] = "review_required"
        raise PresentationDriftError(
            receipt, "material drift lacks a reviewed approval"
        )
    try:
        approval = load_json(approval_path)
    except (OSError, ValueError) as error:
        receipt["status"] = "invalid_approval"
        raise PresentationDriftError(receipt, "approval cannot be read") from error
    if not isinstance(approval, Mapping) or set(approval) != {
        "schema_version",
        "decision",
        "prior_revision",
        "source_snapshot_sha256",
        "parent_map_sha256",
        "candidate_map_sha256",
        "report_sha256",
        "reviewed_at",
        "rationale",
    }:
        receipt["status"] = "invalid_approval"
        raise PresentationDriftError(receipt, "approval schema is invalid")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "decision": "accept_refit",
        "prior_revision": report["prior_revision"],
        "source_snapshot_sha256": report["source_snapshot_sha256"],
        "parent_map_sha256": report["parent_map_sha256"],
        "candidate_map_sha256": report["candidate_map_sha256"],
        "report_sha256": report["report_sha256"],
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        receipt["status"] = "stale_approval"
        raise PresentationDriftError(
            receipt, "approval differs from exact parent, source, or candidate"
        )
    reviewed_at = str(approval.get("reviewed_at") or "")
    rationale = str(approval.get("rationale") or "").strip()
    try:
        date.fromisoformat(reviewed_at)
    except ValueError as error:
        receipt["status"] = "invalid_approval"
        raise PresentationDriftError(receipt, "approval date is invalid") from error
    if len(rationale) < 20:
        receipt["status"] = "invalid_approval"
        raise PresentationDriftError(receipt, "approval lacks review date or rationale")
    receipt["status"] = "pass_reviewed_refit"
    receipt["reviewed_approval_sha256"] = hashlib.sha256(
        approval_path.read_bytes()
    ).hexdigest()
    return receipt
