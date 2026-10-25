"""Preserve reviewed map topics when a release adds only held raw sources.

The immutable Hub parent is the authority for the eight work topic fields.
This path is deliberately unavailable when any released observation, work, or
non-topic map point changes. Other release paths must regenerate their topics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .composite_region import TOPIC_FIELDS
from .region_continuity import (
    IDENTITY_FIELDS,
    PRIOR_OBSERVATION_FIELDS,
    PRIOR_WORK_FIELDS,
)


def held_only_continuity_candidate(
    receipt: Mapping[str, Any], prior_revision: str
) -> bool:
    """Enter the strict freeze gate only for a zero-visible-delta release.

    Other reviewed releases can still run their existing topic path while held
    profiles remain in the raw source. The exact row and map checks below are
    authoritative once this receipt identifies a held-only candidate.
    """

    if receipt.get("prior_revision") != prior_revision:
        return False
    previous = receipt.get("prior_visible_observations")
    current = receipt.get("current_visible_observations")
    if type(previous) is not int or type(current) is not int or previous != current:
        return False
    if type(receipt.get("new_source_gains")) is not int or receipt["new_source_gains"]:
        return False
    change_lists = (
        "reviewed_losses",
        "verified_gains",
        "reviewed_composite_observation_gains",
        "reviewed_composite_observation_losses",
        "approved_sullivan_held_losses",
        "approved_sullivan_rekey_losses",
        "approved_sullivan_additional_losses",
        "approved_sullivan_binding_changes",
        "unapproved_mapped_binding_changes",
        "unexplained_losses",
        "unexplained_gains",
    )
    return all(receipt.get(field) == [] for field in change_lists)


def _value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Mapping):
        return {key: _value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _value(value.tolist())
    return value


def _rows(value: Any) -> list[dict[str, Any]]:
    return value.to_dict("records") if hasattr(value, "to_dict") else list(value)


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    key_fields: tuple[str, ...],
    required_fields: tuple[str, ...],
    label: str,
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    indexed: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        missing = set(required_fields).difference(row)
        if missing:
            raise ValueError(f"Held-only {label} lacks fields: {sorted(missing)}")
        key = tuple(str(row[field] or "") for field in key_fields)
        if not key[0] or key in indexed:
            raise ValueError(
                f"Held-only {label} has missing or duplicate identity: {key}"
            )
        indexed[key] = row
    if not indexed:
        raise ValueError(f"Held-only {label} is empty")
    return indexed


def _require_unchanged_rows(
    prior: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    key_fields: tuple[str, ...],
    fields: tuple[str, ...],
    label: str,
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    old = _index_rows(prior, key_fields, fields, f"parent {label}")
    new = _index_rows(current, key_fields, fields, f"current {label}")
    if old.keys() != new.keys():
        raise ValueError(f"Held-only released {label} identities changed")
    for key, before in old.items():
        after = new[key]
        for field in fields:
            if _value(before[field]) != _value(after[field]):
                raise ValueError(
                    f"Held-only released {label} payload changed: {key}: {field}"
                )
    return old


def _points(artifact: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    points = artifact.get("points")
    if not isinstance(points, list) or artifact.get("point_count") != len(points):
        raise ValueError(f"Held-only {label} map point count is invalid")
    result = _index_rows(points, ("work_id",), ("work_id", "keyword_ids"), label)
    return {key[0]: row for key, row in result.items()}


def _require_parent_topics(
    prior_works: Mapping[tuple[str, ...], Mapping[str, Any]],
    prior_artifact: Mapping[str, Any],
    prior_points: Mapping[str, Mapping[str, Any]],
) -> None:
    keywords = prior_artifact.get("keywords")
    if not isinstance(keywords, list) or prior_artifact.get("keyword_count") != len(
        keywords
    ):
        raise ValueError("Held-only parent map keyword catalog is invalid")
    catalog = _index_rows(
        keywords,
        ("keyword_id",),
        (
            "keyword_id",
            "label",
            "extracted_label",
            "label_reviewed",
            "level",
            "parent_keyword_id",
        ),
        "parent map keywords",
    )
    for work_id, point in prior_points.items():
        row = prior_works.get((work_id,))
        if row is None:
            raise ValueError(f"Held-only parent map point lacks work: {work_id}")
        ids = point["keyword_ids"]
        if ids != [row["keyword_id"], row["detail_keyword_id"]]:
            raise ValueError(f"Held-only parent work/map topics differ: {work_id}")
        for level, prefix in enumerate(("keyword", "detail_keyword")):
            item = catalog.get((ids[level],))
            if item is None or any(
                item[field] != expected
                for field, expected in (
                    ("label", row[prefix]),
                    ("extracted_label", row[f"{prefix}_extracted"]),
                    ("label_reviewed", row[f"{prefix}_label_reviewed"]),
                    ("level", level),
                    ("parent_keyword_id", None if level == 0 else ids[0]),
                )
            ):
                raise ValueError(f"Held-only parent topic metadata differs: {work_id}")


def preserve_held_only_topics(
    works: Any,
    *,
    prior_works: Sequence[Mapping[str, Any]],
    prior_profiles: Sequence[Mapping[str, Any]],
    current_profiles: Any,
    prior_artifact: Mapping[str, Any],
    candidate_artifact: Mapping[str, Any],
    held_observation_ids: set[str],
) -> Any:
    """Copy parent topics only after proving all published inputs are unchanged."""

    if not held_observation_ids:
        raise ValueError("Held-only topic preservation needs held source rows")
    _require_unchanged_rows(
        prior_profiles,
        _rows(current_profiles),
        IDENTITY_FIELDS,
        PRIOR_OBSERVATION_FIELDS,
        "observation",
    )
    old_works = _require_unchanged_rows(
        prior_works,
        _rows(works),
        ("work_id",),
        PRIOR_WORK_FIELDS,
        "work",
    )
    if any(set(TOPIC_FIELDS).difference(row) for row in old_works.values()):
        raise ValueError("Held-only parent works lack frozen topic fields")
    old_points = _points(prior_artifact, "parent")
    new_points = _points(candidate_artifact, "candidate")
    if old_points.keys() != new_points.keys():
        raise ValueError("Held-only mapped work IDs changed")
    _require_parent_topics(old_works, prior_artifact, old_points)
    for work_id, before in old_points.items():
        after = new_points[work_id]
        if {key: value for key, value in before.items() if key != "keyword_ids"} != {
            key: value for key, value in after.items() if key != "keyword_ids"
        }:
            raise ValueError(f"Held-only non-topic map point changed: {work_id}")
    output = works.copy()
    for index, work_id in output["work_id"].items():
        before = old_works[(str(work_id),)]
        for field in TOPIC_FIELDS:
            output.at[index, field] = before[field]
    return output


def validate_held_only_stage(
    candidate_artifact: Mapping[str, Any],
    prior_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact parent map payload before the first atomic Hub write."""

    if candidate_artifact.get("points") != prior_artifact.get("points"):
        raise ValueError("Held-only staged map points differ from parent")
    if candidate_artifact.get("keywords") != prior_artifact.get("keywords"):
        raise ValueError("Held-only staged map keywords differ from parent")
    current = {
        key: value
        for key, value in candidate_artifact.items()
        if key != "generated_at_utc"
    }
    previous = {
        key: value for key, value in prior_artifact.items() if key != "generated_at_utc"
    }
    if current != previous:
        raise ValueError("Held-only staged map metadata differs from parent")
    return {
        "status": "pass",
        "point_count": candidate_artifact["point_count"],
        "keyword_count": candidate_artifact["keyword_count"],
    }
