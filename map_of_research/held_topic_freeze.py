"""Preserve reviewed map topics when a release adds only held raw sources.

The immutable Hub parent is the authority for the eight work topic fields.
This path is deliberately unavailable when any released observation, work, or
non-topic map point changes. Other release paths must regenerate their topics.
"""

from __future__ import annotations

import math
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .composite_region import TOPIC_FIELDS
from .region_continuity import (
    IDENTITY_FIELDS,
    PRIOR_OBSERVATION_FIELDS,
    PRIOR_WORK_FIELDS,
)

ORDER_ONLY_VARIANTS = frozenset(
    {"title_variants", "venue_variants", "citation_variants", "author_variants"}
)
HIDDEN_COORDINATES = frozenset({"x", "y", "tsne_x", "tsne_y"})
TRANSIENT_WORK_FIELDS = frozenset(
    {"map_region_prior_eligible_override", "map_region_prior_excluded_override"}
)
DOCUMENTED_OVERRIDE_WORK_ID = (
    "work-7312a4280e990b03ec1e95cecf82ace2ce437d15413b87634e9d5544f918c3be"
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


def validate_held_only_table_parity(
    parent: Mapping[str, Path],
    staged: Mapping[str, Path],
    *,
    configs: tuple[str, ...] = (
        "people",
        "works",
        "authorships",
        "profile_publications",
    ),
) -> dict[str, int]:
    """Compare published schema, row order, and every value before a Hub write."""

    import pyarrow.parquet as parquet

    counts = {}
    for config in configs:
        before = parquet.ParquetFile(parent[config])
        after = parquet.ParquetFile(staged[config])
        if (
            not before.schema_arrow.equals(after.schema_arrow, check_metadata=True)
            or before.metadata.num_rows != after.metadata.num_rows
        ):
            raise ValueError(f"Held-only staged {config} schema or row count changed")
        old_rows = (
            row
            for batch in before.iter_batches(batch_size=128)
            for row in batch.to_pylist()
        )
        new_rows = (
            row
            for batch in after.iter_batches(batch_size=128)
            for row in batch.to_pylist()
        )
        for index, (old, new) in enumerate(
            zip_longest(old_rows, new_rows, fillvalue=None)
        ):
            if old != new:
                raise ValueError(f"Held-only staged {config} row changed: {index}")
        counts[config] = before.metadata.num_rows
    return counts


def validate_held_only_candidate_works(parent: Path, staged: Path) -> int:
    """Prove work payload before parent bytes replace derived producer drift.

    The exact 38f held-only comparison found only variant permutations, cleared
    coordinates on 37 hidden works, one documented override, and two transient
    producer columns after the eight topic fields were restored. Other changes
    require a separate review, including changes on hidden work identities.
    """

    import pyarrow.parquet as parquet

    before = parquet.ParquetFile(parent)
    after = parquet.ParquetFile(staged)
    old_schema = before.schema_arrow
    new_schema = after.schema_arrow
    extra = set(new_schema.names) - set(old_schema.names)
    if (
        set(old_schema.names) - set(new_schema.names)
        or extra - TRANSIENT_WORK_FIELDS
        or [name for name in new_schema.names if name not in extra] != old_schema.names
        or before.metadata.num_rows != after.metadata.num_rows
        or any(
            not old_schema.field(name).equals(
                new_schema.field(name), check_metadata=True
            )
            for name in old_schema.names
        )
    ):
        raise ValueError("Held-only candidate works schema, order, or count changed")
    old_rows = (
        row
        for batch in before.iter_batches(batch_size=128)
        for row in batch.to_pylist()
    )
    new_rows = (
        row for batch in after.iter_batches(batch_size=128) for row in batch.to_pylist()
    )
    seen: set[str] = set()
    cleared_coordinate_work_ids: set[str] = set()
    for index, (old, new) in enumerate(zip_longest(old_rows, new_rows)):
        if old is None or new is None or old["work_id"] != new["work_id"]:
            raise ValueError(
                f"Held-only candidate work identity/order changed: {index}"
            )
        work_id = str(old["work_id"])
        if not work_id or work_id in seen:
            raise ValueError(f"Held-only candidate repeated work identity: {work_id}")
        seen.add(work_id)
        cleared_coordinates: set[str] = set()
        for field in old_schema.names:
            previous, current = old[field], new[field]
            if previous == current:
                continue
            if (
                field in ORDER_ONLY_VARIANTS
                and isinstance(previous, list)
                and isinstance(current, list)
                and Counter(previous) == Counter(current)
            ):
                continue
            if (
                field in HIDDEN_COORDINATES
                and old["map_eligible"] is False
                and new["map_eligible"] is False
                and previous is not None
                and current is None
            ):
                cleared_coordinates.add(field)
                continue
            if (
                field == "map_region_documented_override"
                and work_id == DOCUMENTED_OVERRIDE_WORK_ID
                and previous is False
                and current is True
            ):
                continue
            raise ValueError(
                f"Held-only candidate work payload changed: {work_id}: {field}"
            )
        if cleared_coordinates:
            if cleared_coordinates != HIDDEN_COORDINATES:
                raise ValueError(
                    f"Held-only candidate cleared partial coordinates: {work_id}"
                )
            cleared_coordinate_work_ids.add(work_id)
    if len(cleared_coordinate_work_ids) > 37:
        raise ValueError("Held-only candidate cleared unexpected hidden coordinates")
    return len(seen)


def retain_held_only_parent_works(
    staged: Mapping[str, Path], *, prior_revision: str, hf_token: str
) -> tuple[Any, dict[str, int]]:
    """Reuse parent work bytes only after proving all non-work tables are exact.

    `preserve_held_only_topics` has already checked every released observation,
    work identity/fingerprint, and non-topic map point. The remaining work-row
    differences are derived variant/layout values and transient producer fields.
    The final four-table and full-map gates still run before the upload.
    """

    import pandas
    from huggingface_hub import hf_hub_download

    from .region_continuity import require_revision

    revision = require_revision(prior_revision)
    parent = {
        config: Path(
            hf_hub_download(
                "ccm/cmu-engineering-publications",
                f"{config}/train-00000-of-00001.parquet",
                repo_type="dataset",
                revision=revision,
                token=hf_token or None,
            )
        )
        for config in (
            "people",
            "works",
            "authorships",
            "profile_publications",
        )
    }
    validate_held_only_table_parity(
        parent,
        staged,
        configs=("people", "authorships", "profile_publications"),
    )
    validate_held_only_candidate_works(parent["works"], staged["works"])
    shutil.copyfile(parent["works"], staged["works"])
    counts = validate_held_only_table_parity(parent, staged)
    return pandas.read_parquet(parent["works"]), counts
