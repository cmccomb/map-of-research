"""Focused regressions for the frozen composite release presentation gate."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pandas
import pytest

from map_of_research.composite_region import (
    BLANK_TOPICS,
    COORDINATE_FIELDS,
    HIDDEN_MATH_ATTRIBUTION_FIELDS,
    TOPIC_FIELDS,
    VARIANT_FIELDS,
    CompositeRegionPlan,
    _digest_row,
    _require_hidden_math_attribution,
    _reviewed_variant_order_matches,
    _work_payload_exceptions,
    freeze_composite_presentation,
)


def test_retained_visible_nonpoint_keeps_prior_blank_topics():
    """A work without eligible membership must not acquire a map topic."""

    work_id = "work-retained-nonpoint"
    prior = {
        "work_id": work_id,
        **BLANK_TOPICS,
        **{field: math.nan for field in COORDINATE_FIELDS},
        **{field: [] for field in VARIANT_FIELDS},
        "map_region_documented_override": False,
    }
    current = {
        **prior,
        "map_eligible": True,
        "map_region_documented_override": True,
        "has_included_source_profile": False,
        "memberships": [],
        "map_region_prior_eligible_override": False,
        "map_region_prior_excluded_override": False,
    }
    plan = CompositeRegionPlan(frozenset(), frozenset(), (), "test")

    result = freeze_composite_presentation(
        pandas.DataFrame([current]),
        [prior],
        {"points": []},
        plan,
        set(),
    ).iloc[0]

    assert {field: result[field] for field in TOPIC_FIELDS} == BLANK_TOPICS
    assert (
        result["map_region_documented_override"]
        == prior["map_region_documented_override"]
    )


def test_hidden_math_allows_only_exact_empty_attribution():
    """The reviewed hold clears affiliation; unrelated payload still fails parity."""

    work_id = "work-reviewed-hidden-math"
    before = {
        "work_id": work_id,
        "title": "Frozen Scholar title",
        "map_eligible": True,
        **BLANK_TOPICS,
        **{
            field: True if field == "collaboration_eligible" else ["old member"]
            for field in HIDDEN_MATH_ATTRIBUTION_FIELDS
        },
    }
    after = copy.deepcopy(before)
    after["map_eligible"] = False
    for field in HIDDEN_MATH_ATTRIBUTION_FIELDS:
        after[field] = False if field == "collaboration_eligible" else []
    exceptions = _work_payload_exceptions(
        work_id, affected=frozenset({work_id}), hidden_math={work_id}
    )

    _require_hidden_math_attribution(after, work_id)
    assert _digest_row(before, exclude=exceptions) == _digest_row(
        after, exclude=exceptions
    )
    changed_title = {**after, "title": "Unreviewed title"}
    assert _digest_row(before, exclude=exceptions) != _digest_row(
        changed_title, exclude=exceptions
    )
    with pytest.raises(ValueError, match="Hidden Math attribution drift"):
        _require_hidden_math_attribution({**after, "faculty": ["unexpected"]}, work_id)


def test_manual_upload_uses_immutable_producer_and_composite_manifests():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/upload.yml"
    ).read_text(encoding="utf-8")
    assert (
        "ref: ${{ github.event_name == 'workflow_dispatch' && github.sha || 'main' }}"
        in workflow
    )
    assert 'MAP_PRODUCER_COMMIT="$(git -C source rev-parse HEAD)"' in workflow
    assert (
        "--math-region-review-manifest source/data/curated/scholar-region-review.json"
    ) in workflow
    assert (
        "--sullivan-continuity-addendum "
        "source/data/curated/sullivan-continuity-addendum-bb88.json"
    ) in workflow


def test_only_two_exact_reviewed_sullivan_variant_orders_are_accepted():
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "data/curated/composite-sullivan-work-digests.json"
        ).read_text(encoding="utf-8")
    )
    exceptions = fixture["order_only_variant_exceptions"]
    assert len(exceptions) == 2
    for exception in exceptions.values():
        assert _reviewed_variant_order_matches(exception["bb88_order"], exception)
        assert _reviewed_variant_order_matches(exception["run04_order"], exception)
        changed = [*exception["run04_order"]]
        changed[0] = "Unreviewed title or venue"
        assert not _reviewed_variant_order_matches(changed, exception)
        assert not _reviewed_variant_order_matches(changed[:1], exception)
