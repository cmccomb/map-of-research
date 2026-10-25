"""Focused regressions for the frozen composite release presentation gate."""

from __future__ import annotations

import copy
import math

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
