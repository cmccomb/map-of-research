"""Focused regressions for the frozen composite release presentation gate."""

from __future__ import annotations

import copy
import hashlib
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
    apply_composite_region_decisions,
    freeze_composite_presentation,
)
from map_of_research.region_continuity import (
    WORK_FINGERPRINT_FIELDS,
    _fingerprint_fields,
)
from map_of_research.region_review import ReviewedRegionDecision


def _synthetic_bernhard_release():
    """Exercise all 280 source-bound IDs and 284 frozen observations."""

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "data/curated/region-remediation-bernhard.json").read_text()
    )
    review = json.loads((root / "data/curated/scholar-region-review.json").read_text())
    restore = frozenset(manifest["restore_eligible_work_ids"])
    reexclude = frozenset(manifest["restore_outlier_work_ids"])
    math_rows = {item["work_id"]: item for item in review["decisions"]}
    before_works = {}
    history = {}
    before_profiles = {}
    after_profiles = {}
    staged = []
    decisions = []
    nonmath = []
    for index, work_id in enumerate(sorted(restore | reexclude)):
        old_visible = work_id in reexclude
        old = {
            "work_id": work_id,
            "title": f"Frozen work {index}",
            "quality_assessment_version": "test",
            "map_eligible": old_visible,
            "map_region_outlier": not old_visible,
            "map_exclusion_reasons": [] if old_visible else ["low_information_region"],
        }
        before_works[work_id] = old
        history[work_id] = {
            "map_eligible": not old_visible,
            "map_region_outlier": old_visible,
        }
        current = {**old, **history[work_id]}
        current["map_exclusion_reasons"] = (
            ["low_information_region"] if old_visible else []
        )
        staged.append(current)
        reviewed = math_rows.get(work_id)
        if reviewed is None:
            nonmath.append(work_id)
            person_id = f"person-test-{index}"
            scholar_id = f"scholar-test-{index}"
            author_pub_id = f"pub-test-{index}"
        else:
            person_id = reviewed["person_id"]
            scholar_id = reviewed["scholar_id"]
            author_pub_id = reviewed["author_pub_id"]
        identity = (person_id, scholar_id, author_pub_id, "scholar")
        raw = json.dumps({"work_id": work_id})
        before_profiles[identity] = {
            "work_id": work_id,
            "source_record_json": raw,
            "map_eligible": old_visible,
        }
        after_profiles[identity] = {
            **before_profiles[identity],
            "map_eligible": reviewed["decision"] == "include"
            if reviewed is not None
            else old_visible,
        }
        if reviewed is not None:
            decisions.append(
                ReviewedRegionDecision(
                    work_id=work_id,
                    person_id=person_id,
                    scholar_id=scholar_id,
                    author_pub_id=author_pub_id,
                    capture_row_index=reviewed["capture_row_index"],
                    source_record_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    work_fingerprint_sha256=_fingerprint_fields(
                        old, (*WORK_FINGERPRINT_FIELDS, "quality_assessment_version")
                    ),
                    captured_title=reviewed["captured_title"],
                    transition=reviewed["transition"],
                    decision=reviewed["decision"],
                    evidence_url=reviewed["evidence_url"],
                    evidence_scope=reviewed["evidence_scope"],
                    saved_scholar_url=reviewed["saved_scholar_url"],
                    reason=reviewed["reason"],
                    reviewed_at=reviewed["reviewed_at"],
                )
            )
    for offset, work_id in enumerate(nonmath[:4]):
        identity = (
            f"person-extra-{offset}",
            f"scholar-extra-{offset}",
            f"pub-extra-{offset}",
            "scholar",
        )
        before_profiles[identity] = {
            "work_id": work_id,
            "source_record_json": json.dumps({"extra": offset}),
            "map_eligible": bool(before_works[work_id]["map_eligible"]),
        }
        after_profiles[identity] = dict(before_profiles[identity])
    return (
        pandas.DataFrame(staged),
        before_works,
        history,
        before_profiles,
        after_profiles,
        CompositeRegionPlan(restore, reexclude, tuple(decisions), "test"),
        nonmath,
    )


def test_only_reviewed_math_changes_bernhard_visibility():
    (staged, old, history, old_profiles, current_profiles, plan, nonmath) = (
        _synthetic_bernhard_release()
    )
    assert len(nonmath) == 249
    assert len(old_profiles) == len(current_profiles) == 284
    assert len(plan.gains) == 4
    assert len(plan.losses) == 2

    result = apply_composite_region_decisions(
        staged,
        prior_works=old,
        historical_works=history,
        prior_profiles=old_profiles,
        current_profiles=current_profiles,
        plan=plan,
    )
    by_id = {row.work_id: row for row in result.itertuples(index=False)}
    for work_id in nonmath:
        assert by_id[work_id].map_eligible == old[work_id]["map_eligible"]
        assert by_id[work_id].map_region_outlier == old[work_id]["map_region_outlier"]
        assert (
            by_id[work_id].map_exclusion_reasons
            == old[work_id]["map_exclusion_reasons"]
        )
    assert {
        work_id
        for work_id in old
        if by_id[work_id].map_eligible and not old[work_id]["map_eligible"]
    } == plan.gains
    assert {
        work_id
        for work_id in old
        if old[work_id]["map_eligible"] and not by_id[work_id].map_eligible
    } == plan.losses


def test_unreviewed_bernhard_source_change_fails_closed():
    (staged, old, history, old_profiles, current_profiles, plan, nonmath) = (
        _synthetic_bernhard_release()
    )
    identity = next(
        identity
        for identity, row in current_profiles.items()
        if row["work_id"] == nonmath[0]
    )
    current_profiles[identity] = {
        **current_profiles[identity],
        "source_record_json": "changed source record",
    }
    with pytest.raises(ValueError, match="Non-Sullivan Scholar observation source"):
        apply_composite_region_decisions(
            staged,
            prior_works=old,
            historical_works=history,
            prior_profiles=old_profiles,
            current_profiles=current_profiles,
            plan=plan,
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
