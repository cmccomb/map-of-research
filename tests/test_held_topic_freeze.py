import copy
import math

import numpy
import pandas
import pytest

from map_of_research.composite_region import TOPIC_FIELDS, UNCLASSIFIED_TOPICS
from map_of_research.held_topic_freeze import (
    _value,
    held_only_continuity_candidate,
    preserve_held_only_topics,
    validate_held_only_stage,
)
from map_of_research.region_continuity import (
    PRIOR_OBSERVATION_FIELDS,
    PRIOR_WORK_FIELDS,
)


def continuity_receipt():
    return {
        "prior_revision": "a" * 40,
        "prior_visible_observations": 100,
        "current_visible_observations": 100,
        "new_source_gains": 0,
        "reviewed_losses": [],
        "verified_gains": [],
        "reviewed_composite_observation_gains": [],
        "reviewed_composite_observation_losses": [],
        "approved_sullivan_held_losses": [],
        "approved_sullivan_rekey_losses": [],
        "approved_sullivan_additional_losses": [],
        "approved_sullivan_binding_changes": [],
        "unapproved_mapped_binding_changes": [],
        "unexplained_losses": [],
        "unexplained_gains": [],
    }


def test_held_only_candidate_requires_zero_visible_delta() -> None:
    receipt = continuity_receipt()
    assert held_only_continuity_candidate(receipt, "a" * 40)
    for field, value in (
        ("current_visible_observations", 101),
        ("new_source_gains", 1),
        ("verified_gains", ["person/new/released/scholar"]),
        ("reviewed_losses", ["person/old/released/scholar"]),
        ("approved_sullivan_binding_changes", [{"observation": "changed"}]),
    ):
        changed = {**receipt, field: value}
        assert not held_only_continuity_candidate(changed, "a" * 40)
    assert not held_only_continuity_candidate(receipt, "b" * 40)
    assert not held_only_continuity_candidate(
        {key: value for key, value in receipt.items() if key != "new_source_gains"},
        "a" * 40,
    )


def held_only_case():
    profile = {field: None for field in PRIOR_OBSERVATION_FIELDS}
    profile.update(
        person_id="person-one",
        scholar_id="ScholarOne",
        author_pub_id="ScholarOne:one",
        source_type="scholar",
        work_id="work-one",
        map_eligible=True,
        source_record_json='{"title":"Prior work"}',
        title="Prior work",
    )
    work = {field: None for field in PRIOR_WORK_FIELDS}
    work.update(
        work_id="work-one",
        title="Prior work",
        map_eligible=True,
        map_region_outlier=False,
        department_ids=["physics"],
    )
    work.update(
        keyword_id="reviewed-broad",
        keyword="Reviewed broad topic",
        keyword_extracted="Extracted broad topic",
        keyword_label_reviewed=True,
        detail_keyword_id="reviewed-detail",
        detail_keyword="Reviewed detailed topic",
        detail_keyword_extracted="Extracted detailed topic",
        detail_keyword_label_reviewed=True,
    )
    current_work = {**work, **dict.fromkeys(TOPIC_FIELDS, "reclustered")}
    current_work["keyword_label_reviewed"] = False
    current_work["detail_keyword_label_reviewed"] = False
    point = {
        "work_id": "work-one",
        "title": "Prior work",
        "x": 0.1,
        "faculty_ids": ["person-one"],
        "keyword_ids": ["reviewed-broad", "reviewed-detail"],
    }
    prior_artifact = {
        "generated_at_utc": "2026-10-24T00:00:00+00:00",
        "point_count": 1,
        "keyword_count": 2,
        "catalogs": {"faculty": [{"person_id": "person-one"}]},
        "points": [point],
        "keywords": [
            {
                "keyword_id": "reviewed-broad",
                "label": "Reviewed broad topic",
                "extracted_label": "Extracted broad topic",
                "label_reviewed": True,
                "level": 0,
                "parent_keyword_id": None,
            },
            {
                "keyword_id": "reviewed-detail",
                "label": "Reviewed detailed topic",
                "extracted_label": "Extracted detailed topic",
                "label_reviewed": True,
                "level": 1,
                "parent_keyword_id": "reviewed-broad",
            },
        ],
    }
    candidate_artifact = copy.deepcopy(prior_artifact)
    candidate_artifact["points"][0]["keyword_ids"] = [
        "reclustered-broad",
        "reclustered-detail",
    ]
    return {
        "works": pandas.DataFrame([current_work]),
        "prior_works": [work],
        "prior_profiles": [profile],
        "current_profiles": pandas.DataFrame([profile]),
        "prior_artifact": prior_artifact,
        "candidate_artifact": candidate_artifact,
        "held_observation_ids": {"observation-held"},
    }


def set_cell(case, table, field, value):
    case[table].loc[0, field] = value


def test_held_only_topics_copy_all_eight_reviewed_fields() -> None:
    case = held_only_case()
    result = preserve_held_only_topics(**case)
    for field in TOPIC_FIELDS:
        assert result.iloc[0][field] == case["prior_works"][0][field]
    assert case["works"].iloc[0]["keyword_id"] == "reclustered"


def test_held_only_freeze_preserves_pending_review_topic_and_stage() -> None:
    case = held_only_case()
    case["prior_works"][0].update(UNCLASSIFIED_TOPICS)
    ids = [
        UNCLASSIFIED_TOPICS["keyword_id"],
        UNCLASSIFIED_TOPICS["detail_keyword_id"],
    ]
    case["prior_artifact"]["points"][0]["keyword_ids"] = ids
    for level, prefix in enumerate(("keyword", "detail_keyword")):
        keyword = case["prior_artifact"]["keywords"][level]
        keyword.update(
            keyword_id=ids[level],
            label=UNCLASSIFIED_TOPICS[prefix],
            extracted_label=UNCLASSIFIED_TOPICS[f"{prefix}_extracted"],
            label_reviewed=False,
            parent_keyword_id=None if level == 0 else ids[0],
        )
    result = preserve_held_only_topics(**case)
    assert {field: result.iloc[0][field] for field in TOPIC_FIELDS} == (
        UNCLASSIFIED_TOPICS
    )
    staged = copy.deepcopy(case["prior_artifact"])
    staged["points"][0]["keyword_ids"] = [
        result.iloc[0]["keyword_id"],
        result.iloc[0]["detail_keyword_id"],
    ]
    staged["generated_at_utc"] = "2026-10-25T00:00:00+00:00"
    assert validate_held_only_stage(staged, case["prior_artifact"])["status"] == (
        "pass"
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda case: case["held_observation_ids"].clear(),
            "needs held source rows",
        ),
        (
            lambda case: set_cell(
                case, "current_profiles", "author_pub_id", "ScholarOne:changed"
            ),
            "observation identities changed",
        ),
        (
            lambda case: set_cell(
                case, "current_profiles", "source_record_json", '{"title":"Changed"}'
            ),
            "observation payload changed",
        ),
        (
            lambda case: set_cell(case, "works", "work_id", "work-changed"),
            "work identities changed",
        ),
        (
            lambda case: set_cell(case, "works", "title", "Changed"),
            "work payload changed",
        ),
        (
            lambda case: case["candidate_artifact"]["points"][0].__setitem__(
                "work_id", "work-changed"
            ),
            "mapped work IDs changed",
        ),
        (
            lambda case: case["candidate_artifact"]["points"][0].__setitem__(
                "faculty_ids", ["person-two"]
            ),
            "non-topic map point changed",
        ),
        (
            lambda case: case["prior_works"][0].__setitem__(
                "keyword", "Unreviewed replacement"
            ),
            "parent topic metadata differs",
        ),
    ],
)
def test_held_only_topics_reject_changed_published_data(change, message) -> None:
    case = held_only_case()
    change(case)
    with pytest.raises(ValueError, match=message):
        preserve_held_only_topics(**case)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda case: case["prior_profiles"].clear(), "parent observation is empty"),
        (
            lambda case: case["prior_profiles"].append(
                copy.deepcopy(case["prior_profiles"][0])
            ),
            "duplicate identity",
        ),
        (
            lambda case: case["prior_works"][0].pop("title"),
            "parent work lacks fields",
        ),
        (
            lambda case: case["prior_works"][0].pop("keyword"),
            "parent works lack frozen topic fields",
        ),
        (
            lambda case: case["prior_artifact"].__setitem__("point_count", 2),
            "parent map point count is invalid",
        ),
        (
            lambda case: case["prior_artifact"].__setitem__("keyword_count", 3),
            "parent map keyword catalog is invalid",
        ),
        (
            lambda case: [
                artifact["points"][0].__setitem__("work_id", "work-other")
                for artifact in (
                    case["prior_artifact"],
                    case["candidate_artifact"],
                )
            ],
            "parent map point lacks work",
        ),
        (
            lambda case: case["prior_artifact"]["points"][0].__setitem__(
                "keyword_ids", ["wrong-broad", "wrong-detail"]
            ),
            "parent work/map topics differ",
        ),
    ],
)
def test_held_only_topics_reject_incomplete_parent(change, message) -> None:
    case = held_only_case()
    change(case)
    with pytest.raises(ValueError, match=message):
        preserve_held_only_topics(**case)


def test_held_only_normalizes_parent_and_current_list_null_values() -> None:
    case = held_only_case()
    case["prior_profiles"][0]["year"] = None
    case["current_profiles"].loc[0, "year"] = math.nan
    case["prior_profiles"][0]["source_verification_urls"] = ["https://example.test"]
    case["current_profiles"].at[0, "source_verification_urls"] = [
        "https://example.test"
    ]
    assert _value({"urls": numpy.array(["https://example.test"])}) == {
        "urls": ["https://example.test"]
    }
    assert preserve_held_only_topics(**case).iloc[0]["keyword"] == (
        "Reviewed broad topic"
    )


def test_held_only_stage_requires_exact_parent_map_except_timestamp() -> None:
    parent = held_only_case()["prior_artifact"]
    candidate = copy.deepcopy(parent)
    candidate["generated_at_utc"] = "2026-10-25T00:00:00+00:00"
    assert validate_held_only_stage(candidate, parent) == {
        "status": "pass",
        "point_count": 1,
        "keyword_count": 2,
    }
    for field, message in (
        ("points", "points differ"),
        ("keywords", "keywords differ"),
        ("catalogs", "metadata differs"),
    ):
        changed = copy.deepcopy(candidate)
        changed[field].append({"unexpected": True}) if isinstance(
            changed[field], list
        ) else changed[field].update(unexpected=True)
        with pytest.raises(ValueError, match=message):
            validate_held_only_stage(changed, parent)
