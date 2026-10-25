import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy
import pandas
import pytest

from map_of_research.composite_region import TOPIC_FIELDS, UNCLASSIFIED_TOPICS
from map_of_research.held_topic_freeze import (
    DOCUMENTED_OVERRIDE_WORK_ID,
    _value,
    held_only_continuity_candidate,
    preserve_held_only_topics,
    retain_held_only_parent_works,
    validate_held_only_candidate_works,
    validate_held_only_stage,
    validate_held_only_table_parity,
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


def _parquet_case(tmp_path: Path):
    parent = {}
    staged = {}
    rows = {
        "people": [{"person_id": "person-one", "mapped_work_count": 1}],
        "works": [
            {
                "work_id": "work-one",
                "title_variants": ["reviewed", "Reviewed"],
                "x": 0.1,
                "y": 0.2,
                "tsne_x": 0.3,
                "tsne_y": 0.4,
                "map_eligible": False,
                "observation_ids": ["observation-one"],
                "author_pub_ids": ["ScholarOne:one"],
                "work_match_method": "exact",
            }
        ],
        "authorships": [{"observation_id": "observation-one", "work_id": "work-one"}],
        "profile_publications": [
            {
                "observation_id": "observation-one",
                "author_pub_id": "ScholarOne:one",
                "source_record_json": '{"title":"reviewed"}',
            }
        ],
    }
    for name, values in rows.items():
        for label, destination in (("parent", parent), ("staged", staged)):
            path = tmp_path / label / name / "train-00000-of-00001.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pandas.DataFrame(values).to_parquet(path, index=False)
            destination[name] = path
    return parent, staged


def test_held_only_table_gate_rejects_changed_raw_row_and_schema(
    tmp_path: Path,
) -> None:
    parent, staged = _parquet_case(tmp_path)
    assert validate_held_only_table_parity(parent, staged) == {
        name: 1 for name in parent
    }
    changed = pandas.read_parquet(staged["profile_publications"])
    changed.loc[0, "source_record_json"] = '{"title":"changed"}'
    changed.to_parquet(staged["profile_publications"], index=False)
    with pytest.raises(ValueError, match="profile_publications row changed"):
        validate_held_only_table_parity(parent, staged)
    changed["extra"] = "transient"
    changed.to_parquet(staged["profile_publications"], index=False)
    with pytest.raises(ValueError, match="schema or row count changed"):
        validate_held_only_table_parity(parent, staged)


def test_held_only_reuses_exact_parent_works_only_after_nonwork_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, staged = _parquet_case(tmp_path)
    changed = pandas.read_parquet(staged["works"])
    changed.at[0, "title_variants"] = ["Reviewed", "reviewed"]
    for field in ("x", "y", "tsne_x", "tsne_y"):
        changed[field] = pandas.Series([None], dtype="Float64")
    changed["map_region_prior_eligible_override"] = True
    changed.to_parquet(staged["works"], index=False)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=lambda _repo, name, **_kwargs: str(
                parent[name.split("/", 1)[0]]
            )
        ),
    )
    works, counts = retain_held_only_parent_works(
        staged, prior_revision="a" * 40, hf_token=""
    )
    assert counts == {name: 1 for name in parent}
    assert works.iloc[0]["x"] == 0.1
    assert staged["works"].read_bytes() == parent["works"].read_bytes()
    assert not any("map_region_prior" in name for name in works.columns)


@pytest.mark.parametrize(
    "changed_table", ["profile_publications", "authorships", "people"]
)
def test_held_only_reuse_rejects_nonwork_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_table: str
) -> None:
    parent, staged = _parquet_case(tmp_path)
    frame = pandas.read_parquet(staged[changed_table])
    first = frame.columns[0]
    frame.loc[0, first] = "changed"
    frame.to_parquet(staged[changed_table], index=False)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=lambda _repo, name, **_kwargs: str(
                parent[name.split("/", 1)[0]]
            )
        ),
    )
    with pytest.raises(ValueError, match=f"staged {changed_table} row changed"):
        retain_held_only_parent_works(staged, prior_revision="a" * 40, hf_token="")


def test_held_only_reuse_rejects_changed_work_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, staged = _parquet_case(tmp_path)
    frame = pandas.read_parquet(staged["works"])
    frame.loc[0, "work_id"] = "work-other"
    frame.to_parquet(staged["works"], index=False)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=lambda _repo, name, **_kwargs: str(
                parent[name.split("/", 1)[0]]
            )
        ),
    )
    with pytest.raises(ValueError, match="work identity/order changed"):
        retain_held_only_parent_works(staged, prior_revision="a" * 40, hf_token="")


@pytest.mark.parametrize(
    "field", ["observation_ids", "author_pub_ids", "work_match_method"]
)
def test_held_only_candidate_rejects_hidden_work_payload(
    tmp_path: Path, field: str
) -> None:
    parent, staged = _parquet_case(tmp_path)
    frame = pandas.read_parquet(staged["works"])
    frame.at[0, field] = "changed" if field == "work_match_method" else ["changed"]
    frame.to_parquet(staged["works"], index=False)
    with pytest.raises(
        ValueError, match=f"candidate work payload changed: work-one: {field}"
    ):
        validate_held_only_candidate_works(parent["works"], staged["works"])


def test_held_only_candidate_rejects_new_work_column(tmp_path: Path) -> None:
    parent, staged = _parquet_case(tmp_path)
    frame = pandas.read_parquet(staged["works"])
    frame["unexpected"] = True
    frame.to_parquet(staged["works"], index=False)
    with pytest.raises(ValueError, match="candidate works schema"):
        validate_held_only_candidate_works(parent["works"], staged["works"])


def test_held_only_candidate_rejects_variant_content_and_partial_coordinates(
    tmp_path: Path,
) -> None:
    parent, staged = _parquet_case(tmp_path)
    frame = pandas.read_parquet(staged["works"])
    frame.at[0, "title_variants"] = ["different", "reviewed"]
    frame.to_parquet(staged["works"], index=False)
    with pytest.raises(ValueError, match="candidate work payload changed"):
        validate_held_only_candidate_works(parent["works"], staged["works"])
    frame = pandas.read_parquet(parent["works"])
    frame["x"] = pandas.Series([None], dtype="Float64")
    frame.to_parquet(staged["works"], index=False)
    with pytest.raises(ValueError, match="cleared partial coordinates"):
        validate_held_only_candidate_works(parent["works"], staged["works"])


def test_held_only_candidate_accepts_only_pinned_documented_override(
    tmp_path: Path,
) -> None:
    parent, staged = _parquet_case(tmp_path)
    for path, value in ((parent["works"], False), (staged["works"], True)):
        frame = pandas.read_parquet(path)
        frame.loc[0, "work_id"] = DOCUMENTED_OVERRIDE_WORK_ID
        frame["map_region_documented_override"] = value
        frame.to_parquet(path, index=False)
    assert validate_held_only_candidate_works(parent["works"], staged["works"]) == 1
    frame = pandas.read_parquet(staged["works"])
    frame.loc[0, "work_id"] = "work-other"
    frame.to_parquet(staged["works"], index=False)
    with pytest.raises(ValueError, match="work identity/order changed"):
        validate_held_only_candidate_works(parent["works"], staged["works"])


def test_held_only_candidate_rejects_duplicate_parent_work_identity(
    tmp_path: Path,
) -> None:
    parent, staged = _parquet_case(tmp_path)
    for path in (parent["works"], staged["works"]):
        frame = pandas.read_parquet(path)
        pandas.concat([frame, frame], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="repeated work identity"):
        validate_held_only_candidate_works(parent["works"], staged["works"])


def test_held_only_candidate_rejects_unreviewed_coordinate_clearance(
    tmp_path: Path,
) -> None:
    parent, staged = _parquet_case(tmp_path)
    frame = pandas.concat(
        [pandas.read_parquet(parent["works"])] * 38, ignore_index=True
    )
    frame["work_id"] = [f"work-{index}" for index in range(38)]
    frame.to_parquet(parent["works"], index=False)
    for field in ("x", "y", "tsne_x", "tsne_y"):
        frame[field] = pandas.Series([None] * 38, dtype="Float64")
    frame.to_parquet(staged["works"], index=False)
    with pytest.raises(ValueError, match="unexpected hidden coordinates"):
        validate_held_only_candidate_works(parent["works"], staged["works"])


def test_held_only_reuse_rejects_extra_released_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, staged = _parquet_case(tmp_path)
    frame = pandas.read_parquet(staged["profile_publications"])
    extra = frame.iloc[[0]].copy()
    extra.loc[:, "observation_id"] = "observation-extra"
    extra.loc[:, "author_pub_id"] = "ScholarOne:extra"
    pandas.concat([frame, extra], ignore_index=True).to_parquet(
        staged["profile_publications"], index=False
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=lambda _repo, name, **_kwargs: str(
                parent[name.split("/", 1)[0]]
            )
        ),
    )
    with pytest.raises(ValueError, match="profile_publications schema or row count"):
        retain_held_only_parent_works(staged, prior_revision="a" * 40, hf_token="")


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
    changed = copy.deepcopy(candidate)
    changed["keywords"].reverse()
    with pytest.raises(ValueError, match="keywords differ"):
        validate_held_only_stage(changed, parent)
    with_two_points = copy.deepcopy(candidate)
    with_two_points["points"].append(
        {**with_two_points["points"][0], "work_id": "work-other"}
    )
    with_two_points["point_count"] = 2
    reordered = copy.deepcopy(with_two_points)
    reordered["points"].reverse()
    with pytest.raises(ValueError, match="points differ"):
        validate_held_only_stage(reordered, with_two_points)
