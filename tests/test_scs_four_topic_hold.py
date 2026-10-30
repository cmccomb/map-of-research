"""The exact SCS topic hold must not erase genuine source or authorship rows."""

from __future__ import annotations

import pandas
import pytest

from map_of_research.scs_four_topic_hold import (
    HOLDS,
    PERSON_ID,
    PRIOR_REVISION,
    REASON,
    SCHOLAR_ID,
    SNAPSHOT_SHA256,
    apply_scs_four_topic_holds,
)


def _tables() -> tuple[pandas.DataFrame, pandas.DataFrame]:
    works = pandas.DataFrame(
        [
            {
                "work_id": HOLDS[0][0],
                "title": HOLDS[0][2],
                "person_ids": [PERSON_ID],
                "scholar_ids": [SCHOLAR_ID],
                "author_pub_ids": [HOLDS[0][1]],
                "observation_ids": ["observation-star"],
                "observation_count": 1,
                "map_eligible": True,
                "map_region_outlier": False,
                "map_region_documented_override": False,
                "map_exclusion_reasons": [],
            },
            {
                "work_id": HOLDS[1][0],
                "title": HOLDS[1][2],
                "person_ids": [PERSON_ID],
                "scholar_ids": [SCHOLAR_ID],
                "author_pub_ids": [HOLDS[1][1]],
                "observation_ids": ["observation-santa"],
                "observation_count": 1,
                "map_eligible": False,
                "map_region_outlier": True,
                "map_region_documented_override": False,
                "map_exclusion_reasons": ["low_information_region"],
            },
            {
                "work_id": "work-unchanged",
                "map_eligible": True,
                "map_region_outlier": False,
                "map_region_documented_override": False,
                "map_exclusion_reasons": [],
            },
        ]
    )
    profiles = pandas.DataFrame(
        [
            {
                "work_id": work_id,
                "author_pub_id": author_pub_id,
                "person_id": PERSON_ID,
                "scholar_id": SCHOLAR_ID,
                "source_type": "scholar",
                "title": title,
                "source_record_json": f'{{"author_pub_id":"{author_pub_id}"}}',
                "map_eligible": True,
            }
            for work_id, author_pub_id, title in HOLDS
        ]
    )
    return works, profiles


def test_exact_topic_hold_preserves_source_and_authorship() -> None:
    works, profiles = _tables()
    original_profiles = profiles.copy(deep=True)
    output, receipt = apply_scs_four_topic_holds(
        works,
        profiles,
        [{"work_id": "work-unchanged"}],
        prior_revision=PRIOR_REVISION,
        snapshot_sha256=SNAPSHOT_SHA256,
    )
    assert receipt is not None
    assert receipt["raw_and_authorship_retained"] is True
    assert {row["work_id"] for row in receipt["held_works"]} == {
        work_id for work_id, _, _ in HOLDS
    }
    for work_id, _, _ in HOLDS:
        row = output.loc[output["work_id"] == work_id].iloc[0]
        assert not row["map_eligible"]
        assert not row["map_region_outlier"]
        assert row["map_exclusion_reasons"] == [REASON]
    assert output.loc[output["work_id"] == "work-unchanged", "map_eligible"].all()
    pandas.testing.assert_frame_equal(profiles, original_profiles)
    assert works.loc[works["work_id"] == HOLDS[0][0], "map_eligible"].all()


def test_topic_hold_is_exact_release_only() -> None:
    works, profiles = _tables()
    output, receipt = apply_scs_four_topic_holds(
        works,
        profiles,
        [],
        prior_revision="another-parent",
        snapshot_sha256="another-source",
    )
    assert receipt is None
    pandas.testing.assert_frame_equal(output, works)
    with pytest.raises(ValueError, match="exact parent and source"):
        apply_scs_four_topic_holds(
            works,
            profiles,
            [],
            prior_revision=PRIOR_REVISION,
            snapshot_sha256="another-source",
        )


def test_topic_hold_rejects_wrong_owner_or_prior_work() -> None:
    works, profiles = _tables()
    profiles.loc[0, "person_id"] = "person-someone-else"
    with pytest.raises(ValueError, match="source does not match"):
        apply_scs_four_topic_holds(
            works,
            profiles,
            [],
            prior_revision=PRIOR_REVISION,
            snapshot_sha256=SNAPSHOT_SHA256,
        )
    profiles.loc[0, "person_id"] = PERSON_ID
    with pytest.raises(ValueError, match="prior public work"):
        apply_scs_four_topic_holds(
            works,
            profiles,
            [{"work_id": HOLDS[0][0]}],
            prior_revision=PRIOR_REVISION,
            snapshot_sha256=SNAPSHOT_SHA256,
        )


@pytest.mark.parametrize("failure", ["missing", "duplicate"])
def test_topic_hold_requires_one_exact_work(failure: str) -> None:
    works, profiles = _tables()
    if failure == "missing":
        works = works.loc[works["work_id"] != HOLDS[0][0]].copy()
    else:
        works = pandas.concat([works, works.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="work is not unique"):
        apply_scs_four_topic_holds(
            works,
            profiles,
            [],
            prior_revision=PRIOR_REVISION,
            snapshot_sha256=SNAPSHOT_SHA256,
        )


@pytest.mark.parametrize("failure", ["missing", "duplicate"])
def test_topic_hold_requires_one_exact_source_observation(failure: str) -> None:
    works, profiles = _tables()
    if failure == "missing":
        profiles = profiles.loc[profiles["author_pub_id"] != HOLDS[0][1]].copy()
    else:
        profiles = pandas.concat([profiles, profiles.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="source observation is not unique"):
        apply_scs_four_topic_holds(
            works,
            profiles,
            [],
            prior_revision=PRIOR_REVISION,
            snapshot_sha256=SNAPSHOT_SHA256,
        )


def _held_parent(
    works: pandas.DataFrame, profiles: pandas.DataFrame
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prior_works = works.loc[works["work_id"].isin([item[0] for item in HOLDS])].copy()
    prior_works["map_eligible"] = False
    prior_works["map_region_outlier"] = False
    prior_works["map_exclusion_reasons"] = [[REASON], [REASON]]
    return prior_works.to_dict("records"), profiles.to_dict("records")


def test_topic_hold_carries_from_exact_published_state_without_erasing_source() -> None:
    works, profiles = _tables()
    prior_works, prior_profiles = _held_parent(works, profiles)
    original_works = works.copy(deep=True)
    original_profiles = profiles.copy(deep=True)
    output, receipt = apply_scs_four_topic_holds(
        works,
        profiles,
        prior_works,
        prior_profiles,
        prior_revision="af0e147a935f5f2a557ea6b3f837890c46ede0c2",
        snapshot_sha256="3059c70d73c6a820e8e1f89158424f291ff56991be5b73eaa7538e861f37eb52",
    )
    assert receipt is not None
    assert receipt["carried_forward"] is True
    assert receipt["raw_and_authorship_retained"] is True
    assert set(output.loc[output["map_eligible"], "work_id"]) == {"work-unchanged"}
    for work_id, _, _ in HOLDS:
        row = output.loc[output["work_id"] == work_id].iloc[0]
        assert row["map_exclusion_reasons"] == [REASON]
        assert not row["map_region_outlier"]
    pandas.testing.assert_frame_equal(works, original_works)
    pandas.testing.assert_frame_equal(profiles, original_profiles)


def test_topic_hold_carry_requires_prior_source_table() -> None:
    works, profiles = _tables()
    prior_works, _ = _held_parent(works, profiles)
    with pytest.raises(ValueError, match="needs prior source observations"):
        apply_scs_four_topic_holds(
            works,
            profiles,
            prior_works,
            prior_revision="af0e147a935f5f2a557ea6b3f837890c46ede0c2",
            snapshot_sha256="3059c70d73c6a820e8e1f89158424f291ff56991be5b73eaa7538e861f37eb52",
        )


def test_topic_hold_carry_rejects_an_alternate_region_parent() -> None:
    works, profiles = _tables()
    prior_works, prior_profiles = _held_parent(works, profiles)
    with pytest.raises(ValueError, match="current public parent"):
        apply_scs_four_topic_holds(
            works,
            profiles,
            prior_works,
            prior_profiles,
            prior_revision="af0e147a935f5f2a557ea6b3f837890c46ede0c2",
            snapshot_sha256="3059c70d73c6a820e8e1f89158424f291ff56991be5b73eaa7538e861f37eb52",
            comparison_revision="6964644510ac2debbc1b94e2fa4a1e07a8a93c25",
        )


@pytest.mark.parametrize(
    "tamper,match",
    [
        ("prior_missing", "prior works are incomplete"),
        ("prior_unheld", "public work changed"),
        ("prior_source_changed", "prior source changed"),
        ("current_source_changed", "prior source changed"),
        ("current_member_changed", "public work changed"),
        ("current_work_changed", "public work changed"),
        ("current_override", "public work changed"),
    ],
)
def test_topic_hold_rejects_changed_source_or_work(tamper: str, match: str) -> None:
    works, profiles = _tables()
    prior_works, prior_profiles = _held_parent(works, profiles)
    if tamper == "prior_missing":
        prior_works.pop()
    elif tamper == "prior_unheld":
        prior_works[0]["map_exclusion_reasons"] = []
    elif tamper == "prior_source_changed":
        prior_profiles[0]["source_record_json"] = '{"changed":true}'
    elif tamper == "current_source_changed":
        profiles.loc[0, "source_record_json"] = '{"changed":true}'
    elif tamper == "current_member_changed":
        works.at[0, "person_ids"] = ["person-someone-else"]
    elif tamper == "current_work_changed":
        works.at[0, "title"] = "A different work"
    elif tamper == "current_override":
        works.at[0, "map_region_documented_override"] = True
    with pytest.raises(ValueError, match=match):
        apply_scs_four_topic_holds(
            works,
            profiles,
            prior_works,
            prior_profiles,
            prior_revision="af0e147a935f5f2a557ea6b3f837890c46ede0c2",
            snapshot_sha256="3059c70d73c6a820e8e1f89158424f291ff56991be5b73eaa7538e861f37eb52",
        )
