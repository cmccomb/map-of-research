"""The two DOI repairs retain their metadata while topic placement is held."""

from dataclasses import replace

import pandas
import pytest

from map_of_research.doi_aliases import DEFAULT_DOI_ALIASES_PATH, load_doi_aliases
from map_of_research.physics_map_holds import (
    PENDING_TOPIC_PLACEMENT_REASON,
    PHYSICS_PENDING_TOPIC_PLACEMENTS,
    hold_pending_physics_placements,
)


def held_work_state():
    works = pandas.DataFrame(
        [
            {
                "work_id": item.work_id,
                "doi": item.doi,
                "title": item.title,
                "year": item.year,
                "work_match_method": "reviewed_version_alias",
                "map_eligible": True,
                "map_exclusion_reasons": [],
            }
            for item in PHYSICS_PENDING_TOPIC_PLACEMENTS
        ]
    )
    profiles = pandas.DataFrame(
        [
            {
                "observation_id": item.observation_id,
                "scholar_id": item.scholar_id,
                "author_pub_id": item.author_pub_id,
                "work_id": item.work_id,
                "source_type": "scholar",
                "doi": item.doi,
                "work_match_method": "reviewed_version_alias",
                "map_eligible": True,
                "map_exclusion_reasons": [],
            }
            for item in PHYSICS_PENDING_TOPIC_PLACEMENTS
        ]
    )
    return works, profiles, load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)


def test_exact_physics_holds_leave_source_rows_intact() -> None:
    works, profiles, aliases = held_work_state()
    original_profiles = profiles.copy(deep=True)
    output = hold_pending_physics_placements(works, profiles, aliases)
    assert output["work_id"].tolist() == works["work_id"].tolist()
    assert not output["map_eligible"].any()
    assert output["map_exclusion_reasons"].tolist() == [
        [PENDING_TOPIC_PLACEMENT_REASON],
        [PENDING_TOPIC_PLACEMENT_REASON],
    ]
    assert works["map_eligible"].all()
    pandas.testing.assert_frame_equal(profiles, original_profiles)


def test_physics_hold_is_inactive_without_either_target() -> None:
    works, profiles, aliases = held_work_state()
    unrelated = works.iloc[0:0].copy()
    unrelated_profiles = profiles.iloc[0:0].copy()
    assert (
        hold_pending_physics_placements(unrelated, unrelated_profiles, aliases)
        is unrelated
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing_work",
        "missing_profile",
        "duplicate_profile",
        "missing_alias",
        "wrong_alias_relationship",
        "wrong_alias_doi",
        "wrong_alias_title",
        "wrong_alias_year",
        "wrong_work_doi",
        "wrong_work_title",
        "wrong_work_year",
        "wrong_work_method",
        "already_held_work",
        "work_reason",
        "wrong_observation_id",
        "wrong_scholar_id",
        "wrong_author_pub_id",
        "wrong_source_type",
        "wrong_profile_doi",
        "wrong_profile_method",
        "ineligible_profile",
        "profile_reason",
    ],
)
def test_physics_hold_rejects_changed_lineage(change) -> None:
    works, profiles, aliases = held_work_state()
    item = PHYSICS_PENDING_TOPIC_PLACEMENTS[0]
    key = (item.scholar_id, item.author_pub_id)
    if change == "missing_work":
        works = works.iloc[[1]].copy()
    elif change == "missing_profile":
        profiles = profiles.iloc[[1]].copy()
    elif change == "duplicate_profile":
        profiles = pandas.concat([profiles, profiles.iloc[[0]]], ignore_index=True)
    elif change == "missing_alias":
        aliases.pop(key)
    elif change == "wrong_alias_relationship":
        aliases[key] = replace(aliases[key], relationship="version_of_record")
    elif change == "wrong_alias_doi":
        aliases[key] = replace(aliases[key], doi="10.1000/other")
    elif change == "wrong_alias_title":
        aliases[key] = replace(aliases[key], observed_title="Other")
    elif change == "wrong_alias_year":
        aliases[key] = replace(aliases[key], observed_year=2020)
    elif change == "wrong_work_doi":
        works.loc[0, "doi"] = "10.1000/other"
    elif change == "wrong_work_title":
        works.loc[0, "title"] = "Other"
    elif change == "wrong_work_year":
        works.loc[0, "year"] = 2020
    elif change == "wrong_work_method":
        works.loc[0, "work_match_method"] = "doi"
    elif change == "already_held_work":
        works.loc[0, "map_eligible"] = False
    elif change == "work_reason":
        works.at[0, "map_exclusion_reasons"] = ["other"]
    elif change == "wrong_observation_id":
        profiles.loc[0, "observation_id"] = "observation-other"
    elif change == "wrong_scholar_id":
        profiles.loc[0, "scholar_id"] = "other"
    elif change == "wrong_author_pub_id":
        profiles.loc[0, "author_pub_id"] = "other:row"
    elif change == "wrong_source_type":
        profiles.loc[0, "source_type"] = "curated_verified"
    elif change == "wrong_profile_doi":
        profiles.loc[0, "doi"] = "10.1000/other"
    elif change == "wrong_profile_method":
        profiles.loc[0, "work_match_method"] = "doi"
    elif change == "ineligible_profile":
        profiles.loc[0, "map_eligible"] = False
    elif change == "profile_reason":
        profiles.at[0, "map_exclusion_reasons"] = ["other"]
    with pytest.raises(ValueError, match="Physics topic-placement"):
        hold_pending_physics_placements(works, profiles, aliases)
