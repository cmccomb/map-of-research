"""The exact SCS topic hold must not erase genuine source or authorship rows."""

from __future__ import annotations

import pandas
import pytest

from map_of_research.scs_four_topic_hold import (
    HOLDS,
    PERSON_ID,
    PRIOR_REVISION,
    REASON,
    SNAPSHOT_SHA256,
    apply_scs_four_topic_holds,
)


def _tables() -> tuple[pandas.DataFrame, pandas.DataFrame]:
    works = pandas.DataFrame(
        [
            {
                "work_id": HOLDS[0][0],
                "map_eligible": True,
                "map_region_outlier": False,
                "map_region_documented_override": False,
                "map_exclusion_reasons": [],
            },
            {
                "work_id": HOLDS[1][0],
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
                "title": title,
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
