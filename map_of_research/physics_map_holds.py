"""Exact map holds while two reviewed Physics works await topic placement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .doi_aliases import ReviewedDoiAlias

PENDING_TOPIC_PLACEMENT_REASON = "topic_placement_pending"


@dataclass(frozen=True)
class PendingTopicPlacement:
    work_id: str
    doi: str
    title: str
    year: int
    observation_id: str
    scholar_id: str
    author_pub_id: str


PHYSICS_PENDING_TOPIC_PLACEMENTS = (
    PendingTopicPlacement(
        work_id="work-b33b19863dc6eaeb1919045ad850ae25ba419ab998ecb8738ec7c79174afffa8",
        doi="10.48550/arxiv.2112.04569",
        title="Toward a resolution of the NN controversy",
        year=2021,
        observation_id="observation-5ae5fa6b40393062c0ab6e440dfbe286606c1166c27d062e5e448b58f2a3db1b",
        scholar_id="8AqWr_4AAAAJ",
        author_pub_id="8AqWr_4AAAAJ:XvxMoLDsR5gC",
    ),
    PendingTopicPlacement(
        work_id="work-e40eef6b5a2616eae3f6a2868d2dbea1b8b842f514d3e2c76a5c9b0fb4cb4309",
        doi="10.48550/arxiv.1804.08406",
        title="LGBT+ inclusivity in physics and astronomy: A best practices guide",
        year=2018,
        observation_id="observation-6960e3f9d4f3c7cac1fdb153a9e6a8acf591b884c9b4accdb5d9ee037d527b12",
        scholar_id="JbSQsXAAAAAJ",
        author_pub_id="JbSQsXAAAAAJ:2osOgNQ5qMEC",
    ),
)


def hold_pending_physics_placements(
    works: Any,
    profiles: Any,
    doi_aliases: Mapping[tuple[str, str], ReviewedDoiAlias],
) -> Any:
    """Keep DOI metadata but omit only the two unplaced works from the map."""

    expected_work_ids = {item.work_id for item in PHYSICS_PENDING_TOPIC_PLACEMENTS}
    expected_author_pub_ids = {
        item.author_pub_id for item in PHYSICS_PENDING_TOPIC_PLACEMENTS
    }
    present_work_ids = set(works["work_id"]) & expected_work_ids
    present_author_pub_ids = set(profiles["author_pub_id"]) & expected_author_pub_ids
    if not present_work_ids and not present_author_pub_ids:
        return works
    if (
        present_work_ids != expected_work_ids
        or present_author_pub_ids != expected_author_pub_ids
    ):
        raise ValueError("Physics topic-placement hold has incomplete work lineage")

    output = works.copy()
    for item in PHYSICS_PENDING_TOPIC_PLACEMENTS:
        alias = doi_aliases.get((item.scholar_id, item.author_pub_id))
        if (
            alias is None
            or alias.relationship != "arxiv_preprint_doi"
            or alias.doi != item.doi
            or alias.observed_title != item.title
            or alias.observed_year != item.year
        ):
            raise ValueError(
                f"Physics topic-placement DOI review changed: {item.work_id}"
            )

        work_rows = output.index[output["work_id"].eq(item.work_id)]
        profile_rows = profiles.loc[profiles["work_id"].eq(item.work_id)]
        if len(work_rows) != 1 or len(profile_rows) != 1:
            raise ValueError(
                f"Physics topic-placement work lineage changed: {item.work_id}"
            )
        index = work_rows[0]
        work = output.loc[index]
        profile = profile_rows.iloc[0]
        if (
            str(work["doi"]).casefold() != item.doi
            or work["title"] != item.title
            or work["year"] != item.year
            or work["work_match_method"] != "reviewed_version_alias"
            or not bool(work["map_eligible"])
            or list(work["map_exclusion_reasons"])
            or profile["observation_id"] != item.observation_id
            or profile["scholar_id"] != item.scholar_id
            or profile["author_pub_id"] != item.author_pub_id
            or profile["source_type"] != "scholar"
            or str(profile["doi"]).casefold() != item.doi
            or profile["work_match_method"] != "reviewed_version_alias"
            or not bool(profile["map_eligible"])
            or list(profile["map_exclusion_reasons"])
        ):
            raise ValueError(f"Physics topic-placement source changed: {item.work_id}")
        output.at[index, "map_eligible"] = False
        output.at[index, "map_exclusion_reasons"] = [PENDING_TOPIC_PLACEMENT_REASON]
    return output
