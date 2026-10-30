"""Exact, source-backed topic holds for the four-profile SCS release.

The two code-language-model papers below are genuine Daniel Fried works. A
global t-SNE topic refit can place them in an unrelated low-information or
mathematics region, depending on numerical details of the build machine.
Keep their raw observations and authorships, but withhold their map points
until a reviewed topic assignment is available.
"""

from __future__ import annotations

from typing import Any

PRIOR_REVISION = "6964644510ac2debbc1b94e2fa4a1e07a8a93c25"
SNAPSHOT_SHA256 = "9cfe0e2d6157d0d70371172e0164683baabacaa1bc3daef3931db03a4bb9c017"
PERSON_ID = "person-danielfried-9300b30d"
REASON = "reviewed_topic_hold"

HOLDS = (
    (
        "work-249dd3e34976c53c7e1b3bd0f7ac5a68466979d2d67f9c8f3a283eb5c61e9de1",
        "sJDqACEAAAAJ:r0BpntZqJG4C",
        "Starcoder: may the source be with you!",
    ),
    (
        "work-e9b6da02fcf0310dc757a98c23ca7e5ac66b5046a889fb80ba82d5cfbe816415",
        "sJDqACEAAAAJ:_Qo2XoVZTnwC",
        "Santacoder: don't reach for the stars!",
    ),
)


def apply_scs_four_topic_holds(
    works: Any,
    profiles: Any,
    prior_works: list[dict[str, Any]],
    *,
    prior_revision: str,
    snapshot_sha256: str,
) -> tuple[Any, dict[str, Any] | None]:
    """Withhold only the two exact new map points, preserving source and author.

    This gate is inactive for other source snapshots or Hub parents. A partial
    match fails rather than silently applying a release-specific exception.
    """

    if prior_revision != PRIOR_REVISION and snapshot_sha256 != SNAPSHOT_SHA256:
        return works, None
    if prior_revision != PRIOR_REVISION or snapshot_sha256 != SNAPSHOT_SHA256:
        raise ValueError("SCS topic hold requires the exact parent and source")

    prior_ids = {str(row["work_id"]) for row in prior_works}
    output = works.copy()
    applied: list[dict[str, str]] = []
    for work_id, author_pub_id, title in HOLDS:
        if work_id in prior_ids:
            raise ValueError(f"SCS topic hold targets a prior public work: {work_id}")
        work_matches = output.index[output["work_id"] == work_id].tolist()
        if len(work_matches) != 1:
            raise ValueError(f"SCS topic hold work is not unique: {work_id}")
        profile_matches = profiles.loc[profiles["author_pub_id"] == author_pub_id]
        if len(profile_matches) != 1:
            raise ValueError(
                f"SCS topic hold source observation is not unique: {author_pub_id}"
            )
        profile = profile_matches.iloc[0]
        if (
            str(profile["work_id"]) != work_id
            or str(profile["person_id"]) != PERSON_ID
            or str(profile["title"]) != title
            or not bool(profile["map_eligible"])
        ):
            raise ValueError(f"SCS topic hold source does not match: {author_pub_id}")
        index = work_matches[0]
        reasons = [
            str(reason)
            for reason in output.at[index, "map_exclusion_reasons"]
            if str(reason) != "low_information_region"
        ]
        output.at[index, "map_eligible"] = False
        output.at[index, "map_region_outlier"] = False
        output.at[index, "map_region_documented_override"] = False
        output.at[index, "map_exclusion_reasons"] = list(
            dict.fromkeys([*reasons, REASON])
        )
        applied.append({"work_id": work_id, "author_pub_id": author_pub_id})
    return output, {
        "prior_revision": prior_revision,
        "snapshot_sha256": snapshot_sha256,
        "reason": REASON,
        "held_works": applied,
        "raw_and_authorship_retained": True,
    }
