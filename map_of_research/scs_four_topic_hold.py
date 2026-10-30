"""Exact, source-backed topic holds for the four-profile SCS release.

The two code-language-model papers below are genuine Daniel Fried works. A
global t-SNE topic refit can place them in an unrelated low-information or
mathematics region, depending on numerical details of the build machine.
Keep their raw observations and authorships, but withhold their map points
until a reviewed topic assignment is available.
"""

from __future__ import annotations

from typing import Any

from .region_continuity import (
    FINGERPRINT_FIELDS,
    WORK_FINGERPRINT_FIELDS,
    _fingerprint_fields,
)

PRIOR_REVISION = "6964644510ac2debbc1b94e2fa4a1e07a8a93c25"
SNAPSHOT_SHA256 = "9cfe0e2d6157d0d70371172e0164683baabacaa1bc3daef3931db03a4bb9c017"
PERSON_ID = "person-danielfried-9300b30d"
SCHOLAR_ID = "sJDqACEAAAAJ"
REASON = "reviewed_topic_hold"
MEMBERSHIP_FIELDS = (
    "person_ids",
    "scholar_ids",
    "author_pub_ids",
    "observation_ids",
    "observation_count",
)
PRIOR_WORK_FIELDS = ("map_exclusion_reasons", *MEMBERSHIP_FIELDS)

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
    prior_profiles: list[dict[str, Any]] | None = None,
    *,
    prior_revision: str,
    snapshot_sha256: str,
    comparison_revision: str | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    """Withhold the two exact map points, preserving source and author.

    The first release requires exact parent and source pins. Later releases
    inherit the hold only from two identically held public works with unchanged
    source fingerprints and work memberships. A partial match fails closed.
    """

    initial_parent = prior_revision == PRIOR_REVISION
    initial_source = snapshot_sha256 == SNAPSHOT_SHA256
    if initial_parent != initial_source:
        raise ValueError("SCS topic hold requires the exact parent and source")

    initial_release = initial_parent and initial_source
    prior_by_id = {
        work_id: [row for row in prior_works if str(row["work_id"]) == work_id]
        for work_id, _, _ in HOLDS
    }
    if comparison_revision is not None and comparison_revision != prior_revision:
        if initial_release or any(prior_by_id.values()):
            raise ValueError("SCS topic hold requires the current public parent")
        return works, None
    if not initial_release:
        counts = [len(rows) for rows in prior_by_id.values()]
        if counts == [0, 0]:
            return works, None
        if counts != [1, 1]:
            raise ValueError("SCS topic hold prior works are incomplete or repeated")
        if prior_profiles is None:
            raise ValueError("SCS topic hold needs prior source observations")

    output = works.copy()
    applied: list[dict[str, str]] = []
    for work_id, author_pub_id, title in HOLDS:
        if initial_release and prior_by_id[work_id]:
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
            or str(profile["scholar_id"]) != SCHOLAR_ID
            or str(profile["source_type"]) != "scholar"
            or str(profile["title"]) != title
            or not bool(profile["map_eligible"])
        ):
            raise ValueError(f"SCS topic hold source does not match: {author_pub_id}")
        index = work_matches[0]
        current_work = output.loc[index].to_dict()
        if not initial_release:
            prior_work = prior_by_id[work_id][0]
            prior_source = [
                row
                for row in prior_profiles
                if str(row["author_pub_id"]) == author_pub_id
            ]
            if (
                len(prior_source) != 1
                or str(prior_source[0]["work_id"]) != work_id
                or str(prior_source[0]["person_id"]) != PERSON_ID
                or str(prior_source[0]["scholar_id"]) != SCHOLAR_ID
                or str(prior_source[0]["source_type"]) != "scholar"
                or str(prior_source[0]["title"]) != title
                or not bool(prior_source[0]["map_eligible"])
                or _fingerprint_fields(prior_source[0], FINGERPRINT_FIELDS)
                != _fingerprint_fields(profile.to_dict(), FINGERPRINT_FIELDS)
            ):
                raise ValueError(
                    f"SCS topic hold prior source changed: {author_pub_id}"
                )
            if (
                bool(prior_work["map_eligible"])
                or prior_work["map_exclusion_reasons"] != [REASON]
                or str(prior_work["title"]) != title
                or _fingerprint_fields(prior_work, WORK_FINGERPRINT_FIELDS)
                != _fingerprint_fields(current_work, WORK_FINGERPRINT_FIELDS)
                or any(
                    prior_work[field] != current_work[field]
                    for field in MEMBERSHIP_FIELDS
                )
                or bool(current_work["map_region_documented_override"])
                or any(
                    reason not in ("low_information_region", REASON)
                    for reason in current_work["map_exclusion_reasons"]
                )
            ):
                raise ValueError(f"SCS topic hold public work changed: {work_id}")
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
        "carried_forward": not initial_release,
        "reason": REASON,
        "held_works": applied,
        "raw_and_authorship_retained": True,
    }
