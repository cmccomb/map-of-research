"""One release approval for the previously visible McManus poster point.

This file deliberately knows one observation. A provisional hold, even one
validated against a complete Scholar capture, does not by itself authorize a
loss from the published map.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bio_release_gate import POSTER_ID, POSTER_SOURCE_SHA256, POSTER_WORK_ID
from .io import load_json, sha256_file
from .profile_releases import ProfileRelease
from .provisional_holds import ProvisionalHold

APPROVAL_ID = "bio-mcmanus-rustbelt-2021-poster-one-time-loss"
POSTER_CACHE_SHA256 = "012435c098eddc0ad41b27ce214ad0f00ea89b8311ac9edbe5b7148900f29096"
POSTER_HELD_IDS_SHA256 = (
    "5009f431f17b2eaecdd281dc6ed51e01f8a4914074e85d7f7ab4e3273ca0e2aa"
)
POSTER_CAPTURE_ROW_INDEX = 60
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
FIELDS = frozenset(
    {
        "schema_version",
        "approval_id",
        "prior_revision",
        "source_snapshot_sha256",
        "profile_releases_sha256",
        "provisional_holds_sha256",
        "source_cache_sha256",
        "expected_loss_count",
        "loss",
    }
)
LOSS_FIELDS = frozenset(
    {
        "person_id",
        "scholar_id",
        "author_pub_id",
        "source_type",
        "prior_work_id",
        "source_record_sha256",
        "capture_row_index",
        "reason",
        "evidence_url",
    }
)
ObservationKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class PosterLossApproval:
    """A reviewed manifest tied to one immutable parent and one exact row."""

    prior_revision: str
    source_snapshot_sha256: str
    manifest_sha256: str
    reason: str
    evidence_url: str

    def require_exact_loss(
        self,
        *,
        prior_revision: str,
        losses: set[ObservationKey],
        previous_rows: Mapping[ObservationKey, Mapping[str, Any]],
        current_rows: Mapping[ObservationKey, Mapping[str, Any]],
        previous_works: Mapping[str, Mapping[str, Any]],
        final_works: Mapping[str, Mapping[str, Any]],
    ) -> ObservationKey:
        """Consume the approval only for the single real, source-bound loss."""

        prior = previous_rows.get(POSTER_ID)
        current = current_rows.get(POSTER_ID)
        old_work = previous_works.get(POSTER_WORK_ID)
        new_work = final_works.get(POSTER_WORK_ID)
        if (
            prior_revision != self.prior_revision
            or POSTER_ID not in losses
            or prior is None
            or current is None
            or old_work is None
            or str(prior.get("work_id") or "") != POSTER_WORK_ID
            or str(current.get("work_id") or "") != POSTER_WORK_ID
            or not bool(prior.get("map_eligible"))
            or not bool(old_work.get("map_eligible"))
            or bool(current.get("map_eligible"))
            or new_work is None
            or bool(new_work.get("map_eligible"))
            or current.get("review_attribution_status") != "held"
            or "attribution_provisional_hold"
            not in (current.get("map_exclusion_reasons") or [])
            or current.get("review_attribution_evidence_url") != self.evidence_url
            or any(
                hashlib.sha256(
                    str(row.get("source_record_json") or "").encode()
                ).hexdigest()
                != POSTER_SOURCE_SHA256
                for row in (prior, current)
            )
        ):
            raise ValueError(
                "McManus poster approval does not match the exact prior and held row"
            )
        return POSTER_ID


def load_poster_loss_approval(
    path: Path,
    *,
    prior_revision: str,
    source_snapshot_sha256: str,
    profile_releases_sha256: str,
    provisional_holds_sha256: str,
    release_decisions: tuple[ProfileRelease, ...],
    validated_holds: Mapping[tuple[str, str], ProvisionalHold],
) -> PosterLossApproval:
    """Validate the reviewed, single-use manifest against source decisions."""

    document = load_json(path)
    if (
        not isinstance(document, dict)
        or set(document) != FIELDS
        or document["schema_version"] != 1
        or document["approval_id"] != APPROVAL_ID
        or type(document["expected_loss_count"]) is not int
        or document["expected_loss_count"] != 1
        or not isinstance(document["loss"], dict)
        or set(document["loss"]) != LOSS_FIELDS
    ):
        raise ValueError("McManus poster approval must name exactly one loss")
    loss = document["loss"]
    if (
        not isinstance(document["prior_revision"], str)
        or REVISION.fullmatch(document["prior_revision"]) is None
        or document["prior_revision"] != prior_revision
        or not isinstance(document["source_snapshot_sha256"], str)
        or SHA256.fullmatch(document["source_snapshot_sha256"]) is None
        or document["source_snapshot_sha256"] != source_snapshot_sha256
        or document["profile_releases_sha256"] != profile_releases_sha256
        or document["provisional_holds_sha256"] != provisional_holds_sha256
        or document["source_cache_sha256"] != POSTER_CACHE_SHA256
        or tuple(
            loss[field]
            for field in ("person_id", "scholar_id", "author_pub_id", "source_type")
        )
        != POSTER_ID
        or loss["prior_work_id"] != POSTER_WORK_ID
        or loss["source_record_sha256"] != POSTER_SOURCE_SHA256
        or loss["capture_row_index"] != POSTER_CAPTURE_ROW_INDEX
    ):
        raise ValueError(
            "McManus poster approval differs from its immutable source or parent"
        )
    matches = [item for item in release_decisions if item.scholar_id == POSTER_ID[1]]
    hold = validated_holds.get((POSTER_ID[1], POSTER_ID[2]))
    if (
        len(matches) != 1
        or matches[0].person_id != POSTER_ID[0]
        or matches[0].state != "released"
        or matches[0].expected_source_rows != 62
        or matches[0].cache_sha256 != POSTER_CACHE_SHA256
        or matches[0].expected_provisional_hold_rows != 1
        or matches[0].provisional_hold_ids_sha256 != POSTER_HELD_IDS_SHA256
        or hold is None
        or hold.capture_row_index != POSTER_CAPTURE_ROW_INDEX
        or not isinstance(loss["reason"], str)
        or not loss["reason"].strip()
        or loss["reason"] != hold.reason
        or not isinstance(loss["evidence_url"], str)
        or not loss["evidence_url"].startswith("https://scholar.google.com/citations?")
        or loss["evidence_url"] != hold.evidence_url
    ):
        raise ValueError("McManus poster approval differs from the validated hold")
    return PosterLossApproval(
        prior_revision=prior_revision,
        source_snapshot_sha256=source_snapshot_sha256,
        manifest_sha256=sha256_file(path),
        reason=hold.reason,
        evidence_url=hold.evidence_url,
    )


def approval_document(
    *,
    prior_revision: str,
    source_snapshot_sha256: str,
    profile_releases_sha256: str,
    provisional_holds_sha256: str,
    hold: ProvisionalHold,
) -> dict[str, Any]:
    """Build an inspectable proposed manifest after the final parent is known."""

    return {
        "schema_version": 1,
        "approval_id": APPROVAL_ID,
        "prior_revision": prior_revision,
        "source_snapshot_sha256": source_snapshot_sha256,
        "profile_releases_sha256": profile_releases_sha256,
        "provisional_holds_sha256": provisional_holds_sha256,
        "source_cache_sha256": POSTER_CACHE_SHA256,
        "expected_loss_count": 1,
        "loss": {
            **dict(
                zip(
                    ("person_id", "scholar_id", "author_pub_id", "source_type"),
                    POSTER_ID,
                    strict=True,
                )
            ),
            "prior_work_id": POSTER_WORK_ID,
            "source_record_sha256": POSTER_SOURCE_SHA256,
            "capture_row_index": POSTER_CAPTURE_ROW_INDEX,
            "reason": hold.reason,
            "evidence_url": hold.evidence_url,
        },
    }
