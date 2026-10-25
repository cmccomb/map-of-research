"""Convert the pinned Neumayer bibliography into a held snapshot candidate."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

from .registry import Registry

NEUMAYER_PERSON_ID = "person-robinneumayer-a5f54a8c"
NEUMAYER_SOURCE_DIR = Path("data/source_reviews/robin-neumayer-2026-10-25")


def load_neumayer_candidate_rows(
    path: Path, registry: Registry
) -> list[dict[str, Any]]:
    """Return all 31 typed, release-held rows without assigning Scholar metadata."""

    staged = runpy.run_path(str(path / "load_staged.py"))
    source_rows = staged["load_staged_neumayer_rows"](path)
    person = registry.people_by_id.get(NEUMAYER_PERSON_ID)
    if person is None or person.scholar_id:
        raise ValueError(
            "Neumayer candidate needs the unresolved Math registry identity"
        )
    memberships = [
        membership
        for membership in registry.memberships
        if membership.person_id == NEUMAYER_PERSON_ID and membership.included
    ]
    if not memberships or {membership.department_id for membership in memberships} != {
        "math"
    }:
        raise ValueError("Neumayer candidate needs the evidenced Math membership")
    membership_rows = [
        {
            "department_id": membership.department_id,
            "role": membership.role,
            "included": membership.included,
            "legacy_label": membership.legacy_label,
            "source_url": membership.source_url,
            "verified_at": membership.verified_at,
        }
        for membership in memberships
    ]
    rows = []
    for source in source_rows:
        kind = source["source_kind"]
        rows.append(
            {
                "scholar_id": "",
                "person_id": NEUMAYER_PERSON_ID,
                "display_name": person.display_name,
                "faculty": person.display_name,
                "department_ids": ["math"],
                "memberships": membership_rows,
                "author_pub_id": source["source_native_id"],
                "title": source["title"],
                "authors": source["authors"],
                "year": source["year"],
                "venue": "",
                "citation": "",
                "citation_count": 0,
                "source_url": source["source_work_url"],
                "source_record_json": json.dumps(
                    source,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "fetched_at_utc": "",
                "source_type": "researcher_cv"
                if kind == "researcher_cv"
                else "author_site",
                "source_doi": source["doi"] or "",
                "source_native_id": source["source_native_id"],
                "source_verified_at": "2026-10-25",
                "source_verification_urls": source["source_verification_urls"],
                "external_coauthors": [],
                "source_citation_count_known": False,
                "source_review_state": "source_only_release_held",
                "source_output_type": source["publication_type"],
                "source_collaboration_eligible": False,
            }
        )
    if (
        len(rows) != 31
        or len({row["author_pub_id"] for row in rows}) != 31
        or sum(row["source_type"] == "author_site" for row in rows) != 30
        or sum(row["source_type"] == "researcher_cv" for row in rows) != 1
        or sum(bool(row["source_doi"]) for row in rows) != 18
        or sum(row["year"] is None for row in rows) != 2
    ):
        raise ValueError("Neumayer candidate whole-person source coverage changed")
    return rows
