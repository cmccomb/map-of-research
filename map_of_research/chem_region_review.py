"""Exact Chemistry overrides for the frozen Bernhard region transition.

The complete Scholar caches and frozen source rows stay untouched.  This
review only changes publication-map and collaboration eligibility for 19
source-bound observations after checking primary evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from .region_review import ReviewedRegionDecision
from .registry import Registry

DEFAULT_CHEM_REGION_REVIEW_PATH = Path("data/curated/scholar-chem-region-review.json")
CHEM_REVIEW_SHA256 = "e2b888a3293b9515491770cf13ccfc0dbdd75e31bc4fda671698f5acbb41ec9b"
SOURCE_COMMIT = "8be6a2400cf72dd959c874bfcada8624540820ef"
SOURCE_SHA256 = "f8d7ffec683eb1830dadc77e612cbad7f3fa18ff199696041a71a9b8edbcefa2"
HISTORICAL_REVISION = "15498d5f78e2e269f4b11c630bc9b4ef56efb628"
PARENT_REVISION = "bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd"
DECISION_COUNTS = {
    ("restore", "include"): 3,
    ("restore", "hold"): 3,
    ("restore", "exclude"): 1,
    ("reexclude", "include"): 7,
    ("reexclude", "hold"): 2,
    ("reexclude", "exclude"): 3,
}
EXCLUSION_CODES = {"false_profile_attribution", "corrupted_duplicate"}


def load_chem_region_review(
    path: Path,
    *,
    snapshot_frame: object,
    snapshot_sha256: str,
    cache_dir: Path,
    registry: Registry,
) -> tuple[ReviewedRegionDecision, ...]:
    """Bind the 19 decisions to exact source rows and eight complete caches."""

    raw_manifest = path.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != CHEM_REVIEW_SHA256:
        raise ValueError("Chemistry region decision manifest bytes changed")
    source = json.loads(raw_manifest)
    if (
        snapshot_sha256 != SOURCE_SHA256
        or not isinstance(source, dict)
        or any(
            source.get(key) != value
            for key, value in {
                "schema_version": 1,
                "purpose": "frozen_bernhard_chemistry_region_review",
                "source_commit": SOURCE_COMMIT,
                "snapshot_sha256": SOURCE_SHA256,
                "historical_region_revision": HISTORICAL_REVISION,
                "current_hub_revision": PARENT_REVISION,
            }.items()
        )
    ):
        raise ValueError("Chemistry region review source or parent changed")
    profiles = source.get("profiles")
    entries = source.get("decisions")
    if (
        not isinstance(profiles, dict)
        or len(profiles) != 8
        or not isinstance(entries, list)
        or len(entries) != 19
    ):
        raise ValueError("Chemistry region review lacks exact 19-row scope")
    captured = {}
    for scholar_id, profile in profiles.items():
        cache_path = cache_dir / f"{scholar_id}.json"
        raw_cache = cache_path.read_bytes()
        if hashlib.sha256(raw_cache).hexdigest() != profile["cache_sha256"]:
            raise ValueError(f"Chemistry complete Scholar cache changed: {scholar_id}")
        cache = json.loads(raw_cache)
        rows = cache.get("publications")
        capture = cache.get("browser_capture")
        if (
            not isinstance(rows, list)
            or len(rows) != profile["expected_source_rows"]
            or cache.get("publication_count") != len(rows)
            or cache.get("scholar_id") != scholar_id
            or not isinstance(capture, dict)
            or capture.get("coverage") != "complete_visible_profile"
            or capture.get("more_remain") is not False
        ):
            raise ValueError(f"Chemistry Scholar profile is incomplete: {scholar_id}")
        indexed = {row["author_pub_id"]: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(
                f"Chemistry Scholar cache has duplicate rows: {scholar_id}"
            )
        captured[scholar_id] = indexed

    requested = {item["author_pub_id"] for item in entries}
    source_rows = [
        row
        for row in snapshot_frame.itertuples(index=False)
        if getattr(row, "source_type", "") == "scholar"
        and row.author_pub_id in requested
    ]
    indexed_source = {row.author_pub_id: row for row in source_rows}
    if len(indexed_source) != 19 or len(source_rows) != 19:
        raise ValueError("Chemistry review source observation set changed")
    reviewed = []
    works = set()
    counts = Counter()
    conference_outputs = 0
    for entry in entries:
        work_id = entry["work_id"]
        scholar_id = entry["scholar_id"]
        author_pub_id = entry["author_pub_id"]
        decision = entry["decision"]
        transition = entry["transition"]
        code = entry.get("reason_code", "")
        output_type = entry.get("output_type", "")
        row = indexed_source[author_pub_id]
        cache_row = captured[scholar_id].get(author_pub_id)
        person_id = entry["person_id"]
        person = registry.people_by_id.get(person_id)
        evidence = urlparse(entry["evidence_url"])
        if (
            work_id in works
            or not work_id.startswith("work-")
            or (transition, decision) not in DECISION_COUNTS
            or (decision == "exclude" and code not in EXCLUSION_CODES)
            or (decision != "exclude" and bool(code))
            or output_type not in {"", "conference_presentation"}
            or (
                output_type == "conference_presentation"
                and (
                    decision != "include"
                    or transition != "restore"
                    or person_id != "person-donahue-f4c5ce86"
                )
            )
            or row.person_id != person_id
            or row.scholar_id != scholar_id
            or row.title != entry["captured_title"]
            or row.source_url != entry["saved_scholar_url"]
            or hashlib.sha256(row.source_record_json.encode()).hexdigest()
            != entry["source_record_sha256"]
            or not isinstance(cache_row, dict)
            or cache_row.get("capture_row_index") != entry["capture_row_index"]
            or cache_row.get("pub_url") != row.source_url
            or person is None
            or person.scholar_id != scholar_id
            or not any(
                item.person_id == person_id
                and item.department_id == "chem"
                and item.included
                for item in registry.memberships
            )
            or evidence.scheme != "https"
            or not evidence.netloc
            or not entry["reason"].strip()
            or entry["reviewed_at"] != "2026-10-25"
        ):
            raise ValueError(f"Chemistry exact source decision drift: {work_id}")
        reviewed.append(
            ReviewedRegionDecision(
                work_id=work_id,
                person_id=person_id,
                scholar_id=scholar_id,
                author_pub_id=author_pub_id,
                capture_row_index=entry["capture_row_index"],
                source_record_sha256=entry["source_record_sha256"],
                work_fingerprint_sha256=entry["work_fingerprint_sha256"],
                captured_title=entry["captured_title"],
                transition=transition,
                decision=decision,
                evidence_url=entry["evidence_url"],
                evidence_scope="exact_work_primary_evidence"
                if decision == "include"
                else "reviewed_unmapped_work",
                saved_scholar_url=entry["saved_scholar_url"],
                reason=entry["reason"],
                reviewed_at=entry["reviewed_at"],
                reason_code=code,
                output_type=output_type,
            )
        )
        works.add(work_id)
        counts[(transition, decision)] += 1
        conference_outputs += output_type == "conference_presentation"
    if (
        conference_outputs != 2
        or counts != DECISION_COUNTS
        or set(captured) != {item.scholar_id for item in reviewed}
    ):
        raise ValueError("Chemistry region decision totals changed")
    return tuple(reviewed)
