"""Exact Math decisions for the frozen bb88 ordinary-continuity release.

The saved Scholar capture remains the publication source. This manifest binds
individual visibility decisions to immutable source rows and separate primary
evidence; it does not infer DOI aliases or repair damaged titles.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .io import load_json, sha256_file
from .registry import Registry

DEFAULT_REGION_REVIEW_PATH = Path("data/curated/scholar-region-review.json")
SOURCE_COMMIT = "8ba438215b50ff877f1a4dd0ce21ca2ed5f9b2d0"
SNAPSHOT_SHA256 = "f8d7ffec683eb1830dadc77e612cbad7f3fa18ff199696041a71a9b8edbcefa2"
HISTORICAL_REVISION = "15498d5f78e2e269f4b11c630bc9b4ef56efb628"
CURRENT_REVISION = "bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd"


@dataclass(frozen=True)
class ReviewedRegionDecision:
    work_id: str
    person_id: str
    scholar_id: str
    author_pub_id: str
    capture_row_index: int
    source_record_sha256: str
    work_fingerprint_sha256: str
    captured_title: str
    transition: str
    decision: str
    evidence_url: str
    evidence_scope: str
    saved_scholar_url: str
    reason: str
    reviewed_at: str
    reason_code: str = ""
    output_type: str = ""

    @property
    def observation_identity(self) -> tuple[str, str, str, str]:
        return self.person_id, self.scholar_id, self.author_pub_id, "scholar"

    @property
    def visible(self) -> bool:
        return self.decision == "include"


def _https(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def load_region_review(
    path: Path,
    *,
    snapshot_frame: object,
    snapshot_sha256: str,
    cache_dir: Path,
    registry: Registry,
    exclusions: set[tuple[str, str]],
    holds: set[tuple[str, str]],
) -> tuple[ReviewedRegionDecision, ...]:
    """Require the exact 31 reviewed rows in the unchanged frozen source."""

    source = load_json(path)
    if (
        not isinstance(source, dict)
        or any(
            source.get(key) != expected
            for key, expected in {
                "schema_version": 1,
                "purpose": "frozen_bernhard_math_region_review",
                "source_commit": SOURCE_COMMIT,
                "snapshot_sha256": SNAPSHOT_SHA256,
                "historical_region_revision": HISTORICAL_REVISION,
                "current_hub_revision": CURRENT_REVISION,
            }.items()
        )
        or snapshot_sha256 != SNAPSHOT_SHA256
    ):
        raise ValueError("Math region review is not bound to the frozen repair")
    profiles = source.get("profiles")
    entries = source.get("decisions")
    if (
        not isinstance(profiles, dict)
        or not isinstance(entries, list)
        or len(entries) != 31
        or any(not isinstance(entry, dict) for entry in entries)
    ):
        raise ValueError("Math region review requires 31 exact decisions")
    expected_ids = {
        item.get("author_pub_id") for item in entries if isinstance(item, dict)
    }
    selected_rows = [
        row
        for row in snapshot_frame.itertuples(index=False)
        if getattr(row, "source_type", "") == "scholar"
        and str(row.author_pub_id) in expected_ids
    ]
    snapshot = {str(row.author_pub_id): row for row in selected_rows}
    if len(selected_rows) != 31 or len(snapshot) != 31:
        raise ValueError("Math region review source rows differ")

    cached: dict[str, dict[str, object]] = {}
    if len(profiles) != 12:
        raise ValueError("Math region review profile set differs")
    for scholar_id, expected in profiles.items():
        if (
            not isinstance(scholar_id, str)
            or not isinstance(expected, dict)
            or type(expected.get("expected_source_rows")) is not int
            or re.fullmatch(r"[0-9a-f]{64}", str(expected.get("cache_sha256"))) is None
        ):
            raise ValueError("Math region review profile source is invalid")
        cache_path = cache_dir / f"{scholar_id}.json"
        if (
            not cache_path.is_file()
            or sha256_file(cache_path) != expected["cache_sha256"]
        ):
            raise ValueError(f"Math region review cache changed: {scholar_id}")
        cache = load_json(cache_path)
        publications = cache.get("publications") if isinstance(cache, dict) else None
        capture = cache.get("browser_capture") if isinstance(cache, dict) else None
        if (
            not isinstance(cache, dict)
            or cache.get("scholar_id") != scholar_id
            or cache.get("publication_count") != expected["expected_source_rows"]
            or not isinstance(publications, list)
            or len(publications) != expected["expected_source_rows"]
            or not isinstance(capture, dict)
            or capture.get("coverage") != "complete_visible_profile"
            or capture.get("more_remain") is not False
        ):
            raise ValueError(f"Math region review capture is incomplete: {scholar_id}")
        cached[scholar_id] = {
            row.get("author_pub_id"): row
            for row in publications
            if isinstance(row, dict)
        }
        if len(cached[scholar_id]) != len(publications):
            raise ValueError(f"Math region review cache IDs repeat: {scholar_id}")

    decisions: list[ReviewedRegionDecision] = []
    seen_works: set[str] = set()
    seen_ids: set[str] = set()
    counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        decision = entry.get("decision")
        transition = entry.get("transition")
        work_id = entry.get("work_id")
        scholar_id = entry.get("scholar_id")
        author_pub_id = entry.get("author_pub_id")
        person_id = entry.get("person_id")
        index = entry.get("capture_row_index")
        reason = entry.get("reason")
        source_hash = entry.get("source_record_sha256")
        work_hash = entry.get("work_fingerprint_sha256")
        url = entry.get("evidence_url")
        saved_url = entry.get("saved_scholar_url")
        scope = entry.get("evidence_scope")
        date = entry.get("reviewed_at")
        if (
            decision not in {"include", "exclude", "hold"}
            or transition not in {"restore", "reexclude"}
            or not isinstance(work_id, str)
            or not work_id.startswith("work-")
            or work_id in seen_works
            or not isinstance(scholar_id, str)
            or scholar_id not in cached
            or not isinstance(author_pub_id, str)
            or not author_pub_id.startswith(f"{scholar_id}:")
            or author_pub_id in seen_ids
            or not isinstance(person_id, str)
            or type(index) is not int
            or index <= 0
            or not isinstance(reason, str)
            or not reason.strip()
            or re.fullmatch(r"[0-9a-f]{64}", str(source_hash)) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(work_hash)) is None
            or not isinstance(url, str)
            or not _https(url)
            or not isinstance(saved_url, str)
            or not _https(saved_url)
            or not isinstance(date, str)
        ):
            raise ValueError("Math region review has invalid exact decision fields")
        try:
            dt.date.fromisoformat(date)
        except ValueError as error:
            raise ValueError("Math region review date is invalid") from error
        if (
            (
                decision == "include"
                and (
                    scope != "exact_work_primary_evidence"
                    or "scholar.google.com" in url
                )
            )
            or (
                decision == "hold"
                and scope not in {"unresolved_work_lead", "saved_scholar_only"}
            )
            or (
                decision == "exclude"
                and scope
                not in {
                    "record_type_example_plus_saved_row",
                    "exact_nonpublication_or_duplicate",
                }
            )
            or (
                decision == "include"
                and (scholar_id, author_pub_id) in exclusions | holds
            )
            or (decision == "exclude" and (scholar_id, author_pub_id) not in exclusions)
            or (decision == "hold" and (scholar_id, author_pub_id) not in holds)
        ):
            raise ValueError(
                "Math region decision conflicts with exact review manifests"
            )
        raw = snapshot[author_pub_id]
        cache_row = cached[scholar_id].get(author_pub_id)
        person = registry.people_by_id.get(person_id)
        if (
            raw.scholar_id != scholar_id
            or raw.person_id != person_id
            or raw.title != entry.get("captured_title")
            or raw.source_url != saved_url
            or hashlib.sha256(raw.source_record_json.encode()).hexdigest()
            != source_hash
            or not isinstance(cache_row, dict)
            or cache_row.get("capture_row_index") != index
            or cache_row.get("pub_url") != saved_url
            or person is None
            or person.scholar_id != scholar_id
        ):
            raise ValueError(
                f"Math region decision differs from raw row: {author_pub_id}"
            )
        home = registry.primary_affiliation(person_id)
        if (
            home.home_type != entry.get("primary_home_type")
            or home.home_id != entry.get("primary_home_id")
            or home.school_id != entry.get("primary_school_id")
            or not any(
                membership.person_id == person_id
                and membership.department_id == "math"
                and membership.included
                for membership in registry.memberships
            )
        ):
            raise ValueError(
                f"Math region primary home or membership changed: {person_id}"
            )
        decisions.append(
            ReviewedRegionDecision(
                work_id,
                person_id,
                scholar_id,
                author_pub_id,
                index,
                source_hash,
                work_hash,
                str(entry["captured_title"]),
                transition,
                decision,
                url,
                scope,
                saved_url,
                reason,
                date,
            )
        )
        counts[(transition, decision)] = counts.get((transition, decision), 0) + 1
        seen_works.add(work_id)
        seen_ids.add(author_pub_id)
    if counts != {
        ("restore", "include"): 4,
        ("restore", "exclude"): 1,
        ("restore", "hold"): 12,
        ("reexclude", "include"): 12,
        ("reexclude", "exclude"): 2,
    } or set(cached) != {row.scholar_id for row in decisions}:
        raise ValueError("Math region decision totals differ from the 31-row review")
    return tuple(decisions)
