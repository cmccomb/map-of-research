"""Preserve the 19 reviewed Chemistry decisions in the next full rebuild."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .chem_region_review import CHEM_REVIEW_SHA256, DEFAULT_CHEM_REGION_REVIEW_PATH
from .contexts import DEFAULT_CONTEXTS_PATH
from .exclusions import DEFAULT_EXCLUSIONS_PATH
from .io import sha256_file
from .profile_releases import DEFAULT_PROFILE_RELEASES_PATH
from .provisional_holds import DEFAULT_PROVISIONAL_HOLDS_PATH

PRIOR_REVISION = "dcbdfacd7e8cfeaefd4c91081c4b347b7c690061"
SOURCE_COMMIT = "45112c6860d380e40c3f60efc4faddf3003f7a89"
SOURCE_SHA256 = "d08fb3a422df8dee3b0d0f75a29d68fef9b7121229a9cf6d418917e1d4a19550"
CURATED_SHA256 = {
    "exclusions": "286a44caa1b6b803a15ddd858785094e7454b2cb51ea6b2d5591583b4103a6a3",
    "holds": "53fd5226b5d44f92ed3cdc5a6e64fad3396566fa4d4b19c0a1bfacbcbd09958c",
    "contexts": "ec2f3f67c0448834717e3977c9a282556cccc010191b1b920bdaaa2ccc6db661",
    "releases": "64160ce6ca3ad0c28253e050486258ebf0ed7235ffc461af5019feae3a5bc15d",
}
PROFILE_FIELDS = (
    "observation_id",
    "title",
    "authors",
    "year",
    "venue",
    "citation",
    "citation_count",
    "source_url",
    "doi",
    "source_doi",
    "source_native_id",
    "source_verified_at",
    "source_verification_urls",
    "map_eligible",
    "collaboration_eligible",
    "publication_type",
    "map_exclusion_reasons",
    "review_exclusion_code",
    "review_exclusion_reason",
    "review_evidence_url",
    "review_verified_at",
    "review_attribution_status",
    "review_attribution_evidence_url",
    "context_review_reason",
    "context_evidence_url",
    "context_verified_at",
)
WORK_FIELDS = (
    "title",
    "authors",
    "year",
    "venue",
    "citation",
    "citation_count",
    "doi",
    "source_urls",
    "source_types",
    "source_native_ids",
    "source_verified_at",
    "source_verification_urls",
    "person_ids",
    "faculty",
    "memberships",
    "department_ids",
    "school_ids",
    "external_coauthor_ids",
    "map_eligible",
    "collaboration_eligible",
    "publication_type",
    "map_region_outlier",
    "map_exclusion_reasons",
    "context_evidence_urls",
)
AUTHORSHIP_FIELDS = (
    "observation_id",
    "work_id",
    "person_id",
    "scholar_id",
    "author_pub_id",
    "source_type",
    "contributor_role",
)


def _value(value: Any) -> Any:  # pragma: no cover - exact dcb/W3 release
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    return value


def _index(
    rows: Iterable[Mapping[str, Any]], field: str
) -> dict[str, Mapping[str, Any]]:  # pragma: no cover - exact dcb/W3 release
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row[field])
        if key in result:
            raise ValueError(f"Chemistry carry-forward repeats {field}: {key}")
        result[key] = row
    return result


def validate_chem_carryforward(  # pragma: no cover - exact dcb/W3 release
    *,
    prior_revision: str,
    source_commit: str,
    source_sha256: str,
    source_rows: Iterable[Mapping[str, Any]],
    prior_profiles: Iterable[Mapping[str, Any]],
    prior_works: Iterable[Mapping[str, Any]],
    prior_authorships: Iterable[Mapping[str, Any]],
    current_profiles: Iterable[Mapping[str, Any]],
    current_works: Iterable[Mapping[str, Any]],
    current_authorships: Iterable[Mapping[str, Any]],
    review_path: Path = DEFAULT_CHEM_REGION_REVIEW_PATH,
    exclusions_path: Path = DEFAULT_EXCLUSIONS_PATH,
    holds_path: Path = DEFAULT_PROVISIONAL_HOLDS_PATH,
    contexts_path: Path = DEFAULT_CONTEXTS_PATH,
    releases_path: Path = DEFAULT_PROFILE_RELEASES_PATH,
) -> dict[str, Any]:
    """Require exact source, published semantics, and 10/9 authored partition."""

    if (
        prior_revision != PRIOR_REVISION
        or source_commit != SOURCE_COMMIT
        or source_sha256 != SOURCE_SHA256
        or sha256_file(review_path) != CHEM_REVIEW_SHA256
        or any(
            sha256_file(path) != CURATED_SHA256[name]
            for name, path in (
                ("exclusions", exclusions_path),
                ("holds", holds_path),
                ("contexts", contexts_path),
                ("releases", releases_path),
            )
        )
    ):
        raise ValueError("Chemistry carry-forward parent, source, or review changed")
    decisions = json.loads(review_path.read_text())["decisions"]
    if (
        len(decisions) != 19
        or Counter(item["decision"] for item in decisions)
        != {"include": 10, "hold": 5, "exclude": 4}
        or len({item["author_pub_id"] for item in decisions}) != 19
        or len({item["work_id"] for item in decisions}) != 19
    ):
        raise ValueError("Chemistry carry-forward decision set changed")
    requested = {item["author_pub_id"] for item in decisions}
    work_ids = {item["work_id"] for item in decisions}
    source = _index(
        (row for row in source_rows if row["author_pub_id"] in requested),
        "author_pub_id",
    )
    old_profiles = _index(
        (row for row in prior_profiles if row["author_pub_id"] in requested),
        "author_pub_id",
    )
    new_profiles = _index(
        (row for row in current_profiles if row["author_pub_id"] in requested),
        "author_pub_id",
    )
    old_works = _index(
        (row for row in prior_works if row["work_id"] in work_ids), "work_id"
    )
    new_works = _index(
        (row for row in current_works if row["work_id"] in work_ids), "work_id"
    )
    if any(
        len(rows) != 19
        for rows in (source, old_profiles, new_profiles, old_works, new_works)
    ):
        raise ValueError("Chemistry carry-forward lacks an exact source or work row")
    old_authored = Counter(
        tuple(row[field] for field in AUTHORSHIP_FIELDS)
        for row in prior_authorships
        if row["work_id"] in work_ids
    )
    new_authored = Counter(
        tuple(row[field] for field in AUTHORSHIP_FIELDS)
        for row in current_authorships
        if row["work_id"] in work_ids
    )
    if (
        sum(old_authored.values()) != 10
        or sum(new_authored.values()) != 10
        or old_authored != new_authored
    ):
        raise ValueError("Chemistry carry-forward authorship count changed")
    for item in decisions:
        observation_id = item["author_pub_id"]
        work_id = item["work_id"]
        original = source[observation_id]
        before = old_profiles[observation_id]
        after = new_profiles[observation_id]
        prior_work = old_works[work_id]
        current_work = new_works[work_id]
        visible = item["decision"] == "include"
        expected_reason = (
            []
            if visible
            else ["attribution_provisional_hold"]
            if item["decision"] == "hold"
            else [f"reviewed_{item['reason_code']}"]
        )
        expected_type = item.get("output_type") or "publication"
        if (
            original["person_id"] != item["person_id"]
            or original["scholar_id"] != item["scholar_id"]
            or original["title"] != item["captured_title"]
            or original["source_url"] != item["saved_scholar_url"]
            or hashlib.sha256(original["source_record_json"].encode()).hexdigest()
            != item["source_record_sha256"]
            or any(
                row["person_id"] != item["person_id"]
                or row["scholar_id"] != item["scholar_id"]
                or row["work_id"] != work_id
                or hashlib.sha256(row["source_record_json"].encode()).hexdigest()
                != item["source_record_sha256"]
                for row in (before, after)
            )
            or any(
                bool(row["map_eligible"]) != visible
                or bool(row["collaboration_eligible"]) != visible
                or row["publication_type"] != expected_type
                or _value(row["map_exclusion_reasons"]) != expected_reason
                for row in (before, after, prior_work, current_work)
            )
            or sum(
                count
                for authored, count in old_authored.items()
                if authored[0] == before["observation_id"]
            )
            != int(visible)
            or sum(
                count
                for authored, count in new_authored.items()
                if authored[0] == after["observation_id"]
            )
            != int(visible)
        ):
            raise ValueError(
                f"Chemistry carry-forward source or visibility drift: {observation_id}"
            )
        for field in PROFILE_FIELDS:
            if _value(before[field]) != _value(after[field]):
                raise ValueError(
                    f"Chemistry carry-forward profile drift: {observation_id}/{field}"
                )
        for field in WORK_FIELDS:
            if _value(prior_work[field]) != _value(current_work[field]):
                raise ValueError(
                    f"Chemistry carry-forward work drift: {work_id}/{field}"
                )
        if item["decision"] == "exclude" and (
            before["review_exclusion_code"] != item["reason_code"]
            or before["review_evidence_url"] != item["evidence_url"]
            or before["review_verified_at"] != item["reviewed_at"]
        ):
            raise ValueError(f"Chemistry exclusion evidence drift: {observation_id}")
        if item["decision"] == "hold" and (
            before["review_attribution_status"] != "held"
            or before["review_attribution_evidence_url"] != item["evidence_url"]
        ):
            raise ValueError(f"Chemistry hold evidence drift: {observation_id}")
        if item.get("output_type") and (
            before["context_evidence_url"] != item["evidence_url"]
            or before["context_verified_at"] != item["reviewed_at"]
        ):
            raise ValueError(f"Chemistry presentation evidence drift: {observation_id}")
    return {
        "review_sha256": CHEM_REVIEW_SHA256,
        "included": 10,
        "held": 5,
        "excluded": 4,
        "authorships": 10,
    }
