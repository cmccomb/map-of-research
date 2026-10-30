"""Bind a public row subset to a complete, privately retained browser capture.

The private capture and row review never belong in the source repository or Hub.
Their hashes and aggregate counts travel with the selected public cache rows.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_METADATA_FIELDS = frozenset(
    {
        "observed_visible_rows",
        "private_capture_sha256",
        "private_review_sha256",
        "selected_ids_sha256",
        "withheld_rows",
        "withheld_hold_rows",
        "withheld_excluded_rows",
        "withheld_ids_sha256",
    }
)
_PUBLIC_CACHE_FIELDS = frozenset(
    {
        "schema_version",
        "scholar_id",
        "display_name",
        "fetched_at_utc",
        "publication_count",
        "publications",
        "browser_capture",
    }
)
_PUBLIC_CAPTURE_FIELDS = frozenset(
    {
        "method",
        "profile_url",
        "captured_at_utc",
        "coverage",
        "more_remain",
        "displayed_count",
        "limit",
        "lifetime_publication_completeness",
        "profile_verified_email_domain",
        "cache_scope",
        "selected_author_pub_ids",
        "private_rendered_html_sha256",
        *_METADATA_FIELDS,
    }
)
_PUBLIC_ROW_FIELDS = frozenset(
    {
        "title",
        "author",
        "pub_year",
        "venue",
        "num_citations",
        "pub_url",
        "author_pub_id",
        "capture_row_index",
        "citation_count_display",
    }
)


@dataclass(frozen=True)
class ReviewedPublicSubset:
    observed_visible_rows: int
    private_capture_sha256: str
    private_review_sha256: str
    selected_ids_sha256: str
    withheld_rows: int
    withheld_hold_rows: int
    withheld_excluded_rows: int
    withheld_ids_sha256: str
    private_rendered_html_sha256: str = ""


def ids_sha256(ids: set[str]) -> str:
    """Hash exact Scholar IDs in the existing hold-manifest encoding."""

    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def parse_public_subset(value: Any) -> ReviewedPublicSubset:
    """Require complete, count-balanced and non-disclosing subset metadata."""

    if not isinstance(value, dict) or not set(value) >= _METADATA_FIELDS:
        raise ValueError("Reviewed public subset metadata is incomplete")
    if set(value) - (_METADATA_FIELDS | {"private_rendered_html_sha256"}):
        raise ValueError("Reviewed public subset metadata has unknown fields")
    observed = value["observed_visible_rows"]
    withheld = value["withheld_rows"]
    held = value["withheld_hold_rows"]
    excluded = value["withheld_excluded_rows"]
    if (
        type(observed) is not int
        or observed <= 1
        or type(withheld) is not int
        or not 0 < withheld < observed
        or type(held) is not int
        or held < 0
        or type(excluded) is not int
        or excluded < 0
        or held + excluded != withheld
    ):
        raise ValueError("Reviewed public subset counts are invalid")
    for field in (
        "private_capture_sha256",
        "private_review_sha256",
        "selected_ids_sha256",
        "withheld_ids_sha256",
    ):
        if not isinstance(value[field], str) or not _DIGEST.fullmatch(value[field]):
            raise ValueError(f"Reviewed public subset {field} is invalid")
    html_digest = value.get("private_rendered_html_sha256", "")
    if html_digest and (
        not isinstance(html_digest, str) or not _DIGEST.fullmatch(html_digest)
    ):
        raise ValueError("Reviewed public subset private HTML digest is invalid")
    return ReviewedPublicSubset(
        observed_visible_rows=observed,
        private_capture_sha256=value["private_capture_sha256"],
        private_review_sha256=value["private_review_sha256"],
        selected_ids_sha256=value["selected_ids_sha256"],
        withheld_rows=withheld,
        withheld_hold_rows=held,
        withheld_excluded_rows=excluded,
        withheld_ids_sha256=value["withheld_ids_sha256"],
        private_rendered_html_sha256=html_digest,
    )


def validate_public_subset_cache(
    scholar_id: str, cache: dict[str, Any]
) -> ReviewedPublicSubset:
    """Verify that the cache contains exactly its selected public citations.

    The private hashes are provenance anchors, not a substitute for the local
    pre-commit comparison against the private capture and independent review.
    """

    rows = cache.get("publications")
    receipt = cache.get("browser_capture")
    if not isinstance(rows, list) or not isinstance(receipt, dict):
        raise ValueError(f"Reviewed public subset cache is incomplete: {scholar_id}")
    if set(cache) - _PUBLIC_CACHE_FIELDS or set(receipt) - _PUBLIC_CAPTURE_FIELDS:
        raise ValueError(
            f"Private payload field entered public subset cache: {scholar_id}"
        )
    if receipt.get("cache_scope") != "reviewed_public_subset":
        raise ValueError(f"Reviewed public subset scope is missing: {scholar_id}")
    subset = parse_public_subset(
        {field: receipt[field] for field in _METADATA_FIELDS if field in receipt}
        | (
            {"private_rendered_html_sha256": receipt["private_rendered_html_sha256"]}
            if "private_rendered_html_sha256" in receipt
            else {}
        )
    )
    count = len(rows)
    selected = receipt.get("selected_author_pub_ids")
    if (
        receipt.get("coverage") != "complete_visible_profile"
        or receipt.get("more_remain") is not False
        or receipt.get("displayed_count") != subset.observed_visible_rows
        or cache.get("publication_count") != count
        or count != subset.observed_visible_rows - subset.withheld_rows
        or type(receipt.get("limit")) is not int
        or receipt["limit"] < subset.observed_visible_rows
        or parse_qs(urlparse(str(receipt.get("profile_url") or "")).query).get("user")
        != [scholar_id]
        or not isinstance(selected, list)
        or len(selected) != count
        or any(
            not isinstance(item, str) or not item.startswith(f"{scholar_id}:")
            for item in selected
        )
        or selected != sorted(set(selected))
        or ids_sha256(set(selected)) != subset.selected_ids_sha256
    ):
        raise ValueError(
            f"Reviewed public subset receipt is inconsistent: {scholar_id}"
        )
    row_ids: list[str] = []
    indices: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Reviewed public subset row is invalid: {scholar_id}")
        if set(row) != _PUBLIC_ROW_FIELDS:
            raise ValueError(
                f"Private payload field entered public subset row: {scholar_id}"
            )
        row_id = row.get("author_pub_id")
        index = row.get("capture_row_index")
        if (
            not isinstance(row_id, str)
            or not row_id.startswith(f"{scholar_id}:")
            or type(index) is not int
            or not 1 <= index <= subset.observed_visible_rows
            or index in indices
        ):
            raise ValueError(
                f"Reviewed public subset row identity is invalid: {scholar_id}"
            )
        citation = parse_qs(urlparse(str(row.get("pub_url") or "")).query).get(
            "citation_for_view"
        )
        if citation != [row_id]:
            raise ValueError(
                f"Reviewed public subset citation URL differs: {scholar_id}"
            )
        row_ids.append(row_id)
        indices.add(index)
    if sorted(row_ids) != selected:
        raise ValueError(f"Unselected citation leaked into public cache: {scholar_id}")
    return subset


def require_public_subset_boundary(
    tables: dict[str, Any],
    release_receipt: list[dict[str, Any]],
    *,
    artifact: dict[str, Any] | None = None,
) -> None:
    """Reject selected-profile leakage in every public relationship and map.

    The original private row identities are intentionally unavailable here.
    The cache-bound selected allowlist is the only admissible Scholar identity
    set for each reviewed public subset.
    """

    subsets = [
        item
        for item in release_receipt
        if item.get("cache_scope") == "reviewed_public_subset"
    ]
    if not subsets:
        return
    profiles = tables["profile_publications"]
    authorships = tables["authorships"]
    works = tables["works"]
    required_profiles = {
        "person_id",
        "scholar_id",
        "author_pub_id",
        "work_id",
        "map_eligible",
    }
    required_authorships = {"scholar_id", "author_pub_id", "work_id"}
    required_works = {"work_id", "scholar_ids", "author_pub_ids", "map_eligible"}
    if (
        not required_profiles <= set(profiles.columns)
        or not required_authorships <= set(authorships.columns)
        or not required_works <= set(works.columns)
    ):
        raise ValueError("Public subset boundary lacks relationship columns")
    work_by_id = {str(row.work_id): row for row in works.itertuples(index=False)}
    if len(work_by_id) != len(works):
        raise ValueError("Public subset boundary has duplicate work IDs")
    points = artifact.get("points") if artifact is not None else None
    if artifact is not None and not isinstance(points, list):
        raise ValueError("Public subset map lacks points")
    for decision in subsets:
        scholar_id = str(decision["scholar_id"])
        person_id = str(decision["person_id"])
        selected_list = decision["selected_author_pub_ids"]
        selected = set(selected_list)
        if (
            len(selected) != len(selected_list)
            or ids_sha256(selected) != decision["selected_ids_sha256"]
        ):
            raise ValueError(f"Public subset selected receipt changed: {scholar_id}")
        raw = profiles.loc[profiles["scholar_id"] == scholar_id]
        if (
            len(raw) != len(selected)
            or set(raw["author_pub_id"].astype(str)) != selected
            or raw["person_id"].ne(person_id).any()
        ):
            raise ValueError(
                f"Unselected citation leaked into public profiles: {scholar_id}"
            )
        mapped_person_work_ids = {
            str(row.work_id)
            for row in profiles.loc[
                (profiles["person_id"] == person_id) & profiles["map_eligible"]
            ].itertuples(index=False)
        }
        for item in authorships.itertuples(index=False):
            if str(item.scholar_id) == scholar_id and (
                str(item.author_pub_id) not in selected
                or str(item.work_id) not in work_by_id
            ):
                raise ValueError(
                    f"Unselected citation leaked into public authorships: {scholar_id}"
                )
        for item in works.itertuples(index=False):
            scholar_ids = {str(value) for value in item.scholar_ids}
            pub_ids = {str(value) for value in item.author_pub_ids}
            own_ids = {value for value in pub_ids if value.startswith(f"{scholar_id}:")}
            if own_ids - selected or (scholar_id in scholar_ids and not own_ids):
                raise ValueError(
                    f"Unselected citation leaked into public works: {scholar_id}"
                )
        if points is not None:
            for point in points:
                if not isinstance(point, dict):
                    raise ValueError("Public subset map has an invalid point")
                faculty_ids = point.get("faculty_ids")
                if not isinstance(faculty_ids, list):
                    raise ValueError("Public subset map point lacks faculty IDs")
                if person_id in faculty_ids:
                    work_id = str(point.get("work_id") or "")
                    if work_id not in work_by_id or not bool(
                        work_by_id[work_id].map_eligible
                    ):
                        raise ValueError(
                            f"Unselected work leaked into public map: {scholar_id}"
                        )
                    mapped_work = work_by_id[work_id]
                    own_source = scholar_id in mapped_work.scholar_ids or any(
                        str(value).startswith(f"{scholar_id}:")
                        for value in mapped_work.author_pub_ids
                    )
                    if own_source and work_id not in mapped_person_work_ids:
                        raise ValueError(
                            f"Unselected work leaked into public map: {scholar_id}"
                        )
                citation = parse_qs(
                    urlparse(str(point.get("source_url") or "")).query
                ).get("citation_for_view")
                if (
                    citation
                    and citation[0].startswith(f"{scholar_id}:")
                    and citation[0] not in selected
                ):
                    raise ValueError(
                        f"Unselected citation URL leaked into public map: {scholar_id}"
                    )
    # Relationship columns are checked above, and this final scan catches a
    # citation identity hidden in another public list/URL/metadata field.
    selected_by_scholar = {
        str(item["scholar_id"]): set(item["selected_author_pub_ids"])
        for item in subsets
    }
    scholar_pattern = "|".join(re.escape(item) for item in selected_by_scholar)
    citation_pattern = re.compile(
        rf"(?<![A-Za-z0-9_-])(?:{scholar_pattern}):[A-Za-z0-9_-]+"
    )

    def scan(value: Any, surface: str) -> None:
        if isinstance(value, str):
            for match in citation_pattern.finditer(unquote(value)):
                citation_id = match.group(0)
                scholar, _ = citation_id.split(":", 1)
                if citation_id not in selected_by_scholar[scholar]:
                    raise ValueError(
                        f"Unselected citation leaked into public {surface}: {scholar}"
                    )
        elif isinstance(value, dict):
            for key, item in value.items():
                scan(key, surface)
                scan(item, surface)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                scan(item, surface)

    for name, frame in tables.items():
        if not hasattr(frame, "itertuples"):
            raise ValueError(f"Public subset boundary lacks tabular {name}")
        for row in frame.itertuples(index=False, name=None):
            scan(row, name)
    if artifact is not None:
        scan(artifact, "map artifact")
