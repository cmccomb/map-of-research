"""Controlled local reconciliation of private captures to public subset caches.

Run this before releasing a reviewed subset. Hosted CI can check the resulting
receipt against public hashes and counts, but it cannot independently inspect
private capture bytes that are deliberately absent from Git and the Hub.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .io import load_json, sha256_file
from .public_subsets import (
    ids_sha256,
    parse_public_subset,
    validate_public_subset_cache,
)

_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "base_source_status_commit",
        "registry_producer_commit",
        "profiles",
        "totals",
        "selected_ids_sha256",
        "withheld_ids_sha256",
        "private_capture_files_or_html_copied",
    }
)
_PROFILE_FIELDS = frozenset(
    {
        "slug",
        "person_id",
        "scholar_id",
        "expected_source_rows",
        "selected_author_pub_ids",
        "public_subset",
        "cache_sha256",
    }
)
_TOTAL_FIELDS = frozenset(
    {
        "observed_visible_rows",
        "selected_rows",
        "withheld_hold_rows",
        "withheld_excluded_rows",
        "withheld_rows",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "source_selection_manifest_sha256",
        "source_snapshot_sha256",
        "selected_ids_sha256",
        "withheld_ids_sha256",
        "totals",
        "profiles",
        "private_evidence_verified_locally",
        "hosted_ci_private_evidence_limit",
    }
)
_RECEIPT_PROFILE_FIELDS = frozenset(
    {
        "person_id",
        "scholar_id",
        "cache_sha256",
        "private_capture_sha256",
        "private_review_sha256",
        "selected_ids_sha256",
        "withheld_ids_sha256",
        "observed_visible_rows",
        "selected_rows",
        "withheld_rows",
        "withheld_hold_rows",
        "withheld_excluded_rows",
    }
)
_HOSTED_PRIVATE_LIMIT = (
    "Hosted CI can verify public hashes and counts, but cannot reread "
    "private full captures or independent reviews."
)


def _validate_selection_manifest(manifest: Any) -> dict[str, Any]:
    """Reject private row fields in the public selected-ID status manifest."""

    if (
        not isinstance(manifest, dict)
        or set(manifest) != _MANIFEST_FIELDS
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "local_source_candidate_only"
        or manifest.get("private_capture_files_or_html_copied") is not False
        or not isinstance(manifest.get("profiles"), list)
        or not manifest["profiles"]
        or not isinstance(manifest.get("totals"), dict)
        or set(manifest["totals"]) != _TOTAL_FIELDS
        or not all(
            type(manifest["totals"][field]) is int and manifest["totals"][field] >= 0
            for field in _TOTAL_FIELDS
        )
    ):
        raise ValueError(
            "Public subset selection manifest has invalid or private fields"
        )
    for field in (
        "base_source_status_commit",
        "registry_producer_commit",
        "selected_ids_sha256",
        "withheld_ids_sha256",
    ):
        length = 40 if field.endswith("commit") else 64
        if re.fullmatch(rf"[0-9a-f]{{{length}}}", str(manifest.get(field))) is None:
            raise ValueError(f"Public subset selection manifest {field} is invalid")
    selected_ids: set[str] = set()
    seen_scholars: set[str] = set()
    for entry in manifest["profiles"]:
        if not isinstance(entry, dict) or set(entry) != _PROFILE_FIELDS:
            raise ValueError(
                "Public subset profile entry has invalid or private fields"
            )
        scholar_id = entry.get("scholar_id")
        person_id = entry.get("person_id")
        selected = entry.get("selected_author_pub_ids")
        subset = parse_public_subset(entry.get("public_subset"))
        if (
            not isinstance(entry.get("slug"), str)
            or not entry["slug"]
            or not isinstance(scholar_id, str)
            or not scholar_id
            or scholar_id in seen_scholars
            or not isinstance(person_id, str)
            or not person_id.startswith("person-")
            or not isinstance(selected, list)
            or any(
                not isinstance(item, str) or not item.startswith(f"{scholar_id}:")
                for item in selected
            )
            or selected != sorted(set(selected))
            or ids_sha256(set(selected)) != subset.selected_ids_sha256
            or entry.get("expected_source_rows") != len(selected)
            or len(selected) != subset.observed_visible_rows - subset.withheld_rows
            or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("cache_sha256"))) is None
        ):
            raise ValueError(f"Public subset selected IDs differ: {scholar_id}")
        selected_ids.update(selected)
        seen_scholars.add(scholar_id)
    if (
        manifest["selected_ids_sha256"] != ids_sha256(selected_ids)
        or manifest["totals"]["observed_visible_rows"]
        != sum(
            item["public_subset"]["observed_visible_rows"]
            for item in manifest["profiles"]
        )
        or manifest["totals"]["selected_rows"] != len(selected_ids)
        or any(
            manifest["totals"][field]
            != sum(item["public_subset"][field] for item in manifest["profiles"])
            for field in (
                "withheld_rows",
                "withheld_hold_rows",
                "withheld_excluded_rows",
            )
        )
    ):
        raise ValueError("Public subset selection aggregate differs")
    return manifest


def _one(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Private subset evidence needs one {pattern}: {matches}")
    return matches[0]


def _decision(item: dict[str, Any]) -> str:
    """Normalize the existing independent review vocabularies."""

    if "release_disposition" in item:
        value = item["release_disposition"]
        if value == "retain_candidate_after_release_gates":
            return "selected"
        if isinstance(value, str) and value.startswith("hold_"):
            return "hold"
    elif "map_disposition" in item:
        value = item["map_disposition"]
        if isinstance(value, str) and value.startswith("candidate_"):
            return "selected"
        if isinstance(value, str) and value.startswith("hold_"):
            return "hold"
    elif "disposition" in item or "category" in item:
        value = item.get("disposition", item.get("category"))
        if value in {
            "retain_candidate",
            "map_candidate_after_normal_gates",
            "attribution_clear_unique_candidate",
            "release_candidate",
        }:
            return "selected"
        if value in {"hold", "provisional_hold", "private_raw_hold"} or (
            isinstance(value, str) and value.startswith("hold_")
        ):
            return "hold"
        if value in {
            "exclude_from_release",
            "exclude_from_authored_and_mapped",
            "exclude_artifact_from_work_map",
            "hard_exclusion",
        }:
            return "exclude"
    raise ValueError(f"Unknown private row-review disposition: {item}")


def _publication(row: dict[str, Any]) -> dict[str, Any]:
    shown = str(row.get("citations") or "").strip()
    if shown and not shown.isdecimal():
        raise ValueError("Private capture citation count is not numeric")
    return {
        "title": row["title"],
        "author": row["authors"],
        "pub_year": row["year"],
        "venue": row["venue"],
        "num_citations": int(shown or "0"),
        "pub_url": row["citation_url"],
        "author_pub_id": row["author_pub_id"],
        "capture_row_index": row["index"],
        "citation_count_display": shown,
    }


def validate_private_public_subsets(
    *,
    source_root: Path,
    private_outputs: Path,
    selection_manifest_path: Path,
    snapshot_path: Path,
) -> dict[str, Any]:
    """Compare every public row byte-for-byte in content to a private raw row."""

    import pandas

    manifest = _validate_selection_manifest(load_json(selection_manifest_path))
    if private_outputs.resolve().is_relative_to(source_root.resolve()):
        raise ValueError(
            "Private capture/review evidence must remain outside source Git"
        )
    if selection_manifest_path.resolve().is_relative_to(private_outputs.resolve()):
        raise ValueError("Public subset selection manifest must be in the source tree")
    if not selection_manifest_path.resolve().is_relative_to(source_root.resolve()):
        raise ValueError("Public subset selection manifest differs from source tree")
    snapshot = pandas.read_parquet(snapshot_path)
    if not {"scholar_id", "author_pub_id", "source_record_json"} <= set(snapshot):
        raise ValueError("Public snapshot lacks source-row provenance")
    checked_profiles: list[dict[str, Any]] = []
    all_selected: set[str] = set()
    all_withheld: set[str] = set()
    for entry in manifest["profiles"]:
        slug = entry.get("slug")
        scholar_id = entry.get("scholar_id")
        person_id = entry.get("person_id")
        # _validate_selection_manifest already checked identity, uniqueness,
        # and exact entry shape before any private file is consulted.
        raw_path = _one(private_outputs, f"{slug}-chrome-visible-*.raw.json")
        review_path = _one(private_outputs, f"{slug}-*review-*.json")
        capture_receipt_path = _one(
            private_outputs, f"{slug}-chrome-capture-receipt-*.json"
        )
        html_path = _one(private_outputs, f"{slug}-chrome-rendered-*.private.html")
        raw = load_json(raw_path)
        review = load_json(review_path)
        capture_receipt = load_json(capture_receipt_path)
        claimed_raw = capture_receipt.get("raw_json") or capture_receipt.get(
            "files", {}
        ).get("raw")
        claimed_html = capture_receipt.get("private_rendered_html") or (
            capture_receipt.get("files", {}).get("private_rendered_html")
        )
        claimed_rows = capture_receipt.get("observed_rows", capture_receipt.get("rows"))
        claimed_unique = capture_receipt.get("unique_citation_ids", claimed_rows)
        cache_path = source_root / "data" / "authors" / f"{scholar_id}.json"
        cache = load_json(cache_path)
        subset = validate_public_subset_cache(scholar_id, cache)
        if (
            not isinstance(raw, dict)
            or not isinstance(review, dict)
            or not isinstance(capture_receipt, dict)
            or raw.get("displayed_count") != subset.observed_visible_rows
            or raw.get("more_remain") is not False
            or raw.get("show_more_disabled") is not True
            or raw.get("challenge") is not False
            or raw.get("page_error") not in (None, False, "")
            or capture_receipt.get("challenge_or_refusal") is True
            or capture_receipt.get("person_id") != person_id
            or capture_receipt.get("scholar_id") != scholar_id
            or review.get("person_id") != person_id
            or review.get("scholar_id") != scholar_id
            or claimed_rows != subset.observed_visible_rows
            or claimed_unique != subset.observed_visible_rows
            or not isinstance(claimed_raw, dict)
            or claimed_raw.get("sha256") != sha256_file(raw_path)
            or claimed_raw.get("bytes") != raw_path.stat().st_size
            or subset.private_capture_sha256 != sha256_file(raw_path)
            or subset.private_review_sha256 != sha256_file(review_path)
            or subset.private_rendered_html_sha256 != sha256_file(html_path)
            or (
                claimed_html is not None
                and claimed_html.get("sha256") != sha256_file(html_path)
            )
            or entry.get("cache_sha256") != sha256_file(cache_path)
            or parse_public_subset(entry.get("public_subset")) != subset
        ):
            raise ValueError(
                f"Private/public subset source binding differs: {scholar_id}"
            )
        if any((source_root / "data" / "captures").glob(f"{slug}-*")):
            raise ValueError(f"Private full capture entered source tree: {scholar_id}")
        raw_rows = raw.get("rows")
        review_rows = review.get("row_reviews") or review.get("rows")
        if (
            not isinstance(raw_rows, list)
            or not isinstance(review_rows, list)
            or len(raw_rows) != subset.observed_visible_rows
            or len(review_rows) != subset.observed_visible_rows
        ):
            raise ValueError(f"Private capture/review rows differ: {scholar_id}")
        selected_rows: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        held_ids: set[str] = set()
        excluded_ids: set[str] = set()
        for index, (raw_row, reviewed) in enumerate(
            zip(raw_rows, review_rows, strict=True), 1
        ):
            if not isinstance(raw_row, dict) or not isinstance(reviewed, dict):
                raise ValueError(f"Private review row is invalid: {scholar_id}:{index}")
            row_id = raw_row.get("author_pub_id")
            reviewed_title = reviewed.get(
                "title",
                reviewed.get("title_as_captured", reviewed.get("captured_title")),
            )
            if (
                raw_row.get("index") != index
                or reviewed.get("index", reviewed.get("row_number")) != index
                or reviewed.get("author_pub_id") != row_id
                or reviewed_title != raw_row.get("title")
                or not isinstance(row_id, str)
                or not row_id.startswith(f"{scholar_id}:")
            ):
                raise ValueError(
                    f"Private capture/review row differs: {scholar_id}:{index}"
                )
            citation = urlparse(str(raw_row.get("citation_url") or ""))
            if (
                citation.scheme != "https"
                or citation.netloc
                not in {
                    "scholar.google.com",
                    "scholar.google.ca",
                    "scholar.google.co.jp",
                }
                or citation.path != "/citations"
                or parse_qs(citation.query).get("user") != [scholar_id]
                or parse_qs(citation.query).get("citation_for_view") != [row_id]
            ):
                raise ValueError(
                    f"Private citation identity differs: {scholar_id}:{index}"
                )
            disposition = _decision(reviewed)
            if disposition == "selected":
                if row_id in selected_ids:
                    raise ValueError(
                        f"Private selected ID repeats: {scholar_id}:{index}"
                    )
                selected_ids.add(row_id)
                selected_rows.append(_publication(raw_row))
            elif disposition == "hold":
                held_ids.add(row_id)
            else:
                excluded_ids.add(row_id)
        withheld_ids = held_ids | excluded_ids
        if (
            len(selected_ids) + len(withheld_ids) != subset.observed_visible_rows
            or selected_ids & withheld_ids
            or len(held_ids) != subset.withheld_hold_rows
            or len(excluded_ids) != subset.withheld_excluded_rows
            or ids_sha256(withheld_ids) != subset.withheld_ids_sha256
            or ids_sha256(selected_ids) != subset.selected_ids_sha256
            or selected_rows != cache["publications"]
            or entry.get("selected_author_pub_ids") != sorted(selected_ids)
        ):
            raise ValueError(
                f"Private review partition differs from public cache: {scholar_id}"
            )
        public_rows = snapshot.loc[snapshot["scholar_id"] == scholar_id]
        if (
            len(public_rows) != len(selected_ids)
            or set(public_rows["author_pub_id"].astype(str)) != selected_ids
            or any(
                json.loads(str(row.source_record_json)) != cache_row
                for row, cache_row in zip(
                    public_rows.sort_values("author_pub_id").itertuples(index=False),
                    sorted(
                        cache["publications"], key=lambda item: item["author_pub_id"]
                    ),
                    strict=True,
                )
            )
        ):
            raise ValueError(
                f"Public snapshot differs from selected cache: {scholar_id}"
            )
        all_selected.update(selected_ids)
        all_withheld.update(withheld_ids)
        checked_profiles.append(
            {
                "person_id": person_id,
                "scholar_id": scholar_id,
                "cache_sha256": sha256_file(cache_path),
                "private_capture_sha256": subset.private_capture_sha256,
                "private_review_sha256": subset.private_review_sha256,
                "selected_ids_sha256": subset.selected_ids_sha256,
                "withheld_ids_sha256": subset.withheld_ids_sha256,
                "observed_visible_rows": subset.observed_visible_rows,
                "selected_rows": len(selected_ids),
                "withheld_rows": len(withheld_ids),
                "withheld_hold_rows": len(held_ids),
                "withheld_excluded_rows": len(excluded_ids),
            }
        )
    if all_selected & all_withheld or (
        manifest.get("selected_ids_sha256") != ids_sha256(all_selected)
        or manifest.get("withheld_ids_sha256") != ids_sha256(all_withheld)
    ):
        raise ValueError("Aggregate private/public selection IDs differ")
    totals = {
        "observed_visible_rows": sum(
            item["observed_visible_rows"] for item in checked_profiles
        ),
        "selected_rows": len(all_selected),
        "withheld_hold_rows": sum(
            item["withheld_hold_rows"] for item in checked_profiles
        ),
        "withheld_excluded_rows": sum(
            item["withheld_excluded_rows"] for item in checked_profiles
        ),
        "withheld_rows": len(all_withheld),
    }
    return {
        "schema_version": 1,
        "status": "private_full_capture_review_reconciled_to_public_subset",
        "source_selection_manifest_sha256": sha256_file(selection_manifest_path),
        "source_snapshot_sha256": sha256_file(snapshot_path),
        "selected_ids_sha256": ids_sha256(all_selected),
        "withheld_ids_sha256": ids_sha256(all_withheld),
        "totals": totals,
        "profiles": checked_profiles,
        "private_evidence_verified_locally": True,
        "hosted_ci_private_evidence_limit": _HOSTED_PRIVATE_LIMIT,
    }


def require_private_preflight_receipt(
    *,
    receipt_path: Path | None,
    selection_manifest_path: Path | None,
    snapshot_path: Path,
    cache_dir: Path,
    release_decisions: tuple[Any, ...],
) -> dict[str, Any] | None:
    """Fail closed on a subset release without its local private audit receipt.

    The receipt is generated by :func:`validate_private_public_subsets` while
    the private evidence is available locally. This check validates its public
    anchors, not the absent private bytes themselves.
    """

    subsets = {
        decision.scholar_id: decision
        for decision in release_decisions
        if decision.public_subset is not None
    }
    if not subsets:
        return None
    if receipt_path is None or selection_manifest_path is None:
        raise ValueError(
            "Reviewed public subset requires a private local preflight receipt"
        )
    receipt = load_json(receipt_path)
    selection = _validate_selection_manifest(load_json(selection_manifest_path))
    if (
        not isinstance(receipt, dict)
        or set(receipt) != _RECEIPT_FIELDS
        or receipt.get("schema_version") != 1
        or receipt.get("status")
        != "private_full_capture_review_reconciled_to_public_subset"
        or receipt.get("private_evidence_verified_locally") is not True
        or receipt.get("source_snapshot_sha256") != sha256_file(snapshot_path)
        or receipt.get("source_selection_manifest_sha256")
        != sha256_file(selection_manifest_path)
        or not isinstance(receipt.get("profiles"), list)
        or not isinstance(receipt.get("totals"), dict)
        or set(receipt["totals"]) != _TOTAL_FIELDS
        or receipt.get("hosted_ci_private_evidence_limit") != _HOSTED_PRIVATE_LIMIT
        or receipt.get("selected_ids_sha256") != selection["selected_ids_sha256"]
        or receipt.get("withheld_ids_sha256") != selection["withheld_ids_sha256"]
        or receipt.get("totals") != selection["totals"]
    ):
        raise ValueError("Private subset preflight receipt differs from public source")
    by_scholar = {
        item.get("scholar_id"): item
        for item in receipt["profiles"]
        if isinstance(item, dict)
    }
    if any(
        not isinstance(item, dict) or set(item) != _RECEIPT_PROFILE_FIELDS
        for item in receipt["profiles"]
    ):
        raise ValueError("Private subset receipt contains unexpected public fields")
    selected = {
        item.get("scholar_id"): item
        for item in selection["profiles"]
        if isinstance(item, dict)
    }
    if (
        len(by_scholar) != len(receipt["profiles"])
        or len(selected) != len(selection["profiles"])
        or set(by_scholar) != set(subsets)
        or set(selected) != set(subsets)
    ):
        raise ValueError(
            "Private subset preflight profiles differ from release manifest"
        )
    for scholar_id, decision in subsets.items():
        audit = by_scholar[scholar_id]
        source = selected[scholar_id]
        subset = decision.public_subset
        cache_path = cache_dir / f"{scholar_id}.json"
        cache = load_json(cache_path)
        if (
            audit.get("person_id") != decision.person_id
            or source.get("person_id") != decision.person_id
            or audit.get("cache_sha256") != decision.cache_sha256
            or source.get("cache_sha256") != decision.cache_sha256
            or sha256_file(cache_path) != decision.cache_sha256
            or validate_public_subset_cache(scholar_id, cache) != subset
            or parse_public_subset(source.get("public_subset")) != subset
            or source.get("selected_author_pub_ids")
            != cache["browser_capture"]["selected_author_pub_ids"]
            or source.get("expected_source_rows") != decision.expected_source_rows
            or audit.get("private_capture_sha256") != subset.private_capture_sha256
            or audit.get("private_review_sha256") != subset.private_review_sha256
            or audit.get("selected_ids_sha256") != subset.selected_ids_sha256
            or audit.get("withheld_ids_sha256") != subset.withheld_ids_sha256
            or audit.get("observed_visible_rows") != subset.observed_visible_rows
            or audit.get("selected_rows") != decision.expected_source_rows
            or audit.get("withheld_rows") != subset.withheld_rows
            or audit.get("withheld_hold_rows") != subset.withheld_hold_rows
            or audit.get("withheld_excluded_rows") != subset.withheld_excluded_rows
        ):
            raise ValueError(
                f"Private subset preflight row binding differs: {scholar_id}"
            )
    return {
        "status": "pass",
        "receipt_sha256": sha256_file(receipt_path),
        "source_selection_manifest_sha256": sha256_file(selection_manifest_path),
        "profile_count": len(subsets),
        "selected_rows": sum(item.expected_source_rows for item in subsets.values()),
        "withheld_rows": sum(
            item.public_subset.withheld_rows for item in subsets.values()
        ),
        "private_evidence_limit": (
            "Public verification checks the locally generated receipt and byte "
            "hashes; private full capture/review content is verified only by "
            "controlled local preflight."
        ),
    }
