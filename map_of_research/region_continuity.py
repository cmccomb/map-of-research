"""Keep weak-region decisions stable for unchanged published observations.

This is release change control, not a claim that a previously mapped row is a
valid publication. Current exact reviews and source changes always take priority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any
from urllib.parse import urlparse

from .doi_aliases import DOI_PATTERN

REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
CONTINUITY_VERSION = "exact-observation-region-continuity-v1"
IDENTITY_FIELDS = ("person_id", "scholar_id", "author_pub_id", "source_type")
FINGERPRINT_FIELDS = (
    "source_record_json",
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
    "metadata_correction_fields",
    "metadata_correction_basis",
    "metadata_correction_evidence_url",
    "metadata_correction_verified_at",
    "publication_type",
    "source_document_id",
    "quality_assessment_version",
)
PRIOR_WORK_FIELDS = (
    "work_id",
    "map_eligible",
    "map_region_outlier",
    "department_ids",
    "quality_assessment_version",
)
PRIOR_OBSERVATION_FIELDS = (
    *IDENTITY_FIELDS,
    "work_id",
    "map_eligible",
    "review_exclusion_code",
    "context_verified_at",
    *tuple(field for field in FINGERPRINT_FIELDS if field not in IDENTITY_FIELDS),
)
ObservationKey = tuple[str, str, str, str]


class RegionReleaseDiffError(ValueError):
    """A release changed existing mapped observations without exact review."""

    def __init__(self, receipt: dict[str, Any]) -> None:
        self.receipt = receipt
        super().__init__(
            "Region release-diff gate failed: "
            f"{len(receipt['unexplained_losses'])} unexplained losses, "
            f"{len(receipt['unexplained_gains'])} unexplained gains; "
            f"first IDs: {receipt['unexplained_losses'][:2]} "
            f"{receipt['unexplained_gains'][:2]}"
        )


def require_revision(value: str) -> str:
    """Reject moving refs and malformed baseline identifiers."""

    if not REVISION_PATTERN.fullmatch(value):
        raise ValueError("Prior dataset revision must be an immutable 40-hex SHA")
    return value


def _value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "tolist"):
        return _value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    return value


def _identity(row: Mapping[str, Any]) -> ObservationKey:
    return (
        str(row.get("person_id") or ""),
        str(row.get("scholar_id") or ""),
        str(row.get("author_pub_id") or ""),
        str(row.get("source_type") or ""),
    )


def _fingerprint(row: Mapping[str, Any]) -> str:
    payload = {field: _value(row.get(field)) for field in FINGERPRINT_FIELDS}
    for field in ("year", "citation_count"):
        if payload[field] is not None:
            payload[field] = int(payload[field])
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _indexed_observations(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[
    dict[ObservationKey, Mapping[str, Any]],
    dict[str, dict[ObservationKey, str]],
]:
    by_identity: dict[ObservationKey, Mapping[str, Any]] = {}
    eligible_by_work: dict[str, dict[ObservationKey, str]] = defaultdict(dict)
    for row in rows:
        identity = _identity(row)
        if identity in by_identity:
            raise ValueError(f"Duplicate profile observation identity: {identity}")
        by_identity[identity] = row
        if bool(row.get("map_eligible")):
            eligible_by_work[str(row.get("work_id") or "")][identity] = _fingerprint(
                row
            )
    return by_identity, dict(eligible_by_work)


def _indexed_works(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        work_id = str(row.get("work_id") or "")
        if not work_id or work_id in result:
            raise ValueError(f"Missing or duplicate work identity: {work_id}")
        result[work_id] = row
    return result


def _visible_observations(
    observations: Mapping[ObservationKey, Mapping[str, Any]],
    works: Mapping[str, Mapping[str, Any]],
) -> set[ObservationKey]:
    return {
        identity
        for identity, row in observations.items()
        if bool(row.get("map_eligible"))
        and (work := works.get(str(row.get("work_id") or ""))) is not None
        and bool(work.get("map_eligible"))
    }


def _reviewed_loss(row: Mapping[str, Any] | None) -> bool:
    if row is None or bool(row.get("map_eligible")):
        return False
    reasons = _value(row.get("map_exclusion_reasons")) or []
    exclusion_code = str(row.get("review_exclusion_code") or "")
    return bool(
        (exclusion_code and f"reviewed_{exclusion_code}" in reasons)
        or (
            row.get("context_verified_at")
            and any(
                reason in {"reviewed_withdrawn_article", "reviewed_context_not_mapped"}
                for reason in reasons
            )
        )
    )


def _independent_evidence(urls: Any) -> bool:
    for url in _value(urls) or []:
        parsed = urlparse(str(url))
        hostname = (parsed.hostname or "").casefold()
        if (
            parsed.scheme == "https"
            and hostname
            and hostname != "scholar.google.com"
            and not hostname.endswith(".scholar.google.com")
        ):
            return True
    return False


def _valid_review_date(value: Any) -> bool:
    try:
        text = str(value or "")
        return date.fromisoformat(text).isoformat() == text
    except ValueError:
        return False


def _verified_gain(row: Mapping[str, Any], work: Mapping[str, Any]) -> bool:
    if not (
        str(work.get("title") or "").strip()
        and _value(work.get("year")) is not None
        and str(work.get("authors") or "").strip()
        and str(work.get("venue") or "").strip()
        and _valid_review_date(work.get("source_verified_at"))
        and _independent_evidence(work.get("source_verification_urls"))
    ):
        return False
    doi = str(work.get("doi") or "")
    has_doi = bool(DOI_PATTERN.fullmatch(doi))
    native_ids = _value(work.get("source_native_ids")) or []
    has_native_id = any(
        bool(re.fullmatch(r"[a-z][a-z0-9-]*:[a-z0-9][a-z0-9._:/-]*", str(value)))
        for value in native_ids
    )
    if row.get("work_match_method") == "reviewed_version_alias":
        return bool(
            has_doi
            and _valid_review_date(row.get("source_verified_at"))
            and _independent_evidence(row.get("source_verification_urls"))
        )
    return bool(
        "curated_verified" in (_value(work.get("source_types")) or [])
        and (has_doi or has_native_id)
    )


def _display(identity: ObservationKey) -> str:
    return "/".join(identity)


def reconcile_region_decisions(
    works: Any,
    profile_publications: Any,
    prior_works: Iterable[Mapping[str, Any]],
    prior_profile_publications: Iterable[Mapping[str, Any]],
    *,
    prior_revision: str,
) -> tuple[Any, dict[str, Any]]:
    """Apply the prior region decision only to the exact same eligible rows.

    Any new/changed observation set uses the current audit. The release gate
    stops unexplained visible losses and promotions before any Hub upload.
    """

    require_revision(prior_revision)
    previous_works = _indexed_works(prior_works)
    previous_rows, previous_members = _indexed_observations(prior_profile_publications)
    current_rows, current_members = _indexed_observations(
        profile_publications.to_dict("records")
    )
    output = works.copy()
    output["map_region_prior_eligible_override"] = False
    output["map_region_prior_excluded_override"] = False
    stable_works = 0
    for index, row in output.iterrows():
        work_id = str(row.work_id)
        prior = previous_works.get(work_id)
        current_work_members = current_members.get(work_id)
        if (
            prior is None
            or not current_work_members
            or previous_members.get(work_id) != current_work_members
        ):
            continue
        if prior.get("quality_assessment_version") != row.quality_assessment_version:
            continue
        previous_outlier = bool(prior.get("map_region_outlier"))
        previous_mapped = bool(prior.get("map_eligible"))
        current_outlier = bool(row.map_region_outlier)
        current_mapped = bool(row.map_eligible)
        if not (previous_mapped or previous_outlier) or not (
            current_mapped or current_outlier
        ):
            continue
        stable_works += 1
        if previous_mapped and current_outlier:
            output.at[index, "map_eligible"] = True
            output.at[index, "map_region_outlier"] = False
            output.at[index, "map_exclusion_reasons"] = [
                reason
                for reason in row.map_exclusion_reasons
                if reason != "low_information_region"
            ]
            output.at[index, "map_region_prior_eligible_override"] = True
        elif previous_outlier and current_mapped:
            # A revised region rule may deliberately admit a documented work.
            # Let the per-observation release gate verify every new member.
            if bool(row.get("map_region_documented_override", False)) and any(
                _verified_gain(current_rows[identity], row)
                for identity in current_work_members
            ):
                continue
            output.at[index, "map_eligible"] = False
            output.at[index, "map_region_outlier"] = True
            output.at[index, "map_exclusion_reasons"] = [
                *row.map_exclusion_reasons,
                "low_information_region",
            ]
            output.at[index, "map_region_prior_excluded_override"] = True

    final_works = _indexed_works(output.to_dict("records"))
    before = _visible_observations(previous_rows, previous_works)
    after = _visible_observations(current_rows, final_works)
    losses = before - after
    gains = after - before
    reviewed_losses = sorted(
        identity for identity in losses if _reviewed_loss(current_rows.get(identity))
    )
    unexplained_losses = sorted(losses - set(reviewed_losses))
    verified_gains = sorted(
        identity
        for identity in gains
        if identity in previous_rows
        and bool(previous_rows[identity].get("map_eligible"))
        and bool(
            previous_works[str(previous_rows[identity]["work_id"])].get(
                "map_region_outlier"
            )
        )
        and _verified_gain(
            current_rows[identity],
            final_works[str(current_rows[identity]["work_id"])],
        )
    )
    # A prior weak-region exclusion remains excluded unless independently
    # verified. Newly collected Scholar rows use the current quality audit.
    new_gains = sorted(identity for identity in gains if identity not in previous_rows)
    unexplained_gains = sorted(gains - set(verified_gains) - set(new_gains))
    receipt = {
        "version": CONTINUITY_VERSION,
        "prior_revision": prior_revision,
        "prior_visible_observations": len(before),
        "current_visible_observations": len(after),
        "stable_works": stable_works,
        "prior_eligible_overrides": int(
            output["map_region_prior_eligible_override"].sum()
        ),
        "prior_excluded_overrides": int(
            output["map_region_prior_excluded_override"].sum()
        ),
        "reviewed_losses": [_display(item) for item in reviewed_losses],
        "verified_gains": [_display(item) for item in verified_gains],
        "new_source_gains": len(new_gains),
        "unexplained_losses": [_display(item) for item in unexplained_losses],
        "unexplained_gains": [_display(item) for item in unexplained_gains],
    }
    if unexplained_losses or unexplained_gains:
        raise RegionReleaseDiffError(receipt)
    return output, receipt
