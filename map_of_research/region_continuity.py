"""Keep weak-region decisions stable for unchanged published observations.

This is release change control, not a claim that a previously mapped row is a
valid publication. Current exact reviews and source changes always take priority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import string
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any
from urllib.parse import unquote_plus, urlparse, urlsplit, urlunsplit

from .doi_aliases import DOI_PATTERN
from .region_review import ReviewedRegionDecision

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
WORK_FINGERPRINT_FIELDS = (
    "title",
    "authors",
    "year",
    "venue",
    "citation",
    "citation_count",
    "doi",
    "publication_type",
    "source_document_id",
    "source_urls",
    "source_types",
    "source_native_ids",
    "source_verified_at",
    "source_verification_urls",
)
PRIOR_WORK_FIELDS = (
    "work_id",
    "map_eligible",
    "map_region_outlier",
    "department_ids",
    "quality_assessment_version",
    *WORK_FINGERPRINT_FIELDS,
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
    return _fingerprint_fields(row, FINGERPRINT_FIELDS)


def _fingerprint_fields(row: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    payload = {field: _value(row.get(field)) for field in fields}
    for field in ("year", "citation_count"):
        if field in payload and payload[field] is not None:
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


_TRACKING_QUERY_KEYS = frozenset(
    {"fbclid", "gclid", "gbraid", "wbraid", "msclkid", "mc_cid", "mc_eid", "igshid"}
)
_UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
_PERCENT_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")


def _verification_url_path(path: str) -> str:
    """Resolve equivalent escaped characters and dot segments in a URL path."""

    def unescape(match: re.Match[str]) -> str:
        char = chr(int(match.group(1), 16))
        return char if char in _UNRESERVED else f"%{match.group(1).upper()}"

    segments = _PERCENT_ESCAPE.sub(unescape, path or "/").split("/")
    normalized: list[str] = []
    for index, segment in enumerate(segments):
        if segment == ".":
            if index == len(segments) - 1:
                normalized.append("")
        elif segment == "..":
            if len(normalized) > 1:
                normalized.pop()
            if index == len(segments) - 1:
                normalized.append("")
        else:
            normalized.append(segment)
    return "/".join(normalized) or "/"


def _verification_url_identity(value: str) -> str:
    """Compare the cited document, ignoring URL presentation and tracking."""

    parsed = urlsplit(value.strip())
    hostname = (parsed.hostname or "").casefold()
    if not hostname:
        return value.strip()
    try:
        port = parsed.port
    except ValueError:
        return value.strip()
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None and port != 443:
        netloc += f":{port}"
    query = "&".join(
        part
        for part in parsed.query.split("&")
        if part
        and not (
            (key := unquote_plus(part.partition("=")[0]).casefold()).startswith("utm_")
            or key in _TRACKING_QUERY_KEYS
        )
    )
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            netloc,
            _verification_url_path(parsed.path),
            query,
            "",
        )
    )


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


def _verification_evidence(
    row: Mapping[str, Any], work: Mapping[str, Any]
) -> set[tuple[str, str]]:
    """Identify publication evidence without display metadata or list order."""

    evidence: set[tuple[str, str]] = set()
    for value in (row.get("doi"), row.get("source_doi"), work.get("doi")):
        doi = str(value or "").strip().casefold()
        if DOI_PATTERN.fullmatch(doi):
            evidence.add(("doi", doi))
    for value in (
        row.get("source_native_id"),
        *(_value(work.get("source_native_ids")) or []),
    ):
        native_id = str(value or "").strip().casefold()
        if re.fullmatch(r"[a-z][a-z0-9-]*:[a-z0-9][a-z0-9._:/-]*", native_id):
            evidence.add(("native_id", native_id))
    for urls in (
        row.get("source_verification_urls"),
        work.get("source_verification_urls"),
    ):
        for value in _value(urls) or []:
            url = str(value).strip()
            if _independent_evidence([url]):
                evidence.add(("verification_url", _verification_url_identity(url)))
    if "curated_verified" in (_value(work.get("source_types")) or []):
        evidence.add(("source_type", "curated_verified"))
    return evidence


def _display(identity: ObservationKey) -> str:
    return "/".join(identity)


def reconcile_region_decisions(
    works: Any,
    profile_publications: Any,
    prior_works: Iterable[Mapping[str, Any]],
    prior_profile_publications: Iterable[Mapping[str, Any]],
    *,
    prior_revision: str,
    reviewed_math_decisions: tuple[ReviewedRegionDecision, ...] = (),
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
        prior_departments = {
            str(value) for value in _value(prior.get("department_ids")) or [] if value
        }
        current_departments = {
            str(value) for value in _value(row.department_ids) or [] if value
        }
        if prior_departments != current_departments:
            continue
        if _fingerprint_fields(prior, WORK_FINGERPRINT_FIELDS) != _fingerprint_fields(
            row, WORK_FINGERPRINT_FIELDS
        ):
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
            output.at[index, "map_eligible"] = False
            output.at[index, "map_region_outlier"] = True
            output.at[index, "map_exclusion_reasons"] = [
                *row.map_exclusion_reasons,
                "low_information_region",
            ]
            output.at[index, "map_region_prior_excluded_override"] = True

    reviewed_gains: set[ObservationKey] = set()
    reviewed_losses: set[ObservationKey] = set()
    if reviewed_math_decisions:
        if prior_revision != "bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd":
            raise ValueError("Math region review requires the frozen bb88 baseline")
        output_indices = {str(row.work_id): index for index, row in output.iterrows()}
        if len(output_indices) != len(output):
            raise ValueError("Math region review found duplicate work IDs")
        for decision in reviewed_math_decisions:
            identity = decision.observation_identity
            prior_row = previous_rows.get(identity)
            current_row = current_rows.get(identity)
            prior_work = previous_works.get(decision.work_id)
            index = output_indices.get(decision.work_id)
            if (
                prior_row is None
                or current_row is None
                or prior_work is None
                or index is None
                or prior_row.get("work_id") != decision.work_id
                or current_row.get("work_id") != decision.work_id
                or hashlib.sha256(
                    str(prior_row.get("source_record_json") or "").encode()
                ).hexdigest()
                != decision.source_record_sha256
                or hashlib.sha256(
                    str(current_row.get("source_record_json") or "").encode()
                ).hexdigest()
                != decision.source_record_sha256
                or _fingerprint_fields(
                    prior_work,
                    (*WORK_FINGERPRINT_FIELDS, "quality_assessment_version"),
                )
                != decision.work_fingerprint_sha256
                or bool(prior_row.get("map_eligible")) is not True
                or bool(prior_work.get("map_eligible"))
                != (decision.transition == "reexclude")
                or bool(prior_work.get("map_region_outlier"))
                != (decision.transition == "restore")
                or bool(current_row.get("map_eligible")) != decision.visible
                or len(
                    [
                        item
                        for item in current_rows.values()
                        if item.get("work_id") == decision.work_id
                    ]
                )
                != 1
            ):
                raise ValueError(
                    "Math region review no longer matches exact prior/current "
                    f"observation: {decision.author_pub_id}"
                )
            if decision.transition == "restore" and decision.visible:
                if not bool(output.at[index, "map_region_outlier"]):
                    raise ValueError("Reviewed Math gain is no longer a region outlier")
                output.at[index, "map_eligible"] = True
                output.at[index, "map_region_outlier"] = False
                output.at[index, "map_exclusion_reasons"] = [
                    reason
                    for reason in output.at[index, "map_exclusion_reasons"]
                    if reason != "low_information_region"
                ]
                reviewed_gains.add(identity)
            elif decision.transition == "reexclude" and not decision.visible:
                reviewed_losses.add(identity)
            if bool(output.at[index, "map_eligible"]) != decision.visible:
                raise ValueError(
                    "Math region final visibility differs from reviewed decision: "
                    f"{decision.author_pub_id}"
                )

    final_works = _indexed_works(output.to_dict("records"))
    before = _visible_observations(previous_rows, previous_works)
    after = _visible_observations(current_rows, final_works)
    losses = before - after
    gains = after - before
    approved_losses = sorted(
        identity
        for identity in losses
        if _reviewed_loss(current_rows.get(identity)) or identity in reviewed_losses
    )
    unexplained_losses = sorted(losses - set(approved_losses))
    verified_gains = sorted(
        identity
        for identity in gains
        if identity in reviewed_gains
        or (
            identity in previous_rows
            and bool(previous_rows[identity].get("map_eligible"))
            and bool(
                previous_works[str(previous_rows[identity]["work_id"])].get(
                    "map_region_outlier"
                )
            )
            and bool(
                _verification_evidence(
                    current_rows[identity],
                    final_works[str(current_rows[identity]["work_id"])],
                )
                - _verification_evidence(
                    previous_rows[identity],
                    previous_works[str(previous_rows[identity]["work_id"])],
                )
            )
            and _verified_gain(
                current_rows[identity],
                final_works[str(current_rows[identity]["work_id"])],
            )
        )
    )
    # A prior weak-region exclusion remains excluded unless independently
    # verified. Newly collected Scholar rows use the current quality audit.
    new_gains = sorted(identity for identity in gains if identity not in previous_rows)
    unexplained_gains = sorted(gains - set(verified_gains) - set(new_gains))
    if reviewed_math_decisions and (
        set(reviewed_gains) != set(verified_gains)
        or set(reviewed_losses) != set(approved_losses)
        or len(reviewed_gains) != 4
        or len(reviewed_losses) != 2
        or new_gains
    ):
        raise ValueError(
            "Math region review changed observations outside its exact +4/-2 scope"
        )
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
        "reviewed_losses": [_display(item) for item in approved_losses],
        "verified_gains": [_display(item) for item in verified_gains],
        "reviewed_math_gains": [_display(item) for item in sorted(reviewed_gains)],
        "new_source_gains": len(new_gains),
        "unexplained_losses": [_display(item) for item in unexplained_losses],
        "unexplained_gains": [_display(item) for item in unexplained_gains],
    }
    if unexplained_losses or unexplained_gains:
        raise RegionReleaseDiffError(receipt)
    return output, receipt
