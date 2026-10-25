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
from .sullivan_continuity_addendum import (
    ADDENDUM_SHA256 as SULLIVAN_ADDENDUM_SHA256,
)
from .sullivan_continuity_addendum import (
    BINDINGS_SHA256 as SULLIVAN_BINDINGS_SHA256,
)
from .sullivan_continuity_addendum import (
    LOSS_IDS_SHA256 as SULLIVAN_ADDENDUM_LOSS_IDS_SHA256,
)
from .sullivan_continuity_addendum import (
    TARGETS_SHA256 as SULLIVAN_TARGETS_SHA256,
)
from .sullivan_continuity_addendum import (
    ExactSullivanContinuityAddendum,
    decision_set_digests,
)
from .sullivan_loss_approval import (
    APPROVAL_SHA256 as SULLIVAN_APPROVAL_SHA256,
)
from .sullivan_loss_approval import (
    APPROVED_IDS_SHA256 as SULLIVAN_APPROVED_IDS_SHA256,
)
from .sullivan_loss_approval import (
    PRIOR_REVISION as SULLIVAN_PRIOR_REVISION,
)
from .sullivan_loss_approval import (
    ExactSullivanLossApproval,
)

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
    "collaboration_eligible",
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
            f"{len(receipt['unexplained_gains'])} unexplained gains, "
            f"{len(receipt.get('unapproved_mapped_binding_changes', []))} "
            "mapped binding changes, "
            f"{len(receipt.get('unused_sullivan_binding_approvals', []))} "
            "unused binding approvals, "
            f"{len(receipt.get('unused_sullivan_loss_approvals', []))} "
            "unused Sullivan approvals; "
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
    sullivan_loss_approval: ExactSullivanLossApproval | None = None,
    sullivan_continuity_addendum: ExactSullivanContinuityAddendum | None = None,
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

    final_works = _indexed_works(output.to_dict("records"))
    before = _visible_observations(previous_rows, previous_works)
    after = _visible_observations(current_rows, final_works)
    losses = before - after
    gains = after - before
    # Visibility alone is insufficient: a still-visible observation can move
    # to another eligible work or stop contributing a collaboration edge.
    # Neither change is covered by a Sullivan visible-row loss approval.
    mapped_binding_changes = [
        {
            "observation": _display(identity),
            "prior_work_id": str(previous_rows[identity].get("work_id") or ""),
            "current_work_id": str(current_rows[identity].get("work_id") or ""),
            "prior_collaboration_eligible": bool(
                previous_rows[identity].get("collaboration_eligible")
            ),
            "current_collaboration_eligible": bool(
                current_rows[identity].get("collaboration_eligible")
            ),
        }
        for identity in sorted(before & after)
        if str(previous_rows[identity].get("work_id") or "")
        != str(current_rows[identity].get("work_id") or "")
        or bool(previous_rows[identity].get("collaboration_eligible"))
        != bool(current_rows[identity].get("collaboration_eligible"))
    ]
    approved_binding_changes = set()
    approved_additional_losses: set[ObservationKey] = set()
    required_curated_work_ids: set[str] = set()
    if sullivan_continuity_addendum is not None:
        if (
            not isinstance(
                sullivan_continuity_addendum, ExactSullivanContinuityAddendum
            )
            or sullivan_loss_approval is None
            or prior_revision != SULLIVAN_PRIOR_REVISION
            or sullivan_continuity_addendum.manifest_sha256 != SULLIVAN_ADDENDUM_SHA256
            or len(sullivan_continuity_addendum.loss_identities) != 5
            or len(sullivan_continuity_addendum.binding_changes) != 3
            or len(sullivan_continuity_addendum.required_curated_work_ids) != 2
            or decision_set_digests(sullivan_continuity_addendum)
            != (
                SULLIVAN_ADDENDUM_LOSS_IDS_SHA256,
                SULLIVAN_BINDINGS_SHA256,
                SULLIVAN_TARGETS_SHA256,
            )
        ):
            raise ValueError("Sullivan continuity addendum is not the reviewed set")
        approved_additional_losses = set(sullivan_continuity_addendum.loss_identities)
        approved_binding_changes = set(sullivan_continuity_addendum.binding_changes)
        required_curated_work_ids = set(
            sullivan_continuity_addendum.required_curated_work_ids
        )
    actual_binding_changes = {
        (
            tuple(change["observation"].split("/")),
            change["prior_work_id"],
            change["current_work_id"],
            change["prior_collaboration_eligible"],
            change["current_collaboration_eligible"],
        )
        for change in mapped_binding_changes
    }
    unapproved_mapped_binding_changes = [
        change
        for change in mapped_binding_changes
        if (
            tuple(change["observation"].split("/")),
            change["prior_work_id"],
            change["current_work_id"],
            change["prior_collaboration_eligible"],
            change["current_collaboration_eligible"],
        )
        not in approved_binding_changes
    ]
    unused_sullivan_binding_approvals = (
        approved_binding_changes - actual_binding_changes
    )
    missing_curated_targets = sorted(
        work_id
        for work_id in required_curated_work_ids
        if not (
            (work := final_works.get(work_id)) is not None
            and bool(work.get("map_eligible"))
            and bool(work.get("collaboration_eligible"))
        )
    )
    reviewed_losses = sorted(
        identity for identity in losses if _reviewed_loss(current_rows.get(identity))
    )
    otherwise_unexplained_losses = losses - set(reviewed_losses)
    approved_sullivan_losses: set[ObservationKey] = set()
    unused_sullivan_approvals: set[ObservationKey] = set()
    if sullivan_loss_approval is not None:
        if not isinstance(sullivan_loss_approval, ExactSullivanLossApproval):
            raise ValueError("Sullivan loss approval has the wrong type")
        approved_sullivan_losses = set(sullivan_loss_approval.identities)
        approval_digest = hashlib.sha256(
            "\n".join(
                "/".join(identity) for identity in sorted(approved_sullivan_losses)
            ).encode()
        ).hexdigest()
        if (
            prior_revision != SULLIVAN_PRIOR_REVISION
            or sullivan_loss_approval.manifest_sha256 != SULLIVAN_APPROVAL_SHA256
            or approval_digest != SULLIVAN_APPROVED_IDS_SHA256
            or len(approved_sullivan_losses) != 22
            or len(sullivan_loss_approval.held_identities) != 18
            or len(sullivan_loss_approval.rekey_identities) != 4
            or not sullivan_loss_approval.held_identities.isdisjoint(
                sullivan_loss_approval.rekey_identities
            )
            or approved_sullivan_losses
            != set(
                sullivan_loss_approval.held_identities
                | sullivan_loss_approval.rekey_identities
            )
        ):
            raise ValueError("Sullivan exact-loss approval is not the reviewed set")
        approved_sullivan_losses |= approved_additional_losses
        unused_sullivan_approvals = approved_sullivan_losses - losses
    # In this one-time source transition the manifest is the entire permitted
    # prior-visible loss set. Generic reviewed-loss exemptions still apply to
    # ordinary releases, but cannot admit a 23rd Sullivan-release loss.
    losses_requiring_approval = (
        losses if sullivan_loss_approval is not None else otherwise_unexplained_losses
    )
    unexplained_losses = sorted(losses_requiring_approval - approved_sullivan_losses)
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
        "approved_sullivan_held_losses": (
            [_display(item) for item in sorted(sullivan_loss_approval.held_identities)]
            if sullivan_loss_approval is not None
            else []
        ),
        "approved_sullivan_rekey_losses": (
            [_display(item) for item in sorted(sullivan_loss_approval.rekey_identities)]
            if sullivan_loss_approval is not None
            else []
        ),
        "approved_sullivan_additional_losses": [
            _display(item) for item in sorted(approved_additional_losses)
        ],
        "approved_sullivan_binding_changes": [
            change
            for change in mapped_binding_changes
            if (
                tuple(change["observation"].split("/")),
                change["prior_work_id"],
                change["current_work_id"],
                change["prior_collaboration_eligible"],
                change["current_collaboration_eligible"],
            )
            in approved_binding_changes
        ],
        "sullivan_continuity_addendum_sha256": (
            sullivan_continuity_addendum.manifest_sha256
            if sullivan_continuity_addendum is not None
            else None
        ),
        "sullivan_loss_approval_sha256": (
            sullivan_loss_approval.manifest_sha256
            if sullivan_loss_approval is not None
            else None
        ),
        "unused_sullivan_loss_approvals": [
            _display(item) for item in sorted(unused_sullivan_approvals)
        ],
        "verified_gains": [_display(item) for item in verified_gains],
        "new_source_gains": len(new_gains),
        "unapproved_mapped_binding_changes": unapproved_mapped_binding_changes,
        "unused_sullivan_binding_approvals": [
            {
                "observation": _display(identity),
                "prior_work_id": prior_work_id,
                "current_work_id": current_work_id,
                "prior_collaboration_eligible": prior_collaboration_eligible,
                "current_collaboration_eligible": current_collaboration_eligible,
            }
            for (
                identity,
                prior_work_id,
                current_work_id,
                prior_collaboration_eligible,
                current_collaboration_eligible,
            ) in sorted(unused_sullivan_binding_approvals)
        ],
        "missing_sullivan_curated_targets": missing_curated_targets,
        "unexplained_losses": [_display(item) for item in unexplained_losses],
        "unexplained_gains": [_display(item) for item in unexplained_gains],
    }
    if (
        unexplained_losses
        or unexplained_gains
        or unapproved_mapped_binding_changes
        or unused_sullivan_binding_approvals
        or missing_curated_targets
        or unused_sullivan_approvals
    ):
        raise RegionReleaseDiffError(receipt)
    return output, receipt
