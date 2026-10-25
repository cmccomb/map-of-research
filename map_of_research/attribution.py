"""Positive author attribution for Scholar profiles with known mixed identities."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .io import load_json

DEFAULT_ATTRIBUTION_REVIEWS_PATH = Path("data/curated/scholar-attribution-reviews.json")


@dataclass(frozen=True)
class ConfirmedAttribution:
    evidence_url: str
    bibliography_entry: int | None
    verified_at: str
    evidence_basis: str = ""


@dataclass(frozen=True)
class AttributionReviews:
    held_profiles: frozenset[str]
    confirmed: dict[tuple[str, str], ConfirmedAttribution]


def _https_url(value: object) -> bool:
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme == "https" and bool(parsed.netloc)


def _date(value: object) -> bool:
    try:
        dt.date.fromisoformat(str(value or ""))
    except ValueError:
        return False
    return True


def load_attribution_reviews(path: Path) -> AttributionReviews:
    """Load a profile hold and exact IDs independently attributed to its owner."""

    source = load_json(path)
    if not isinstance(source, dict) or source.get("schema_version") != 1:
        raise ValueError(f"Unsupported Scholar-attribution schema: {path}")
    profiles = source.get("profiles")
    entries = source.get("confirmed_observations")
    if not isinstance(profiles, list) or not isinstance(entries, list):
        raise ValueError("Scholar attribution needs profile and observation lists")
    held_profiles: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("Scholar attribution profile must be an object")
        scholar_id = str(profile.get("scholar_id") or "").strip()
        capture_sha256 = str(profile.get("capture_sha256") or "").strip()
        captured_count = profile.get("captured_observations")
        if (
            not scholar_id
            or scholar_id in held_profiles
            or not re.fullmatch(r"[0-9a-f]{64}", capture_sha256)
            or not isinstance(captured_count, int)
            or captured_count <= 0
        ):
            raise ValueError(
                "Scholar attribution profile has invalid identity or capture"
            )
        if not _https_url(profile.get("evidence_url")) or not _date(
            profile.get("verified_at")
        ):
            raise ValueError("Scholar attribution profile needs dated HTTPS evidence")
        held_profiles.add(scholar_id)
    confirmed: dict[tuple[str, str], ConfirmedAttribution] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Scholar attribution observation must be an object")
        scholar_id = str(entry.get("scholar_id") or "").strip()
        author_pub_id = str(entry.get("author_pub_id") or "").strip()
        bibliography_entry = entry.get("bibliography_entry")
        evidence_basis = entry.get("evidence_basis", "")
        key = (scholar_id, author_pub_id)
        has_bibliography_entry = (
            isinstance(bibliography_entry, int)
            and not isinstance(bibliography_entry, bool)
            and bibliography_entry > 0
        )
        has_source_basis = (
            bibliography_entry is None
            and isinstance(evidence_basis, str)
            and bool(evidence_basis.strip())
        )
        if (
            scholar_id not in held_profiles
            or not author_pub_id.startswith(f"{scholar_id}:")
            or author_pub_id == f"{scholar_id}:"
            or key in confirmed
            or not (has_bibliography_entry or has_source_basis)
        ):
            raise ValueError("Scholar attribution observation has invalid identity")
        if not _https_url(entry.get("evidence_url")) or not _date(
            entry.get("verified_at")
        ):
            raise ValueError(
                "Scholar attribution observation needs dated HTTPS evidence"
            )
        confirmed[key] = ConfirmedAttribution(
            evidence_url=str(entry["evidence_url"]).strip(),
            bibliography_entry=bibliography_entry,
            verified_at=str(entry["verified_at"]).strip(),
            evidence_basis=evidence_basis.strip()
            if isinstance(evidence_basis, str)
            else "",
        )
    return AttributionReviews(frozenset(held_profiles), confirmed)
