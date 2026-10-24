"""Reviewed, source-backed decisions about individual Scholar observations."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .io import load_json

DEFAULT_EXCLUSIONS_PATH = Path("data/curated/scholar-exclusions.json")
EXCLUSION_SCHEMA_VERSION = 1
REASON_CODES = frozenset(
    {"nonpublication", "false_profile_attribution", "corrupted_duplicate"}
)


@dataclass(frozen=True)
class ReviewedExclusion:
    reason_code: str
    reason: str
    evidence_url: str
    verified_at: str


def load_exclusions(path: Path) -> dict[tuple[str, str], ReviewedExclusion]:
    """Validate a checked-in review manifest keyed by Scholar row identity."""

    source = load_json(path)
    if not isinstance(source, dict) or source.get("schema_version") != (
        EXCLUSION_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported Scholar-exclusion schema: {path}")
    entries = source.get("exclusions")
    if not isinstance(entries, list):
        raise ValueError("Scholar exclusions must be a list")
    exclusions: dict[tuple[str, str], ReviewedExclusion] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Scholar exclusion must be an object")
        scholar_id = str(entry.get("scholar_id") or "").strip()
        author_pub_id = str(entry.get("author_pub_id") or "").strip()
        if not scholar_id or not author_pub_id.startswith(f"{scholar_id}:"):
            raise ValueError("Scholar exclusion has an invalid observation identity")
        reason_code = str(entry.get("reason_code") or "").strip()
        reason = str(entry.get("reason") or "").strip()
        if reason_code not in REASON_CODES or not reason:
            raise ValueError("Scholar exclusion needs a known code and specific reason")
        evidence_url = str(entry.get("evidence_url") or "").strip()
        parsed = urlparse(evidence_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Scholar exclusion needs an HTTPS evidence URL")
        verified_at = str(entry.get("verified_at") or "").strip()
        try:
            dt.date.fromisoformat(verified_at)
        except ValueError as error:
            raise ValueError(
                "Scholar exclusion verified_at must use YYYY-MM-DD"
            ) from error
        key = (scholar_id, author_pub_id)
        if key in exclusions:
            raise ValueError(f"Duplicate Scholar exclusion: {key}")
        exclusions[key] = ReviewedExclusion(
            reason_code=reason_code,
            reason=reason,
            evidence_url=evidence_url,
            verified_at=verified_at,
        )
    return exclusions
