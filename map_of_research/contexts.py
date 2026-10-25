"""Source-backed publication context for individual Scholar observations."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .io import load_json

DEFAULT_CONTEXTS_PATH = Path("data/curated/scholar-contexts.json")
CONTEXT_SCHEMA_VERSION = 1
CONTEXT_TYPES = frozenset(
    {
        "patent",
        "edited_book",
        "panel_transcript",
        "withdrawn_article",
        "conference_presentation",
    }
)
PATENT_PUBLICATION_NUMBER = re.compile(r"^[A-Z]{2}\d{5,}[A-Z]\d?$")


@dataclass(frozen=True)
class ReviewedContext:
    publication_type: str
    document_id: str
    reason: str
    evidence_url: str
    verified_at: str
    display_contributors: str = ""
    map_eligible: bool = True


def load_contexts(path: Path) -> dict[tuple[str, str], ReviewedContext]:
    """Validate exact, evidence-backed Scholar row context annotations."""

    source = load_json(path)
    if not isinstance(source, dict) or source.get("schema_version") != (
        CONTEXT_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported Scholar-context schema: {path}")
    entries = source.get("contexts")
    if not isinstance(entries, list):
        raise ValueError("Scholar contexts must be a list")
    contexts: dict[tuple[str, str], ReviewedContext] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Scholar context must be an object")
        scholar_id = str(entry.get("scholar_id") or "").strip()
        author_pub_id = str(entry.get("author_pub_id") or "").strip()
        if not scholar_id or not author_pub_id.startswith(f"{scholar_id}:"):
            raise ValueError("Scholar context has an invalid observation identity")
        publication_type = str(entry.get("publication_type") or "").strip()
        if publication_type not in CONTEXT_TYPES:
            raise ValueError("Scholar context needs a known publication type")
        document_id = str(entry.get("document_id") or "").strip().upper()
        if publication_type == "patent":
            if not PATENT_PUBLICATION_NUMBER.fullmatch(document_id):
                raise ValueError("Patent context needs a publication number")
        elif document_id:
            raise ValueError("Only patent contexts may specify a document ID")
        reason = str(entry.get("reason") or "").strip()
        if not reason:
            raise ValueError("Scholar context needs a specific reason")
        display_contributors = str(entry.get("display_contributors") or "").strip()
        if display_contributors and publication_type != "edited_book":
            raise ValueError(
                "Only edited-book contexts may override displayed contributors"
            )
        map_eligible = entry.get("map_eligible", True)
        if not isinstance(map_eligible, bool):
            raise ValueError("Scholar context map_eligible must be boolean")
        evidence_url = str(entry.get("evidence_url") or "").strip()
        parsed = urlparse(evidence_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Scholar context needs an HTTPS evidence URL")
        verified_at = str(entry.get("verified_at") or "").strip()
        try:
            dt.date.fromisoformat(verified_at)
        except ValueError as error:
            raise ValueError(
                "Scholar context verified_at must use YYYY-MM-DD"
            ) from error
        key = (scholar_id, author_pub_id)
        if key in contexts:
            raise ValueError(f"Duplicate Scholar context: {key}")
        contexts[key] = ReviewedContext(
            publication_type=publication_type,
            document_id=document_id,
            reason=reason,
            evidence_url=evidence_url,
            verified_at=verified_at,
            display_contributors=display_contributors,
            map_eligible=map_eligible,
        )
    return contexts
