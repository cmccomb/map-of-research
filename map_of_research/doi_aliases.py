"""Reviewed DOI links for exact Scholar observations."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .io import load_json

DEFAULT_DOI_ALIASES_PATH = Path("data/curated/scholar-doi-aliases.json")
DOI_ALIAS_SCHEMA_VERSION = 1
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
ARXIV_PREPRINT_DOI_PATTERN = re.compile(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})")


@dataclass(frozen=True)
class ReviewedDoiAlias:
    doi: str
    observed_title: str
    observed_year: int | None
    observed_venue: str | None
    relationship: str
    evidence_urls: tuple[str, ...]
    verified_at: str


def load_doi_aliases(path: Path) -> dict[tuple[str, str], ReviewedDoiAlias]:
    """Load exact, independently evidenced version links without changing raw rows."""

    source = load_json(path)
    if not isinstance(source, dict) or source.get("schema_version") != (
        DOI_ALIAS_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported Scholar DOI-alias schema: {path}")
    entries = source.get("aliases")
    if not isinstance(entries, list):
        raise ValueError("Scholar DOI aliases must be a list")
    aliases: dict[tuple[str, str], ReviewedDoiAlias] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Scholar DOI alias must be an object")
        scholar_id = str(entry.get("scholar_id") or "").strip()
        author_pub_id = str(entry.get("author_pub_id") or "").strip()
        if not scholar_id or not author_pub_id.startswith(f"{scholar_id}:"):
            raise ValueError("Scholar DOI alias has an invalid observation identity")
        relationship = str(entry.get("relationship") or "").strip()
        if relationship not in {
            "preprint_of",
            "version_of_record",
            "arxiv_preprint_doi",
        }:
            raise ValueError("Scholar DOI alias needs a supported DOI relationship")
        if relationship == "arxiv_preprint_doi":
            if "published_doi" in entry:
                raise ValueError("arXiv preprint DOI must not be a published-work DOI")
            doi = str(entry.get("preprint_doi") or "").strip().casefold()
            arxiv_match = ARXIV_PREPRINT_DOI_PATTERN.fullmatch(doi)
            if not arxiv_match:
                raise ValueError("Scholar DOI alias needs a valid arXiv preprint DOI")
            observed_venue = str(entry.get("observed_venue") or "").strip()
            if not observed_venue or f"arxiv:{arxiv_match.group(1)}" not in (
                observed_venue.casefold()
            ):
                raise ValueError("arXiv preprint DOI needs its exact observed venue")
        else:
            if "preprint_doi" in entry or "observed_venue" in entry:
                raise ValueError("Published-work DOI alias has preprint-only fields")
            doi = str(entry.get("published_doi") or "").strip().casefold()
            if not DOI_PATTERN.fullmatch(doi):
                raise ValueError("Scholar DOI alias needs a valid published DOI")
            observed_venue = None
        title = str(entry.get("observed_title") or "").strip()
        year = entry.get("observed_year")
        if (
            not title
            or "observed_year" not in entry
            or (year is not None and type(year) is not int)
        ):
            raise ValueError(
                "Scholar DOI alias needs the exact observed title and year"
            )
        evidence_urls = entry.get("evidence_urls")
        if (
            not isinstance(evidence_urls, list)
            or len(evidence_urls) < 2
            or len(set(evidence_urls)) != len(evidence_urls)
            or any(
                not isinstance(url, str)
                or urlparse(url).scheme != "https"
                or not urlparse(url).netloc
                for url in evidence_urls
            )
        ):
            raise ValueError("Scholar DOI alias needs distinct HTTPS source URLs")
        if relationship == "arxiv_preprint_doi" and not any(
            (parsed := urlparse(url)).hostname == "arxiv.org"
            and parsed.path.rstrip("/") == f"/abs/{arxiv_match.group(1)}"
            for url in evidence_urls
        ):
            raise ValueError("arXiv preprint DOI needs the matching arXiv record")
        verified_at = str(entry.get("verified_at") or "").strip()
        try:
            dt.date.fromisoformat(verified_at)
        except ValueError as error:
            raise ValueError(
                "Scholar DOI alias verified_at must use YYYY-MM-DD"
            ) from error
        key = (scholar_id, author_pub_id)
        if key in aliases:
            raise ValueError(f"Duplicate Scholar DOI alias: {key}")
        aliases[key] = ReviewedDoiAlias(
            doi=doi,
            observed_title=title,
            observed_year=year,
            observed_venue=observed_venue,
            relationship=relationship,
            evidence_urls=tuple(evidence_urls),
            verified_at=verified_at,
        )
    return aliases


def validate_doi_aliases(
    frame: Any,
    aliases: dict[tuple[str, str], ReviewedDoiAlias],
    exclusions: Any,
) -> None:
    """Fail before publication if an alias is stale or points to an excluded row."""

    import pandas

    for key, alias in aliases.items():
        scholar_id, author_pub_id = key
        rows = frame.loc[
            (frame["source_type"] == "scholar")
            & (frame["scholar_id"] == scholar_id)
            & (frame["author_pub_id"] == author_pub_id)
        ]
        if len(rows) != 1:
            raise ValueError(f"Scholar DOI alias must match one raw row: {key}")
        row = rows.iloc[0]
        observed_year_matches = (
            bool(pandas.isna(row["year"]))
            if alias.observed_year is None
            else bool(
                not pandas.isna(row["year"]) and row["year"] == alias.observed_year
            )
        )
        if row["title"] != alias.observed_title or not observed_year_matches:
            raise ValueError(f"Scholar DOI alias observed title or year changed: {key}")
        if alias.observed_venue is not None and row.get("venue") != (
            alias.observed_venue
        ):
            raise ValueError(f"Scholar DOI alias observed venue changed: {key}")
        if key in exclusions:
            raise ValueError(f"Scholar DOI alias also has a reviewed exclusion: {key}")
        source_doi = str(row.get("source_doi") or "").strip().casefold()
        if source_doi and source_doi != alias.doi:
            raise ValueError(f"Scholar DOI alias conflicts with a source DOI: {key}")
