"""Reviewed links from exact Scholar observations to published-work DOIs."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .io import load_json

DEFAULT_DOI_ALIASES_PATH = Path("data/curated/scholar-doi-aliases.json")
DOI_ALIAS_SCHEMA_VERSION = 1
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
CELL_REVIEWED_SPARSE_TITLES = {
    "10.1016/j.molcel.2008.06.010": {
        "identity": ("28yZqbcAAAAJ", "28yZqbcAAAAJ:UeHWp8X0CEIC"),
        "title": "Getting the message out",
        "year": 2008,
        "evidence_urls": frozenset(
            {
                "https://www.cell.com/molecular-cell/fulltext/S1097-2765(08)00428-0",
                "https://pubmed.ncbi.nlm.nih.gov/18614041/",
                "https://doi.org/10.1016/j.molcel.2008.06.010",
            }
        ),
    },
    "10.1016/j.neuron.2017.06.006": {
        "identity": ("FXqaMl4AAAAJ", "FXqaMl4AAAAJ:UebtZRa9Y70C"),
        "title": "Constructing the External World",
        "year": 2017,
        "evidence_urls": frozenset(
            {
                "https://www.sciencedirect.com/science/article/pii/S0896627317305044",
                "https://pubmed.ncbi.nlm.nih.gov/28641104/",
                "https://doi.org/10.1016/j.neuron.2017.06.006",
            }
        ),
    },
}


@dataclass(frozen=True)
class ReviewedDoiAlias:
    doi: str
    observed_title: str
    observed_year: int | None
    relationship: str
    evidence_urls: tuple[str, ...]
    verified_at: str
    allow_short_title_region_override: bool = False


def _publisher_url_for_doi(url: str, doi: str) -> bool:
    parsed = urlparse(url)
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == "link.springer.com"
        and parsed.path.casefold().endswith("/" + doi)
    )


def _short_title_evidence_matches(
    doi: str,
    evidence_urls: tuple[str, ...] | list[str],
    verification_urls: tuple[str, ...] | list[str] | None = None,
) -> bool:
    """Require a publisher DOI link or the exact reviewed Cell DOI/PII chain."""

    cell_review = CELL_REVIEWED_SPARSE_TITLES.get(doi)
    if cell_review is not None:
        required_urls = cell_review["evidence_urls"]
        return required_urls.issubset(evidence_urls) and (
            verification_urls is None or required_urls.issubset(verification_urls)
        )
    return any(
        _publisher_url_for_doi(url, doi)
        and (verification_urls is None or url in verification_urls)
        for url in evidence_urls
    )


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
        doi = str(entry.get("published_doi") or "").strip().casefold()
        if not DOI_PATTERN.fullmatch(doi):
            raise ValueError("Scholar DOI alias needs a valid published DOI")
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
        relationship = str(entry.get("relationship") or "").strip()
        if relationship not in {"preprint_of", "version_of_record"}:
            raise ValueError("Scholar DOI alias needs a supported version relationship")
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
        verified_at = str(entry.get("verified_at") or "").strip()
        try:
            dt.date.fromisoformat(verified_at)
        except ValueError as error:
            raise ValueError(
                "Scholar DOI alias verified_at must use YYYY-MM-DD"
            ) from error
        short_title_override = entry.get("allow_short_title_region_override", False)
        if type(short_title_override) is not bool:
            raise ValueError("Scholar short-title override must be a boolean")
        cell_review = CELL_REVIEWED_SPARSE_TITLES.get(doi)
        if short_title_override and (
            year is None
            or (len(title) >= 30 and cell_review is None)
            or (
                cell_review is not None
                and (
                    (scholar_id, author_pub_id) != cell_review["identity"]
                    or (title, year) != (cell_review["title"], cell_review["year"])
                )
            )
            or not _short_title_evidence_matches(doi, evidence_urls)
        ):
            raise ValueError(
                "Scholar short-title override needs a short exact title, year, "
                "and reviewed publisher/identifier evidence for its DOI"
            )
        key = (scholar_id, author_pub_id)
        if key in aliases:
            raise ValueError(f"Duplicate Scholar DOI alias: {key}")
        aliases[key] = ReviewedDoiAlias(
            doi=doi,
            observed_title=title,
            observed_year=year,
            relationship=relationship,
            evidence_urls=tuple(evidence_urls),
            verified_at=verified_at,
            allow_short_title_region_override=short_title_override,
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
        if alias.allow_short_title_region_override:
            source = urlparse(str(row.get("source_url") or ""))
            query = parse_qs(source.query)
            if (
                source.scheme != "https"
                or source.hostname != "scholar.google.com"
                or source.path != "/citations"
                or query.get("user") != [scholar_id]
                or query.get("citation_for_view") != [author_pub_id]
                or not isinstance(row.get("authors"), str)
                or not row["authors"].strip()
                or not isinstance(row.get("venue"), str)
                or not row["venue"].strip()
            ):
                raise ValueError(
                    "Scholar short-title override needs the exact citation "
                    f"and metadata: {key}"
                )
        if key in exclusions:
            raise ValueError(f"Scholar DOI alias also has a reviewed exclusion: {key}")
        source_doi = str(row.get("source_doi") or "").strip().casefold()
        if source_doi and source_doi != alias.doi:
            raise ValueError(f"Scholar DOI alias conflicts with a source DOI: {key}")
