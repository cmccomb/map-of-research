"""Reviewed links from exact Scholar observations to published-work DOIs."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .io import load_json

DEFAULT_DOI_ALIASES_PATH = Path("data/curated/scholar-doi-aliases.json")
DOI_ALIAS_SCHEMA_VERSION = 1
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ReviewedAliasTarget:
    person_id: str
    author_pub_id: str
    observation_id: str
    observed_title: str
    observed_year: int | None
    source_record_sha256: str
    capture_path: Path
    capture_sha256: str


@dataclass(frozen=True)
class ReviewedDoiAlias:
    doi: str
    observed_title: str
    observed_year: int | None
    relationship: str
    evidence_urls: tuple[str, ...]
    verified_at: str
    observed_source_record_sha256: str = ""
    target_observation: ReviewedAliasTarget | None = None


def _observation_id(scholar_id: str, author_pub_id: str) -> str:
    value = f"{scholar_id}\0{author_pub_id}"
    return "observation-" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _exact_year(value: Any, expected: int | None) -> bool:
    import pandas

    if expected is None:
        return bool(pandas.isna(value))
    return bool(not pandas.isna(value) and value == expected)


def _reviewed_target(
    entry: dict[str, Any], doi: str, source_root: Path
) -> ReviewedAliasTarget | None:
    raw_target = entry.get("target_observation")
    if raw_target is None:
        if "observed_source_record_sha256" in entry:
            raise ValueError("Pinned Scholar source needs a target observation")
        return None
    scholar_hash = entry.get("observed_source_record_sha256")
    if not isinstance(scholar_hash, str) or not SHA256_PATTERN.fullmatch(scholar_hash):
        raise ValueError("Reviewed target needs the pinned Scholar source record")
    if not isinstance(raw_target, dict):
        raise ValueError("Reviewed alias target must be an object")
    person_id = raw_target.get("person_id")
    author_pub_id = raw_target.get("author_pub_id")
    observation_id = raw_target.get("observation_id")
    title = raw_target.get("observed_title")
    year = raw_target.get("observed_year")
    source_hash = raw_target.get("source_record_sha256")
    capture_value = raw_target.get("capture_path")
    capture_hash = raw_target.get("capture_sha256")
    capture_path = Path(capture_value) if isinstance(capture_value, str) else Path("")
    if (
        not isinstance(person_id, str)
        or not person_id.startswith("person-")
        or not isinstance(author_pub_id, str)
        or not author_pub_id.startswith(f"author-site:{person_id}:")
        or observation_id != _observation_id("", author_pub_id)
        or not isinstance(title, str)
        or not title.strip()
        or "observed_year" not in raw_target
        or (year is not None and type(year) is not int)
        or raw_target.get("source_doi") != doi
        or not isinstance(source_hash, str)
        or not SHA256_PATTERN.fullmatch(source_hash)
        or not isinstance(capture_value, str)
        or capture_path.parts[:2] != ("data", "author_sites")
        or capture_path.suffix != ".json"
        or ".." in capture_path.parts
        or not isinstance(capture_hash, str)
        or not SHA256_PATTERN.fullmatch(capture_hash)
    ):
        raise ValueError("Reviewed alias target identity or source pin is invalid")
    return ReviewedAliasTarget(
        person_id=person_id,
        author_pub_id=author_pub_id,
        observation_id=observation_id,
        observed_title=title,
        observed_year=year,
        source_record_sha256=source_hash,
        capture_path=source_root / capture_path,
        capture_sha256=capture_hash,
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
            observed_source_record_sha256=str(
                entry.get("observed_source_record_sha256") or ""
            ),
            target_observation=_reviewed_target(entry, doi, path.resolve().parents[2]),
        )
    return aliases


def validate_doi_aliases(
    frame: Any,
    aliases: dict[tuple[str, str], ReviewedDoiAlias],
    exclusions: Any,
) -> None:
    """Fail before publication if an alias is stale or points to an excluded row."""

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
        observed_year_matches = _exact_year(row["year"], alias.observed_year)
        if row["title"] != alias.observed_title or not observed_year_matches:
            raise ValueError(f"Scholar DOI alias observed title or year changed: {key}")
        if key in exclusions:
            raise ValueError(f"Scholar DOI alias also has a reviewed exclusion: {key}")
        source_doi = str(row.get("source_doi") or "").strip().casefold()
        if source_doi and source_doi != alias.doi:
            raise ValueError(f"Scholar DOI alias conflicts with a source DOI: {key}")
        target = alias.target_observation
        if target is None:
            continue
        source_record = row.get("source_record_json")
        if (
            not isinstance(source_record, str)
            or hashlib.sha256(source_record.encode("utf-8")).hexdigest()
            != alias.observed_source_record_sha256
        ):
            raise ValueError(f"Scholar DOI alias source record changed: {key}")
        target_rows = frame.loc[
            (frame["source_type"] == "author_site")
            & (frame["person_id"] == target.person_id)
            & (frame["author_pub_id"] == target.author_pub_id)
        ]
        if len(target_rows) != 1:
            raise ValueError(f"Scholar DOI alias needs one exact target row: {key}")
        target_row = target_rows.iloc[0]
        if (
            _observation_id("", target.author_pub_id) != target.observation_id
            or target_row["title"] != target.observed_title
            or not _exact_year(target_row["year"], target.observed_year)
            or str(target_row.get("source_doi") or "").strip().casefold() != alias.doi
        ):
            raise ValueError(f"Scholar DOI alias target identity changed: {key}")
        target_record = target_row.get("source_record_json")
        if (
            not isinstance(target_record, str)
            or hashlib.sha256(target_record.encode("utf-8")).hexdigest()
            != target.source_record_sha256
        ):
            raise ValueError(f"Scholar DOI alias target source record changed: {key}")
        capture = target.capture_path
        if (
            not capture.is_file()
            or hashlib.sha256(capture.read_bytes()).hexdigest() != target.capture_sha256
        ):
            raise ValueError(f"Scholar DOI alias target capture changed: {key}")
