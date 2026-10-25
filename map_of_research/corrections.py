"""Reviewed bibliographic repairs to exact Scholar observations.

The raw snapshot and ``source_record_json`` remain untouched. Corrections are
applied to effective publication fields only after snapshot validation and
before embeddings, quality assessment, and work matching.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .io import load_json

DEFAULT_CORRECTIONS_PATH = Path("data/curated/scholar-metadata-corrections.json")
CORRECTIONS_SCHEMA_VERSION = 1
REQUIRED_FIELDS = frozenset({"title", "year", "venue"})
FIELDS = REQUIRED_FIELDS | {"authors"}


@dataclass(frozen=True)
class ReviewedCorrection:
    expected: dict[str, str | int | None]
    corrected: dict[str, str | int]
    basis: str
    evidence_url: str
    verified_at: str


def _valid_fields(value: Any, *, complete: bool) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    fields = set(value)
    if complete and fields not in (REQUIRED_FIELDS, FIELDS):
        return False
    if not complete and not fields <= FIELDS:
        return False
    return all(
        (complete and field_value is None)
        or (type(field_value) is int and 1500 <= field_value <= 2100)
        if field == "year"
        else isinstance(field_value, str)
        and (bool(field_value.strip()) or (complete and field == "venue"))
        for field, field_value in value.items()
    )


def load_corrections(path: Path) -> dict[tuple[str, str], ReviewedCorrection]:
    """Load exact, source-backed repairs with pre-correction field guards."""

    source = load_json(path)
    if not isinstance(source, dict) or source.get("schema_version") != (
        CORRECTIONS_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported Scholar-correction schema: {path}")
    entries = source.get("corrections")
    if not isinstance(entries, list):
        raise ValueError("Scholar corrections must be a list")
    corrections: dict[tuple[str, str], ReviewedCorrection] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Scholar correction must be an object")
        scholar_id = str(entry.get("scholar_id") or "").strip()
        author_pub_id = str(entry.get("author_pub_id") or "").strip()
        if not scholar_id or not author_pub_id.startswith(f"{scholar_id}:"):
            raise ValueError("Scholar correction has an invalid observation identity")
        expected = entry.get("expected")
        corrected = entry.get("corrected")
        if not _valid_fields(expected, complete=True) or not _valid_fields(
            corrected, complete=False
        ):
            raise ValueError("Scholar correction has invalid bibliographic fields")
        if "authors" in corrected and "authors" not in expected:
            raise ValueError("Scholar author correction needs exact observed authors")
        if any(expected[field] == value for field, value in corrected.items()):
            raise ValueError("Scholar correction must change every listed field")
        basis = str(entry.get("basis") or "").strip()
        if not basis:
            raise ValueError("Scholar correction needs a specific basis")
        evidence_url = str(entry.get("evidence_url") or "").strip()
        parsed = urlparse(evidence_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Scholar correction needs an HTTPS evidence URL")
        verified_at = str(entry.get("verified_at") or "").strip()
        try:
            dt.date.fromisoformat(verified_at)
        except ValueError as error:
            raise ValueError(
                "Scholar correction verified_at must use YYYY-MM-DD"
            ) from error
        key = (scholar_id, author_pub_id)
        if key in corrections:
            raise ValueError(f"Duplicate Scholar correction: {key}")
        corrections[key] = ReviewedCorrection(
            expected=expected,
            corrected=corrected,
            basis=basis,
            evidence_url=evidence_url,
            verified_at=verified_at,
        )
    return corrections


def apply_corrections(
    frame: Any,
    corrections: dict[tuple[str, str], ReviewedCorrection],
) -> tuple[Any, int]:
    """Change effective fields only; retain source identity and raw source JSON."""

    import pandas

    scholar_rows = frame.loc[
        frame["source_type"] == "scholar", ["scholar_id", "author_pub_id"]
    ]
    present_profiles = {str(value) for value in scholar_rows["scholar_id"]}
    observed = Counter(
        (str(row.scholar_id), str(row.author_pub_id))
        for row in scholar_rows.itertuples(index=False)
    )
    for key in corrections:
        if key[0] in present_profiles and observed[key] != 1:
            raise ValueError(
                "Scholar correction ID must occur exactly once for a present "
                f"profile: {key} (found {observed[key]})"
            )

    output = frame.copy()
    for field in (
        "metadata_correction_fields",
        "metadata_correction_basis",
        "metadata_correction_evidence_url",
        "metadata_correction_verified_at",
    ):
        output[field] = ""
    applied = 0
    for index, row in output.iterrows():
        key = (str(row["scholar_id"]), str(row["author_pub_id"]))
        correction = corrections.get(key)
        if correction is None:
            continue
        if row["source_type"] != "scholar":
            raise ValueError(f"Scholar correction targets a non-Scholar row: {key}")
        for field, expected in correction.expected.items():
            matches = (
                bool(pandas.isna(row[field]))
                if expected is None
                else bool(not pandas.isna(row[field]) and row[field] == expected)
            )
            if not matches:
                raise ValueError(
                    f"Scholar correction source {field} changed for {key}: "
                    f"expected {expected!r}, got {row[field]!r}"
                )
        for field, value in correction.corrected.items():
            output.at[index, field] = value
        output.at[index, "metadata_correction_fields"] = ",".join(
            sorted(correction.corrected)
        )
        output.at[index, "metadata_correction_basis"] = correction.basis
        output.at[index, "metadata_correction_evidence_url"] = correction.evidence_url
        output.at[index, "metadata_correction_verified_at"] = correction.verified_at
        applied += 1
    return output, applied
