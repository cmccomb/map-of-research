"""Evidence gate for Panangaden's held whole-researcher source review."""

from __future__ import annotations

import csv

from scripts.validate_panangaden_source_review import SOURCE_DIR, validate


def test_panangaden_source_review_retains_versions_and_thesis_genres() -> None:
    assert validate() == {
        "source_observations": 14,
        "provisional_work_families": 8,
        "orcid": 5,
        "arxiv": 7,
        "caltech_thesis": 1,
        "mcgill_thesis": 1,
    }
    with (SOURCE_DIR / "observations-14.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    masters = [row for row in rows if row["work_family"] == "masters-energy-fcs"]
    assert {row["source"] for row in masters} == {"arxiv", "mcgill_thesis"}
    assert {row["genre"] for row in masters} == {"preprint", "masters_thesis"}
    assert all(row["work_family"] != "energy-conservation" for row in masters)
    assert [
        row["source"] for row in rows if row["work_family"] == "doctoral-thesis"
    ] == ["caltech_thesis"]
