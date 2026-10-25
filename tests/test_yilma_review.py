"""Exact Yilma Scholar version links and publisher-verified bylines."""

from __future__ import annotations

import json

import pandas
import pytest

from map_of_research.corrections import (
    DEFAULT_CORRECTIONS_PATH,
    apply_corrections,
    load_corrections,
)
from map_of_research.dataset import build_dataset_tables
from map_of_research.doi_aliases import (
    DEFAULT_DOI_ALIASES_PATH,
    load_doi_aliases,
    validate_doi_aliases,
)
from map_of_research.registry import load_registry

SCHOLAR_ID = "AlNKTGsAAAAJ"
FULL_AUTHORS = (
    "Daniel Král', Chun-Hung Liu, Jean-Sébastien Sereni, "
    "Peter Whalen, Zelealem B. Yilma"
)
VERSION_ROWS = {
    "IjCSPb-OGe4C": (
        "Multiple Petersen subdivisions in permutation graphs",
        2012,
        "10.37236/2239",
        "preprint_of",
    ),
    "eQOLeE2rZwMC": (
        "A new bound for the 2/3 conjecture",
        2012,
        "10.1017/s0963548312000612",
        "preprint_of",
    ),
    "u-x6o8ySG0sC": (
        "A new bound for the 2/3 conjecture",
        2013,
        "10.1017/s0963548312000612",
        "version_of_record",
    ),
    "MXK_kJrjxJIC": (
        "The minimum number of spanning trees in regular multigraphs",
        2022,
        "10.37236/10911",
        "version_of_record",
    ),
}


def test_yilma_aliases_bind_only_four_verified_source_rows() -> None:
    reviewed = load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)
    yilma = {key: value for key, value in reviewed.items() if key[0] == SCHOLAR_ID}
    assert {key[1].split(":")[-1] for key in yilma} == set(VERSION_ROWS)
    rows = []
    for suffix, (title, year, doi, relationship) in VERSION_ROWS.items():
        key = (SCHOLAR_ID, f"{SCHOLAR_ID}:{suffix}")
        alias = yilma[key]
        assert (alias.observed_title, alias.observed_year) == (title, year)
        assert (alias.doi, alias.relationship) == (doi, relationship)
        rows.append(
            {
                "source_type": "scholar",
                "scholar_id": SCHOLAR_ID,
                "author_pub_id": key[1],
                "title": title,
                "year": year,
                "source_doi": "",
            }
        )
    validate_doi_aliases(pandas.DataFrame(rows), yilma, {})


def test_yilma_corrected_bylines_join_two_versions_without_mutating_raw_source() -> (
    None
):
    registry = load_registry()
    person_id = "person-zelealemyilma-fb547edb"
    assert registry.people_by_id[person_id].display_name == "Zelealem Yilma"
    reviewed_corrections = load_corrections(DEFAULT_CORRECTIONS_PATH)
    keys = [
        (SCHOLAR_ID, f"{SCHOLAR_ID}:eQOLeE2rZwMC"),
        (SCHOLAR_ID, f"{SCHOLAR_ID}:u-x6o8ySG0sC"),
    ]
    corrections = {key: reviewed_corrections[key] for key in keys}
    observations = pandas.DataFrame(
        [
            {
                "scholar_id": SCHOLAR_ID,
                "person_id": person_id,
                "display_name": "Zelealem Yilma",
                "faculty": "Zelealem Yilma",
                "department_ids": ["math"],
                "memberships": [
                    {"department_id": "math", "role": "teaching", "included": True}
                ],
                "author_pub_id": key[1],
                **correction.expected,
                "citation": "",
                "citation_count": 1,
                "source_url": "https://scholar.google.com/citations?user=AlNKTGsAAAAJ",
                "source_record_json": json.dumps(correction.expected),
                "fetched_at_utc": "2026-10-25T00:00:00+00:00",
                "source_type": "scholar",
                "source_doi": "",
                "source_native_id": "",
                "source_verified_at": "",
                "source_verification_urls": [],
                "external_coauthors": [],
                "embedding": [0.0] * 768,
            }
            for key, correction in corrections.items()
        ]
    )
    aliases = {
        key: value
        for key, value in load_doi_aliases(DEFAULT_DOI_ALIASES_PATH).items()
        if key in keys
    }
    validate_doi_aliases(observations, aliases, {})
    effective, count = apply_corrections(observations, corrections)
    assert count == 2
    assert effective["authors"].tolist() == [FULL_AUTHORS, FULL_AUTHORS]
    assert effective["metadata_correction_fields"].tolist() == ["authors", "authors"]
    assert effective["source_record_json"].equals(observations["source_record_json"])
    tables = build_dataset_tables(effective, registry, doi_aliases=aliases)
    raw = tables["profile_publications"]
    works = tables["works"]
    assert len(raw) == 2
    assert len(works) == 1
    assert set(raw["work_match_method"]) == {"reviewed_version_alias"}
    assert works.iloc[0]["doi"] == "10.1017/s0963548312000612"
    assert works.iloc[0]["authors"] == FULL_AUTHORS
    assert works.iloc[0]["year"] == 2013
    assert set(raw["source_record_json"]) == set(observations["source_record_json"])

    with pytest.raises(ValueError, match="source authors changed"):
        apply_corrections(
            observations.assign(authors="Publisher changed the byline"),
            corrections,
        )
