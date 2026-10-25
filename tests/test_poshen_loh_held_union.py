"""Guard the held Loh union against source loss and graph promotion."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from scripts.build_poshen_loh_held_union import (
    ARXIV_FILE,
    AUTHOR_FILE,
    ELEMENTS_FILE,
    REVIEWS,
    UNION_FILE,
    UNION_MANIFEST,
    build,
)

ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_deterministic_held_union_preserves_every_pinned_row() -> None:
    union_bytes, cases_bytes, manifest_bytes = build()
    assert union_bytes == (REVIEWS / UNION_FILE).read_bytes()
    assert (
        cases_bytes
        == (
            REVIEWS / "2026-10-25-math-poshen-loh-held-reconciliation-cases.json"
        ).read_bytes()
    )
    assert manifest_bytes == (REVIEWS / UNION_MANIFEST).read_bytes()
    observations = [json.loads(line) for line in union_bytes.splitlines()]
    assert len(observations) == 169
    assert Counter(row["collection"] for row in observations) == {
        "author_site": 46,
        "cmu_elements": 84,
        "arxiv_exact_author": 38,
        "publisher_registered_doi": 1,
    }
    indexed = {row["observation_id"]: row for row in observations}
    assert len(indexed) == 169
    for row in _rows(REVIEWS / AUTHOR_FILE):
        key = f"AUTHOR:{row['source_section']}:{row['section_row']}"
        assert indexed[key]["raw_fields"] == row
    for row in _rows(REVIEWS / ELEMENTS_FILE):
        assert indexed[f"ELEMENTS:{row['display_row']}"]["raw_fields"] == row
    for row in _rows(REVIEWS / ARXIV_FILE):
        assert indexed[f"ARXIV:{row['arxiv_id']}"]["raw_fields"] == row
    assert {row["graph_eligible"] for row in observations} == {False}
    assert {row["hold_reason"] for row in observations} == {
        "no_verified_scholar_or_whole_career_bibliography"
    }
    manifest = json.loads(manifest_bytes)
    assert manifest["release_eligible"] is False
    assert manifest["scholar_account_status"] == "no_verified_account"
    assert manifest["coauthor_display_truncated_elements_rows"] == 45
    assert manifest["elements_only_display_rows"] == [1, 2, 3, 4, 5, 12, 20]
    assert indexed["CROSSREF:10.1201/b16132-60"]["genre_as_source_displays"] == (
        "book-chapter"
    )
    edited = indexed["AUTHOR:Olympiad related:5"]
    assert edited["genre_as_source_displays"] == "Olympiad related"
    assert edited["genre_interpretation"] == "edited_book"


def test_distinct_manifestations_and_unresolved_identity_stay_held() -> None:
    observations = {
        row["observation_id"]: row
        for row in (
            json.loads(line) for line in (REVIEWS / UNION_FILE).read_text().splitlines()
        )
    }
    assert observations["ELEMENTS:82"]["genre_as_source_displays"] == "CHAPTER"
    assert observations["ELEMENTS:84"]["genre_as_source_displays"] == (
        "JOURNAL ARTICLE"
    )
    assert (
        observations["ELEMENTS:82"]["raw_fields"]["doi_as_displayed"]
        != (observations["ELEMENTS:84"]["raw_fields"]["doi_as_displayed"])
    )
    assert observations["ARXIV:2011.04650"]["date_as_source_displays"] == ("2020-11-08")
    assert observations["ELEMENTS:1"]["raw_fields"]["year_in_citation"] == ("2026")
    cases = json.loads(
        (
            REVIEWS / "2026-10-25-math-poshen-loh-held-reconciliation-cases.json"
        ).read_text()
    )
    assert {case["case_id"] for case in cases} >= {
        "2003_2006_shortest_paths",
        "2013_probabilistic_method_chapter",
        "2023_2024_covid_survey",
    }
    assert (
        next(case for case in cases if case["case_id"] == "2023_2024_covid_survey")[
            "status"
        ]
        == "possible_version_relationship_unresolved"
    )
    with (ROOT / "registry/people.csv").open(newline="", encoding="utf-8") as handle:
        people = list(csv.DictReader(handle))
    person = next(
        row for row in people if row["person_id"] == "person-poshenloh-b545d989"
    )
    assert person["scholar_id"] == ""
