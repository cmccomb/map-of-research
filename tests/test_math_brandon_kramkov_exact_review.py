"""Full-capture Brandon/Kramkov rows exercise exact eligibility and work joins."""

import json
from pathlib import Path

import pandas

from map_of_research.dataset import build_dataset_tables
from map_of_research.doi_aliases import (
    DEFAULT_DOI_ALIASES_PATH,
    load_doi_aliases,
    validate_doi_aliases,
)
from map_of_research.exclusions import DEFAULT_EXCLUSIONS_PATH, load_exclusions
from map_of_research.registry import load_registry
from tests.registry_helpers import write_registry

FIXTURE_PATH = Path(__file__).parent / "fixtures/math_brandon_kramkov_exact_rows.json"
BRANDON = "9eegHpAAAAAJ"
KRAMKOV = "InIipCMAAAAJ"
ALIAS_SUFFIXES = {
    BRANDON: {
        "2osOgNQ5qMEC",
        "YsMSGLbcyi4C",
        "u-x6o8ySG0sC",
        "LkGwnXOMwfcC",
        "roLk4NBRz8UC",
        "9yKSN-GCB0IC",
        "WF5omc3nYNoC",
    },
    KRAMKOV: {
        "zYLM7Y9cAGgC",
        "d1gkVwhDpl0C",
        "5nxA0vEk-isC",
        "u5HHmVD_uO8C",
    },
}
EXCLUSION_SUFFIXES = {
    BRANDON: {"MXK_kJrjxJIC"},
    KRAMKOV: {
        "4TOpqqG69KYC",
        "qxL8FJ1GzNcC",
        "qjMakFHDy7sC",
        "Zph67rFs4hoC",
        "KlAtU1dfN6UC",
    },
}


def _keys(suffixes: dict[str, set[str]]) -> set[tuple[str, str]]:
    return {
        (scholar_id, f"{scholar_id}:{suffix}")
        for scholar_id, values in suffixes.items()
        for suffix in values
    }


def _capture_rows() -> list[dict[str, object]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["rows"]


def _observations() -> pandas.DataFrame:
    output = []
    for row in _capture_rows():
        scholar_id = row["scholar_id"]
        display_name = "Deborah Brandon" if scholar_id == BRANDON else "Dmitry Kramkov"
        output.append(
            {
                **row,
                "person_id": "person-brandon"
                if scholar_id == BRANDON
                else "person-kramkov",
                "display_name": display_name,
                "faculty": display_name,
                "department_ids": ["math"],
                "memberships": [
                    {"department_id": "math", "role": "faculty", "included": True}
                ],
                "citation": row["venue"],
                "citation_count": 1,
                "embedding": [0.0, 1.0],
                "external_coauthors": [],
                "source_type": "scholar",
                "source_url": f"https://scholar.google.com/citations?user={scholar_id}",
                "source_record_json": json.dumps(row, ensure_ascii=False),
                "source_doi": "",
                "source_verified_at": "",
                "source_verification_urls": [],
                "fetched_at_utc": "2026-10-25T04:25:59+00:00",
            }
        )
    return pandas.DataFrame(output)


def test_exact_decisions_match_captured_scholar_rows() -> None:
    captured = {
        (row["scholar_id"], row["author_pub_id"]): row for row in _capture_rows()
    }
    aliases = load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    alias_keys = _keys(ALIAS_SUFFIXES)
    exclusion_keys = _keys(EXCLUSION_SUFFIXES)
    assert len(alias_keys) == 11
    assert len(exclusion_keys) == 6
    assert not alias_keys & exclusion_keys
    for key in alias_keys:
        assert aliases[key].observed_title == captured[key]["title"]
        assert aliases[key].observed_year == captured[key]["year"]
    for key in exclusion_keys:
        assert exclusions[key].reason_code == "nonpublication"
        assert captured[key]["title"]
    for key in (
        (BRANDON, f"{BRANDON}:5nxA0vEk-isC"),
        (BRANDON, f"{BRANDON}:IjCSPb-OGe4C"),
        (KRAMKOV, f"{KRAMKOV}:0EnyYjriUFMC"),
    ):
        assert key not in aliases
        assert key not in exclusions


def test_reviewed_versions_join_and_raw_observations_survive(tmp_path) -> None:
    registry_paths = write_registry(
        tmp_path / "registry",
        people=[
            {
                "person_id": "person-brandon",
                "display_name": "Deborah Brandon",
                "scholar_id": BRANDON,
            },
            {
                "person_id": "person-kramkov",
                "display_name": "Dmitry Kramkov",
                "scholar_id": KRAMKOV,
            },
        ],
        memberships=[
            {
                "person_id": person_id,
                "department_id": "math",
                "role": "faculty",
                "included": "true",
                "source_url": "https://www.cmu.edu/math/people/faculty/index.html",
                "verified_at": "2026-10-25",
            }
            for person_id in ("person-brandon", "person-kramkov")
        ],
        departments=[
            {
                "department_id": "math",
                "school_id": "mcs",
                "title": "Mathematical Sciences",
                "directory_url": "https://www.cmu.edu/math/people/faculty/index.html",
            }
        ],
        schools=[
            {
                "school_id": "mcs",
                "title": "Mellon College of Science",
                "directory_url": "https://www.cmu.edu/mcs/",
            }
        ],
    )
    registry = load_registry(*registry_paths)
    raw = _observations()
    original = raw[["title", "year", "venue", "source_record_json"]].copy()
    all_aliases = load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)
    aliases = {key: all_aliases[key] for key in _keys(ALIAS_SUFFIXES)}
    all_exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    exclusions = {key: all_exclusions[key] for key in _keys(EXCLUSION_SUFFIXES)}
    validate_doi_aliases(raw, aliases, exclusions)
    tables = build_dataset_tables(raw, registry, exclusions, aliases)
    assert raw[["title", "year", "venue", "source_record_json"]].equals(original)
    publications = tables["profile_publications"].set_index("author_pub_id")
    works = tables["works"].set_index("work_id")
    assert len(publications) == 20
    assert len(works) == 14
    assert (
        sum(
            not publications.loc[citation_id, "map_eligible"]
            for _, citation_id in _keys(EXCLUSION_SUFFIXES)
        )
        == 6
    )
    for _, citation_id in _keys(EXCLUSION_SUFFIXES):
        publication = publications.loc[citation_id]
        assert publication["review_exclusion_code"] == "nonpublication"
        assert not publication["map_eligible"]
        assert not works.loc[publication["work_id"], "map_eligible"]

    def same_work(scholar_id: str, *suffixes: str):
        citation_ids = [f"{scholar_id}:{suffix}" for suffix in suffixes]
        work_ids = {
            publications.loc[citation_id, "work_id"] for citation_id in citation_ids
        }
        assert len(work_ids) == 1
        return works.loc[next(iter(work_ids))]

    heat = same_work(BRANDON, "2osOgNQ5qMEC", "YsMSGLbcyi4C", "u-x6o8ySG0sC")
    assert heat["doi"] == "10.1137/0522005"
    assert heat["year"] == 1991
    assert heat["venue"] == "SIAM Journal on Mathematical Analysis 22 (1), 72-106, 1991"
    assert same_work(BRANDON, "LkGwnXOMwfcC", "roLk4NBRz8UC")["doi"] == (
        "10.1117/12.174243"
    )
    assert same_work(BRANDON, "9yKSN-GCB0IC", "WF5omc3nYNoC")["doi"] == (
        "10.1117/12.148401"
    )
    hedging = same_work(KRAMKOV, "zYLM7Y9cAGgC", "d1gkVwhDpl0C")
    assert hedging["doi"] == "10.1016/j.spa.2007.04.014"
    assert hedging["year"] == 2007
    martingales = same_work(KRAMKOV, "5nxA0vEk-isC", "u5HHmVD_uO8C")
    assert martingales["doi"] == "10.1016/j.spa.2013.06.017"
    assert martingales["year"] == 2014

    for citation_id in (
        f"{BRANDON}:5nxA0vEk-isC",
        f"{BRANDON}:IjCSPb-OGe4C",
        f"{KRAMKOV}:0EnyYjriUFMC",
    ):
        assert publications.loc[citation_id, "map_eligible"]
