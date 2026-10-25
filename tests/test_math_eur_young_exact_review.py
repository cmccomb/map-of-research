"""Full-capture Eur/Young rows exercise exact producer repairs and work joins."""

import json
from pathlib import Path

import pandas

from map_of_research.corrections import (
    DEFAULT_CORRECTIONS_PATH,
    apply_corrections,
    load_corrections,
)
from map_of_research.dataset import build_dataset_tables, normalize_title
from map_of_research.doi_aliases import (
    DEFAULT_DOI_ALIASES_PATH,
    load_doi_aliases,
    validate_doi_aliases,
)
from map_of_research.exclusions import DEFAULT_EXCLUSIONS_PATH, load_exclusions
from map_of_research.registry import load_registry
from tests.registry_helpers import write_registry

FIXTURE_PATH = Path(__file__).parent / "fixtures/math_eur_young_exact_rows.json"
EUR = "8Ufu7vkAAAAJ"
YOUNG = "l0rzNYsAAAAJ"
REVIEWED_ALIAS_SUFFIXES = {
    EUR: {"W7OEmFMy1HYC", "WF5omc3nYNoC", "_FxGoFyzp5QC", "0EnyYjriUFMC"},
    YOUNG: {"d1gkVwhDpl0C", "aqlVkmm33-oC", "W7OEmFMy1HYC"},
}
REVIEWED_CORRECTION_SUFFIXES = {
    EUR: {"hqOjcs7Dif8C"},
    YOUNG: {"ufrVoPGSRksC", "mVmsd5A6BfQC"},
}


def _capture_rows() -> list[dict[str, object]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["rows"]


def _keys(suffixes: dict[str, set[str]]) -> set[tuple[str, str]]:
    return {
        (scholar_id, f"{scholar_id}:{suffix}")
        for scholar_id, values in suffixes.items()
        for suffix in values
    }


def _observations() -> pandas.DataFrame:
    output = []
    for row in _capture_rows():
        scholar_id = row["scholar_id"]
        display_name = "Christopher Eur" if scholar_id == EUR else "Michael Young"
        output.append(
            {
                **row,
                "person_id": "person-eur" if scholar_id == EUR else "person-young",
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
                "fetched_at_utc": "2026-10-25T04:19:32+00:00",
            }
        )
    return pandas.DataFrame(output)


def test_reviewed_entries_guard_exact_saved_capture_rows() -> None:
    captured = {
        (row["scholar_id"], row["author_pub_id"]): row for row in _capture_rows()
    }
    aliases = load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)
    corrections = load_corrections(DEFAULT_CORRECTIONS_PATH)
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    alias_keys = _keys(REVIEWED_ALIAS_SUFFIXES)
    correction_keys = _keys(REVIEWED_CORRECTION_SUFFIXES)
    assert len(alias_keys) == 7
    assert len(correction_keys) == 3
    for key in alias_keys:
        assert aliases[key].observed_title == captured[key]["title"]
        assert aliases[key].observed_year == captured[key]["year"]
    for key in correction_keys:
        assert corrections[key].expected == {
            field: captured[key][field] for field in ("title", "year", "venue")
        }
    proposal = (YOUNG, f"{YOUNG}:UeHWp8X0CEIC")
    assert captured[proposal]["title"].endswith("Proposal for research groups at IMA")
    assert captured[proposal]["year"] == captured[proposal]["venue"] == ""
    assert exclusions[proposal].reason_code == "nonpublication"
    assert exclusions[proposal].evidence_url.endswith(
        f"citation_for_view={proposal[1]}"
    )
    assert proposal not in aliases
    assert proposal not in corrections


def test_complete_capture_repair_joins_works_without_mutating_raw_rows(
    tmp_path,
) -> None:
    registry_paths = write_registry(
        tmp_path / "registry",
        people=[
            {
                "person_id": "person-eur",
                "display_name": "Christopher Eur",
                "scholar_id": EUR,
            },
            {
                "person_id": "person-young",
                "display_name": "Michael Young",
                "scholar_id": YOUNG,
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
            for person_id in ("person-eur", "person-young")
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
    aliases = {key: all_aliases[key] for key in _keys(REVIEWED_ALIAS_SUFFIXES)}
    proposal = (YOUNG, f"{YOUNG}:UeHWp8X0CEIC")
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    selected_exclusions = {proposal: exclusions[proposal]}
    validate_doi_aliases(raw, aliases, selected_exclusions)
    effective, applied = apply_corrections(
        raw, load_corrections(DEFAULT_CORRECTIONS_PATH)
    )
    assert applied == 3
    assert raw[["title", "year", "venue", "source_record_json"]].equals(original)
    assert effective["source_record_json"].equals(raw["source_record_json"])
    tables = build_dataset_tables(effective, registry, selected_exclusions, aliases)
    publications = tables["profile_publications"].set_index("author_pub_id")
    works = tables["works"].set_index("work_id")
    assert len(publications) == 15
    assert len(works) == 8
    proposal_publication = publications.loc[proposal[1]]
    assert not proposal_publication["map_eligible"]
    assert proposal_publication["review_exclusion_code"] == "nonpublication"
    assert not works.loc[proposal_publication["work_id"], "map_eligible"]

    def same_work(scholar_id: str, *suffixes: str):
        ids = [f"{scholar_id}:{suffix}" for suffix in suffixes]
        work_ids = {publications.loc[citation_id, "work_id"] for citation_id in ids}
        assert len(work_ids) == 1
        return works.loc[next(iter(work_ids))]

    eur_reciprocal = same_work(EUR, "W7OEmFMy1HYC", "WF5omc3nYNoC")
    assert eur_reciprocal["doi"] == "10.4418/2021.76.2.10"
    assert eur_reciprocal["year"] == 2021
    assert eur_reciprocal["venue"] == "Le Matematiche 76 (2), 447-459, 2021"

    eur_survey = same_work(EUR, "_FxGoFyzp5QC", "0EnyYjriUFMC")
    assert eur_survey["doi"] == "10.1090/bull/1803"
    assert eur_survey["year"] == 2024
    assert (
        eur_survey["title"]
        == "Essence of independence: Hodge theory of matroids since June Huh"
    )

    young_power = same_work(YOUNG, "d1gkVwhDpl0C", "aqlVkmm33-oC", "W7OEmFMy1HYC")
    assert young_power["doi"] == "10.1007/s10878-016-0103-z"
    assert young_power["year"] == 2017
    assert young_power["title"] == (
        "Note on power propagation time and lower bounds for the power "
        "domination number"
    )

    young_arxiv = same_work(YOUNG, "ufrVoPGSRksC", "mVmsd5A6BfQC")
    assert young_arxiv["doi"] == ""
    assert young_arxiv["year"] == 2018
    assert young_arxiv["title"] == (
        "Lower Bounds for the Exponential Domination Number of C_m × C_n"
    )
    assert publications.loc[f"{YOUNG}:ufrVoPGSRksC", "year"] == 2018
    assert publications.loc[f"{YOUNG}:mVmsd5A6BfQC", "year"] == 2018

    for suffixes in (
        ("4DMP91E08xMC", "LkGwnXOMwfcC"),
        ("R3hNpaxXUhUC", "dhFuZR0502QC"),
    ):
        same_work(YOUNG, *suffixes)
        assert not any((YOUNG, f"{YOUNG}:{suffix}") in aliases for suffix in suffixes)


def test_identical_title_and_year_need_no_extra_alias() -> None:
    rows = {row["author_pub_id"]: row for row in _capture_rows()}
    for first, second in (
        ("4DMP91E08xMC", "LkGwnXOMwfcC"),
        ("R3hNpaxXUhUC", "dhFuZR0502QC"),
    ):
        left = rows[f"{YOUNG}:{first}"]
        right = rows[f"{YOUNG}:{second}"]
        assert normalize_title(left["title"]) == normalize_title(right["title"])
        assert left["year"] == right["year"]
