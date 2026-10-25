"""Deterministically assemble Loh source observations for review only.

This module does not feed the collector or publisher. It preserves one row per
observed source record and attaches candidate cross-references without merging
versions or asserting a complete researcher profile.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = ROOT / "registry/reviews"
SOURCES = ROOT / "data/source_reviews/po-shen-loh-2026-10-25"
PREFIX = "2026-10-25-math-poshen-loh-"
AUTHOR_FILE = PREFIX + "author-site-observations.csv"
ELEMENTS_FILE = PREFIX + "elements-observations.csv"
AUTHOR_ELEMENTS_FILE = PREFIX + "author-elements-crosswalk.csv"
ELEMENTS_ONLY_FILE = PREFIX + "elements-only-crosswalk.csv"
ARXIV_FILE = PREFIX + "arxiv-author-crosswalk.csv"
BIB_MANIFEST = PREFIX + "bibliography-manifest.json"
ARXIV_MANIFEST = PREFIX + "arxiv-author-crosswalk-manifest.json"
AUTHOR_HTML = PREFIX + "author-page.html"
UNION_FILE = PREFIX + "held-source-union.jsonl"
UNION_MANIFEST = PREFIX + "held-source-union-manifest.json"
CASES_FILE = PREFIX + "held-reconciliation-cases.json"
PERSON_ID = "person-poshenloh-b545d989"
HOLD = "no_verified_scholar_or_whole_career_bibliography"
DOI_EVIDENCE = {
    "chapter_2013": "crossref-2013-probabilistic-method.json",
    "journal_2003": "crossref-2003-shortest-paths-journal.json",
    "chapter_2006": "crossref-2006-shortest-paths-chapter.json",
    "lavrov_2016": "crossref-2016-lavrov-loh.json",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(value: str, prefix: str) -> list[str]:
    return [f"{prefix}{part}" for part in value.split(";") if part]


def _author_id(section: str, row: str) -> str:
    return f"AUTHOR:{section}:{row}"


def _arxiv_author_ids(value: str) -> list[str]:
    return [_author_id(*part.split(":", 1)) for part in value.split(";") if part]


def _record(
    observation_id: str,
    collection: str,
    source_file: str,
    source_url: str,
    source_sha256: str,
    title: str,
    genre: str,
    date: str,
    candidate_links: list[str],
    raw_fields: dict[str, str],
    genre_interpretation: str | None = None,
) -> dict:
    return {
        "observation_id": observation_id,
        "person_id": PERSON_ID,
        "collection": collection,
        "source_file": source_file,
        "source_url": source_url,
        "source_sha256": source_sha256,
        "title_as_source_displays": title,
        "genre_as_source_displays": genre,
        "genre_interpretation": genre_interpretation,
        "date_as_source_displays": date,
        "candidate_correspondences": sorted(set(candidate_links)),
        "raw_fields": raw_fields,
        "graph_eligible": False,
        "hold_reason": HOLD,
    }


def build() -> tuple[bytes, bytes, bytes]:
    bibliography = _json(REVIEWS / BIB_MANIFEST)
    arxiv_manifest = _json(REVIEWS / ARXIV_MANIFEST)
    assert bibliography["person_id"] == PERSON_ID
    assert bibliography["release_eligible"] is False
    assert bibliography["scholar_account_status"] == "no_verified_account"
    assert bibliography["author_site"]["observed_entries"] == 46
    assert bibliography["cmu_elements"]["displayed_records"] == 84
    assert bibliography["cmu_elements"]["coauthors_truncated_rows"] == 45
    assert bibliography["cmu_elements"]["strict_normalized_title_duplicates"] == 29
    assert arxiv_manifest["captured_entries"] == 38
    assert arxiv_manifest["opensearch_total_results"] == 38
    assert (
        _sha(REVIEWS / AUTHOR_FILE) == bibliography["author_site"]["review_csv_sha256"]
    )
    assert _sha(REVIEWS / AUTHOR_HTML) == bibliography["author_site"]["html_sha256"]
    assert (
        _sha(REVIEWS / ELEMENTS_FILE)
        == bibliography["cmu_elements"]["review_csv_sha256"]
    )

    author = _csv(REVIEWS / AUTHOR_FILE)
    elements = _csv(REVIEWS / ELEMENTS_FILE)
    crosswalk = _csv(REVIEWS / AUTHOR_ELEMENTS_FILE)
    elements_only = _csv(REVIEWS / ELEMENTS_ONLY_FILE)
    arxiv = _csv(REVIEWS / ARXIV_FILE)
    assert (
        len(author),
        len(elements),
        len(crosswalk),
        len(elements_only),
        len(arxiv),
    ) == (
        46,
        84,
        46,
        7,
        38,
    )
    author_ids = [
        _author_id(row["source_section"], row["section_row"]) for row in author
    ]
    element_ids = [f"ELEMENTS:{row['display_row']}" for row in elements]
    arxiv_ids = [f"ARXIV:{row['arxiv_id']}" for row in arxiv]
    assert len(set(author_ids)) == 46
    assert element_ids == [f"ELEMENTS:{number}" for number in range(1, 85)]
    assert len(set(arxiv_ids)) == 38
    assert set(author_ids) == {
        _author_id(row["source_section"], row["section_row"]) for row in crosswalk
    }
    assert {int(row["cmu_elements_display_row"]) for row in elements_only} == {
        1,
        2,
        3,
        4,
        5,
        12,
        20,
    }
    author_to_elements = {
        _author_id(row["source_section"], row["section_row"]): _ids(
            row["cmu_elements_display_rows"], "ELEMENTS:"
        )
        for row in crosswalk
    }
    assert {link for links in author_to_elements.values() for link in links} <= set(
        element_ids
    )
    assert sum(bool(links) for links in author_to_elements.values()) == 45
    assert len({link for links in author_to_elements.values() for link in links}) == 77
    element_to_authors: dict[str, list[str]] = defaultdict(list)
    for author_id, links in author_to_elements.items():
        for link in links:
            element_to_authors[link].append(author_id)
    assert set(element_ids) - set(element_to_authors) == {
        f"ELEMENTS:{number}" for number in (1, 2, 3, 4, 5, 12, 20)
    }

    arxiv_to_elements: dict[str, list[str]] = {}
    arxiv_to_authors: dict[str, list[str]] = {}
    for row in arxiv:
        arxiv_id = f"ARXIV:{row['arxiv_id']}"
        arxiv_to_elements[arxiv_id] = _ids(
            row["cmu_elements_display_rows"], "ELEMENTS:"
        )
        arxiv_to_authors[arxiv_id] = _arxiv_author_ids(row["author_site_section_rows"])
        assert arxiv_to_elements[arxiv_id]
        assert set(arxiv_to_elements[arxiv_id]) <= set(element_ids)
        assert set(arxiv_to_authors[arxiv_id]) <= set(author_ids)
    assert sum(bool(links) for links in arxiv_to_authors.values()) == 35
    assert {key for key, links in arxiv_to_authors.items() if not links} == {
        "ARXIV:1910.06709",
        "ARXIV:2011.04650",
        "ARXIV:2504.10664",
    }
    element_to_arxiv: dict[str, list[str]] = defaultdict(list)
    author_to_arxiv: dict[str, list[str]] = defaultdict(list)
    for arxiv_id, links in arxiv_to_elements.items():
        for link in links:
            element_to_arxiv[link].append(arxiv_id)
        for link in arxiv_to_authors[arxiv_id]:
            author_to_arxiv[link].append(arxiv_id)

    dois = {
        key: _json(SOURCES / filename)["message"]
        for key, filename in DOI_EVIDENCE.items()
    }
    chapter = dois["chapter_2013"]
    assert chapter["DOI"] == "10.1201/b16132-60"
    assert chapter["type"] == "book-chapter"
    assert chapter["title"] == ["The Probabilistic Method"]
    assert chapter["page"] == "1026-1037"
    assert [(person["given"], person["family"]) for person in chapter["author"]] == [
        ("Alan", "Frieze"),
        ("Po-Shen", "Loh"),
    ]
    assert chapter["published"]["date-parts"] == [[2013, 11, 26]]
    assert all(
        "the probabilistic method" not in row["title_as_listed"].lower()
        for row in author
    )
    assert all(
        "the probabilistic method" not in row["title_as_displayed"].lower()
        for row in elements
    )
    assert dois["journal_2003"]["DOI"] == "10.7155/jgaa.00071"
    assert dois["journal_2003"]["type"] == "journal-article"
    assert dois["chapter_2006"]["DOI"] == "10.1142/9789812773296_0013"
    assert dois["chapter_2006"]["type"] == "book-chapter"
    assert dois["journal_2003"]["title"] == dois["chapter_2006"]["title"]
    assert dois["lavrov_2016"]["DOI"] == "10.1002/rsa.20592"
    assert dois["lavrov_2016"]["type"] == "journal-article"

    observations: list[dict] = []
    for row in author:
        observation_id = _author_id(row["source_section"], row["section_row"])
        observations.append(
            _record(
                observation_id,
                "author_site",
                AUTHOR_FILE,
                row["source_page_url"],
                row["source_page_sha256"],
                row["title_as_listed"],
                row["source_section"],
                row["year_first_in_bibliographic_tail"],
                author_to_elements[observation_id] + author_to_arxiv[observation_id],
                row,
                "edited_book"
                if observation_id == "AUTHOR:Olympiad related:5"
                else None,
            )
        )
    for row in elements:
        observation_id = f"ELEMENTS:{row['display_row']}"
        observations.append(
            _record(
                observation_id,
                "cmu_elements",
                ELEMENTS_FILE,
                row["source_page_url"],
                row["source_page_sha256"],
                row["title_as_displayed"],
                row["output_type_as_displayed"],
                row["year_in_citation"],
                element_to_authors[observation_id] + element_to_arxiv[observation_id],
                row,
            )
        )
    for row in arxiv:
        observation_id = f"ARXIV:{row['arxiv_id']}"
        observations.append(
            _record(
                observation_id,
                "arxiv_exact_author",
                ARXIV_FILE,
                row["arxiv_url"],
                arxiv_manifest["atom_sha256"],
                row["arxiv_title"],
                "preprint",
                row["submitted_date"],
                arxiv_to_elements[observation_id] + arxiv_to_authors[observation_id],
                row,
            )
        )
    observations.append(
        _record(
            "CROSSREF:10.1201/b16132-60",
            "publisher_registered_doi",
            DOI_EVIDENCE["chapter_2013"],
            "https://api.crossref.org/works/10.1201%2Fb16132-60",
            _sha(SOURCES / DOI_EVIDENCE["chapter_2013"]),
            chapter["title"][0],
            chapter["type"],
            "2013-11-26",
            [],
            {
                "DOI": chapter["DOI"],
                "title": chapter["title"],
                "author": chapter["author"],
                "type": chapter["type"],
                "container-title": chapter["container-title"],
                "published": chapter["published"],
                "page": chapter["page"],
            },
        )
    )
    assert len(observations) == 169
    assert len({row["observation_id"] for row in observations}) == 169
    assert all(row["graph_eligible"] is False for row in observations)
    assert {row["hold_reason"] for row in observations} == {HOLD}

    cases = [
        {
            "case_id": "2013_probabilistic_method_chapter",
            "status": "verified_source_observation_missing_from_both_lists",
            "observation_ids": ["CROSSREF:10.1201/b16132-60"],
            "decision": (
                "Publisher-registered 2013 book chapter by Frieze and Loh; "
                "separate from the author-page edited Olympiad book."
            ),
            "evidence": ["10.1201/b16132-60", bibliography["cv_evidence"][1]["url"]],
        },
        {
            "case_id": "2003_2006_shortest_paths",
            "status": "distinct_genres_verified_relationship_open",
            "observation_ids": ["AUTHOR:Papers:41", "ELEMENTS:82", "ELEMENTS:84"],
            "decision": (
                "Same title, author, and pages, but two registered DOIs identify "
                "a 2003 journal article and a 2006 book chapter. Derivation or "
                "reprint relationship is not asserted."
            ),
            "evidence": ["10.7155/jgaa.00071", "10.1142/9789812773296_0013"],
        },
        {
            "case_id": "2013_2016_anarchy_versions",
            "status": "distinct_genres_verified_common_lineage",
            "observation_ids": [
                "AUTHOR:Papers:22",
                "ELEMENTS:22",
                "ELEMENTS:41",
                "ELEMENTS:45",
                "ARXIV:1307.3113",
            ],
            "decision": (
                "Author citation and older CV identify the 2013 conference "
                "and 2016 journal versions; CMU also lists the 2013 preprint. "
                "Retain all manifestations."
            ),
            "evidence": [
                "10.1145/2729978",
                "10.1007/978-3-319-03536-9_17",
                "arXiv:1307.3113",
                bibliography["cv_evidence"][1]["url"],
            ],
        },
        {
            "case_id": "2020_f_saturated_versions",
            "status": "distinct_genres_verified_title_variant",
            "observation_ids": [
                "AUTHOR:Papers:5",
                "ELEMENTS:8",
                "ELEMENTS:10",
                "ARXIV:1907.01603",
            ],
            "decision": (
                "Author-linked arXiv ID and journal citation resolve the title "
                "expansion; journal and preprint remain separate manifestations."
            ),
            "evidence": ["10.1016/j.ejc.2020.103185", "arXiv:1907.01603"],
        },
        {
            "case_id": "rainbow_revision_date",
            "status": "first_submission_verified",
            "observation_ids": ["ELEMENTS:1", "ARXIV:2011.04650"],
            "decision": (
                "CMU's 2026 date is an arXiv v4 display date; the captured "
                "arXiv result gives first submission 2020-11-08."
            ),
            "evidence": ["arXiv:2011.04650"],
        },
        {
            "case_id": "lavrov_loh_title_and_year",
            "status": "doi_identity_verified_dates_distinct",
            "observation_ids": ["ELEMENTS:20"],
            "decision": (
                "Publisher DOI fixes the journal article identity. Crossref "
                "gives 2015 online publication while the CMU card and CV use "
                "the 2016 volume; CV reverses the first two title words."
            ),
            "evidence": ["10.1002/rsa.20592", bibliography["cv_evidence"][1]["url"]],
        },
        {
            "case_id": "2023_2024_covid_survey",
            "status": "possible_version_relationship_unresolved",
            "observation_ids": ["ELEMENTS:4", "ELEMENTS:5"],
            "decision": (
                "Seven-person bylines overlap, but titles, output types, and "
                "DOIs differ. Content-level relationship remains open."
            ),
            "evidence": ["10.4018/IJHISI.353901", "10.1101/2023.07.20.23292950"],
        },
        {
            "case_id": "edited_olympiad_volume",
            "status": "editor_versus_author_genre_verified",
            "observation_ids": ["AUTHOR:Olympiad related:5"],
            "decision": (
                "Author page and older CV identify Loh as one of three "
                "editors of a 2005 Olympiad volume. Do not treat it as a "
                "research journal article."
            ),
            "evidence": [bibliography["cv_evidence"][1]["url"]],
        },
    ]
    assert (
        next(row for row in elements if row["display_row"] == "82")["doi_as_displayed"]
        == dois["chapter_2006"]["DOI"]
    )
    assert (
        next(row for row in elements if row["display_row"] == "84")["doi_as_displayed"]
        == dois["journal_2003"]["DOI"]
    )
    assert (
        next(row for row in elements if row["display_row"] == "20")["doi_as_displayed"]
        == dois["lavrov_2016"]["DOI"]
    )
    assert (
        next(row for row in arxiv if row["arxiv_id"] == "2011.04650")["submitted_date"]
        == "2020-11-08"
    )
    assert {row["observation_id"] for row in observations} >= {
        observation_id for case in cases for observation_id in case["observation_ids"]
    }

    union_bytes = b"".join(
        (
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in observations
    )
    cases_bytes = (json.dumps(cases, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    input_names = [
        AUTHOR_FILE,
        ELEMENTS_FILE,
        AUTHOR_ELEMENTS_FILE,
        ELEMENTS_ONLY_FILE,
        ARXIV_FILE,
        BIB_MANIFEST,
        ARXIV_MANIFEST,
        AUTHOR_HTML,
    ]
    manifest = {
        "schema_version": 1,
        "person_id": PERSON_ID,
        "review_state": "held_incomplete_whole_researcher",
        "scholar_account_status": "no_verified_account",
        "release_eligible": False,
        "counts": dict(Counter(row["collection"] for row in observations)),
        "total_observations_not_works": len(observations),
        "author_rows_with_elements_candidate": 45,
        "elements_cards_with_author_candidate": 77,
        "elements_only_display_rows": [1, 2, 3, 4, 5, 12, 20],
        "arxiv_rows_with_elements_candidate": 38,
        "arxiv_rows_with_author_candidate": 35,
        "coauthor_display_truncated_elements_rows": 45,
        "source_input_sha256": {name: _sha(REVIEWS / name) for name in input_names},
        "supplemental_doi_sha256": {
            name: _sha(SOURCES / name) for name in DOI_EVIDENCE.values()
        },
        "union_file": UNION_FILE,
        "union_sha256": hashlib.sha256(union_bytes).hexdigest(),
        "cases_file": CASES_FILE,
        "cases_sha256": hashlib.sha256(cases_bytes).hexdigest(),
        "limits": [
            "No exact Google Scholar account is verified.",
            (
                "The 2021-stated author page omits later CMU records; the "
                "84-card CMU list is not verified as a complete career."
            ),
            (
                "CMU Elements displays truncated coauthors on 45 cards and "
                "can use revision dates."
            ),
            (
                "The prior arXiv capture pins a 38-row CSV and Atom hash but "
                "not the raw Atom body in Git."
            ),
            (
                "Candidate cross-references do not merge manifestations or "
                "authorize graph edges."
            ),
        ],
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    return union_bytes, cases_bytes, manifest_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="check committed outputs without writing"
    )
    args = parser.parse_args()
    generated = build()
    paths = (REVIEWS / UNION_FILE, REVIEWS / CASES_FILE, REVIEWS / UNION_MANIFEST)
    if args.check:
        for path, content in zip(paths, generated, strict=True):
            assert path.read_bytes() == content, path.name
    else:
        for path, content in zip(paths, generated, strict=True):
            path.write_bytes(content)
    print(
        json.dumps(
            {
                "observations_not_works": 169,
                "cases": 8,
                "release_eligible": False,
                "mode": "check" if args.check else "write",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
