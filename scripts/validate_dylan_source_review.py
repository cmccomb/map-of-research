"""Validate Dylan Quintana's held, source-only career bibliography review."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

REVIEW_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/dylan-quintana-2026-10-25"
)
PERSON_ID = "person-dylanquintana-ca051f35"
OPENALEX_AUTHOR_ID = "https://openalex.org/A5015885996"
NAMESAKE_AUTHOR_ID = "https://openalex.org/A5148776245"
CORE_DOIS = {
    "10.1103/physrevb.92.054435",
    "10.1107/s1600576716016423",
    "10.1137/1.9781611975994.85",
    "10.1109/focs46700.2020.00048",
}
EXPECTED_COUNTS = {
    "peer_reviewed_journal_articles": 2,
    "peer_reviewed_conference_papers": 2,
    "doctoral_dissertations": 1,
    "preprint_manifestations": 3,
    "conference_abstract_manifestations": 5,
    "student_research_reports": 1,
    "duplicate_metadata_rows": 2,
    "openalex_exact_cluster_works_count_claim": 12,
    "openalex_filtered_work_rows": 11,
    "crosswalk_manifestations": 16,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _html(root: Path, name: str) -> bytes:
    return gzip.decompress((root / name).read_bytes())


def _crossref(root: Path, name: str) -> dict:
    return _json(root / name)["message"]


def validate_review(root: Path = REVIEW_ROOT) -> tuple[dict, list[dict[str, str]]]:
    """Preserve all raw rows, source decisions, and publication holds."""
    root = Path(root)
    manifest = _json(root / "manifest.json")
    _require(manifest["status"] == "held_source_review", "review must stay held")
    _require(manifest["person_id"] == PERSON_ID, "person ID changed")
    _require(manifest["verified_google_scholar_id"] is None, "Scholar ID unverified")
    for field in (
        "lifetime_complete",
        "release_eligible",
        "map_eligible",
        "collaboration_eligible",
    ):
        _require(manifest[field] is False, f"{field} must stay false")
    _require(manifest["counts"] == EXPECTED_COUNTS, "review counts changed")

    files = manifest["files"]
    actual = {path.name for path in root.iterdir() if path.is_file()}
    _require(
        actual == set(files) | {"README.md", "manifest.json"},
        "source file inventory differs from manifest",
    )
    for name, entry in files.items():
        data = (root / name).read_bytes()
        _require(len(data) == entry["bytes"], f"{name} length changed")
        _require(
            hashlib.sha256(data).hexdigest() == entry["sha256"],
            f"{name} hash changed",
        )

    profile = _html(root, "cmu-profile.html.gz")
    interview = _html(root, "cmu-interview.html.gz")
    thesis = _html(root, "chicago-dissertation.html.gz")
    _require(
        b"Dylan Quintana" in profile and b"University of Chicago" in profile,
        "CMU person identity changed",
    )
    _require(
        b"materials science and engineering" in interview.lower()
        and b"2021" in interview
        and b"research since completing his doctoral thesis" in interview,
        "CMU career and point-in-time research evidence changed",
    )
    _require(
        b"Adventures in Decoding Direct Sum Codes" in thesis
        and b"10.6082/uchicago.3032" in thesis
        and b"Dissertation" in thesis,
        "UChicago dissertation evidence changed",
    )
    _require(
        b"10.1103/PhysRevB.92.054435" in _html(root, "arxiv-materials-2014.html.gz")
        and b"Dylan Quintana" in _html(root, "arxiv-materials-2014.html.gz"),
        "2014 materials preprint relation changed",
    )
    _require(
        b"Full version of paper from SODA 2020"
        in _html(root, "arxiv-soda-2020.html.gz"),
        "SODA full-version relation changed",
    )
    _require(
        b"Dylan Quintana" in _html(root, "arxiv-focs-2020.html.gz")
        and b"2011.05500" in _html(root, "arxiv-focs-2020.html.gz"),
        "FOCS preprint identity changed",
    )
    _require(
        b"Carnegie Mellon University" in _html(root, "aps-prb-2015.html.gz")
        and b"Dylan Quintana" in _html(root, "nist-prb-2015.html.gz")
        and b"Dylan Quintana" in _html(root, "aps-march-2014.html.gz"),
        "materials affiliation or byline changed",
    )
    _require(
        b"Dylan J. Quintanta" in _html(root, "nist-bland-2016.html.gz"),
        "NIST spelling conflict changed",
    )
    _require(
        b"Dylan Quintana, MD" in _html(root, "idaho-md-namesake.html.gz")
        and b"College of Idaho" in _html(root, "idaho-md-namesake.html.gz"),
        "separate physician evidence changed",
    )

    author_search = _json(root / "openalex-author-search.json")
    authors = {author["id"]: author for author in author_search["results"]}
    _require(
        authors[OPENALEX_AUTHOR_ID]["works_count"] == 12
        and authors[OPENALEX_AUTHOR_ID]["orcid"] is None
        and NAMESAKE_AUTHOR_ID in authors,
        "OpenAlex identity clusters or count changed",
    )
    works = _json(root / "openalex-works.json")
    _require(
        works["meta"]["count"] == len(works["results"]) == 11,
        "OpenAlex filtered work count changed",
    )
    source_by_id = {work["id"].rsplit("/", 1)[-1]: work for work in works["results"]}
    _require(len(source_by_id) == 11, "OpenAlex raw IDs are not distinct")

    with (root / "candidate-crosswalk-16.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    _require(len(rows) == 16, "crosswalk manifestation count changed")
    ids = [row["source_record_id"] for row in rows]
    _require(len(ids) == len(set(ids)), "crosswalk source IDs duplicated")
    _require(
        {row["openalex_id"] for row in rows if row["openalex_id"]} == set(source_by_id),
        "raw OpenAlex row missing from crosswalk",
    )
    for row in rows:
        _require(row["primary_source_url"], "candidate lost source URL")
        _require(row["scholar_publication_id"] == "", "Scholar work ID invented")
        for field in (
            "lifetime_complete",
            "release_eligible",
            "map_eligible",
            "collaboration_eligible",
        ):
            _require(row[field] == "False", f"candidate {field} escaped hold")
        if row["openalex_id"]:
            source = source_by_id[row["openalex_id"]]
            _require(
                row["year_as_source"] == str(source["publication_year"])
                and row["title_as_source"] == source["display_name"]
                and row["genre_as_source"] == source["type"]
                and row["doi_as_source"] == (source["doi"] or "")
                and row["byline_as_source"]
                == "; ".join(
                    authorship["author"]["display_name"]
                    for authorship in source["authorships"]
                ),
                "raw OpenAlex fields changed in crosswalk",
            )
    by_id = {row["source_record_id"]: row for row in rows}
    genre_counts = Counter(row["reviewed_genre"] for row in rows)
    _require(
        genre_counts
        == {
            "research_article": 2,
            "conference_paper": 2,
            "doctoral_dissertation": 1,
            "preprint_full_version": 2,
            "preprint_related_version": 1,
            "conference_abstract": 5,
            "student_research_report": 1,
            "discovery_metadata_duplicate": 2,
        },
        "publication genres changed",
    )
    _require(
        {
            row["doi_as_source"].lower().removeprefix("https://doi.org/")
            for row in rows
            if row["reviewed_genre"] in {"research_article", "conference_paper"}
        }
        == CORE_DOIS,
        "core publication DOI set changed",
    )
    _require(
        by_id["W2998157673"]["year_as_source"] == "2019"
        and by_id["W3202123363"]["reviewed_genre"] == "doctoral_dissertation"
        and by_id["W2564317063"]["attribution_status"] == "discovery_only"
        and by_id["W4287601069"]["attribution_status"] == "confirmed_duplicate"
        and by_id["W2900132984"]["attribution_status"] == "confirmed_duplicate",
        "known source conflicts or holds changed",
    )
    _require(
        by_id["arxiv:1406.7793"]["research_line_or_work"] == "prb-2015"
        and by_id["W3102313022"]["research_line_or_work"] == "soda-2020"
        and by_id["W3103427415"]["research_line_or_work"] == "focs-2020"
        and by_id["W4287601069"]["research_line_or_work"] == "focs-2020",
        "paper and preprint relations changed",
    )

    for name, doi in (
        ("crossref-prb-2015.json", "10.1103/physrevb.92.054435"),
        ("crossref-iucr-2016.json", "10.1107/s1600576716016423"),
        ("crossref-soda-2020.json", "10.1137/1.9781611975994.85"),
        ("crossref-focs-2020.json", "10.1109/focs46700.2020.00048"),
    ):
        deposit = _crossref(root, name)
        _require(deposit["DOI"].lower() == doi, f"{name} DOI changed")
        _require(
            any(author.get("family") == "Quintana" for author in deposit["author"]),
            f"{name} Dylan byline changed",
        )
    soda = _crossref(root, "crossref-soda-2020.json")
    iucr = _crossref(root, "crossref-iucr-2016.json")
    _require(
        soda["published"]["date-parts"][0][:2] == [2020, 1]
        and iucr["author"][2]["family"] == "Quintana",
        "2020 SODA date or 2016 publisher surname conflict changed",
    )
    version = _json(root / "datacite-thesis-v1.json")["data"]["attributes"]
    concept = _json(root / "datacite-thesis-concept.json")["data"]["attributes"]
    _require(
        version["url"]
        == concept["url"]
        == "https://knowledge.uchicago.edu/records/3c4qz-van37"
        and any(
            related["relatedIdentifier"] == "10.6082/he9v3-2g538"
            for related in version["relatedIdentifiers"]
        ),
        "dissertation DOI relation changed",
    )

    pdfs = _json(root / "pdf-evidence-index.json")
    _require(len(pdfs) == 5, "inspected PDF count changed")
    _require(
        pdfs["cmu-selected-2018-2023"]["source_sha256"]
        == "727f50a85a0af3397dc6395566a0e8d719101c27a30865e4667c2df0ec67af8a"
        and "two selected 2020 coding"
        in pdfs["cmu-selected-2018-2023"]["inspected_evidence"]
        and all(not item["content_committed"] for item in pdfs.values()),
        "selected list or PDF evidence changed",
    )
    review = _json(root / "identity-and-attribution-review.json")
    _require(
        review["person_id"] == PERSON_ID
        and "No exact Google Scholar" in review["scholar_account_status"]
        and "Lifetime completeness is unestablished" in review["completeness"]
        and "separate Dylan Quintana MD" in review["namesake_exclusion"],
        "identity, namesake, or completeness hold changed",
    )
    return manifest, rows


def load_held_candidates(root: Path = REVIEW_ROOT) -> list[dict]:
    """Expose source observations without canonical IDs or release eligibility."""
    manifest, rows = validate_review(root)
    return [
        {
            "person_id": manifest["person_id"],
            "source_record_id": row["source_record_id"],
            "reviewed_genre": row["reviewed_genre"],
            "attribution_status": row["attribution_status"],
            "canonical_work_id": None,
            "canonical_author_ids": [],
            "scholar_publication_id": None,
            "release_eligible": False,
            "map_eligible": False,
            "collaboration_eligible": False,
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_ROOT)
    args = parser.parse_args()
    manifest, rows = validate_review(args.review_dir)
    print(
        f"{manifest['person_id']}: {len(rows)} held source manifestations "
        "(four peer-reviewed papers and one dissertation observed); "
        "lifetime completeness unestablished"
    )


if __name__ == "__main__":
    main()
