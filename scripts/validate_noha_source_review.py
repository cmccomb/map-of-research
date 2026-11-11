"""Validate Noha Abdelghany's held, source-only bibliography review."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

REVIEW_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/noha-abdelghany-2026-10-25"
)
EXPECTED_DOIS = {
    "10.13069/jacodesmath.284939",
    "10.1016/j.disc.2020.112036",
    "10.1007/978-3-319-17296-5_18",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _html(root: Path, name: str) -> bytes:
    return gzip.decompress((root / name).read_bytes())


def validate_review(root: Path = REVIEW_ROOT) -> tuple[dict, list[dict[str, str]]]:
    """Check evidence, attribution boundaries, and all release holds."""
    root = Path(root)
    manifest = _json(root / "manifest.json")
    _require(isinstance(manifest, dict), "manifest must be an object")
    _require(manifest["status"] == "held_source_review", "review must stay held")
    _require(manifest["person_id"] == "person-nohaabdelghany-c393dec3", "person ID")
    _require(manifest["verified_google_scholar_id"] is None, "Scholar ID unverified")
    _require(
        manifest["publisher_attributed_orcid"] == "0000-0002-0117-4358",
        "publisher ORCID changed",
    )
    for field in (
        "lifetime_complete",
        "release_eligible",
        "map_eligible",
        "collaboration_eligible",
    ):
        _require(manifest[field] is False, f"{field} must remain false")

    files = manifest["files"]
    actual = {path.name for path in root.iterdir() if path.is_file()}
    _require(
        actual == set(files) | {"manifest.json", "README.md"},
        "source file inventory differs from manifest",
    )
    for name, entry in files.items():
        data = (root / name).read_bytes()
        _require(len(data) == entry["bytes"], f"{name} length changed")
        _require(
            hashlib.sha256(data).hexdigest() == entry["sha256"],
            f"{name} hash changed",
        )

    _require(
        b"Noha Abdelghany" in _html(root, "cmu-profile.html.gz")
        and b"Western Michigan University" in _html(root, "cmu-profile.html.gz"),
        "CMU profile identity changed",
    )
    _require(
        b"Cairo University" in _html(root, "cmu-interview.html.gz")
        and b"coding theory" in _html(root, "cmu-interview.html.gz"),
        "CMU career identity changed",
    )
    _require(
        b"Noha Abdelghany" in _html(root, "wmu-dissertation.html.gz")
        and b"Dr. Jay Wood" in _html(root, "wmu-dissertation.html.gz"),
        "WMU dissertation identity changed",
    )
    _require(
        b"Code" in _html(root, "jacodesmath-2017.html.gz")
        and b"Nefertiti Megahed" in _html(root, "jacodesmath-2017.html.gz"),
        "2017 publisher evidence changed",
    )
    _require(
        b"Noha ElGarem" in _html(root, "springer-elgarem-2015.html.gz")
        and b"Nefertiti Megahed" in _html(root, "springer-elgarem-2015.html.gz")
        and b"Jay A. Wood" in _html(root, "springer-elgarem-2015.html.gz"),
        "2015 alias evidence changed",
    )
    _require(
        b"Noha ElGarem" in _html(root, "anglisticum-elgarem-2013.html.gz"),
        "2013/2016 alias evidence changed",
    )
    _require(
        b"Noha ElGarem" in _html(root, "toronto-elgarem.html.gz")
        and b"MSc 2022" in _html(root, "toronto-elgarem.html.gz"),
        "competing ElGarem identity context changed",
    )
    _require(
        b"Noha Abdelghany" in _html(root, "gulfcoast-namesake.html.gz")
        and b"BSC-2085" in _html(root, "gulfcoast-namesake.html.gz"),
        "unlinked Gulf Coast name context changed",
    )

    selected = _json(root / "cmu-selected-pubs-index.json")
    pdf = _json(root / "jacodesmath-2017-pdf-index.json")
    _require(
        selected["content_committed"] is False
        and selected["source_sha256"]
        == "727f50a85a0af3397dc6395566a0e8d719101c27a30865e4667c2df0ec67af8a",
        "CMU selected-publications PDF evidence changed",
    )
    _require(
        pdf["content_committed"] is False
        and pdf["source_sha256"]
        == "ca8c6e7e57319f76b388bf1ad43c0e2d50ca4807280a41d86499448addacf01f"
        and "Cairo University" in pdf["inspected_evidence"],
        "2017 publisher PDF evidence changed",
    )

    author_search = _json(root / "openalex-abdelghany-author-search.json")
    works = _json(root / "openalex-abdelghany-works.json")
    alias_search = _json(root / "openalex-elgarem-author-search.json")
    alias_works = _json(root / "openalex-elgarem-works.json")
    orcid = _json(root / "orcid-record.json")
    _require(
        author_search["meta"]["count"] == 1
        and author_search["results"][0]["id"] == "https://openalex.org/A5010204925"
        and author_search["results"][0]["orcid"]
        == "https://orcid.org/0000-0002-0117-4358",
        "exact discovery cluster changed",
    )
    _require(
        works["meta"]["count"] == len(works["results"]) == 3,
        "exact cluster query count changed",
    )
    _require(
        alias_search["meta"]["count"] == 1
        and alias_search["results"][0]["id"] == "https://openalex.org/A5001560604"
        and alias_works["meta"]["count"] == len(alias_works["results"]) == 2,
        "unresolved alias cluster changed",
    )
    _require(
        orcid["person"]["name"]["family-name"]["value"] == "Abdelghany"
        and orcid["activities-summary"]["works"]["group"] == []
        and orcid["person"]["researcher-urls"]["researcher-url"] == [],
        "ORCID cannot establish complete works or Scholar account",
    )

    crossref_2017 = _json(root / "crossref-2017.json")["message"]
    crossref_2020 = _json(root / "crossref-2020.json")["message"]
    crossref_alias = _json(root / "crossref-elgarem-2015.json")["message"]
    _require(
        crossref_2017["DOI"].lower() == "10.13069/jacodesmath.284939"
        and [author["family"] for author in crossref_2017["author"]]
        == ["Abdelghany", "Megahed"]
        and crossref_2017["page"] == "115-115",
        "2017 deposit or pagination conflict changed",
    )
    _require(
        crossref_2020["DOI"].lower() == "10.1016/j.disc.2020.112036"
        and [author["family"] for author in crossref_2020["author"]]
        == ["Abdelghany", "Wood"]
        and crossref_2020["author"][0]["ORCID"]
        == "https://orcid.org/0000-0002-0117-4358",
        "2020 deposit identity changed",
    )
    _require(
        crossref_alias["DOI"].lower() == "10.1007/978-3-319-17296-5_18"
        and [author["family"] for author in crossref_alias["author"]]
        == ["ElGarem", "Megahed", "Wood"],
        "2015 alias publisher byline changed",
    )

    review = _json(root / "identity-and-attribution-review.json")
    _require(
        "Unresolved" in review["elgarem_alias_status"]
        and "No verifiable Google Scholar" in review["scholar_account_status"]
        and "lower bound" in review["completeness"],
        "identity or completeness hold changed",
    )
    with (root / "candidate-crosswalk-5.csv").open(newline="") as source:
        candidates = list(csv.DictReader(source))
    _require(len(candidates) == 5, "candidate count changed")
    _require(
        {row["doi"] for row in candidates if row["doi"]} == EXPECTED_DOIS,
        "candidate DOI set changed",
    )
    _require(
        [row["attribution_status"] for row in candidates]
        == ["confirmed_person_candidate"] * 3 + ["unresolved_alias_candidate"] * 2,
        "unresolved alias cannot be promoted",
    )
    _require(
        [row["reviewed_genre"] for row in candidates[:3]]
        == ["research_article", "research_article", "doctoral_dissertation"],
        "confirmed genre decisions changed",
    )
    for row in candidates:
        _require(row["scholar_publication_id"] == "", "Scholar work ID invented")
        _require(
            all(
                row[field] == "False"
                for field in (
                    "scholar_account_verified",
                    "lifetime_complete",
                    "release_eligible",
                    "map_eligible",
                    "collaboration_eligible",
                )
            ),
            "candidate escaped hold",
        )
    _require(
        manifest["counts"]
        == {
            "confirmed_person_candidate_manifestations": 3,
            "confirmed_research_articles": 2,
            "confirmed_dissertations": 1,
            "unresolved_elgarem_alias_manifestations": 2,
            "observed_confirmed_coauthors": 2,
            "orcid_public_work_groups": 0,
            "openalex_exact_cluster_query_works": 3,
        },
        "manifest counts changed",
    )
    return manifest, candidates


def load_held_candidates(root: Path = REVIEW_ROOT) -> list[dict]:
    """Expose source observations without canonical work or author graph IDs."""
    manifest, rows = validate_review(root)
    return [
        {
            "person_id": manifest["person_id"],
            "source_candidate_id": row["candidate_id"],
            "attribution_status": row["attribution_status"],
            "genre": row["reviewed_genre"],
            "doi": row["doi"] or None,
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
        f"{manifest['person_id']}: {len(rows)} held observations "
        "(3 confirmed manifestations, 2 unresolved alias); "
        "lifetime completeness unestablished"
    )


if __name__ == "__main__":
    main()
