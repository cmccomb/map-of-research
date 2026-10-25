"""Validate Marion Oliver's held, source-only bibliography review."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

REVIEW_ROOT = (
    Path(__file__).resolve().parents[1] / "data/source_reviews/marion-oliver-2026-10-25"
)
EXPECTED_DOIS = {
    "10.1007/bf00255666",
    "10.1090/qam/99715",
    "10.1090/qam/449210",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def validate_review(root: Path = REVIEW_ROOT) -> tuple[dict, list[dict[str, str]]]:
    """Verify source integrity, exact identity, candidate scope, and all holds."""
    root = Path(root)
    manifest = _json(root / "manifest.json")
    _require(isinstance(manifest, dict), "manifest must be an object")
    _require(manifest["status"] == "held_source_review", "review must stay held")
    _require(manifest["person_id"] == "person-marionoliver-1975d99a", "person ID")
    _require(manifest["verified_scholar_id"] is None, "Scholar ID unverified")
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

    cmu_page = gzip.decompress((root / "cmu-profile.html.gz").read_bytes())
    retirement = gzip.decompress((root / "cmu-retirement.html.gz").read_bytes())
    springer = gzip.decompress((root / "springer-1972.html.gz").read_bytes())
    _require(b"Marion Oliver" in cmu_page, "official profile identity changed")
    _require(b"Marion Oliver" in retirement, "retirement identity changed")
    _require(
        b"Marion L. Oliver" in springer
        and b"Department of Mathematics" in springer
        and b"Carnegie-Mellon University" in springer,
        "publisher name and institutional affiliation changed",
    )
    archive = _json(root / "cmu-1970-archive-index.json")
    _require(
        archive["content_committed"] is False
        and archive["matching_pdf_page"] == 12
        and "MARION LEON OLIVER" in archive["identity_text"],
        "CMU archival identity evidence changed",
    )

    author = _json(root / "openalex-math-author.json")
    name_search = _json(root / "openalex-name-search.json")
    works = _json(root / "openalex-works-3.json")
    crossref = _json(root / "crossref-3.json")
    with (root / "candidate-crosswalk-3.csv").open(newline="") as source:
        candidates = list(csv.DictReader(source))
    collisions = _json(root / "name-collision-review.json")
    context = _json(root / "context-review.json")

    _require(author["id"] == "https://openalex.org/A5014311491", "author ID")
    _require(
        any(
            item["institution"]["display_name"] == "Carnegie Mellon University"
            for item in author["affiliations"]
        ),
        "OpenAlex Mathematics affiliation changed",
    )
    _require(name_search["meta"]["count"] == 6, "name-search count")
    _require(len(name_search["results"]) == len(collisions) == 6, "collision rows")
    _require(
        {item["id"] for item in name_search["results"]}
        == {item["openalex_author_id"] for item in collisions},
        "name-search row dropped",
    )
    collision_by_id = {
        item["openalex_author_id"].rsplit("/", 1)[-1]: item for item in collisions
    }
    _require(
        collision_by_id["A5014311491"]["review_decision"]
        == "included_source_candidate",
        "exact CMU author decision changed",
    )
    _require(
        all(
            collision_by_id[key]["review_decision"] == "excluded_medical_namesake"
            for key in ("A5023340417", "A5130714641")
        ),
        "medical namesake clusters cannot be included",
    )
    _require(
        {
            collision_by_id[key]["works_count_as_search"]
            for key in (
                "A5023340417",
                "A5130714641",
            )
        }
        == {41, 8},
        "medical namesake counts changed",
    )

    _require(works["meta"]["count"] == len(works["results"]) == 3, "works")
    work_dois = {
        item["doi"].removeprefix("https://doi.org/").lower()
        for item in works["results"]
    }
    _require(work_dois == EXPECTED_DOIS, "OpenAlex DOI set changed")
    _require(set(crossref) == EXPECTED_DOIS, "Crossref DOI set changed")
    _require(len(candidates) == 3, "candidate count changed")
    _require(
        {item["doi"] for item in candidates} == EXPECTED_DOIS,
        "candidate DOI set changed",
    )
    for row in candidates:
        doi = row["doi"]
        publisher = crossref[doi]
        deposited = "; ".join(
            " ".join((author.get("given", ""), author.get("family", ""))).strip()
            for author in publisher["author"]
        )
        _require(row["title_as_publisher"] == publisher["title"][0], "title")
        _require(row["authors_as_deposited"] == deposited, "publisher byline")
        _require(row["reviewed_genre"] == "research_article", "genre")
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
        sum("Williams" in row["authors_as_deposited"] for row in candidates) == 2,
        "Williams coauthor observation count changed",
    )
    _require(
        _json(root / "kilthub-author-search.json") == []
        and _json(root / "kilthub-title-search.json") == [],
        "KiltHub search result changed",
    )
    _require(
        context["verified_google_scholar_account"] is None
        and context["official_researcher_bibliography_located"] is False,
        "whole-researcher source gate changed",
    )
    _require(
        manifest["counts"]
        == {
            "exact_research_article_candidates": 3,
            "with_william_o_williams": 2,
            "openalex_name_search_results": 6,
            "excluded_medical_namesake_clusters": 2,
            "unresolved_name_collision_clusters": 2,
            "kilthub_exact_author_results": 0,
            "kilthub_exact_title_results": 0,
        },
        "manifest counts changed",
    )
    return manifest, candidates


def load_held_candidates(root: Path = REVIEW_ROOT) -> list[dict]:
    """Return source observations with no canonical work or author edges."""
    manifest, rows = validate_review(root)
    return [
        {
            "person_id": manifest["person_id"],
            "source_candidate_id": row["candidate_id"],
            "doi": row["doi"],
            "title_as_publisher": row["title_as_publisher"],
            "authors_as_deposited": row["authors_as_deposited"],
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
        f"{manifest['person_id']}: {len(rows)} research article observations, "
        "all held; lifetime completeness unestablished"
    )


if __name__ == "__main__":
    main()
