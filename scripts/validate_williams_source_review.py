"""Validate the held William O. Williams source review.

This module does not import or mutate the collection or publication pipeline.
Candidate rows remain observations until a verified Scholar account or a separately
approved, source-reconciled whole-researcher release exists.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

REVIEW_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/william-o-williams-2026-10-25"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_review(root: Path = REVIEW_ROOT) -> tuple[dict, list[dict[str, str]]]:
    """Check the frozen sources, their crosswalks, and the release hold."""
    root = Path(root)
    manifest = _json(root / "manifest.json")
    _require(isinstance(manifest, dict), "manifest must be an object")
    _require(manifest["status"] == "held_source_review", "review must stay held")
    _require(
        manifest["person_id"] == "person-williamowilliams-d43b120b",
        "review person changed",
    )
    _require(manifest["verified_scholar_id"] is None, "Scholar ID is unverified")
    for field in ("lifetime_complete", "release_eligible", "map_eligible"):
        _require(manifest[field] is False, f"{field} must be false")
    _require(
        manifest["collaboration_eligible"] is False,
        "collaboration hold must remain false",
    )

    files = manifest["files"]
    actual = {path.name for path in root.iterdir() if path.is_file()}
    _require(
        actual == set(files) | {"manifest.json", "README.md"},
        "source file inventory differs from manifest",
    )
    for name, entry in files.items():
        data = (root / name).read_bytes()
        _require(len(data) == entry["bytes"], f"{name} length changed")
        digest = hashlib.sha256(data).hexdigest()
        _require(digest == entry["sha256"], f"{name} hash changed")

    selected = _json(root / "cmu-selected-8.json")
    pdf_index = _json(root / "author-pdf-index.json")
    thesis_pdf = _json(root / "rice-thesis-pdf-index.json")
    openalex = json.loads(
        gzip.decompress((root / "openalex-works.json.gz").read_bytes())
    )
    crossref = json.loads(gzip.decompress((root / "crossref-57.json.gz").read_bytes()))
    kilthub = _json(root / "kilthub-records.json")
    kilthub_rows = _csv(root / "kilthub-crosswalk-6.csv")
    candidates = _csv(root / "candidate-crosswalk-65.csv")
    rice = _json(root / "rice-ms-thesis-item.json")

    _require(len(selected) == 8, "CMU selected-list length changed")
    _require(
        all("Williams" in item["citation_as_listed"] for item in selected),
        "CMU selected citations changed",
    )
    _require(len(pdf_index) == 8, "researcher PDF inventory length changed")
    _require(
        {item["filename"] for item in pdf_index}
        == {
            "complmusc.pdf",
            "geometric.pdf",
            "marching.pdf",
            "newprimer.pdf",
            "oldnewprimer.pdf",
            "snelson.pdf",
            "thickshell.pdf",
            "vibration.pdf",
        },
        "researcher PDF filenames changed",
    )
    _require(
        all(item["content_committed"] is False for item in pdf_index),
        "PDF content must not be committed",
    )
    _require(thesis_pdf["content_committed"] is False, "thesis PDF must remain out")

    works = openalex["results"]
    _require(openalex["meta"]["count"] == len(works) == 60, "OpenAlex row count")
    _require(
        len({work["id"] for work in works}) == 60,
        "OpenAlex candidate IDs are not unique",
    )
    indexed_dois = {
        work["doi"].removeprefix("https://doi.org/").lower()
        for work in works
        if work.get("doi")
        and not work["doi"].lower().startswith("https://doi.org/10.1184/")
    }
    _require(len(indexed_dois) == 57, "publisher DOI coverage changed")
    _require(set(crossref) == indexed_dois, "Crossref rows are not the DOI set")
    _require(
        all(message["DOI"].lower() == doi for doi, message in crossref.items()),
        "Crossref DOI mismatch",
    )
    marching = _json(root / "crossref-marching.json")
    _require(
        marching["DOI"].lower() == "10.2140/jomms.2007.2.857",
        "Micheletti-Williams publisher DOI mismatch",
    )

    _require(len(kilthub) == len(kilthub_rows) == 6, "KiltHub report count")
    _require(
        {item["id"] for item in kilthub}
        == {int(row["repository_id"]) for row in kilthub_rows},
        "KiltHub crosswalk IDs differ from raw records",
    )
    for item in kilthub:
        _require(
            "Mathematics Technical Report" in item["description"],
            "KiltHub genre changed",
        )
        _require(
            any(
                author["full_name"] == "William O. Williams"
                for author in item["authors"]
            ),
            "KiltHub report lacks the verified author",
        )
    _require(
        rice["metadata"]["dc.creator"][0]["value"] == "Williams, William Orville",
        "Rice thesis creator changed",
    )
    _require(
        rice["metadata"]["dc.date.issued"][0]["value"] == "1964",
        "Rice thesis year changed",
    )

    _require(len(candidates) == 65, "held candidate row count changed")
    by_id = {row["candidate_id"]: row for row in candidates}
    _require(len(by_id) == 65, "candidate IDs are not unique")
    _require(
        {work["id"] for work in works}.issubset(by_id),
        "OpenAlex rows were dropped from the crosswalk",
    )
    _require(
        all(
            row[field] == "False"
            for row in candidates
            for field in (
                "scholar_account_verified",
                "release_eligible",
                "map_eligible",
                "collaboration_eligible",
            )
        ),
        "a candidate row escaped the hold",
    )
    doi_rows = {row["doi"]: row for row in candidates if row["doi"]}
    _require(
        doi_rows["10.1007/bf00247688"]["reviewed_genre"] == "corrigendum",
        "Owen-Williams correction was promoted",
    )
    _require(
        all(
            doi_rows[doi]["reviewed_genre"] == "book_review"
            for doi in ("10.1063/1.2915016", "10.1115/1.3162673", "10.1137/1033045")
        ),
        "review bylines must remain under review",
    )
    _require(
        by_id["https://openalex.org/W3036570908"]["primary_evidence_url"] == "",
        "1979 paper has no located journal primary source",
    )
    _require(
        by_id["rice:1911/88982"]["reviewed_genre"] == "masters_thesis",
        "Rice thesis genre changed",
    )
    counts = manifest["counts"]
    _require(
        counts
        == {
            "official_selected_citations": 8,
            "researcher_pdf_files": 8,
            "openalex_discovery_rows": 60,
            "crossref_deposits_for_openalex_rows": 57,
            "kilthub_technical_reports": 6,
            "rice_masters_theses": 1,
            "held_candidate_rows": 65,
        },
        "manifest counts changed",
    )
    return manifest, candidates


def load_held_candidates(root: Path = REVIEW_ROOT) -> list[dict]:
    """Return source observations with no canonical work or coauthor edges."""
    manifest, candidates = validate_review(root)
    return [
        {
            "person_id": manifest["person_id"],
            "source_candidate_id": row["candidate_id"],
            "title_as_source": row["title_as_source"],
            "year_as_source": row["year_as_source"],
            "doi": row["doi"],
            "authors_as_deposited": row["authors_as_deposited"],
            "reviewed_genre": row["reviewed_genre"],
            "primary_evidence_url": row["primary_evidence_url"],
            "relationship_review": row["relationship_review"],
            "review_note": row["review_note"],
            "canonical_work_id": None,
            "canonical_author_ids": [],
            "scholar_publication_id": None,
            "release_eligible": False,
            "map_eligible": False,
            "collaboration_eligible": False,
        }
        for row in candidates
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_ROOT)
    args = parser.parse_args()
    manifest, candidates = validate_review(args.review_dir)
    print(
        f"{manifest['person_id']}: {len(candidates)} candidate observations, "
        "all held; lifetime completeness unestablished"
    )


if __name__ == "__main__":
    main()
