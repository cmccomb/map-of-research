"""Stage the 47 Schimmerling source manifestations without graph eligibility."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts.validate_schimmerling_source_review import validate

SOURCE_DIR = Path(__file__).resolve().parent
PERSON_ID = "person-ernestschimmerling-2b6af882"
RESEARCH_GENRES = {
    "research_article",
    "research_chapter",
    "research_note",
    "research_preprint",
    "monograph",
}
ELEMENTS_ONLY_GENRES = {
    767570: "research_preprint",
    327734: "article_review",
    420492: "research_preprint",
    58367: "research_preprint",
}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_staged_rows(source_dir: Path = SOURCE_DIR) -> list[dict[str, Any]]:
    """Return the whole visible-source union, with every row release-held."""

    validate(source_dir)
    elements = _csv(source_dir / "elements-45.csv")
    author = _csv(source_dir / "author-34.csv")
    crosswalk = _csv(source_dir / "author-elements-crosswalk-34.csv")
    author_by_key = {(row["section"], row["source_ordinal"]): row for row in author}
    mapped: dict[int, tuple[dict[str, str], dict[str, str]]] = {}
    for cross in crosswalk:
        source_author = author_by_key[(cross["section"], cross["source_ordinal"])]
        for item in cross["elements_object_ids"].split("|"):
            if item:
                mapped[int(item)] = (cross, source_author)
    rows: list[dict[str, Any]] = []
    for raw in elements:
        object_id = int(raw["cmu_elements_object_id"])
        match = mapped.get(object_id)
        genre = match[0]["genre"] if match else ELEMENTS_ONLY_GENRES[object_id]
        authors = [value.strip() for value in raw["authors"].split("|")]
        row = {
            "person_id": PERSON_ID,
            "source_type": "cmu_elements",
            "source_native_id": f"cmu-elements:1550:object:{object_id}",
            "source_url": raw["source_url"],
            "source_title": raw["title"],
            "source_authors": authors,
            "source_year": int(raw["year"]) if raw["year"] else None,
            "source_doi": raw["doi"],
            "reviewed_genre": genre,
            "research_candidate": genre in RESEARCH_GENRES,
            "collaboration_candidate": genre in RESEARCH_GENRES and len(authors) > 1,
            "version_review_pending": bool(
                match and len(match[0]["elements_object_ids"].split("|")) > 1
            )
            or object_id in {420492, 58367},
            "source_record": {
                "elements": raw,
                "author_crosswalk": match[0] if match else None,
                "author_entry": match[1] if match else None,
            },
            "release_eligible": False,
            "map_eligible": False,
            "collaboration_eligible": False,
            "citation_count_known": False,
        }
        rows.append(row)
    appendix_cross = next(
        cross for cross in crosswalk if cross["relation"] == "author_only"
    )
    appendix = author_by_key[
        (appendix_cross["section"], appendix_cross["source_ordinal"])
    ]
    if not all(
        text in appendix["raw_citation"]
        for text in ("Wellorderings, ordinals and cardinals", "2007")
    ):
        raise ValueError("Schimmerling appendix citation changed")
    rows.append(
        {
            "person_id": PERSON_ID,
            "source_type": "author_site",
            "source_native_id": "author-site:schimmerling:other-publications:4",
            "source_url": appendix["source_url"],
            "source_title": "Wellorderings, ordinals and cardinals",
            "source_authors": ["Ernest Schimmerling"],
            "source_year": 2007,
            "source_doi": "",
            "reviewed_genre": "appendix",
            "research_candidate": False,
            "collaboration_candidate": False,
            "version_review_pending": False,
            "source_record": {
                "author_entry": appendix,
                "author_crosswalk": appendix_cross,
            },
            "release_eligible": False,
            "map_eligible": False,
            "collaboration_eligible": False,
            "citation_count_known": False,
        }
    )
    thesis = json.loads((source_dir / "orcid-thesis-decision.json").read_text())
    raw_orcid = json.loads((source_dir / "orcid-1992-thesis.json").read_text())
    rows.append(
        {
            "person_id": PERSON_ID,
            "source_type": "researcher_orcid",
            "source_native_id": thesis["source_native_id"],
            "source_url": thesis["source_url"],
            "source_title": thesis["title"],
            "source_authors": [thesis["author"]],
            "source_year": thesis["year"],
            "source_doi": "",
            "reviewed_genre": "doctoral_dissertation",
            "research_candidate": False,
            "collaboration_candidate": False,
            "version_review_pending": True,
            "source_record": {"orcid": raw_orcid, "genre_decision": thesis},
            "release_eligible": False,
            "map_eligible": False,
            "collaboration_eligible": False,
            "citation_count_known": False,
        }
    )
    if (
        len(rows) != 47
        or len({row["source_native_id"] for row in rows}) != 47
        or sum(row["source_type"] == "cmu_elements" for row in rows) != 45
        or sum(row["source_type"] == "author_site" for row in rows) != 1
        or sum(row["source_type"] == "researcher_orcid" for row in rows) != 1
        or any(
            row["release_eligible"]
            or row["map_eligible"]
            or row["collaboration_eligible"]
            for row in rows
        )
        or sum(row["reviewed_genre"] == "article_review" for row in rows) != 2
        or sum(row["reviewed_genre"] == "book_review" for row in rows) != 2
    ):
        raise ValueError("Schimmerling whole-person source union or hold changed")
    return rows


if __name__ == "__main__":
    staged = load_staged_rows()
    print(
        json.dumps(
            {
                "source_observations": len(staged),
                "research_candidates": sum(row["research_candidate"] for row in staged),
                "collaboration_candidates": sum(
                    row["collaboration_candidate"] for row in staged
                ),
                "release_eligible": False,
            },
            sort_keys=True,
        )
    )
