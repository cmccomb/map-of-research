"""Stage every pinned Cummings source observation without graph eligibility."""

from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path
from typing import Any

SOURCE_DIR = Path(__file__).resolve().parent
PERSON_ID = "person-jamescummings-521af556"


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _held_row(
    source_type: str,
    source_native_id: str,
    source_url: str,
    source_title: str,
    source_year: int | None,
    source_doi: str,
    reviewed_genre: str,
    contributor_role: str,
    source_contributors: list[str] | None,
    source_record: dict[str, Any],
    work_group: str | None = None,
) -> dict[str, Any]:
    return {
        "person_id": PERSON_ID,
        "source_type": source_type,
        "source_native_id": source_native_id,
        "source_url": source_url,
        "source_title": source_title,
        "source_year": source_year,
        "source_doi": source_doi,
        "reviewed_genre": reviewed_genre,
        "contributor_role": contributor_role,
        "source_contributors": source_contributors,
        "work_group_hint": work_group,
        "source_record": source_record,
        "release_eligible": False,
        "map_eligible": False,
        "collaboration_eligible": False,
        "citation_count_known": False,
    }


def load_staged_rows(source_dir: Path = SOURCE_DIR) -> list[dict[str, Any]]:
    """Keep all 214 source observations distinct until work review is complete."""

    validate = runpy.run_path(str(SOURCE_DIR / "validate_capture.py"))[
        "validate_capture"
    ]
    validate(source_dir)
    elements = _csv(source_dir / "elements-108.csv")
    elements_review = _csv(source_dir / "elements-crosswalk-108.csv")
    site_review = _csv(source_dir / "researcher-crosswalk-51.csv")
    orcid_review = _csv(source_dir / "orcid-crosswalk-51.csv")
    orcid_groups = json.loads((source_dir / "orcid-works.json").read_text())["group"]
    publisher = json.loads((source_dir / "publisher-omissions-4.json").read_text())
    rows: list[dict[str, Any]] = []

    for source, review in zip(elements, elements_review, strict=True):
        object_id = source["cmu_elements_object_id"]
        if object_id != review["elements_object_id"]:
            raise ValueError("Cummings Elements crosswalk order changed")
        rows.append(
            _held_row(
                "cmu_elements",
                f"cmu-elements:2104:object:{object_id}",
                source["source_url"],
                source["title"],
                int(source["year"]) if source["year"] else None,
                source["doi"],
                review["reviewed_genre"],
                review["contributor_role"],
                [name.strip() for name in source["authors"].split("|")],
                {"elements": source, "review": review},
                review["work_group"],
            )
        )

    for review in site_review:
        item = int(review["site_item"])
        rows.append(
            _held_row(
                "researcher_site",
                f"researcher-site:jcumming:papers:{item}",
                "https://www.math.cmu.edu/users/jcumming/papers/",
                review["title_as_listed"],
                None,
                "",
                review["reviewed_genre"],
                "author" if item != 14 else "presenter",
                None,
                {"review": review, "citation_as_listed": review["citation_as_listed"]},
            )
        )

    for review, group in zip(orcid_review, orcid_groups, strict=True):
        summaries = group["work-summary"]
        if int(review["put_code"]) not in {
            summary["put-code"] for summary in summaries
        }:
            raise ValueError("Cummings ORCID source order changed")
        rows.append(
            _held_row(
                "researcher_orcid",
                f"orcid:0000-0002-7913-0427:work:{review['put_code']}",
                f"https://pub.orcid.org/v3.0/0000-0002-7913-0427/work/{review['put_code']}",
                review["title"],
                int(review["year"]) if review["year"] else None,
                review["doi"],
                "unresolved_orcid_work",
                "unresolved",
                None,
                {"orcid_group": group, "review": review},
            )
        )

    for review in publisher:
        crossref = json.loads((source_dir / review["crossref_file"]).read_text())[
            "message"
        ]
        rows.append(
            _held_row(
                "publisher_registration",
                f"doi:{review['doi'].casefold()}",
                review["publisher_url"],
                review["title"],
                review["publication_year"],
                review["doi"],
                review["genre"],
                review["role"],
                review["contributors"],
                {"publisher_review": review, "crossref": crossref},
            )
        )

    counts = {
        kind: sum(row["source_type"] == kind for row in rows)
        for kind in {row["source_type"] for row in rows}
    }
    if (
        len(rows) != 214
        or len({row["source_native_id"] for row in rows}) != 214
        or counts
        != {
            "cmu_elements": 108,
            "researcher_site": 51,
            "researcher_orcid": 51,
            "publisher_registration": 4,
        }
        or any(
            row["release_eligible"]
            or row["map_eligible"]
            or row["collaboration_eligible"]
            or row["citation_count_known"]
            for row in rows
        )
    ):
        raise ValueError("Cummings whole-source observation hold changed")
    return rows


if __name__ == "__main__":
    staged = load_staged_rows()
    print(
        json.dumps(
            {
                "source_observations": len(staged),
                "source_types": {
                    kind: sum(row["source_type"] == kind for row in staged)
                    for kind in sorted({row["source_type"] for row in staged})
                },
                "release_eligible": False,
            },
            sort_keys=True,
        )
    )
