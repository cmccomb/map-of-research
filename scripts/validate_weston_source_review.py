"""Validate the held Anthony Weston source review without network access."""

from __future__ import annotations

import csv
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "data/source_reviews/anthony-weston-2026-10-25"
PERSON_ID = "person-anthonyweston-79b583d1"
ORCID = "0000-0002-6345-3408"


class _Titles(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts: list[str] = []
        self.titles: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.depth:
            self.depth += 1
        elif tag == "div" and "__title" in dict(attrs).get("class", ""):
            self.depth = 1
            self.parts = []

    def handle_endtag(self, _tag: str) -> None:
        if self.depth:
            self.depth -= 1
            if not self.depth:
                title = "".join(self.parts).strip()
                self.titles.append(title.removesuffix(" (opens in a new tab)\u2060"))

    def handle_data(self, data: str) -> None:
        if self.depth:
            self.parts.append(data)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate(review: Path = REVIEW, registry: Path = ROOT / "registry") -> dict:
    manifest = json.loads((review / "manifest.json").read_text())
    _require(manifest["schema_version"] == 1, "invalid manifest schema")
    _require(manifest["person_id"] == PERSON_ID, "wrong person")
    _require(
        manifest["orcid"] == ORCID and manifest["scholar_id"] == "", "wrong account"
    )
    _require(
        manifest["publication_release"] == "held"
        and manifest["publication_observations_added"] == 0,
        "publication release must stay held",
    )
    sources = {source["file"]: source for source in manifest["sources"]}
    expected_files = {
        "cmu-math.html",
        "scholars-page-1.html",
        "scholars-page-2.html",
        "orcid-works.json",
    }
    _require(set(sources) == expected_files, "source file set changed")
    for name, source in sources.items():
        _require(source["url"].startswith("https://"), f"non-HTTPS source: {name}")
        digest = hashlib.sha256((review / name).read_bytes()).hexdigest()
        _require(digest == source["sha256"], f"source digest changed: {name}")
    cmu_profile = (review / "cmu-math.html").read_text()
    scholars = (review / "scholars-page-1.html").read_text()
    _require("11914-anthony-ross-weston" in cmu_profile, "CMU profile link missing")
    _require(
        ORCID in scholars and "aweston2@andrew.cmu.edu" in scholars,
        "CMU Scholars identity link missing",
    )

    cmu_rows = _rows(review / "cmu-scholars-29.csv")
    _require(len(cmu_rows) == 29, "not all 29 CMU cards captured")
    for page, expected_count in ((1, 25), (2, 4)):
        raw = (review / f"scholars-page-{page}.html").read_text()
        parser = _Titles()
        parser.feed(raw)
        page_rows = [row for row in cmu_rows if row["source_page"] == str(page)]
        _require(len(page_rows) == expected_count, f"page {page} row count changed")
        _require(
            [row["source_title"] for row in page_rows] == parser.titles,
            f"page {page} source titles differ",
        )
        for row in page_rows:
            if row["doi"]:
                _require(row["doi"] in raw.lower(), f"CMU DOI missing: {row['doi']}")
    _require(
        [int(row["source_ordinal"]) for row in cmu_rows] == list(range(1, 30)),
        "CMU row order changed",
    )
    _require(
        all(row["review_state"] == "held_pending_full_capture" for row in cmu_rows),
        "CMU row released",
    )
    _require(
        all(
            row["source_type"] in {"journal_article", "chapter", "patent"}
            for row in cmu_rows
        ),
        "CMU genre changed",
    )
    cmu_dois = {row["doi"] for row in cmu_rows if row["doi"]}
    _require(len(cmu_dois) == 26, "CMU DOI count changed")

    orcid_groups = json.loads((review / "orcid-works.json").read_text())["group"]
    orcid_rows = _rows(review / "orcid-24.csv")
    _require(len(orcid_rows) == len(orcid_groups) == 24, "ORCID group count changed")
    _require(
        [int(row["source_ordinal"]) for row in orcid_rows] == list(range(1, 25)),
        "ORCID row order changed",
    )
    for row, group in zip(orcid_rows, orcid_groups, strict=True):
        summary = group["work-summary"][0]
        title = summary["title"]["title"]["value"]
        ids = (summary.get("external-ids") or {}).get("external-id") or []
        doi = next(
            (
                entry["external-id-value"].lower()
                for entry in ids
                if entry["external-id-type"] == "doi"
            ),
            "",
        )
        _require(
            row["put_code"] == str(summary["put-code"])
            and row["source_title"] == title
            and row["doi"] == doi,
            "ORCID row differs from pinned source",
        )
        _require(
            row["review_state"] == "held_pending_full_capture", "ORCID row released"
        )
    orcid_dois = {row["doi"] for row in orcid_rows if row["doi"]}
    _require(
        len(orcid_dois) == 23 and len(cmu_dois & orcid_dois) == 23,
        "cross-source DOI overlap changed",
    )
    expected_counts = {
        "cmu_scholars": 29,
        "orcid_groups": 24,
        "exact_doi_overlaps": 23,
        "cmu_dois_absent_orcid": 3,
        "cmu_non_doi_not_in_orcid": 2,
    }
    _require(manifest["source_counts"] == expected_counts, "source counts changed")

    person = next(
        row for row in _rows(registry / "people.csv") if row["person_id"] == PERSON_ID
    )
    memberships = [
        row
        for row in _rows(registry / "memberships.csv")
        if row["person_id"] == PERSON_ID and row["included"] == "true"
    ]
    _require(
        person["orcid"] == ORCID and person["scholar_id"] == "",
        "registry account identity changed",
    )
    _require(
        [(row["department_id"], row["role"]) for row in memberships]
        == [("math", "teaching")],
        "Math membership changed",
    )
    return expected_counts


def load_held_observations(review: Path = REVIEW) -> list[dict]:
    """Preserve the complete visible CMU and ORCID sources without map edges."""

    validate(review)
    manifest = json.loads((review / "manifest.json").read_text())
    rows: list[dict] = []
    for source in _rows(review / "cmu-scholars-29.csv"):
        rows.append(
            {
                "person_id": PERSON_ID,
                "source_observation_id": (
                    f"cmu-scholars:11914:capture:{source['source_ordinal']}"
                ),
                "source_type": "cmu_scholars",
                "source_url": next(
                    item["url"]
                    for item in manifest["sources"]
                    if item["file"] == f"scholars-page-{source['source_page']}.html"
                ),
                "title_as_source": source["source_title"],
                "doi_as_source": source["doi"],
                "genre_as_source": source["source_type"],
                "source_record": source,
                "attribution_review_pending": True,
                "career_scope_verified": False,
                "release_eligible": False,
                "map_eligible": False,
                "collaboration_eligible": False,
            }
        )
    groups = json.loads((review / "orcid-works.json").read_text())["group"]
    for source, group in zip(_rows(review / "orcid-24.csv"), groups, strict=True):
        rows.append(
            {
                "person_id": PERSON_ID,
                "source_observation_id": (f"orcid:{ORCID}:work:{source['put_code']}"),
                "source_type": "researcher_orcid",
                "source_url": (
                    f"https://pub.orcid.org/v3.0/{ORCID}/work/{source['put_code']}"
                ),
                "title_as_source": source["source_title"],
                "doi_as_source": source["doi"],
                "genre_as_source": group["work-summary"][0]["type"],
                "source_record": {"table_row": source, "orcid_group": group},
                "attribution_review_pending": True,
                "career_scope_verified": False,
                "release_eligible": False,
                "map_eligible": False,
                "collaboration_eligible": False,
            }
        )
    if len(rows) != 53 or len({row["source_observation_id"] for row in rows}) != 53:
        raise ValueError("Weston whole-source observation hold changed")
    return rows


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
