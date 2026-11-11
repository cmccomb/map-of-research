"""Validate the release-held Cummings bibliography evidence and crosswalk.

This module does not write a source snapshot or publication dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PERSON_ID = "person-jamescummings-521af556"
ORCID_ID = "0000-0002-7913-0427"
VERSION_PAIRS = (
    ("499952", "608884"),
    ("153674", "608887"),
    ("80684", "608889"),
    ("80690", "608888"),
    ("80687", "428468"),
    ("313301", "428481"),
    ("80706", "313300"),
    ("80712", "608890"),
    ("106520", "608892"),
    ("80707", "608894"),
    ("608891", "608881"),
    ("608893", "608882"),
    ("80686", "612226"),
)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _doi(value: str) -> str:
    return value.lower().replace("\\_", "_").strip()


def validate_capture(path: Path) -> dict[str, int]:
    """Return exact source counts or raise on evidence/identity drift."""

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["schema_version"] != 1
        or manifest["person_id"] != PERSON_ID
        or manifest["primary_department"] != "math"
        or manifest["scholar_id"] is not None
        or manifest["release_eligible"] is not False
    ):
        raise ValueError("Cummings identity or release hold changed")
    counts = manifest["source_counts"]
    if counts != {
        "cmu_elements_records": 108,
        "researcher_site_items": 51,
        "orcid_work_groups": 51,
        "orcid_exact_doi_elements_matches": 49,
        "publisher_missing_publication_manifestations": 4,
        "candidate_duplicate_pairs": 13,
    }:
        raise ValueError("Cummings source inventory changed")

    for item in manifest["sources"] + manifest["review_files"]:
        source = path / item["file"]
        if (
            not source.is_file()
            or hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError(f"Cummings pinned source changed: {item['file']}")

    official = (path / "official-identity.html").read_text(encoding="utf-8")
    first_page = (path / "page-000.html").read_text(encoding="utf-8")
    if "James Cummings" not in official or "jcumming" not in official:
        raise ValueError("CMU identity evidence changed")
    if ORCID_ID not in first_page or "jcumming@andrew.cmu.edu" not in first_page:
        raise ValueError("CMU-linked ORCID identity changed")
    if (
        len(
            re.findall(
                r"<LI\b[^>]*>.*?</LI>",
                (path / "researcher-papers.html").read_text(),
                re.I | re.S,
            )
        )
        != 51
    ):
        raise ValueError("Researcher-maintained bibliography length changed")

    elements_manifest = json.loads((path / "elements-manifest.json").read_text())
    if (
        elements_manifest["cmu_elements_user_id"] != 2104
        or elements_manifest["captured_rows"] != 108
    ):
        raise ValueError("Elements capture identity changed")
    for index, page in enumerate(elements_manifest["pages"]):
        page_file = path / f"page-{index * 25:03d}.html"
        if hashlib.sha256(page_file.read_bytes()).hexdigest() != page["html_sha256"]:
            raise ValueError(f"Elements page {index} changed")
    if (
        hashlib.sha256((path / "elements-108.csv").read_bytes()).hexdigest()
        != elements_manifest["csv_sha256"]
    ):
        raise ValueError("Elements extracted records changed")

    source = _csv(path / "elements-108.csv")
    review = _csv(path / "elements-crosswalk-108.csv")
    site = _csv(path / "researcher-crosswalk-51.csv")
    orcid_review = _csv(path / "orcid-crosswalk-51.csv")
    orcid_raw = json.loads((path / "orcid-works.json").read_text())["group"]
    publisher = json.loads((path / "publisher-omissions-4.json").read_text())
    if not (
        len(source) == len(review) == 108
        and len(site) == len(orcid_review) == len(orcid_raw) == 51
        and len(publisher) == 4
    ):
        raise ValueError("Cummings crosswalk row count changed")
    if Counter(row["type"] for row in source) != {
        "Journal article": 55,
        "Preprint": 14,
        "Presentation": 35,
        "Chapter": 2,
        "Conference": 1,
        "Book": 1,
    }:
        raise ValueError("Elements source genres changed")
    ids = [row["cmu_elements_object_id"] for row in source]
    if len(set(ids)) != 108 or [row["ordinal"] for row in source] != [
        str(i) for i in range(1, 109)
    ]:
        raise ValueError("Elements row identity changed")
    if any(
        row["resolved_author_match"] != "True" or row["linked_user_match"] != "True"
        for row in source
    ):
        raise ValueError("Unresolved or unlinked Elements attribution")
    if any(
        raw["cmu_elements_object_id"] != typed["elements_object_id"]
        or raw["ordinal"] != typed["ordinal"]
        or typed["review_state"] != "release_held"
        or typed["graph_eligible"] != "False"
        for raw, typed in zip(source, review, strict=True)
    ):
        raise ValueError(
            "Cummings review does not preserve each held Elements observation"
        )
    id_to_review = {row["elements_object_id"]: row for row in review}
    for first, second in VERSION_PAIRS:
        if id_to_review[first]["work_group"] != id_to_review[second]["work_group"]:
            raise ValueError(f"Unreconciled version pair: {first}/{second}")
    if len({item for pair in VERSION_PAIRS for item in pair}) != 26:
        raise ValueError("Version pair IDs overlap")

    site_numbers = {int(row["site_item"]) for row in site}
    if site_numbers != set(range(1, 52)) or any(
        row["review_state"] != "release_held" for row in site
    ):
        raise ValueError("Researcher site is not fully crosswalked and held")
    if [row["site_item"] for row in site if not row["elements_object_ids"]] != ["14"]:
        raise ValueError("Researcher-only slides changed")
    if {
        row["site_item"]
        for row in site
        if row["reviewed_genre"] == "published_book_chapter"
    } != {"13", "25"}:
        raise ValueError("Researcher's published chapters changed")
    for row in site:
        for object_id in filter(None, row["elements_object_ids"].split("|")):
            if object_id not in id_to_review or row["site_item"] not in id_to_review[
                object_id
            ]["researcher_site_items"].split("|"):
                raise ValueError("Researcher to Elements crosswalk is not reciprocal")

    exact_doi = 0
    for index, (raw_group, typed) in enumerate(
        zip(orcid_raw, orcid_review, strict=True), 1
    ):
        raw = raw_group["work-summary"]
        if (
            len(raw) != 1
            or typed["orcid_index"] != str(index)
            or typed["put_code"] != str(raw[0]["put-code"])
        ):
            raise ValueError("ORCID group identity changed")
        if typed["review_state"] != "release_held":
            raise ValueError("ORCID work was released")
        if typed["relation"] == "exact_doi":
            matched = typed["elements_object_ids"].split("|")
            if len(matched) != 1 or _doi(typed["doi"]) != _doi(
                next(
                    r["doi"]
                    for r in source
                    if r["cmu_elements_object_id"] == matched[0]
                )
            ):
                raise ValueError("ORCID DOI match changed")
            exact_doi += 1
    if exact_doi != 49 or {
        r["elements_object_ids"] for r in orcid_review if r["relation"] != "exact_doi"
    } != {"341749", "341761"}:
        raise ValueError("ORCID chapter/DOI coverage changed")

    expected_roles = {
        "crossref-large-cardinal.json": ("book-chapter", "author"),
        "crossref-compactness.json": ("book-chapter", "author"),
        "crossref-appalachian.json": ("edited-book", "editor"),
        "crossref-foundations.json": ("edited-book", "editor"),
    }
    existing_dois = {_doi(row["doi"]) for row in source if row["doi"]}
    if len({_doi(row["doi"]) for row in publisher}) != 4:
        raise ValueError("Duplicate publisher omission DOI")
    for row in publisher:
        crossref = json.loads((path / row["crossref_file"]).read_text())["message"]
        expected_type, role = expected_roles[row["crossref_file"]]
        if (
            crossref["type"] != expected_type
            or row["role"] != role
            or _doi(crossref["DOI"]) != _doi(row["doi"])
            or _doi(row["doi"]) in existing_dois
            or not any(person.get("family") == "Cummings" for person in crossref[role])
        ):
            raise ValueError(
                f"Publisher omission attribution changed: {row['review_id']}"
            )
    if sum(row["role"] == "editor" for row in publisher) != 2:
        raise ValueError("Edited volume roles changed")
    if id_to_review["619986"]["contributor_role"] != "editor":
        raise ValueError("2025 edited volume role changed")
    higher = json.loads((path / "crossref-higher-recursion.json").read_text())[
        "message"
    ]
    if (
        higher["type"] != "edited-book"
        or len(higher["author"]) != 4
        or {person["family"] for person in higher["author"]}
        != {"Cummings", "Marks", "Yang", "Yu"}
    ):
        raise ValueError("2025 book editor evidence changed")
    if id_to_review["327935"]["reviewed_genre"] != "book_review":
        raise ValueError("Malformed book review correction changed")
    review_registration = json.loads((path / "crossref-review.json").read_text())[
        "message"
    ]
    if (
        _doi(review_registration["DOI"]) != "10.2307/421078"
        or review_registration["author"][0]["family"] != "Cummings"
    ):
        raise ValueError("Malformed review registration changed")
    if {
        id_to_review[item]["reviewed_genre"] for item in ("313300", "313301", "313302")
    } != {"repository_manuscript"}:
        raise ValueError("Figshare repository record genre changed")
    notes_registration = json.loads((path / "crossref-notes.json").read_text())[
        "message"
    ]
    if (
        id_to_review["80711"]["reviewed_genre"] != "journal_article"
        or notes_registration["type"] != "journal-article"
        or "Notre Dame Journal of Formal Logic"
        not in notes_registration["container-title"]
    ):
        raise ValueError("Notre Dame journal evidence changed")
    education = json.loads((path / "crossref-education-chapter.json").read_text())[
        "message"
    ]
    if (
        id_to_review["787912"]["reviewed_genre"] != "book_chapter"
        or education["type"] != "book-chapter"
        or [person["family"] for person in education["author"]]
        != ["Cummings", "Lockwood", "Weber"]
    ):
        raise ValueError("2018 education chapter evidence changed")

    return counts


if __name__ == "__main__":
    print(json.dumps(validate_capture(Path(__file__).resolve().parent), sort_keys=True))
