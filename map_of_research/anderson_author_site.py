"""Stage Theresa Anderson's complete displayed author bibliography, not Scholar data."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from .registry import Registry

PERSON_ID = "person-theresaanderson-5ed760a9"
SOURCE_URL = "https://faculty.math.cmu.edu/anderson/research/"
CV_URL = (
    "https://faculty.math.cmu.edu/anderson/wp-content/uploads/sites/2/2026/08/"
    "CV2026.pdf"
)
EXPECTED_METADATA_REVIEW_ORDINALS = {21, 22, 23, 24, 25, 33}
EXPECTED_OUTPUT_TYPES = {5: "appendix", 11: "personal_essay", 37: "thesis"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid Anderson source URL: {value}")
    return value


class _AuthorPageEntries(HTMLParser):
    """Extract the same paragraph text and links as the reviewed source CSV."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.heading = False
        self.heading_parts: list[str] = []
        self.active = False
        self.paragraph = False
        self.parts: list[str] = []
        self.links: list[str] = []
        self.entries: list[tuple[str, list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "h3":
            self.heading = True
            self.heading_parts = []
        elif (
            tag == "p"
            and self.active
            and "wp-block-paragraph" in (attributes.get("class") or "").split()
        ):
            self.paragraph = True
            self.parts = []
            self.links = []
        elif tag == "a" and self.paragraph and attributes.get("href"):
            self.links.append(str(attributes["href"]))

    def handle_data(self, data: str) -> None:
        if self.heading:
            self.heading_parts.append(data)
        if self.paragraph:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h3" and self.heading:
            self.heading = False
            self.active = " ".join(" ".join(self.heading_parts).split()) == (
                "Publications and preprints"
            )
        elif tag == "p" and self.paragraph:
            self.paragraph = False
            citation = " ".join(
                " ".join(part.split()) for part in self.parts if part.strip()
            )
            self.entries.append((citation, self.links.copy()))


def load_anderson_author_site_rows(
    path: Path, registry: Registry
) -> list[dict[str, object]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source_type") != "author_site"
        or manifest.get("person_id") != PERSON_ID
        or manifest.get("official_profile_url")
        != "https://www.cmu.edu/math/people/faculty/anderson.html"
        or manifest.get("source_page_url") != SOURCE_URL
        or manifest.get("cv_url") != CV_URL
        or manifest.get("source_selector")
        != "h3 Publications and preprints + 42 sibling p elements"
        or manifest.get("expected_entry_count") != 42
        or manifest.get("expected_cv_pair_count") != 42
        or manifest.get("release_state") != "staged_for_explicit_promotion"
        or manifest.get("release_eligible") is not False
        or manifest.get("scholar_account_status") != "no_verified_account"
        or set(manifest.get("metadata_review_ordinals", []))
        != EXPECTED_METADATA_REVIEW_ORDINALS
        or manifest.get("genre_overrides")
        != {str(index): value for index, value in EXPECTED_OUTPUT_TYPES.items()}
    ):
        raise ValueError("Anderson author-site capture identity or scope changed")
    captured_at = dt.datetime.fromisoformat(manifest["captured_at_utc"])
    if captured_at.tzinfo is None:
        raise ValueError("Anderson author-site capture time lacks timezone")
    person = registry.people_by_id.get(PERSON_ID)
    memberships = [
        item
        for item in registry.memberships
        if item.person_id == PERSON_ID and item.included
    ]
    if (
        person is None
        or person.scholar_id
        or person.homepage_url != manifest["official_profile_url"]
        or not memberships
    ):
        raise ValueError(
            "Anderson author-site capture lacks verified registry identity"
        )
    required_files = {
        "html": "theresa-anderson-2026-10-25.html",
        "source_csv": "theresa-anderson-2026-10-25.source.csv",
        "cv_pdf": "theresa-anderson-2026-10-25.cv.pdf",
        "cv_crosswalk_csv": "theresa-anderson-2026-10-25.cv-crosswalk.csv",
        "review_csv": "theresa-anderson-2026-10-25.review.csv",
    }
    for key, filename in required_files.items():
        if manifest.get(f"{key}_file") != filename:
            raise ValueError(f"Anderson author-site {key} filename changed")
        if _sha256(path.parent / filename) != manifest.get(f"{key}_sha256"):
            raise ValueError(f"Anderson author-site {key} checksum changed")
    html_path = path.parent / required_files["html"]
    html_bytes = html_path.read_bytes()
    page = _AuthorPageEntries()
    page.feed(html_bytes.decode("utf-8"))
    if len(page.entries) != 42:
        raise ValueError("Anderson author page no longer has 42 bibliography entries")

    def read_csv(name: str) -> list[dict[str, str]]:
        with (path.parent / required_files[name]).open(
            newline="", encoding="utf-8"
        ) as handle:
            return list(csv.DictReader(handle))

    source = read_csv("source_csv")
    crosswalk = read_csv("cv_crosswalk_csv")
    reviews = read_csv("review_csv")
    if not (len(source) == len(crosswalk) == len(reviews) == 42):
        raise ValueError("Anderson author bibliography or CV crosswalk is incomplete")
    if len({row["raw_citation"] for row in source}) != 42:
        raise ValueError("Anderson author bibliography repeats a source citation")
    if {int(row["cv_number"]) for row in crosswalk} != set(range(1, 43)):
        raise ValueError("Anderson CV entries do not have a one-to-one crosswalk")
    department_ids = sorted({item.department_id for item in memberships})
    membership_records = [
        {
            "department_id": item.department_id,
            "role": item.role,
            "included": item.included,
            "legacy_label": item.legacy_label,
            "source_url": item.source_url,
            "verified_at": item.verified_at,
        }
        for item in memberships
    ]
    rows: list[dict[str, object]] = []
    for ordinal, (raw, cv, review, html_entry) in enumerate(
        zip(source, crosswalk, reviews, page.entries, strict=True), start=1
    ):
        citation, linked_urls = html_entry
        if (
            int(raw["source_ordinal"]) != ordinal
            or int(cv["author_site_ordinal"]) != ordinal
            or int(review["source_ordinal"]) != ordinal
            or raw["source_url"] != SOURCE_URL
            or raw["raw_citation"] != citation
            or json.loads(raw["linked_urls_json"]) != linked_urls
            or cv["source_citation"] != citation
            or not cv["cv_citation"].strip()
            or hashlib.sha256(citation.encode("utf-8")).hexdigest()
            != review["raw_citation_sha256"]
        ):
            raise ValueError(f"Anderson source or CV row changed: {ordinal}")
        expected_review = (
            "publication_metadata_review"
            if ordinal in EXPECTED_METADATA_REVIEW_ORDINALS
            else "source_exact"
        )
        if (
            review["metadata_review_status"] != expected_review
            or cv["status"]
            != (
                "publication_metadata_review"
                if ordinal in EXPECTED_METADATA_REVIEW_ORDINALS
                else "same_author_listed_entry"
            )
            or review["source_output_type"]
            != EXPECTED_OUTPUT_TYPES.get(ordinal, "publication")
        ):
            raise ValueError(f"Anderson metadata or genre review changed: {ordinal}")
        title = review["title_as_listed"].strip()
        authors = review["authors_as_listed"].strip()
        if (
            not title
            or citation.count(title) != 1
            or not authors.startswith("Anderson, Theresa C.")
            or not citation.startswith(authors)
        ):
            raise ValueError(f"Anderson title or author parsing changed: {ordinal}")
        if ordinal == 5 and ("Kim, Elena" in authors or "Appendix to" in authors):
            raise ValueError("Appendix host author cannot become an appendix coauthor")
        year_text = review["year_for_staging"]
        year = int(year_text) if year_text else None
        expected_year_basis = (
            "cv" if ordinal == 30 else "author_page" if year else "not_stated"
        )
        if review["year_basis"] != expected_year_basis or (
            year is not None
            and str(year) not in (cv["cv_citation"] if ordinal == 30 else citation)
        ):
            raise ValueError(f"Anderson staged year lacks source support: {ordinal}")
        suffix = citation.split(title, 1)[1].strip().lstrip(". ").strip()
        if review["venue_and_status_as_listed"] != suffix:
            raise ValueError(f"Anderson source venue or status changed: {ordinal}")
        links = [_url(value) for value in linked_urls]
        digest = hashlib.sha256(citation.encode("utf-8")).hexdigest()[:20]
        native_id = f"author-site:anderson:{digest}"
        source_record = {
            "author_site_row": raw,
            "cv_crosswalk_row": cv,
            "review_row": review,
        }
        rows.append(
            {
                "scholar_id": "",
                "person_id": PERSON_ID,
                "display_name": person.display_name,
                "faculty": person.display_name,
                "department_ids": department_ids,
                "memberships": membership_records,
                "author_pub_id": f"author-site:{PERSON_ID}:{digest}",
                "title": title,
                "authors": authors,
                "year": year,
                "venue": suffix,
                "citation": "",
                "citation_count": 0,
                "source_url": SOURCE_URL,
                "source_record_json": json.dumps(
                    source_record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "fetched_at_utc": "",
                "source_type": "author_site",
                "source_doi": "",
                "source_native_id": native_id,
                "source_verified_at": captured_at.date().isoformat(),
                "source_verification_urls": [SOURCE_URL, CV_URL, *links],
                "external_coauthors": [],
                "source_output_type": review["source_output_type"],
                "source_citation_count_known": False,
                "source_work_url_as_linked": links[0] if links else "",
            }
        )
    return rows
