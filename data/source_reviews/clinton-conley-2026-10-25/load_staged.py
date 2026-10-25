"""Load Conley's audited sources as held records, outside the map pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

PERSON_ID = "person-clintonconley-626c1f70"
SITE_URL = "https://www.math.cmu.edu/~clintonc/"
CHAPTER_DOI = "10.1142/9789813223523_0002"
REVIEW_DOI = "10.1017/bsl.2016.25"
CHEMISTRY_HOLD_DOI = "10.1016/S0009-2614(99)00018-4"
ATOM = "{http://www.w3.org/2005/Atom}"


class _VisibleList(HTMLParser):
    """Read first links in visible list items, ignoring HTML comments."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_list = False
        self.current: dict[str, Any] | None = None
        self.in_first_anchor = False
        self.rows: list[tuple[str, str]] = []

    def _finish(self) -> None:
        if self.current and self.current["href"]:
            self.rows.append(
                (" ".join("".join(self.current["title"]).split()), self.current["href"])
            )
        self.current = None
        self.in_first_anchor = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "ul":
            self.in_list = True
        elif tag == "li" and self.in_list:
            self._finish()
            self.current = {"href": None, "title": []}
        elif tag == "a" and self.current and self.current["href"] is None:
            self.current["href"] = dict(attrs).get("href")
            self.in_first_anchor = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self.in_first_anchor = False
        elif tag == "ul" and self.in_list:
            self._finish()
            self.in_list = False

    def handle_data(self, data: str) -> None:
        if self.in_first_anchor and self.current:
            self.current["title"].append(data)


def _read_pinned(path: Path, name: str, expected: str) -> bytes:
    data = (path / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Conley source checksum changed: {name}")
    return data


def _csv(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(data.decode("utf-8-sig").splitlines()))


def _https(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"Conley evidence URL must be HTTPS: {url}")
    return url


def _crossref(path: Path, manifest: dict[str, str], file: str, doi: str) -> dict:
    record = json.loads(_read_pinned(path, file, manifest[file]))["message"]
    if record["DOI"].casefold() != doi.casefold():
        raise ValueError(f"Conley DOI identity changed: {file}")
    return record


def load_staged_conley_rows(path: Path) -> list[dict[str, Any]]:
    """Return 33 release-held observations without a snapshot or graph write."""

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("person_id") != PERSON_ID
        or manifest.get("primary_department_id") != "math"
        or manifest.get("scholar_id") is not None
        or manifest.get("release_eligible") is not False
        or manifest.get("whole_career_complete") is not False
        or manifest.get("counts")
        != {
            "researcher_page_visible": 31,
            "arxiv_links": 7,
            "researcher_pdf_links": 24,
            "publisher_omissions_verified": 2,
            "union_distinct_observations": 33,
            "comment_only_not_counted": 2,
            "namesake_identity_hold_not_counted": 1,
        }
    ):
        raise ValueError("Conley identity, scope, or release hold changed")
    pinned = {item["file"]: item["sha256"] for item in manifest["source_files"]}
    if len(pinned) != 10 or any(len(digest) != 64 for digest in pinned.values()):
        raise ValueError("Conley source manifest file coverage changed")
    raw = {name: _read_pinned(path, name, digest) for name, digest in pinned.items()}
    if b"Clinton" not in raw["official.html"]:
        raise ValueError("Conley official identity page changed")

    parser = _VisibleList()
    parser.feed(raw["author-site.html"].decode("utf-8"))
    listed = _csv(raw["author-listed.csv"])
    pdf_review = _csv(raw["pdf-first-page-review.csv"])
    exceptions = _csv(raw["publisher-exceptions.csv"])
    if len(parser.rows) != 31 or len(listed) != 31 or len(pdf_review) != 24:
        raise ValueError("Conley visible-page or PDF review count changed")
    if len(exceptions) != 3:
        raise ValueError("Conley publisher and identity-hold ledger changed")
    if len({row["link"] for row in listed}) != 31:
        raise ValueError("Conley visible source work URL duplicated")

    root = ET.fromstring(raw["arxiv-seven.xml"])
    arxiv: dict[str, dict[str, Any]] = {}
    for entry in root.findall(f"{ATOM}entry"):
        identity = entry.findtext(f"{ATOM}id", default="").rsplit("/", 1)[-1]
        identity = re.sub(r"v\d+$", "", identity)
        if identity in arxiv:
            raise ValueError("Conley arXiv source identity duplicated")
        authors = [
            author.findtext(f"{ATOM}name", default="")
            for author in entry.findall(f"{ATOM}author")
        ]
        arxiv[identity] = {
            "title": " ".join(entry.findtext(f"{ATOM}title", default="").split()),
            "authors": authors,
            "submitted": entry.findtext(f"{ATOM}published"),
            "updated": entry.findtext(f"{ATOM}updated"),
        }
    if len(arxiv) != 7:
        raise ValueError("Conley arXiv primary-record coverage changed")

    pdf_by_number = {int(row["ordinal"]): row for row in pdf_review}
    if set(pdf_by_number) != set(range(8, 32)):
        raise ValueError("Conley PDF first-page review coverage changed")

    rows: list[dict[str, Any]] = []
    for number, (source, (page_title, page_link)) in enumerate(
        zip(listed, parser.rows, strict=True), start=1
    ):
        title = " ".join(source["title_as_listed"].split())
        link = _https(source["link"])
        if (
            source["ordinal"] != str(number)
            or title != page_title
            or link != urljoin(SITE_URL, page_link)
            or source["source_url"] != SITE_URL
            or source["year_as_listed"]
        ):
            raise ValueError(f"Conley visible source row changed: {number}")
        if number <= 7:
            arxiv_id = source["arxiv_id_if_direct"]
            record = arxiv.get(arxiv_id)
            if (
                record is None
                or link != f"https://arxiv.org/abs/{arxiv_id}"
                or record["title"].casefold() != title.casefold()
                or not any("Conley" in author for author in record["authors"])
            ):
                raise ValueError(f"Conley direct arXiv record changed: {number}")
            byline = "; ".join(record["authors"])
            work_record = record
            genre = "research_preprint"
            row_hold = None
        else:
            review = pdf_by_number[number]
            if (
                source["arxiv_id_if_direct"]
                or review["source_work_url"] != link
                or review["researcher_site_title"] != title
                or "conley" not in review["first_page_byline"].casefold()
                or any(
                    not re.fullmatch(r"[0-9a-f]{64}", review[field])
                    for field in ("pdf_sha256", "first_page_text_sha256")
                )
                or int(review["pdf_bytes"]) <= 0
            ):
                raise ValueError(f"Conley PDF primary-record review changed: {number}")
            if number == 30 and review["version_review"] != "material_title_variant":
                raise ValueError("Conley row 30 version hold changed")
            if number == 27 and review["version_review"] != "preliminary_draft":
                raise ValueError("Conley row 27 draft hold changed")
            if number == 21 and review["version_review"] != "placeholder_doi":
                raise ValueError("Conley row 21 placeholder DOI hold changed")
            byline = review["first_page_byline"]
            work_record = review
            genre = "preliminary_draft" if number == 27 else "research_manuscript"
            row_hold = review["row_hold"] or None
        rows.append(
            {
                "person_id": PERSON_ID,
                "primary_department_id": "math",
                "scholar_id": None,
                "source_native_id": f"researcher-bibliography:conley:site-{number:02}",
                "source_kind": "researcher_owned_site",
                "source_url": SITE_URL,
                "source_work_url": link,
                "title": title,
                "authors": byline,
                "publication_type": genre,
                "year": None,
                "year_basis": "author_site_does_not_state_publication_year",
                "doi": None,
                "citation_count": None,
                "map_eligible": False,
                "collaboration_eligible": False,
                "release_state": "source_only_release_held",
                "row_hold": row_hold,
                "source_record": {"author_site": source, "primary_work": work_record},
            }
        )

    ledger = {row["doi"]: row for row in exceptions}
    if set(ledger) != {CHAPTER_DOI, REVIEW_DOI, CHEMISTRY_HOLD_DOI}:
        raise ValueError("Conley publisher omissions or namesake hold changed")
    chapter = _crossref(path, pinned, "doi-registry/chapter.json", CHAPTER_DOI)
    review = _crossref(path, pinned, "doi-registry/review.json", REVIEW_DOI)
    namesake = _crossref(
        path, pinned, "doi-registry/chemistry-identity-hold.json", CHEMISTRY_HOLD_DOI
    )
    if (
        chapter["type"] != "book-chapter"
        or len(chapter["author"]) != 2
        or "Carnegie Mellon" not in str(chapter["author"][0]["affiliation"])
        or review["type"] != "journal-article"
        or len(review["author"]) != 1
        or review["author"][0]["family"] != "Conley"
        or namesake["type"] != "journal-article"
        or ledger[CHEMISTRY_HOLD_DOI]["attribution_status"]
        != "identity_hold_not_importable"
    ):
        raise ValueError("Conley publisher identity or namesake hold changed")
    cambridge = json.loads(raw["cambridge-review-metadata.json"])
    if (
        cambridge["citation_meta"]["citation_doi"] != REVIEW_DOI
        or cambridge["citation_meta"]["citation_author"] != "Clinton T. Conley"
        or cambridge["publisher_type"] != "Reviews"
        or cambridge["publisher_affiliation_email"] != "clintonc@andrew.cmu.edu"
    ):
        raise ValueError("Conley publisher book-review genre or identity changed")
    for doi, work, genre, year, hold in (
        (
            CHAPTER_DOI,
            chapter,
            "book_chapter",
            None,
            "Publisher online date 2017 differs from coauthor catalog book year 2018",
        ),
        (REVIEW_DOI, review, "book_review", 2016, None),
    ):
        evidence = ledger[doi]
        if evidence["attribution_status"] != "verified_cmu_conley_not_on_author_page":
            raise ValueError(f"Conley publisher record is not verified: {doi}")
        rows.append(
            {
                "person_id": PERSON_ID,
                "primary_department_id": "math",
                "scholar_id": None,
                "source_native_id": f"publisher-doi:{doi.lower()}",
                "source_kind": "publisher_verified_omission",
                "source_url": _https(evidence["publisher_or_coauthor_source_url"]),
                "source_work_url": f"https://doi.org/{doi}",
                "title": work["title"][0],
                "authors": "; ".join(
                    " ".join(
                        (author.get("given", ""), author.get("family", ""))
                    ).strip()
                    for author in work["author"]
                ),
                "publication_type": genre,
                "year": year,
                "year_basis": "Cambridge review publication date"
                if genre == "book_review"
                else "publisher_online_2017_coauthor_catalog_book_2018_unresolved",
                "year_facts": {"crossref_online": 2017, "book_catalog": 2018}
                if genre == "book_chapter"
                else {"publisher_publication": 2016},
                "doi": doi,
                "citation_count": None,
                "map_eligible": False,
                "collaboration_eligible": False,
                "release_state": "source_only_release_held",
                "row_hold": hold,
                "source_record": {
                    "publisher_exception": evidence,
                    "doi_registry": work,
                },
            }
        )
    if (
        len(rows) != 33
        or len({row["source_native_id"] for row in rows}) != 33
        or sum(row["row_hold"] is not None for row in rows) != 4
        or any(row["doi"] == CHEMISTRY_HOLD_DOI for row in rows)
        or any(row["map_eligible"] or row["collaboration_eligible"] for row in rows)
    ):
        raise ValueError("Conley whole-source union or held state changed")
    return rows
