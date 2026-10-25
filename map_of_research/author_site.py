"""Load an exact author-owned bibliography without calling it Scholar data."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

from .registry import Registry

AUTHOR_SITE_SCHEMA_VERSION = 1
IYER_PERSON_ID = "person-gautamiyer-3894621b"
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+\Z", re.IGNORECASE)
COMB_MANUSCRIPT_URL = (
    "https://www.math.cmu.edu/~gautam/sj/research/papers/201809-fcomb.pdf"
)
COMB_PUBLISHER_URL = (
    "https://www.intlpress.com/site/pub/files/_fulltext/journals/"
    "cms/2020/0018/0007/CMS-2020-0018-0007-a002.pdf"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _url(value: str, *, secure: bool = True) -> str:
    parsed = urlparse(value)
    allowed_schemes = {"https"} if secure else {"http", "https"}
    if parsed.scheme not in allowed_schemes or not parsed.netloc:
        raise ValueError(f"Invalid author-site URL: {value}")
    return value


def _normalized_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


class _AllPublicationsCounter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._stack: list[str] = []
        self._target_depth = -1
        self._list_depth = -1
        self.count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        parent = self._stack[-1] if self._stack else ""
        self._stack.append(tag)
        if tag == "div" and dict(attrs).get("id") == "pubs-all":
            self._target_depth = len(self._stack)
        elif self._target_depth > 0 and tag == "ul" and self._list_depth < 0:
            self._list_depth = len(self._stack)
        elif tag == "li" and self._list_depth > 0 and parent == "ul":
            self.count += 1

    def handle_endtag(self, tag: str) -> None:
        if len(self._stack) == self._list_depth and tag == "ul":
            self._list_depth = -1
        if len(self._stack) == self._target_depth and tag == "div":
            self._target_depth = -1
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()


def load_author_site_rows(path: Path, registry: Registry) -> list[dict[str, object]]:
    """Return all 47 Iyer observations only if every capture and review pin matches."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != AUTHOR_SITE_SCHEMA_VERSION
        or manifest.get("source_type") != "author_site"
        or manifest.get("person_id") != IYER_PERSON_ID
        or manifest.get("expected_entry_count") != 47
        or manifest.get("expected_doi_review_count") != 20
        or manifest.get("expected_unlinked_doi_review_count") != 3
        or manifest.get("release_state") != "staged_for_explicit_promotion"
        or manifest.get("scholar_account_status") != "no_verified_account"
        or manifest.get("html_file") != "gautam-iyer-2026-10-25.html"
        or manifest.get("review_csv_path")
        != "registry/reviews/2026-10-25-math-gautam-iyer-author-site.csv"
        or manifest.get("doi_audit_csv_path")
        != "registry/reviews/2026-10-25-math-gautam-iyer-doi-audit.csv"
        or manifest.get("unlinked_doi_audit_json_path")
        != "registry/reviews/2026-10-25-math-gautam-iyer-unlinked-doi-audit.json"
    ):
        raise ValueError("Author-site capture identity or exact scope changed")
    person = registry.people_by_id.get(IYER_PERSON_ID)
    if (
        person is None
        or person.scholar_id
        or person.homepage_url != manifest["official_profile_url"]
        or not any(
            item.person_id == IYER_PERSON_ID and item.included
            for item in registry.memberships
        )
    ):
        raise ValueError("Author-site fallback lacks a verified faculty identity")
    source_page = _url(manifest["source_page_url"])
    _url(manifest["official_profile_url"])
    captured_at = dt.datetime.fromisoformat(manifest["captured_at_utc"])
    if captured_at.tzinfo is None:
        raise ValueError("Author-site capture time lacks timezone")
    html_path = path.parent / manifest["html_file"]
    repository_root = path.parent.parent.parent
    csv_path = repository_root / manifest["review_csv_path"]
    audit_path = repository_root / manifest["doi_audit_csv_path"]
    unlinked_audit_path = repository_root / manifest["unlinked_doi_audit_json_path"]
    for source, field in (
        (html_path, "html_sha256"),
        (csv_path, "review_csv_sha256"),
        (audit_path, "doi_audit_csv_sha256"),
        (unlinked_audit_path, "unlinked_doi_audit_json_sha256"),
    ):
        if _sha256(source) != manifest[field]:
            raise ValueError(f"Author-site capture file changed: {source.name}")
    counter = _AllPublicationsCounter()
    counter.feed(html_path.read_text(encoding="utf-8"))
    if counter.count != 47 or manifest["source_selector"] != "div#pubs-all > ul > li":
        raise ValueError("Author-site All Publications section changed")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        extracted = list(csv.DictReader(handle))
    with audit_path.open(newline="", encoding="utf-8") as handle:
        audited = {int(item["row_index"]): item for item in csv.DictReader(handle)}
    unlinked_audit = json.loads(unlinked_audit_path.read_text(encoding="utf-8"))
    if (
        unlinked_audit.get("schema_version") != 1
        or unlinked_audit.get("source_html_sha256") != manifest["html_sha256"]
        or unlinked_audit.get("source_extraction_sha256")
        != manifest["review_csv_sha256"]
        or len(unlinked_audit.get("records", [])) != 3
    ):
        raise ValueError("Author-site unlinked DOI audit scope changed")
    unlinked_reviews = {
        int(item["row_index"]): item for item in unlinked_audit["records"]
    }
    if set(unlinked_reviews) != {32, 33, 34}:
        raise ValueError("Author-site unlinked DOI review rows changed")
    reviews = {int(item["row_index"]): item for item in manifest["doi_reviews"]}
    if (
        len(extracted) != 47
        or len(audited) != 20
        or len(reviews) != 20
        or len(unlinked_reviews) != 3
    ):
        raise ValueError("Author-site bibliography or DOI review is incomplete")
    output_types = {
        int(item["row_index"]): item["publication_type"]
        for item in manifest["publication_type_overrides"]
    }
    if output_types != {44: "thesis"}:
        raise ValueError("Author-site output-type review changed")
    memberships = [
        item
        for item in registry.memberships
        if item.person_id == IYER_PERSON_ID and item.included
    ]
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
    identities: set[str] = set()
    normalized_titles: set[tuple[str, int]] = set()
    for index, original in enumerate(extracted, start=1):
        if (
            int(original["row_index"]) != index
            or original["person_id"] != IYER_PERSON_ID
            or original["source_page_url"] != source_page
            or original["source_page_sha256"] != manifest["html_sha256"]
            or original["captured_at_utc"] != manifest["captured_at_utc"]
            or original["site_list"] != manifest["source_section"]
        ):
            raise ValueError(f"Author-site source row changed: {index}")
        title = original["title_as_listed"].strip()
        authors = original["authors_as_listed"].strip()
        author_year = int(original["year_as_listed"])
        if not title or not authors or "Iyer" not in authors or author_year < 1900:
            raise ValueError(f"Author-site bibliographic row is incomplete: {index}")
        title_key = (re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip(), author_year)
        if title_key in normalized_titles:
            raise ValueError("Author-site title/year is duplicated")
        normalized_titles.add(title_key)
        manuscript_url = _url(original["author_manuscript_url_as_linked"])
        linked_url = _url(original["work_url_as_linked"], secure=False)
        identifier = hashlib.sha256(manuscript_url.encode()).hexdigest()[:20]
        native_id = f"author-site:iyer:{identifier}"
        if native_id in identities:
            raise ValueError("Author-site manuscript identity is duplicated")
        identities.add(native_id)
        review = reviews.get(index)
        unlinked_review = unlinked_reviews.get(index)
        linked_doi = original["doi_as_linked"].casefold()
        if bool(linked_doi) != bool(review):
            raise ValueError(f"Author-site DOI review coverage changed: {index}")
        if review is not None:
            audit = audited[index]
            if (
                not DOI_PATTERN.fullmatch(linked_doi)
                or review["doi"] != linked_doi
                or audit["doi_as_linked"].casefold() != linked_doi
                or review["author_site_year"] != author_year
                or review["publisher_year"] != int(audit["doi_registry_published_year"])
                or review["publisher_title"] != audit["doi_registry_title"]
                or review["publisher_registry_url"] != audit["doi_registry_url"]
                or audit["doi_registry_has_iyer_author"] != "true"
                or (
                    index != 22
                    and audit["title_exact_after_alphanumeric_normalization"] != "true"
                )
                or (
                    index == 22
                    and review["title_match"] != "author_manuscript_matches_publisher"
                )
            ):
                raise ValueError(f"Author-site DOI source review changed: {index}")
            year = review["publisher_year"]
        elif unlinked_review is not None:
            registry_url = _url(unlinked_review["publisher_registry_url"])
            doi = unlinked_review["doi"]
            registry_parts = urlparse(registry_url)
            publisher_authors = unlinked_review["publisher_authors"]
            evidence_urls = unlinked_review["evidence_urls"]
            publisher_title = unlinked_review["publisher_title"]
            if (
                linked_doi
                or unlinked_review["review_origin"] != "independently_verified_unlinked"
                or not DOI_PATTERN.fullmatch(doi)
                or unlinked_review["author_site_title"] != original["title_as_listed"]
                or unlinked_review["author_site_authors"]
                != original["authors_as_listed"]
                or unlinked_review["author_site_year"] != author_year
                or unlinked_review["publisher_year"] != author_year
                or _normalized_title(publisher_title) != _normalized_title(title)
                or registry_parts.netloc != "api.crossref.org"
                or unquote(registry_parts.path.rsplit("/", 1)[-1]).casefold() != doi
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    unlinked_review["publisher_registry_response_sha256"],
                )
                or not isinstance(publisher_authors, list)
                or len(publisher_authors) != len(authors.split(","))
                or "Gautam Iyer" not in publisher_authors
                or "Robert L. Pego" not in publisher_authors
                or not isinstance(evidence_urls, list)
                or len(evidence_urls) != 2
            ):
                raise ValueError(
                    f"Author-site unlinked DOI source review changed: {index}"
                )
            reviewed_at = dt.datetime.fromisoformat(unlinked_review["reviewed_at_utc"])
            if reviewed_at.tzinfo is None:
                raise ValueError("Author-site unlinked DOI review time lacks timezone")
            year = unlinked_review["publisher_year"]
        else:
            year = author_year
        source_urls = [source_page, manuscript_url]
        if review is not None:
            source_urls.append(_url(review["publisher_registry_url"]))
            source_urls.extend(
                _url(url) for url in review.get("additional_evidence_urls", [])
            )
        if unlinked_review is not None:
            source_urls.append(registry_url)
            source_urls.extend(_url(url) for url in evidence_urls)
        if index == 22 and review.get("additional_evidence_urls") != [
            COMB_MANUSCRIPT_URL,
            COMB_PUBLISHER_URL,
        ]:
            raise ValueError("Author-site comb-paper title review is missing")
        record = {"author_site_row": original, "doi_review": review}
        if unlinked_review is not None:
            record["unlinked_doi_review"] = unlinked_review
        rows.append(
            {
                "scholar_id": "",
                "person_id": IYER_PERSON_ID,
                "display_name": person.display_name,
                "faculty": person.display_name,
                "department_ids": department_ids,
                "memberships": membership_records,
                "author_pub_id": f"author-site:{IYER_PERSON_ID}:{identifier}",
                "title": title,
                "authors": authors,
                "year": year,
                "venue": original["venue_as_listed"].strip(),
                "citation": "",
                "citation_count": 0,
                "source_url": manuscript_url,
                "source_record_json": json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "fetched_at_utc": "",
                "source_type": "author_site",
                "source_doi": linked_doi
                or (unlinked_review["doi"] if unlinked_review is not None else ""),
                "source_native_id": native_id,
                "source_verified_at": captured_at.date().isoformat(),
                "source_verification_urls": source_urls,
                "external_coauthors": [],
                "source_output_type": output_types.get(index, "publication"),
                "source_citation_count_known": False,
                "source_work_url_as_linked": linked_url,
            }
        )
    return rows
