"""Read Larsson's 74 pinned observations without admitting them to the graph.

This module has no snapshot or publisher integration. Its output preserves
source data for whole-researcher reconciliation while the global hold remains.
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

PERSON_ID = "person-martinlarsson-970afb46"
OFFICIAL_URL = "https://www.cmu.edu/math/people/faculty/larsson.html"
SITE_URL = "https://sites.google.com/view/martin-larsson"
EXPECTED_DOIS = {
    "10.1093/jrsssb/qkae061",
    "10.1093/jrsssb/qkad120",
    "10.2139/ssrn.3170978",
}
EXPECTED_PUBLISHER_URLS = {
    "10.1093/jrsssb/qkae061": (
        "https://academic.oup.com/jrsssb/article/86/5/1135/7700656"
    ),
    "10.1093/jrsssb/qkad120": (
        "https://academic.oup.com/jrsssb/article/86/1/39/7303215"
    ),
    "10.2139/ssrn.3170978": (
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3170978"
    ),
}
EXPECTED_ADDITIONAL_IDS = {
    "arxiv:2402.00713",
    "arxiv:2601.05217",
    "arxiv:2606.24768",
    "arxiv:2608.09840",
    "arxiv:1008.1754",
    "arxiv:1104.5272",
    "ercim-news:145:optimal-gambling-strategies-when-testing-by-betting",
    "ercim-news:145:a-map-of-sub-gaussian-e-variables",
}
SITE_ROW_HOLDS = {
    1: "article_link_points_to_row_11",
    5: "journal_manifestation_unpaired",
    9: "retitled_journal_manifestation_unpaired",
    22: "truncated_bold_span_title",
    36: "book_chapter_manifestation_unpaired",
    50: "article_link_points_to_row_51",
}


class _PapersList(HTMLParser):
    """Extract citations and links from the author's single named list."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_heading = False
        self.heading_parts: list[str] = []
        self.awaiting_list = False
        self.list_depth = 0
        self.in_entry = False
        self.entry_parts: list[str] = []
        self.entry_links: list[dict[str, str]] = []
        self.current_link: dict[str, str] | None = None
        self.entries: list[tuple[str, list[dict[str, str]]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "h2":
            self.in_heading = True
            self.heading_parts = []
        elif tag == "ul" and self.awaiting_list and not self.list_depth:
            self.list_depth = 1
        elif tag == "ul" and self.list_depth:
            self.list_depth += 1
        elif tag == "li" and self.list_depth == 1:
            self.in_entry = True
            self.entry_parts = []
            self.entry_links = []
        elif tag == "a" and self.in_entry and attributes.get("href"):
            self.current_link = {"label": "", "url": str(attributes["href"])}
            self.entry_links.append(self.current_link)

    def handle_data(self, value: str) -> None:
        if self.in_heading:
            self.heading_parts.append(value)
        if self.in_entry:
            self.entry_parts.append(value)
        if self.current_link is not None:
            self.current_link["label"] += value

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2" and self.in_heading:
            self.in_heading = False
            self.awaiting_list = (
                " ".join("".join(self.heading_parts).split()) == "Papers and preprints"
            )
        elif tag == "a":
            self.current_link = None
        elif tag == "li" and self.in_entry:
            self.in_entry = False
            self.entries.append(
                (" ".join("".join(self.entry_parts).split()), self.entry_links)
            )
        elif tag == "ul" and self.list_depth:
            self.list_depth -= 1
            if not self.list_depth:
                self.awaiting_list = False


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(raw: dict[str, bytes], filename: str) -> dict[str, Any]:
    return json.loads(raw[filename])


def _common_row() -> dict[str, Any]:
    return {
        "person_id": PERSON_ID,
        "display_name": "Martin Larsson",
        "primary_department_id": "math",
        "primary_department_source_url": OFFICIAL_URL,
        "scholar_id": None,
        "scholar_account_status": "no_verified_account",
        "year": None,
        "canonical_title": None,
        "canonical_authors": None,
        "citation_count": None,
        "citation_count_known": False,
        "map_eligible": False,
        "collaboration_eligible": False,
        "release_eligible": False,
        "release_state": "source_only_release_held",
    }


def load_staged_larsson_rows(path: Path) -> list[dict[str, Any]]:
    """Return 74 provenance-rich, graph-ineligible source observations."""

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("person_id") != PERSON_ID
        or manifest.get("official_profile_url") != OFFICIAL_URL
        or manifest.get("author_site_url") != SITE_URL
        or manifest.get("author_site_observations") != 63
        or manifest.get("separate_verified_omissions") != 11
        or manifest.get("reviewed_observations") != 74
        or manifest.get("scholar_account_status") != "no_verified_account"
        or manifest.get("release_state") != "held_for_whole_researcher_review"
        or manifest.get("release_eligible") is not False
        or manifest.get("whole_researcher_complete") is not False
    ):
        raise ValueError("Larsson source identity, scope, or release hold changed")
    hashes = manifest["files_sha256"]
    actual_source_files = {
        source.name
        for source in path.iterdir()
        if source.is_file()
        and source.name not in {"manifest.json", "README.md", "load_staged.py"}
    }
    if set(hashes) != actual_source_files:
        raise ValueError("Larsson source inventory changed")
    raw: dict[str, bytes] = {}
    for filename, digest in hashes.items():
        if Path(filename).name != filename:
            raise ValueError("Larsson source filename is not local")
        raw[filename] = (path / filename).read_bytes()
        if _sha256(raw[filename]) != digest:
            raise ValueError(f"Larsson source checksum changed: {filename}")
    captured = _read_json(raw, "author-site-rows.json")
    if (
        captured.get("person_id") != PERSON_ID
        or captured.get("source_url") != SITE_URL
        or captured.get("linked_from_official_profile") != OFFICIAL_URL
        or captured.get("source_type") != "researcher_site"
        or captured.get("source_html_sha256") != _sha256(raw["author-site.html"])
        or len(captured.get("rows", [])) != 63
    ):
        raise ValueError("Larsson author-site capture identity or count changed")
    page = _PapersList()
    page.feed(raw["author-site.html"].decode("utf-8"))
    if len(page.entries) != 63:
        raise ValueError("Larsson author page no longer has 63 list entries")

    rows: list[dict[str, Any]] = []
    for ordinal, (source, (citation, links)) in enumerate(
        zip(captured["rows"], page.entries, strict=True), start=1
    ):
        digest_fields = {
            name: source[name] for name in ("title", "citation_text", "links")
        }
        source_digest = _sha256(
            json.dumps(
                digest_fields,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if (
            source["index"] != ordinal
            or source["citation_text"] != citation
            or source["links"] != links
            or source["row_sha256"] != source_digest
        ):
            raise ValueError(f"Larsson author-site row changed: {ordinal}")
        row = _common_row()
        row.update(
            {
                "observation_id": f"researcher-site:larsson:{source_digest[:20]}",
                "source_kind": "researcher_owned_site",
                "source_url": SITE_URL,
                "source_title_as_listed": source["title"],
                "source_citation_as_listed": citation,
                "source_links_as_listed": links,
                "source_year_tokens_as_listed": source["years_in_citation"],
                "year_basis": "not_normalized_from_site_citation",
                "authors_basis": "author_owned_list_without_complete_byline",
                "doi": None,
                "reviewed_genre": "papers_and_preprints_unclassified",
                "row_hold": SITE_ROW_HOLDS.get(
                    ordinal, "whole_researcher_reconciliation"
                ),
                "source_record": source,
            }
        )
        rows.append(row)

    omissions = _read_json(raw, "omissions.json")
    if (
        omissions.get("schema_version") != 1
        or omissions.get("person_id") != PERSON_ID
        or len(omissions.get("observations", [])) != 3
        or {item["doi"] for item in omissions["observations"]} != EXPECTED_DOIS
    ):
        raise ValueError("Larsson omitted-work review changed")
    for item in omissions["observations"]:
        doi = item["doi"]
        crossref_file = item["crossref_file"]
        registered = _read_json(raw, crossref_file)["message"]
        registered_authors = [
            (author.get("given", "") + " " + author.get("family", "")).strip()
            for author in registered["author"]
        ]
        if (
            item["crossref_response_sha256"] != _sha256(raw[crossref_file])
            or registered["DOI"].casefold() != doi
            or registered["title"] != [item["title_as_registered"]]
            or registered_authors != item["authors_as_registered"]
            or item["publisher_url"] != EXPECTED_PUBLISHER_URLS[doi]
            or item["registry_url"] != f"https://api.crossref.org/works/{doi}"
            or registered["published"]["date-parts"][0][0] != item["online_year"]
            or "Martin Larsson" not in registered_authors
            or item["listed_in_author_site"] is not False
            or item["scholar_profile_observation"] is not False
        ):
            raise ValueError(f"Larsson omitted-work evidence changed: {doi}")
        row = _common_row()
        row.update(
            {
                "observation_id": f"publisher-registration:{doi}",
                "source_kind": "publisher_registered_journal_discussion"
                if item["source_kind"] == "journal_discussion"
                else "arxiv_with_ssrn_preprint",
                "source_url": item["publisher_url"],
                "source_title_as_listed": item["title_as_registered"],
                "source_citation_as_listed": None,
                "source_links_as_listed": None,
                "source_year_tokens_as_listed": None,
                "authors_as_registered": registered_authors,
                "authors_basis": "publisher_registration",
                "doi": doi,
                "reviewed_genre": item["source_kind"],
                "row_hold": "omitted_from_author_site_whole_researcher_reconciliation",
                "source_record": item,
            }
        )
        if item["source_kind"] == "journal_discussion":
            if (
                item["journal_issue_year"] != 2024
                or registered["page"] != item["pages"]
                or not any(
                    "Mathematical Sciences, Carnegie Mellon" in affiliation["name"]
                    for author in registered["author"]
                    if author.get("given") == "Martin"
                    and author.get("family") == "Larsson"
                    for affiliation in author.get("affiliation", [])
                )
            ):
                raise ValueError(f"Larsson journal affiliation changed: {doi}")
            row["year"] = item["online_year"]
            row["year_basis"] = "publisher_registered_online_year"
            row["journal_issue_year"] = item["journal_issue_year"]
        elif item["source_kind"] == "research_preprint":
            arxiv_file = item["arxiv_file"]
            arxiv_raw = gzip.decompress(raw[arxiv_file])
            arxiv_html = arxiv_raw.decode("utf-8")
            title_meta = (
                'name="citation_title" content="Polynomial processes for power prices"'
            )
            if (
                doi != "10.2139/ssrn.3170978"
                or item["arxiv_id"] != "1710.10293"
                or item["arxiv_page_sha256"] != _sha256(arxiv_raw)
                or item["arxiv_first_year"] != 2017
                or item["online_year"] != 2018
                or title_meta not in arxiv_html
                or not all(
                    f'name="citation_author" content="{author}"' in arxiv_html
                    for author in ("Filipovic, Damir", "Larsson, Martin", "Ware, Tony")
                )
            ):
                raise ValueError("Larsson arXiv and SSRN preprint identity changed")
            row["source_url"] = item["arxiv_url"]
            row["source_years"] = {
                "arxiv_first_submission": 2017,
                "ssrn_registration": 2018,
            }
            row["year_basis"] = "two_preprint_dates_no_canonical_year"
        else:
            raise ValueError(f"Unknown Larsson omitted-work kind: {doi}")
        rows.append(row)

    additions = _read_json(raw, "additional-held-eight.json")
    additional_items = additions.get("observations", [])
    if (
        additions.get("schema_version") != 1
        or additions.get("person_id") != PERSON_ID
        or len(additional_items) != 8
        or {item["observation_id"] for item in additional_items}
        != EXPECTED_ADDITIONAL_IDS
    ):
        raise ValueError("Larsson additional source inventory changed")
    for item in additional_items:
        source_file = item["source_page_file"]
        page_raw = gzip.decompress(raw[source_file])
        page = page_raw.decode("utf-8")
        source_kind = item["source_kind"]
        if (
            item["source_page_sha256"] != _sha256(page_raw)
            or "Martin Larsson" not in item["authors_as_listed"]
            or item["listed_in_author_site"] is not False
            or item["scholar_profile_observation"] is not False
            or item["release_eligible"] is not False
            or item["map_eligible"] is not False
            or item["collaboration_eligible"] is not False
            or item["row_hold"] not in {
                "whole_researcher_reconciliation_and_genre_policy",
                "withdrawn_work_no_collaboration_edge",
                "magazine_genre_and_whole_researcher_reconciliation",
            }
        ):
            raise ValueError("Larsson additional source evidence or hold changed")
        if source_kind in {"research_preprint", "masters_thesis", "withdrawn_preprint"}:
            arxiv_id = item["arxiv_id"]
            citation_date = item["arxiv_first_posted"].replace("-", "/")
            expected_meta = [
                f'name="citation_arxiv_id" content="{arxiv_id}"',
                f'name="citation_title" content="{item["title_as_listed"]}"',
                f'name="citation_date" content="{citation_date}"',
            ]
            expected_authors = []
            for name in item["authors_as_listed"]:
                given, family = name.rsplit(" ", 1)
                expected_authors.append(
                    f'name="citation_author" content="{family}, {given}"'
                )
            if (
                item["observation_id"] != f"arxiv:{arxiv_id}"
                or item["source_url"] != f"https://arxiv.org/abs/{arxiv_id}"
                or source_file != f"arxiv-{arxiv_id}.html.gz"
                or not all(meta in page for meta in expected_meta)
                or not all(author in page for author in expected_authors)
                or item["withdrawn"] is not (source_kind == "withdrawn_preprint")
                or (
                    source_kind == "withdrawn_preprint"
                    and "This paper has been withdrawn" not in page
                )
            ):
                raise ValueError("Larsson additional arXiv evidence changed")
            source_years = {"arxiv_first_posted": item["arxiv_first_posted"]}
        elif source_kind == "magazine_article":
            slug = item["source_url"].rsplit("/", 1)[-1]
            readable_page = html.unescape(page)
            if (
                item["observation_id"] != f"ercim-news:145:{slug}"
                or source_file != f"ercim-{slug}.html.gz"
                or not item["source_url"].startswith(
                    "https://ercim-news.ercim.eu/en145/special/"
                )
                or f"<title>{item['title_as_listed']}</title>" not in page
                or not all(
                    author in readable_page for author in item["authors_as_listed"]
                )
                or not item["published_date"].startswith("2026-07-")
            ):
                raise ValueError("Larsson ERCIM evidence changed")
            source_years = {"published_date": item["published_date"]}
        else:
            raise ValueError("Unknown Larsson additional source kind")
        row = _common_row()
        row.update(
            {
                "observation_id": item["observation_id"],
                "source_kind": source_kind,
                "source_url": item["source_url"],
                "source_title_as_listed": item["title_as_listed"],
                "source_citation_as_listed": None,
                "source_links_as_listed": None,
                "source_year_tokens_as_listed": None,
                "source_years": source_years,
                "year_basis": "source_date_not_normalized_for_publication",
                "authors_as_registered": item["authors_as_listed"],
                "authors_basis": "primary_source_byline",
                "doi": None,
                "reviewed_genre": source_kind,
                "row_hold": item["row_hold"],
                "source_record": item,
            }
        )
        rows.append(row)

    if (
        len(rows) != 74
        or len({row["observation_id"] for row in rows}) != 74
        or sum(row["source_kind"] == "researcher_owned_site" for row in rows) != 63
        or sum(
            row["source_kind"] == "publisher_registered_journal_discussion"
            for row in rows
        )
        != 2
        or sum(row["source_kind"] == "arxiv_with_ssrn_preprint" for row in rows) != 1
        or sum(row["source_kind"] == "research_preprint" for row in rows) != 4
        or sum(row["source_kind"] == "magazine_article" for row in rows) != 2
        or sum(row["source_kind"] == "masters_thesis" for row in rows) != 1
        or sum(row["source_kind"] == "withdrawn_preprint" for row in rows) != 1
        or any(row["map_eligible"] or row["collaboration_eligible"] for row in rows)
    ):
        raise ValueError("Larsson staged source coverage or graph hold changed")
    return rows
