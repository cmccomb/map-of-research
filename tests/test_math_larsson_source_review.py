"""Guard a held Larsson bibliography against partial capture and false identity."""

from __future__ import annotations

import gzip
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "data/source_reviews/martin-larsson-2026-10-25"


class _PapersList(HTMLParser):
    """Read only the first list after the exact author-page heading."""

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(filename: str) -> dict:
    return json.loads((REVIEW / filename).read_text(encoding="utf-8"))


def test_larsson_exact_author_list_and_release_hold() -> None:
    manifest = _record("manifest.json")
    assert manifest["person_id"] == "person-martinlarsson-970afb46"
    assert manifest["scholar_account_status"] == "no_verified_account"
    assert manifest["release_state"] == "held_for_whole_researcher_review"
    assert manifest["release_eligible"] is False
    assert manifest["whole_researcher_complete"] is False
    counts = (manifest["author_site_observations"], manifest["reviewed_observations"])
    assert counts == (
        63,
        74,
    )
    assert manifest["separate_verified_omissions"] == 11
    for filename, digest in manifest["files_sha256"].items():
        assert _sha256(REVIEW / filename) == digest

    captured = _record("author-site-rows.json")
    assert captured["person_id"] == manifest["person_id"]
    assert captured["source_type"] == "researcher_site"
    assert captured["source_html_sha256"] == _sha256(REVIEW / "author-site.html")
    extractor = _PapersList()
    extractor.feed((REVIEW / "author-site.html").read_text(encoding="utf-8"))
    assert len(extractor.entries) == len(captured["rows"]) == 63
    assert [row["index"] for row in captured["rows"]] == list(range(1, 64))
    for row, (citation, links) in zip(captured["rows"], extractor.entries, strict=True):
        assert row["citation_text"] == citation
        assert row["links"] == links
        digest_fields = {key: row[key] for key in ("title", "citation_text", "links")}
        expected = hashlib.sha256(
            json.dumps(
                digest_fields,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        assert row["row_sha256"] == expected


def test_three_separate_omissions_have_exact_primary_attribution() -> None:
    omissions = _record("omissions.json")
    assert omissions["person_id"] == "person-martinlarsson-970afb46"
    items = omissions["observations"]
    assert len(items) == len({item["observation_id"] for item in items}) == 3
    expected_authors = {
        "10.1093/jrsssb/qkae061": [
            "Martin Larsson",
            "Aaditya Ramdas",
            "Johannes Ruf",
        ],
        "10.1093/jrsssb/qkad120": ["Martin Larsson", "Johannes Ruf"],
        "10.2139/ssrn.3170978": [
            "Damir Filipovic",
            "Martin Larsson",
            "Tony Ware",
        ],
    }
    assert {item["doi"] for item in items} == set(expected_authors)
    site = _record("author-site-rows.json")
    site_titles = {row["citation_text"].casefold() for row in site["rows"]}
    for item in items:
        crossref = _record(item["crossref_file"])["message"]
        assert item["listed_in_author_site"] is False
        assert item["scholar_profile_observation"] is False
        assert item["doi"].casefold() == crossref["DOI"].casefold()
        assert item["title_as_registered"] == crossref["title"][0]
        assert item["title_as_registered"].casefold() not in site_titles
        assert item["crossref_response_sha256"] == _sha256(
            REVIEW / item["crossref_file"]
        )
        authors = [
            (author.get("given", "") + " " + author.get("family", "")).strip()
            for author in crossref["author"]
        ]
        assert authors == item["authors_as_registered"] == expected_authors[item["doi"]]
        if item["source_kind"] == "journal_discussion":
            assert any(
                "Mathematical Sciences, Carnegie Mellon" in affiliation["name"]
                for affiliation in crossref["author"][0]["affiliation"]
            )
            assert item["journal_issue_year"] == 2024
        else:
            assert item["source_kind"] == "research_preprint"
            assert item["arxiv_id"] == "1710.10293"
            arxiv_bytes = gzip.decompress((REVIEW / item["arxiv_file"]).read_bytes())
            assert item["arxiv_page_sha256"] == hashlib.sha256(arxiv_bytes).hexdigest()
            arxiv_html = arxiv_bytes.decode("utf-8")
            title_meta = (
                'name="citation_title" content="Polynomial processes for power prices"'
            )
            assert title_meta in arxiv_html
            for name in ("Filipovic, Damir", "Larsson, Martin", "Ware, Tony"):
                assert f'name="citation_author" content="{name}"' in arxiv_html
            assert 'name="citation_arxiv_id" content="1710.10293"' in arxiv_html


def test_broken_article_links_and_truncated_title_remain_visible_for_review() -> None:
    rows = _record("author-site-rows.json")["rows"]
    assert rows[0]["links"][0]["url"] == rows[10]["links"][0]["url"]
    assert rows[49]["links"][0]["url"] == rows[50]["links"][0]["url"]
    assert rows[21]["title"] == "Robust"
    assert rows[21]["citation_text"].startswith(
        "Robust asymptotic growth in stochastic portfolio theory"
    )
    assert _record("moor.2025.1129.crossref.json")["message"]["title"] == [
        "Testing Hypotheses Generated by Constraints"
    ]
    assert _record("ejp.v20-3785.crossref.json")["message"]["title"] == [
        "Matrix-valued Bessel processes"
    ]
    possible_variant = _record("ssrn.2040907.crossref.json")["message"]
    assert possible_variant["title"] == ["Economic Impact of Naked Sovereign CDS Ban"]
    assert [author["family"] for author in possible_variant["author"]] == [
        "Capponi",
        "Larsson",
    ]
    assert possible_variant["DOI"] not in {
        item["doi"] for item in _record("omissions.json")["observations"]
    }


def test_eight_additional_primary_source_records_remain_held() -> None:
    additions = _record("additional-held-eight.json")
    assert additions["person_id"] == "person-martinlarsson-970afb46"
    items = additions["observations"]
    assert len(items) == len({item["observation_id"] for item in items}) == 8
    assert {item["source_kind"] for item in items} == {
        "research_preprint", "masters_thesis", "withdrawn_preprint", "magazine_article"
    }
    assert sum(item["source_kind"] == "research_preprint" for item in items) == 4
    assert sum(item["source_kind"] == "magazine_article" for item in items) == 2
    site = _record("author-site-rows.json")
    site_titles = {row["citation_text"].casefold() for row in site["rows"]}
    for item in items:
        raw = gzip.decompress((REVIEW / item["source_page_file"]).read_bytes())
        page = raw.decode("utf-8")
        assert hashlib.sha256(raw).hexdigest() == item["source_page_sha256"]
        assert item["title_as_listed"].casefold() not in site_titles
        assert "Martin Larsson" in item["authors_as_listed"]
        assert item["listed_in_author_site"] is False
        assert item["scholar_profile_observation"] is False
        assert item["release_eligible"] is False
        assert item["map_eligible"] is False
        assert item["collaboration_eligible"] is False
        if item.get("arxiv_id"):
            arxiv_meta = 'name="citation_arxiv_id" content="{}"'.format(
                item["arxiv_id"]
            )
            title_meta = 'name="citation_title" content="{}"'.format(
                item["title_as_listed"]
            )
            assert arxiv_meta in page
            assert title_meta in page
            assert item["withdrawn"] is (item["source_kind"] == "withdrawn_preprint")
            if item["withdrawn"]:
                assert "This paper has been withdrawn" in page
        else:
            assert item["source_kind"] == "magazine_article"
            assert item["published_date"].startswith("2026-07-")
