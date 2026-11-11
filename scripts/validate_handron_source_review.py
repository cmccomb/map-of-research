"""Validate David Handron's held primary-source observations."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

SOURCE_DIR = (
    Path(__file__).resolve().parents[1] / "data/source_reviews/david-handron-2026-10-25"
)
PERSON_ID = "person-davidhandron-66faa357"
HOLD_REASON = "no_verified_scholar_or_current_complete_author_bibliography"
ATOM = "{http://www.w3.org/2005/Atom}"
OPEN_SEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _strip_markup(raw: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", raw)).split())


def validate(source_dir: Path = SOURCE_DIR) -> dict[str, int]:
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["person_id"] == PERSON_ID
    assert manifest["status"] == "review_only_no_snapshot_no_release"
    assert manifest["scholar_account_status"] == "no_verified_account"
    assert manifest["orcid_status"] == "no_self_asserted_account_found"
    assert manifest["lifetime_publication_completeness"] == "not_established"
    assert manifest["arxiv_queries"] == [
        'au:"David G. C. Handron"',
        'au:"David Handron"',
    ]
    assert manifest["kilthub_search_request"] == {
        "search_for": "David Handron",
        "page_size": 100,
        "pages": [1, 2],
    }
    assert manifest["counts"] == {
        "cmu_selected_citations": 1,
        "author_papers_and_preprints": 2,
        "rice_thesis_records": 1,
        "arxiv_exact_author_entries": 1,
        "crossref_publisher_registered_doi": 1,
        "attributed_source_observations_not_works": 6,
        "provisional_work_groups_not_lifetime_count": 3,
        "uncaptured_conference_leads_not_ingested": 3,
        "kilthub_search_hits": 0,
    }
    assert len(manifest["files"]) == 15
    actual_files = {
        path.name
        for path in source_dir.iterdir()
        if path.is_file()
        and path.name not in {"manifest.json", "README.md", ".gitattributes"}
    }
    assert set(manifest["files"]) == actual_files
    for filename, expected_hash in manifest["files"].items():
        assert hashlib.sha256((source_dir / filename).read_bytes()).hexdigest() == (
            expected_hash
        ), filename

    captures = json.loads((source_dir / "capture-http-responses.json").read_text())
    assert len(captures) == 10
    assert {row["filename"] for row in captures} == {
        name
        for name in manifest["files"]
        if name
        not in {
            "capture-http-responses.json",
            "cmu-selected-1.csv",
            "author-page-papers-2.csv",
            "source-observations-6.csv",
            "uncaptured-conference-leads-3.csv",
        }
    }
    assert {row["status"] for row in captures} == {200}
    for row in captures:
        assert (source_dir / row["filename"]).stat().st_size == row["bytes"]
    assert len(manifest["access_checks"]) == 3
    assert any("Google sign-in" in row["result"] for row in manifest["access_checks"])
    assert any("Rice API item" in row["result"] for row in manifest["access_checks"])
    assert any("403" in row["result"] for row in manifest["access_checks"])

    cmu = (source_dir / "cmu-profile.html").read_text(encoding="utf-8")
    section = cmu.split("<h1> Select Publications </h1>", 1)[1].split(
        'id="footer-grid"', 1
    )[0]
    citations = [
        _strip_markup(raw) for raw in re.findall(r"<p>(.*?)</p>", section, re.S)
    ]
    selected = _rows(source_dir / "cmu-selected-1.csv")
    assert len(selected) == len(citations) == 1
    assert selected[0]["row_no"] == "1"
    assert selected[0]["citation_as_listed"] == citations[0]
    assert selected[0]["source_url"] == (
        "https://www.cmu.edu/math/people/faculty/handron.html"
    )

    personal = (source_dir / "cmu-personal-home.html").read_text(encoding="utf-8")
    assert "Schedule: Fall 2010" in personal
    personal_section = personal.split("<H4>Papers and Preprints</H4>", 1)[1].split(
        "</ul>", 1
    )[0]
    personal_citations = [
        _strip_markup(raw)
        for raw in re.findall(r"<li>(.*?)(?=<li>|$)", personal_section, re.I | re.S)
    ]
    author_rows = _rows(source_dir / "author-page-papers-2.csv")
    assert len(personal_citations) == len(author_rows) == 2
    assert [row["row_no"] for row in author_rows] == ["1", "2"]
    assert [row["citation_as_listed"] for row in author_rows] == personal_citations
    assert [row["source_url"] for row in author_rows] == [
        "https://www.math.cmu.edu/~handron/"
    ] * 2
    assert author_rows[0]["linked_url"] == "https://arxiv.org/abs/math/0406486"
    assert author_rows[1]["linked_url"] == ""

    rice = json.loads((source_dir / "rice-thesis-api.json").read_text())
    rice_meta = rice["metadata"]
    assert rice["uuid"] == "3f2f6423-2661-454a-8d2e-e17e4814b734"
    assert rice_meta["dc.creator"][0]["value"] == "Handron, David Gerard Christian"
    assert rice_meta["dc.title"][0]["value"] == (
        "Generalized billiard paths and Morse theory for manifolds with corners"
    )
    assert rice_meta["dc.date.issued"][0]["value"] == "1999"
    assert rice_meta["dc.type"][0]["value"] == "Thesis"
    assert rice_meta["thesis.degree.grantor"][0]["value"] == "Rice University"

    arxiv = ET.fromstring((source_dir / "arxiv-author-page-1.xml").read_bytes())
    arxiv_second = ET.fromstring((source_dir / "arxiv-author-page-2.xml").read_bytes())
    arxiv_broad = ET.fromstring((source_dir / "arxiv-author-broad.xml").read_bytes())
    assert arxiv.findtext(f"{OPEN_SEARCH}totalResults") == "1"
    assert arxiv_second.findtext(f"{OPEN_SEARCH}totalResults") == "1"
    assert arxiv_second.findall(f"{ATOM}entry") == []
    entries = arxiv.findall(f"{ATOM}entry")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.findtext(f"{ATOM}id") == "http://arxiv.org/abs/math/0406486v2"
    assert entry.findtext(f"{ATOM}title") == (
        "The Morse Complex for a Morse Function on a Manifold with Corners"
    )
    assert [
        author.findtext(f"{ATOM}name") for author in entry.findall(f"{ATOM}author")
    ] == ["David G. C. Handron"]
    assert [
        other.findtext(f"{ATOM}id") for other in arxiv_broad.findall(f"{ATOM}entry")
    ] == [entry.findtext(f"{ATOM}id")]
    assert "David G. C. Handron" in (source_dir / "arxiv-morse-complex.html").read_text(
        encoding="utf-8"
    )

    crossref = json.loads((source_dir / "crossref-billiards-doi.json").read_text())[
        "message"
    ]
    assert crossref["DOI"].lower() == "10.1016/s0166-8641(02)00036-6"
    assert crossref["title"] == [rice_meta["dc.title"][0]["value"]]
    assert crossref["author"][0]["given"] == "David G.C."
    assert crossref["author"][0]["family"] == "Handron"
    assert crossref["type"] == "journal-article"
    assert crossref["container-title"] == ["Topology and its Applications"]
    assert (crossref["volume"], crossref["issue"], crossref["page"]) == (
        "126",
        "1-2",
        "83-118",
    )
    assert json.loads((source_dir / "kilthub-search-page-1.json").read_text()) == []
    assert json.loads((source_dir / "kilthub-search-page-2.json").read_text()) == []

    observations = _rows(source_dir / "source-observations-6.csv")
    assert len(observations) == 6
    assert [row["source_row_id"] for row in observations] == [
        "CMU-01",
        "AUTHOR-01",
        "AUTHOR-02",
        "RICE-01",
        "ARXIV-01",
        "CROSSREF-01",
    ]
    assert Counter(row["source"] for row in observations) == {
        "cmu_selected_profile": 1,
        "author_papers_preprints": 2,
        "rice_repository_thesis": 1,
        "arxiv_exact_author_query": 1,
        "publisher_registered_doi": 1,
    }
    assert Counter(row["provisional_work_group"] for row in observations) == {
        "W01": 1,
        "W02": 3,
        "W03": 2,
    }
    assert observations[3]["source_genre"] == "doctoral_thesis"
    assert observations[5]["source_genre"] == "journal_article"
    assert observations[3]["title_as_listed"] == observations[5]["title_as_listed"]
    assert (
        observations[3]["provisional_work_group"]
        != observations[5]["provisional_work_group"]
    )
    assert observations[4]["source_url"] == "https://arxiv.org/abs/math/0406486v2"
    assert {row["graph_eligible"] for row in observations} == {"false"}
    assert {row["hold_reason"] for row in observations} == {HOLD_REASON}

    leads = _rows(source_dir / "uncaptured-conference-leads-3.csv")
    assert [row["lead_id"] for row in leads] == ["AMS-2002", "AMS-2004", "AMS-2006"]
    assert all(row["graph_eligible"] == "false" for row in leads)
    assert all("denied" in row["capture_status"] for row in leads)
    assert all(row["source_type"].startswith("meeting_") for row in leads)
    return {
        "cmu_selected": len(selected),
        "author_listed": len(author_rows),
        "arxiv_exact_author": len(entries),
        "attributed_source_observations_not_works": len(observations),
        "provisional_work_groups_not_lifetime_count": 3,
        "uncaptured_conference_leads": len(leads),
    }


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
