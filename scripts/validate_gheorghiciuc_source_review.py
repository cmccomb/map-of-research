"""Validate Irina Gheorghiciuc's held primary-source observations."""

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
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/irina-gheorghiciuc-2026-10-25"
)
PERSON_ID = "person-irinagheorghiciuc-c53bff46"
HOLD_REASON = "no_verified_scholar_or_complete_author_bibliography"
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
    assert manifest["counts"] == {
        "cmu_selected_citations": 3,
        "arxiv_exact_author_entries": 4,
        "penn_thesis_listings": 1,
        "dmtcs_publisher_records": 1,
        "jis_publisher_records": 1,
        "attributed_source_observations_not_works": 10,
        "provisional_work_groups_not_lifetime_count": 6,
        "kilthub_search_hits": 1,
        "kilthub_excluded_other_author_theses": 1,
    }
    assert manifest["arxiv_query"] == {
        "search_query": 'au:"Irina Gheorghiciuc"',
        "start": [0, 100],
        "max_results": 100,
    }
    assert manifest["kilthub_search_request"] == {
        "search_for": "Irina Gheorghiciuc",
        "page_size": 100,
        "pages": [1, 2],
    }
    assert len(manifest["files"]) == 19
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
    assert len(captures) == 14
    assert {row["filename"] for row in captures} == {
        name
        for name in manifest["files"]
        if name
        not in {
            "capture-http-responses.json",
            "cmu-selected-3.csv",
            "arxiv-exact-author-4.csv",
            "source-observations-10.csv",
            "kilthub-search-review-1.csv",
        }
    }
    assert Counter(row["status"] for row in captures) == {200: 13, 404: 1}
    assert {row["filename"] for row in captures if row["status"] == 404} == {
        "cmu-personal-research-response.html"
    }
    assert len(manifest["access_checks"]) == 2
    assert any("Google sign-in" in row["result"] for row in manifest["access_checks"])
    assert any("403" in row["result"] for row in manifest["access_checks"])
    assert "RESEARCH" in (source_dir / "cmu-personal-home.html").read_text()

    cmu_html = (source_dir / "cmu-profile.html").read_text(encoding="utf-8")
    section = cmu_html.split("<h1> Select Publications </h1>", 1)[1].split(
        'id="footer-grid"', 1
    )[0]
    citations = [
        _strip_markup(raw) for raw in re.findall(r"<p>(.*?)</p>", section, re.S)
    ]
    selected = _rows(source_dir / "cmu-selected-3.csv")
    assert len(selected) == len(citations) == 3
    assert [row["row_no"] for row in selected] == ["1", "2", "3"]
    assert [row["citation_as_listed"] for row in selected] == citations
    assert [row["source_url"] for row in selected] == [
        "https://www.cmu.edu/math/people/faculty/gheorghiciuc.html"
    ] * 3
    assert citations[2].endswith("Journal of Integer Sequences, 10.")

    page_1 = ET.fromstring((source_dir / "arxiv-author-page-1.xml").read_bytes())
    page_2 = ET.fromstring((source_dir / "arxiv-author-page-2.xml").read_bytes())
    entries = page_1.findall(f"{ATOM}entry")
    assert page_1.findtext(f"{OPEN_SEARCH}totalResults") == "4"
    assert page_2.findtext(f"{OPEN_SEARCH}totalResults") == "4"
    assert len(entries) == 4
    assert page_2.findall(f"{ATOM}entry") == []
    arxiv_rows = _rows(source_dir / "arxiv-exact-author-4.csv")
    assert len(arxiv_rows) == 4
    assert [row["row_no"] for row in arxiv_rows] == ["1", "2", "3", "4"]
    assert len({row["arxiv_id_version"] for row in arxiv_rows}) == 4
    for entry, row in zip(entries, arxiv_rows, strict=True):
        assert entry.findtext(f"{ATOM}id") == (
            "http://arxiv.org/abs/" + row["arxiv_id_version"]
        )
        assert (
            " ".join((entry.findtext(f"{ATOM}title") or "").split())
            == row["title_as_listed"]
        )
        authors = [
            author.findtext(f"{ATOM}name") for author in entry.findall(f"{ATOM}author")
        ]
        assert "Irina Gheorghiciuc" in authors
        assert row["authors_as_listed"] == " | ".join(authors)
        assert (entry.findtext(f"{ATOM}published") or "")[:10] == row["first_posted"]
        assert row["source_url"] == ("https://arxiv.org/abs/" + row["arxiv_id_version"])

    penn = (source_dir / "penn-math-phds.html").read_text(encoding="utf-8")
    assert re.search(
        r"Gheorghiciuc, Irina</td>\s*<td>2004<br />\(H\.S\. Wilf\)</td>"
        r"\s*<td><em>The subword complexity of finite and infinite binary words</em>",
        penn,
    )

    tei = ET.fromstring((source_dir / "dmtcs-3553-tei.xml").read_bytes())
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    assert tei.findtext(".//tei:biblStruct/tei:analytic/tei:title", namespaces=ns) == (
        "On Correlation Polynomials and Subword Complexity"
    )
    assert [
        " ".join(name.itertext())
        for name in tei.findall(
            ".//tei:biblStruct/tei:analytic/tei:author/tei:persName", ns
        )
    ] == ["Irina Gheorghiciuc", "Mark Daniel Ward"]
    assert tei.findtext(".//tei:idno[@type='doi']", namespaces=ns) == (
        "10.46298/dmtcs.3553"
    )
    assert "DMTCS Proceedings vol. AH" in tei.findtext(
        ".//tei:biblScope[@unit='volume']", namespaces=ns
    )

    jis = (source_dir / "jis-gheo.html").read_text(encoding="utf-8")
    assert "Emily Allen and Irina Gheorghiciuc" in jis
    assert "Vol. 17" in jis and "Article 14.10.7" in jis
    assert "November 5 2014" in jis
    jis_index = (source_dir / "jis-vol17.html").read_text(encoding="utf-8")
    assert "Emily Allen and Irina Gheorghiciuc" in jis_index
    assert "Article 14.10.7" in jis_index
    assert (source_dir / "jis-gheo.pdf").read_bytes().startswith(b"%PDF")

    search_1 = json.loads((source_dir / "kilthub-search-page-1.json").read_text())
    search_2 = json.loads((source_dir / "kilthub-search-page-2.json").read_text())
    detail = json.loads((source_dir / "kilthub-article-6715193.json").read_text())
    excluded = _rows(source_dir / "kilthub-search-review-1.csv")
    assert len(search_1) == len(excluded) == 1 and search_2 == []
    assert search_1[0]["id"] == detail["id"] == 6715193
    assert detail["defined_type_name"] == "thesis"
    assert [author["full_name"].strip() for author in detail["authors"]] == [
        "Emily  Allen"
    ]
    assert excluded[0]["attributed_to_irina"] == "false"
    assert excluded[0]["source_url"] == detail["url_public_html"]

    observations = _rows(source_dir / "source-observations-10.csv")
    assert len(observations) == 10
    assert len({row["source_row_id"] for row in observations}) == 10
    assert Counter(row["source"] for row in observations) == {
        "cmu_selected_profile": 3,
        "arxiv_exact_author_query": 4,
        "penn_math_phd_history": 1,
        "publisher_tei": 1,
        "publisher_article": 1,
    }
    assert [row["source_row_id"] for row in observations[:3]] == [
        "CMU-01",
        "CMU-02",
        "CMU-03",
    ]
    assert [row["title_as_listed"] for row in observations[:3]] == [
        row["title_transcription"] for row in selected
    ]
    assert [row["source_row_id"] for row in observations[3:7]] == [
        "ARXIV-01",
        "ARXIV-02",
        "ARXIV-03",
        "ARXIV-04",
    ]
    assert [row["title_as_listed"] for row in observations[3:7]] == [
        row["title_as_listed"] for row in arxiv_rows
    ]
    assert {row["provisional_work_group"] for row in observations} == {
        "W01",
        "W02",
        "W03",
        "W04",
        "W05",
        "W06",
    }
    assert Counter(row["provisional_work_group"] for row in observations) == {
        "W01": 2,
        "W02": 2,
        "W03": 3,
        "W04": 1,
        "W05": 1,
        "W06": 1,
    }
    assert (
        next(row for row in observations if row["source_row_id"] == "PENN-01")[
            "source_genre"
        ]
        == "doctoral_thesis"
    )
    assert (
        next(row for row in observations if row["source_row_id"] == "DMTCS-01")[
            "source_genre"
        ]
        == "conference_proceedings"
    )
    assert (
        next(row for row in observations if row["source_row_id"] == "JIS-01")[
            "source_genre"
        ]
        == "journal_article"
    )
    assert {row["graph_eligible"] for row in observations} == {"false"}
    assert {row["hold_reason"] for row in observations} == {HOLD_REASON}
    assert not any("6715193" in row["source_url"] for row in observations)
    return {
        "cmu_selected": len(selected),
        "arxiv_exact_author": len(arxiv_rows),
        "attributed_source_observations_not_works": len(observations),
        "provisional_work_groups_not_lifetime_count": 6,
        "excluded_other_author_theses": len(excluded),
    }


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
