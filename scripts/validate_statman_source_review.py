"""Validate Richard Statman's held CMU source evidence and attribution."""

from __future__ import annotations

import csv
import gzip
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote, unquote

SOURCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/richard-statman-2026-10-25"
)
PERSON_ID = "person-richardstatman-2e46bf59"
KILTHUB_AUTHOR_ID = 5364608


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _citation(text: str) -> str:
    return re.sub(r"\s+([.,;:])", r"\1", " ".join(text.split()))


def _clean_markup(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]*>", "", value)).split())


def _normalized_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", html.unescape(value).casefold())


def validate(source_dir: Path = SOURCE_DIR) -> dict[str, int]:
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["person_id"] == PERSON_ID
    assert manifest["status"] == "review_only_no_snapshot_no_release"
    assert manifest["scholar_account_status"] == "no_verified_account"
    assert manifest["lifetime_publication_completeness"] == "not_established"
    assert manifest["sources"]["kilthub_statman_author_id"] == KILTHUB_AUTHOR_ID
    assert manifest["sources"]["kilthub_search_request"] == {
        "search_for": "Richard Statman",
        "page_size": 100,
        "pages": [1, 2],
    }
    assert manifest["sources"]["springer_lfcs_publisher"] == (
        "https://link.springer.com/chapter/10.1007/978-3-030-93100-1_20"
    )
    assert manifest["sources"]["excluded_misattributed_orcid"] == (
        "0000-0002-1825-0097"
    )
    assert manifest["sources"]["cambridge_book_crossref"] == (
        "https://api.crossref.org/works/10.1017/CBO9781139032636"
    )
    assert manifest["sources"]["statman_dowek_arxiv"] == (
        "https://arxiv.org/abs/2309.03602"
    )
    assert manifest["sources"]["csauthors_bibliography_discovery"] == (
        "https://www.csauthors.net/richard-statman/"
    )
    assert manifest["counts"] == {
        "cmu_profile_selected": 12,
        "cmu_recent_selected": 4,
        "kilthub_search_hits": 19,
        "kilthub_attributed_reports_or_manuscripts": 17,
        "kilthub_other_author_theses": 2,
        "publisher_confirmed_conference_papers": 1,
        "publisher_confirmed_monographs": 1,
        "historical_report_observations": 1,
        "arxiv_preprint_observations": 1,
        "attributed_source_observations_not_works": 37,
        "third_party_bibliography_candidates": 67,
        "third_party_exact_title_matches_to_attributed_37": 18,
        "third_party_unmatched_titles_needing_primary_review": 49,
        "third_party_doi_leads_with_crossref_registration": 38,
        "third_party_doi_exact_normalized_titles": 27,
        "third_party_doi_title_variants_for_review": 11,
        "third_party_doi_registrations_without_published_date": 6,
        "third_party_non_doi_leads_reviewed": 11,
        "third_party_non_doi_arxiv_journal_version_pairs": 2,
        "third_party_non_doi_crossref_registrations": 7,
    }
    assert len(manifest["files"]) == 39
    for filename, expected_hash in manifest["files"].items():
        assert hashlib.sha256((source_dir / filename).read_bytes()).hexdigest() == (
            expected_hash
        ), filename

    profile = (source_dir / "cmu-profile.html").read_text(encoding="utf-8")
    section = profile.split("<h1> Select Publications </h1>", 1)[1].split(
        'id="footer-grid"', 1
    )[0]
    citations = [
        _citation(html.unescape(re.sub(r"<[^>]+>", "", raw)))
        for raw in re.findall(r"<p>(.*?)</p>", section, re.S)
    ]
    # The extra footer paragraph links to the IEEE Symposium; it is not a paper.
    assert len(citations) == 13
    assert citations[-1] == "IEEE Symposium on Logic in Computer Science"
    selected = _rows(source_dir / "cmu-selected-16.csv")
    assert len(selected) == 16
    assert [row["source_url"] for row in selected[:12]] == [
        manifest["sources"]["cmu_profile"]
    ] * 12
    assert [_citation(row["citation_as_listed"]) for row in selected[:12]] == (
        citations[:12]
    )
    assert [row["row_no"] for row in selected[:12]] == [str(i) for i in range(1, 13)]

    recent_text = " ".join(
        (source_dir / "cmu-sample-publications.txt").read_text().split()
    ).casefold()
    assert "9 papers since 2018" in recent_text
    assert "selected recent publications" in recent_text
    assert [row["source_url"] for row in selected[12:]] == [
        manifest["sources"]["cmu_recent_pdf"]
    ] * 4
    assert [row["row_no"] for row in selected[12:]] == [str(i) for i in range(1, 5)]
    for row in selected[12:]:
        assert row["title_transcription"].casefold() in recent_text

    search_page_1 = json.loads((source_dir / "kilthub-search-page-1.json").read_text())
    search_page_2 = json.loads((source_dir / "kilthub-search-page-2.json").read_text())
    hits = _rows(source_dir / "kilthub-search-19.csv")
    assert len(search_page_1) == len(hits) == 19
    assert search_page_2 == []
    ids = [int(result["id"]) for result in search_page_1]
    assert len(set(ids)) == 19
    assert ids == [int(row["article_id"]) for row in hits]
    assert Counter(row["attributed_to_statman"] for row in hits) == {
        "true": 17,
        "false": 2,
    }
    assert {
        int(row["article_id"])
        for row in hits
        if row["attributed_to_statman"] == "false"
    } == {
        6723047,
        6723053,
    }
    for result, row in zip(search_page_1, hits, strict=True):
        article_id = int(row["article_id"])
        detail = json.loads(
            (source_dir / f"kilthub-article-{article_id}.json").read_text()
        )
        assert detail["id"] == result["id"] == article_id
        assert detail["title"] == result["title"] == row["title"]
        assert detail["url_public_html"] == row["source_url"]
        assert row["source_url"].startswith("https://kilthub.cmu.edu/")
        assert detail["defined_type_name"] == row["repository_type"]
        assert detail["published_date"][:10] == row["repository_date"]
        assert detail["doi"] == row["repository_doi"]
        assert [author["id"] for author in detail["authors"]] == [
            int(value) for value in row["author_ids"].split(" | ")
        ]
        assert row["authors_as_listed"] == " | ".join(
            author["full_name"] for author in detail["authors"]
        )
        attributed = any(
            author["id"] == KILTHUB_AUTHOR_ID for author in detail["authors"]
        )
        assert row["attributed_to_statman"] == str(attributed).lower()
        assert row["pdf_url"] == detail["files"][0]["download_url"]
        if attributed:
            assert row["repository_type"] == "journal contribution"
            assert row["reviewed_genre"] == "cmu_math_report_or_manuscript"
            assert int(row["pdf_bytes"]) == detail["files"][0]["size"]
            assert re.fullmatch(r"[0-9a-f]{64}", row["pdf_sha256"])
        else:
            assert row["repository_type"] == "thesis"
            assert row["reviewed_genre"] == "other_author_thesis"
            assert row["pdf_sha256"] == ""

    observations = _rows(source_dir / "source-observations-37.csv")
    assert len(observations) == 37
    assert len({row["source_row_id"] for row in observations}) == 37
    assert Counter(row["source"] for row in observations) == {
        "cmu_profile_select": 12,
        "cmu_recent_selected": 4,
        "cmu_kilthub_repository": 17,
        "publisher_crossref_verified": 2,
        "coauthor_owned_bibliography": 1,
        "arxiv_primary_record": 1,
    }
    assert {row["source_row_id"] for row in observations[:16]} == {
        f"CMU-{i:02d}" for i in range(1, 17)
    }
    assert {row["source_row_id"] for row in observations[16:33]} == {
        f"KILTHUB-{row['article_id']}"
        for row in hits
        if row["attributed_to_statman"] == "true"
    }
    publisher = observations[33]
    assert publisher["source_row_id"] == "SPRINGER-LFCS-2022-20"
    assert publisher["source"] == "publisher_crossref_verified"
    assert publisher["source_url"] == manifest["sources"]["springer_lfcs_publisher"]
    assert publisher["source_genre"] == "conference paper"
    assert publisher["reviewed_genre"] == "lfcs_proceedings_conference_paper"
    assert publisher["source_year"] == "2022"
    assert "2021-12-16" in publisher["review_note"]
    assert "0000-0002-1825-0097" in publisher["review_note"]
    assert (
        publisher["primary_comparison_url"]
        == manifest["sources"]["springer_lfcs_crossref"]
    )
    crossref = json.loads(
        (source_dir / "springer-lfcs-2022-crossref.json").read_text()
    )["message"]
    assert crossref["DOI"] == "10.1007/978-3-030-93100-1_20"
    assert " ".join(crossref["title"][0].split()) == publisher["source_title"]
    assert [(a["given"], a["family"]) for a in crossref["author"]] == [
        ("Rick", "Statman")
    ]
    assert crossref["author"][0]["ORCID"] == ("https://orcid.org/0000-0002-1825-0097")
    assert crossref["published-online"]["date-parts"] == [[2021, 12, 16]]
    assert crossref["published-print"]["date-parts"] == [[2022]]
    assert crossref["page"] == "316-326"
    assert crossref["type"] == "book-chapter"
    orcid = json.loads(
        (source_dir / "publisher-linked-false-orcid-person.json").read_text()
    )
    assert orcid["path"] == "/0000-0002-1825-0097/person"
    assert orcid["name"]["given-names"]["value"] == "Josiah"
    assert orcid["name"]["family-name"]["value"] == "Carberry"
    book, report, preprint = observations[34:]
    assert (book["source_row_id"], book["source_year"], book["reviewed_genre"]) == (
        "CAMBRIDGE-BOOK-2013",
        "2013",
        "monograph",
    )
    assert book["source_url"] == manifest["sources"]["cambridge_book_publisher"]
    assert (
        book["primary_comparison_url"] == manifest["sources"]["cambridge_book_crossref"]
    )
    registered_book = json.loads(
        (source_dir / "cambridge-book-crossref.json").read_text()
    )["message"]
    assert registered_book["DOI"].casefold() == ("10.1017/CBO9781139032636".casefold())
    assert registered_book["title"] == [book["source_title"]]
    assert registered_book["type"] == "monograph"
    assert [(a["given"], a["family"]) for a in registered_book["author"]] == [
        ("Henk", "Barendregt"),
        ("Wil", "Dekkers"),
        ("Richard", "Statman"),
    ]
    assert registered_book["published-print"]["date-parts"] == [[2013, 6, 20]]
    assert registered_book["published-online"]["date-parts"] == [[2013, 8, 5]]

    assert report["source_row_id"] == "CMU-REPORT-1992-152"
    assert report["source_year"] == "1992"
    assert report["reviewed_genre"] == "cmu_scs_technical_report"
    assert report["source_url"] == manifest["sources"]["dowek_coauthor_bibliography"]
    dowek_page = gzip.decompress(
        (source_dir / "dowek-papers.html.gz").read_bytes()
    ).decode("utf-8")
    assert "On Statman's finite completeness theorem" in " ".join(dowek_page.split())
    assert "CMU-CS-92-152" in dowek_page
    assert "R. Statman, G. Dowek" in dowek_page
    assert preprint["source_row_id"] == "ARXIV-2309.03602"
    assert preprint["source_url"] == manifest["sources"]["statman_dowek_arxiv"]
    assert preprint["source_year"] == "2023"
    assert preprint["reviewed_genre"] == ("arxiv_preprint_possible_1992_report_version")
    arxiv_page = gzip.decompress(
        (source_dir / "arxiv-2309.03602.html.gz").read_bytes()
    ).decode("utf-8")
    for metadata in (
        'name="citation_title" content="On Statman&#39;s Finite Completeness Theorem"',
        'name="citation_author" content="Statman, Richard"',
        'name="citation_author" content="Dowek, Gilles"',
        'name="citation_date" content="2023/09/07"',
        'name="citation_arxiv_id" content="2309.03602"',
    ):
        assert metadata in arxiv_page
    assert "Statman (CMU)" in arxiv_page
    assert report["source_title"] == preprint["source_title"]
    assert report["source_author"] == preprint["source_author"]
    assert "possible" in report["review_note"].lower()
    assert "possible" in preprint["review_note"].lower()

    secondary = json.loads((source_dir / "csauthors-67.json").read_text())
    secondary_page = gzip.decompress(
        (source_dir / "csauthors-page.html.gz").read_bytes()
    )
    assert secondary["schema_version"] == 1
    assert secondary["person_id"] == PERSON_ID
    assert (
        secondary["source_url"]
        == manifest["sources"]["csauthors_bibliography_discovery"]
    )
    assert secondary["source_kind"] == "third_party_bibliography_discovery"
    assert secondary["source_html_sha256"] == hashlib.sha256(secondary_page).hexdigest()
    page_text = secondary_page.decode("utf-8")
    assert "at least 67 papers between 1977 and 2023" in " ".join(page_text.split())
    assert "math.cmu.edu" in page_text
    chunks = re.findall(
        r'<div class="bibentry-([^"]+)">(.*?)(?=<div class="bibentry-|\Z)',
        page_text,
        re.S,
    )
    candidates = secondary["records"]
    assert len(chunks) == len(candidates) == 67
    assert len({row["source_record_key"] for row in candidates}) == 67
    title_lookup: dict[str, list[str]] = {}
    for row in observations:
        title_lookup.setdefault(_normalized_title(row["source_title"]), []).append(
            row["source_row_id"]
        )
    for index, ((kind, fragment), row) in enumerate(
        zip(chunks, candidates, strict=True), start=1
    ):
        title_match = re.search(r'<div class="title">(.*?)</div>', fragment, re.S)
        detail_match = re.search(r'<div class="bibdetails">(.*?)</div>', fragment, re.S)
        ref_match = re.search(r'data-ref="([^"]+)"', fragment)
        doi_match = re.search(
            r'<div class="bibdoi">\s*<a href="([^"]+)"', fragment, re.S
        )
        assert title_match and detail_match and ref_match
        authors = [
            _clean_markup(value)
            for value in re.findall(
                r'<div class="bibauthor">(.*?)</div>', fragment, re.S
            )
        ]
        assert row["index"] == index
        assert row["source_kind"] == kind
        assert row["title_as_displayed"] == _clean_markup(title_match.group(1))
        assert row["authors_as_displayed"] == authors
        assert row["details_as_displayed"] == _clean_markup(detail_match.group(1))
        assert row["source_record_key"] == ref_match.group(1)
        assert row["link_as_displayed"] == (doi_match.group(1) if doi_match else None)
        assert row["exact_normalized_title_matches_in_prior_37"] == (
            title_lookup.get(_normalized_title(row["title_as_displayed"]), [])
        )
        assert row["source_scope"] == "third_party_bibliography_discovery"
        for field in (
            "scholar_profile_observation",
            "release_eligible",
            "map_eligible",
            "collaboration_eligible",
        ):
            assert row[field] is False
        normalized_row = {
            key: value for key, value in row.items() if key != "row_sha256"
        }
        assert (
            row["row_sha256"]
            == hashlib.sha256(
                json.dumps(
                    normalized_row,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
    assert (
        sum(
            bool(row["exact_normalized_title_matches_in_prior_37"])
            for row in candidates
        )
        == 18
    )
    crossref_bundle = json.loads(
        (source_dir / "csauthors-crossref-38.json").read_text()
    )
    assert crossref_bundle["schema_version"] == 1
    assert crossref_bundle["person_id"] == PERSON_ID
    assert crossref_bundle["source"] == (
        "publisher_deposited_crossref_metadata_for_third_party_bibliography_leads"
    )
    publisher_candidates = crossref_bundle["records"]
    assert len(publisher_candidates) == 38
    assert len({row["doi"].casefold() for row in publisher_candidates}) == 38
    candidate_by_key = {row["source_record_key"]: row for row in candidates}
    for row in publisher_candidates:
        candidate = candidate_by_key[row["candidate_source_record_key"]]
        assert candidate["exact_normalized_title_matches_in_prior_37"] == []
        assert row["candidate_index"] == candidate["index"]
        assert row["candidate_title_as_displayed"] == candidate["title_as_displayed"]
        assert row["candidate_link_as_displayed"] == candidate["link_as_displayed"]
        doi_match = re.search(
            r"10\.\d{4,9}/[^?#\s]+",
            unquote(row["candidate_link_as_displayed"]),
            re.I,
        )
        assert doi_match
        doi_from_link = doi_match.group().rstrip(".").replace("\\_", "_")
        assert row["doi"].casefold() == doi_from_link.casefold()
        assert row["crossref_url"] == (
            "https://api.crossref.org/works/" + quote(doi_from_link, safe="")
        )
        registered = row["registered_message"]
        assert registered["DOI"].casefold() == row["doi"].casefold()
        assert registered["title"][0] == row["registered_title"]
        assert row["registered_authors"] == [
            (author.get("given", "") + " " + author.get("family", "")).strip()
            for author in registered["author"]
        ]
        assert (
            sum("statman" in name.casefold() for name in row["registered_authors"]) == 1
        )
        assert row["registered_type"] == registered["type"]
        assert row["registered_published"] == registered.get("published")
        title_status = (
            "exact_normalized"
            if _normalized_title(row["candidate_title_as_displayed"])
            == _normalized_title(row["registered_title"])
            else "variant_requires_review"
        )
        assert row["title_comparison"] == title_status
        assert row["source_scope"] == (
            "publisher_registration_candidate_pending_person_version_review"
        )
        for field in (
            "scholar_profile_observation",
            "release_eligible",
            "map_eligible",
            "collaboration_eligible",
        ):
            assert row[field] is False
        hashed_row = {key: value for key, value in row.items() if key != "row_sha256"}
        assert (
            row["row_sha256"]
            == hashlib.sha256(
                json.dumps(
                    hashed_row,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
    assert (
        sum(
            row["title_comparison"] == "exact_normalized"
            for row in publisher_candidates
        )
        == 27
    )
    assert sum(row["registered_published"] is None for row in publisher_candidates) == 6
    non_doi = json.loads((source_dir / "csauthors-non-doi-11.json").read_text())
    assert non_doi["schema_version"] == 1
    assert non_doi["person_id"] == PERSON_ID
    assert non_doi["source_scope"] == "held_primary_lead_review_not_distinct_works"
    non_doi_rows = non_doi["records"]
    assert [row["index"] for row in non_doi_rows] == [
        7,
        9,
        11,
        20,
        21,
        48,
        53,
        55,
        57,
        61,
        65,
    ]
    assert len({row["source_record_key"] for row in non_doi_rows}) == 11
    assert not {row["source_record_key"] for row in non_doi_rows} & {
        row["candidate_source_record_key"] for row in publisher_candidates
    }
    for row in non_doi_rows:
        candidate = candidate_by_key[row["source_record_key"]]
        assert candidate["index"] == row["index"]
        assert candidate["row_sha256"] == row["source_row_sha256"]
        assert candidate["title_as_displayed"] == row["title_as_displayed"]
        assert candidate["exact_normalized_title_matches_in_prior_37"] == []
        assert row["primary_evidence_url"].startswith("https://")
        for field in (
            "scholar_profile_observation",
            "release_eligible",
            "map_eligible",
            "collaboration_eligible",
        ):
            assert row[field] is False
    assert {row["candidate_source_record_key"] for row in publisher_candidates} | {
        row["source_record_key"] for row in non_doi_rows
    } == {
        row["source_record_key"]
        for row in candidates
        if not row["exact_normalized_title_matches_in_prior_37"]
    }
    registrations = json.loads(
        (source_dir / "csauthors-non-doi-crossref-7.json").read_text()
    )["records"]
    assert len(registrations) == 7
    registration_by_index = {row["index"]: row for row in registrations}
    assert set(registration_by_index) == {9, 11, 21, 55, 57, 61, 65}
    for row in non_doi_rows:
        if row["index"] in registration_by_index:
            registered = registration_by_index[row["index"]]
            assert (
                registered["message"]["DOI"].casefold()
                == row["related_journal_doi"].casefold()
            )
            assert (
                sum(
                    a.get("family", "").casefold() == "statman"
                    for a in registered["message"]["author"]
                )
                == 1
            )
            assert row["publisher_registration_bundle"] == (
                "csauthors-non-doi-crossref-7.json"
            )
        else:
            assert row["publisher_registration_bundle"] is None
    for index, arxiv_id, journal_doi in (
        (7, "1809.08169", "10.46298/lmcs-18(3:35)2022"),
        (20, "0903.1374", "10.2168/LMCS-5(2:6)2009"),
    ):
        row = next(row for row in non_doi_rows if row["index"] == index)
        arxiv_page = gzip.decompress(
            (source_dir / f"arxiv-{arxiv_id}.html.gz").read_bytes()
        ).decode("utf-8")
        assert f'name="citation_arxiv_id" content="{arxiv_id}"' in arxiv_page
        assert "Richard Statman" in arxiv_page
        assert journal_doi.casefold() == row["related_journal_doi"].casefold()
        assert journal_doi.casefold() in unquote(arxiv_page).casefold()
    assert non_doi_rows[0]["existing_observation"] == "CMU-16"
    assert all(row["existing_observation"] is None for row in non_doi_rows[1:])
    assert {row["graph_eligible"] for row in observations} == {"false"}
    assert {row["hold_reason"] for row in observations} == {
        "no_verified_scholar_or_complete_author_bibliography"
    }
    assert sum(bool(row["primary_comparison_url"]) for row in hits) == 6
    assert sum(bool(row["cover_report_number"]) for row in hits) == 15
    return {
        "cmu_selected": len(selected),
        "kilthub_hits": len(hits),
        "kilthub_attributed": 17,
        "excluded_other_author_theses": 2,
        "publisher_confirmed_conference_papers": 1,
        "publisher_confirmed_monographs": 1,
        "historical_report_observations": 1,
        "arxiv_preprint_observations": 1,
        "third_party_bibliography_candidates": 67,
        "third_party_unmatched_titles_needing_primary_review": 49,
        "third_party_doi_leads_with_crossref_registration": 38,
        "third_party_doi_title_variants_for_review": 11,
        "third_party_non_doi_leads_reviewed": 11,
        "held_source_observations_not_works": len(observations),
    }


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
