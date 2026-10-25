"""Validate Richard Statman's held CMU source evidence and attribution."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

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
    assert manifest["counts"] == {
        "cmu_profile_selected": 12,
        "cmu_recent_selected": 4,
        "kilthub_search_hits": 19,
        "kilthub_attributed_reports_or_manuscripts": 17,
        "kilthub_other_author_theses": 2,
        "attributed_source_observations_not_works": 33,
    }
    assert len(manifest["files"]) == 27
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

    observations = _rows(source_dir / "source-observations-33.csv")
    assert len(observations) == 33
    assert len({row["source_row_id"] for row in observations}) == 33
    assert Counter(row["source"] for row in observations) == {
        "cmu_profile_select": 12,
        "cmu_recent_selected": 4,
        "cmu_kilthub_repository": 17,
    }
    assert {row["source_row_id"] for row in observations[:16]} == {
        f"CMU-{i:02d}" for i in range(1, 17)
    }
    assert {row["source_row_id"] for row in observations[16:]} == {
        f"KILTHUB-{row['article_id']}"
        for row in hits
        if row["attributed_to_statman"] == "true"
    }
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
        "held_source_observations_not_works": len(observations),
    }


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
