"""Validate Gregory Johnson's held primary-source observations."""

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
    / "data/source_reviews/gregory-johnson-2026-10-25"
)
PERSON_ID = "person-gregoryjohnson-26c9645f"
HOLD_REASON = "no_verified_scholar_or_current_complete_author_bibliography"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _citation_meta(path: Path) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for key, value in re.findall(
        r'<meta\s+name="(citation_[^"]+)"\s+content="([^"]*)"',
        path.read_text(encoding="utf-8"),
    ):
        values.setdefault(key, []).append(html.unescape(value))
    return values


def validate(source_dir: Path = SOURCE_DIR) -> dict[str, int]:
    manifest = json.loads((source_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["person_id"] == PERSON_ID
    assert manifest["status"] == "review_only_no_snapshot_no_release"
    assert manifest["scholar_account_status"] == "no_verified_account"
    assert manifest["lifetime_publication_completeness"] == "not_established"
    assert manifest["counts"] == {
        "attributed_source_observations_not_works": 8,
        "provisional_manifestations_not_lifetime_count": 6,
        "research_lineages_not_lifetime_count": 2,
        "uncaptured_ams_meeting_leads": 1,
        "excluded_sponsor_only_records": 1,
    }
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
    assert {row["filename"] for row in captures} == {
        name
        for name in manifest["files"]
        if name
        not in {
            "capture-http-responses.json",
            "source-observations-8.csv",
            "genre-and-attribution-holds-2.csv",
        }
    }
    assert Counter(row["status"] for row in captures) == {200: 11, 403: 1}
    for row in captures:
        assert (source_dir / row["filename"]).stat().st_size == row["bytes"]
    assert next(row for row in captures if row["status"] == 403)["filename"] == (
        "ams-2011-denied.html"
    )

    cmu = (source_dir / "cmu-profile.html").read_text()
    old_cmu = (source_dir / "cmu-old-profile.html").read_text()
    assert "greggo@andrew.cmu.edu" in cmu
    assert "greggo@andrew.cmu.edu" in old_cmu
    assert "Abstract Elementary Classes" in old_cmu
    assert "University of Maryland" in cmu
    umd = json.loads((source_dir / "umd-dissertation-api.json").read_text())
    assert umd["uuid"] == "95910634-bbe1-4228-8bf7-b2049a005c6e"
    umd_meta = umd["metadata"]
    assert umd_meta["dc.contributor.author"][0]["value"] == (
        "Johnson, Gregory Mitchell"
    )
    assert umd_meta["dc.type"][0]["value"] == "Dissertation"
    assert umd_meta["dc.date.issued"][0]["value"] == "2008-08-03"
    assert umd_meta["dc.contributor.advisor"][0]["value"] == "Kueker, David W"
    logic = json.loads((source_dir / "crossref-logic-article.json").read_text())[
        "message"
    ]
    assert logic["DOI"] == "10.1215/00294527-2010-022"
    assert len(logic["author"]) == 1
    assert logic["author"][0]["given"] == "Gregory M."
    assert logic["author"][0]["family"] == "Johnson"
    assert logic["type"] == "journal-article"
    assert logic["container-title"] == ["Notre Dame Journal of Formal Logic"]
    assert logic["page"] == "361-371"

    for label, arxiv_id, expected_authors in [
        ("2025", "2509.16778", 19),
        ("2026-feb", "2602.18807", 21),
        ("2026-oct", "2610.31481", 21),
    ]:
        meta = _citation_meta(source_dir / f"arxiv-ai-{label}.html")
        assert meta["citation_arxiv_id"] == [arxiv_id]
        assert len(meta["citation_author"]) == expected_authors
        assert "Johnson, Gregory" in meta["citation_author"]
        assert "Mackey, John" in meta["citation_author"]
    october = (source_dir / "arxiv-ai-2026-oct.html").read_text()
    assert "substantially revised and extended analysis" in october
    assert "2602.18807" in october and "2509.16778" in october

    springer = _citation_meta(source_dir / "springer-ai-2026.html")
    assert springer["citation_doi"] == ["10.1007/978-3-032-29763-1_41"]
    assert springer["citation_publication_date"] == ["2027"]
    assert springer["citation_firstpage"] == ["601"]
    assert springer["citation_lastpage"] == ["610"]
    assert "Johnson, Gregory" in springer["citation_author"]
    chapter = json.loads((source_dir / "crossref-ai-chapter.json").read_text())[
        "message"
    ]
    assert chapter["DOI"] == springer["citation_doi"][0]
    assert chapter["published"]["date-parts"] == [[2026, 6, 27]]
    assert chapter["type"] == "book-chapter"
    assert chapter["page"] == "601-610"
    assert any(
        author["given"] == "Gregory" and author["family"] == "Johnson"
        for author in chapter["author"]
    )
    hcii = (source_dir / "cmu-hcii-2026.html").read_text()
    match = re.search(
        r'<a href="https://link.springer.com/chapter/10.1007/978-3-032-29763-1_41">(.*?)</a>\s*-\s*(.*?)</li>',
        hcii,
        re.S,
    )
    assert match is not None
    hcii_title = html.unescape(re.sub(r"<[^>]+>", "", match[1])).strip()
    hcii_authors = html.unescape(re.sub(r"<[^>]+>", "", match[2])).strip()
    assert "Gregory Johnson" in hcii_authors
    assert "John Mackey" in hcii_authors
    sponsor = (source_dir / "rose-hulman-sponsor.html").read_text()
    assert "Faculty Sponsor" in sponsor and "Gregory Johnson" in sponsor

    observations = _rows(source_dir / "source-observations-8.csv")
    assert len(observations) == 8
    assert [row["source_row_id"] for row in observations] == [
        "UMD-01",
        "CROSSREF-LOGIC-01",
        "ARXIV-2025",
        "ARXIV-2026-FEB",
        "ARXIV-2026-OCT",
        "SPRINGER-01",
        "CROSSREF-AI-01",
        "CMU-HCII-01",
    ]
    assert len({row["provisional_manifestation"] for row in observations}) == 6
    assert Counter(row["research_lineage"] for row in observations) == {
        "LOGIC-01": 2,
        "AI-PROOF-01": 6,
    }
    assert observations[0]["title_as_listed"] == umd_meta["dc.title"][0]["value"]
    assert observations[0]["source_genre"] == "doctoral_dissertation"
    assert observations[1]["title_as_listed"] == logic["title"][0]
    assert observations[1]["source_genre"] == "journal_article"
    assert (
        observations[0]["provisional_manifestation"]
        != observations[1]["provisional_manifestation"]
    )
    assert observations[5]["issued_date_as_listed"] == "2027"
    assert observations[6]["issued_date_as_listed"] == "2026-06-27"
    assert observations[7]["title_as_listed"] == hcii_title
    assert observations[7]["byline_as_listed"] == hcii_authors
    assert {row["graph_eligible"] for row in observations} == {"false"}
    assert {row["hold_reason"] for row in observations} == {HOLD_REASON}
    holds = _rows(source_dir / "genre-and-attribution-holds-2.csv")
    assert [row["hold_id"] for row in holds] == ["AMS-2011", "SPONSOR-2017"]
    assert all(row["graph_eligible"] == "false" for row in holds)
    assert "not an author" in holds[1]["finding"]
    return {
        "attributed_source_observations_not_works": len(observations),
        "provisional_manifestations_not_lifetime_count": 6,
        "research_lineages_not_lifetime_count": 2,
        "uncaptured_meeting_leads": 1,
        "excluded_sponsor_only_records": 1,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
