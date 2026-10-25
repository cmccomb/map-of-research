"""Validate a held, source-only review of Luc C. Tartar's CMU records."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

SOURCE_DIR = (
    Path(__file__).resolve().parents[1] / "data/source_reviews/luc-tartar-2026-10-25"
)
PERSON_ID = "person-lucctartar-8b1dc8e8"
AUTHOR_ID = 5364974


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate(source_dir: Path = SOURCE_DIR) -> dict[str, int]:
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "review_only_no_snapshot_no_release"
    assert manifest["person_id"] == PERSON_ID
    assert manifest["scholar_account_status"] == "no_verified_account"
    assert manifest["lifetime_publication_completeness"] == "not_established"
    assert manifest["kilthub_author_id"] == AUTHOR_ID
    assert manifest["source_counts"] == {
        "cmu_select": 6,
        "kilthub_repository": 10,
        "cna_lecture": 5,
    }
    assert len(manifest["files"]) == 18
    for filename, expected in manifest["files"].items():
        actual = hashlib.sha256((source_dir / filename).read_bytes()).hexdigest()
        assert actual == expected, filename

    profile = (source_dir / "cmu-profile.html").read_text(encoding="utf-8")
    section = profile.split("<h1> Select Publications </h1>", 1)[1].split(
        '<div class="grid column3', 1
    )[0]
    citations = [
        " ".join(html.unescape(re.sub(r"<[^>]+>", "", raw)).split())
        for raw in re.findall(r"<p>(.*?)</p>", section, re.S)
    ]
    selected = _rows(source_dir / "cmu-select-6.csv")
    assert len(citations) == len(selected) == 6
    assert [row["citation"] for row in selected] == citations
    assert selected[0]["year"] == ""
    assert "in preparation" in selected[0]["citation"].casefold()
    assert [row["ordinal"] for row in selected] == [str(i) for i in range(1, 7)]

    first = json.loads((source_dir / "kilthub-search-page-1.json").read_text())
    second = json.loads((source_dir / "kilthub-search-page-2.json").read_text())
    repository = _rows(source_dir / "kilthub-10.csv")
    assert len(first) == len(repository) == 10 and second == []
    ids = [int(item["id"]) for item in first]
    assert len(set(ids)) == 10
    assert ids == [int(row["article_id"]) for row in repository]
    for result, row in zip(first, repository, strict=True):
        article_id = int(row["article_id"])
        detail = json.loads(
            (source_dir / f"kilthub-article-{article_id}.json").read_text()
        )
        assert detail["id"] == result["id"]
        assert detail["title"] == result["title"] == row["title"]
        assert row["source_url"] == detail["url_public_html"]
        assert row["source_url"].startswith("https://kilthub.cmu.edu/")
        assert detail["defined_type_name"] == row["repository_type"]
        assert detail["published_date"][:10] == row["repository_date"]
        assert any(author["id"] == AUTHOR_ID for author in detail["authors"])
        assert row["author_ids"] == " | ".join(
            str(author["id"]) for author in detail["authors"]
        )

    cna_html = (source_dir / "cmu-cna.html").read_text(encoding="latin1")
    cna = _rows(source_dir / "cmu-cna-5.csv")
    assert len(cna) == 5
    assert [row["ordinal"] for row in cna] == [str(i) for i in range(1, 6)]
    for row in cna:
        assert row["filename"] in cna_html
        assert row["source_url"].endswith(row["filename"])
        assert int(row["byte_size"]) > 0
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
    assert "Luc Tartar - Lecture 1" in cna_html
    assert "Lecture 2" in cna_html

    crosswalk = _rows(source_dir / "source-crosswalk-21.csv")
    assert len(crosswalk) == manifest["reviewed_source_rows"] == 21
    assert len({row["source_row_id"] for row in crosswalk}) == 21
    assert len({row["work_group"] for row in crosswalk}) == 19
    assert (
        len({row["work_group"] for row in crosswalk})
        == manifest["distinct_internal_work_groups"]
    )
    assert {row["graph_eligible"] for row in crosswalk} == {"false"}
    assert all(row["review_status"].startswith("hold_") for row in crosswalk)
    assert Counter(row["source"] for row in crosswalk) == {
        "cmu_profile_select": 6,
        "cmu_kilthub_repository": 10,
        "cmu_cna_lecture": 5,
    }
    assert {
        row["source_row_id"]
        for row in crosswalk
        if row["source"] == "cmu_profile_select"
    } == {f"PROFILE-{i:02d}" for i in range(1, 7)}
    assert {
        row["source_row_id"]
        for row in crosswalk
        if row["source"] == "cmu_kilthub_repository"
    } == {f"KILTHUB-{article_id}" for article_id in ids}
    assert {
        row["source_row_id"] for row in crosswalk if row["source"] == "cmu_cna_lecture"
    } == {f"CNA-{i:02d}" for i in range(1, 6)}
    grouped = Counter(row["work_group"] for row in crosswalk)
    assert {key for key, count in grouped.items() if count > 1} == {"P02", "P03"}
    assert grouped["P02"] == grouped["P03"] == 2
    return {
        "cmu_selected_rows": len(selected),
        "kilthub_rows": len(repository),
        "cna_notes": len(cna),
        "source_rows": len(crosswalk),
        "internal_groups_not_lifetime_works": len(grouped),
    }


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
