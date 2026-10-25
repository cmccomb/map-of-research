"""Check Jane Panangaden's pinned, release-held publication source review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

SOURCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/jane-panangaden-2026-10-25"
)
DOI_FAMILIES = {
    "10.1007/s43034-026-00512-1": "furstenberg",
    "10.1063/5.0179805": "modular-curves",
    "10.1007/s00023-016-0543-0": "reflection-probabilities",
    "10.1007/s11005-015-0769-7": "energy-conservation",
    "10.1007/s11005-015-0748-z": "reflectionless-cmv",
}
ARXIV_FAMILIES = {
    "2310.12468": "higher-weight",
    "2110.01678": "masters-energy-fcs",
    "2006.16897": "modular-curves",
    "1509.08525": "reflection-probabilities",
    "1407.8127": "reflectionless-cmv",
    "2410.22701": "furstenberg",
    "1409.8610": "energy-conservation",
}
FIELDNAMES = (
    "source",
    "source_id",
    "title",
    "source_year",
    "genre",
    "doi",
    "work_family",
    "source_url",
)
ATOM = "{http://www.w3.org/2005/Atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


def source_observations(source_dir: Path = SOURCE_DIR) -> list[dict[str, str]]:
    """Preserve every visible record and group only verified version relationships."""
    orcid = json.loads((source_dir / "orcid-record.json").read_text())
    assert orcid["orcid-identifier"]["path"] == "0000-0002-6214-2130"
    name = orcid["person"]["name"]
    assert (name["given-names"]["value"], name["family-name"]["value"]) == (
        "Jane",
        "Panangaden",
    )
    employments = orcid["activities-summary"]["employments"]["affiliation-group"]
    assert any(
        summary["employment-summary"]["organization"]["name"]
        == "Carnegie Mellon University"
        and summary["employment-summary"]["source"]["source-name"]["value"]
        == "Jane Panangaden"
        for group in employments
        for summary in group["summaries"]
    )
    rows: list[dict[str, str]] = []
    for group in orcid["activities-summary"]["works"]["group"]:
        assert len(group["work-summary"]) == 1
        work = group["work-summary"][0]
        ids = group["external-ids"]["external-id"]
        assert len(ids) == 1 and ids[0]["external-id-type"] == "doi"
        doi = ids[0]["external-id-value"].lower()
        assert doi in DOI_FAMILIES
        assert work["source"]["source-name"]["value"] == "Jane Panangaden"
        rows.append(
            {
                "source": "orcid",
                "source_id": str(work["put-code"]),
                "title": work["title"]["title"]["value"],
                "source_year": work["publication-date"]["year"]["value"],
                "genre": "journal_article",
                "doi": doi,
                "work_family": DOI_FAMILIES[doi],
                "source_url": f"https://orcid.org/0000-0002-6214-2130/work/{work['put-code']}",
            }
        )
    assert len(rows) == len(DOI_FAMILIES) == 5
    assert {row["doi"] for row in rows} == set(DOI_FAMILIES)

    root = ET.parse(source_dir / "arxiv-author-query.xml").getroot()
    assert root.findtext(f"{OPENSEARCH}totalResults") == "7"
    entries = root.findall(f"{ATOM}entry")
    assert len(entries) == 7
    for entry in entries:
        source_id = entry.findtext(f"{ATOM}id") or ""
        match = re.fullmatch(r"http://arxiv\.org/abs/(\d+\.\d+)v\d+", source_id)
        assert match is not None
        arxiv_id = match.group(1)
        assert arxiv_id in ARXIV_FAMILIES
        authors = [
            author.findtext(f"{ATOM}name") for author in entry.findall(f"{ATOM}author")
        ]
        assert "Jane Panangaden" in authors
        assert "Prakash Panangaden" not in authors
        rows.append(
            {
                "source": "arxiv",
                "source_id": arxiv_id,
                "title": " ".join((entry.findtext(f"{ATOM}title") or "").split()),
                "source_year": (entry.findtext(f"{ATOM}published") or "")[:4],
                "genre": "preprint",
                "doi": "",
                "work_family": ARXIV_FAMILIES[arxiv_id],
                "source_url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        )
    assert {row["source_id"] for row in rows if row["source"] == "arxiv"} == set(
        ARXIV_FAMILIES
    )

    caltech = (source_dir / "caltech-phd.html").read_text()
    assert "Panangaden, Jane Mariam" in caltech
    assert "Quantum Statistical Mechanics, Noncommutative Geometry, and" in caltech
    assert 'content="10.7907/xrba-8471"' in caltech
    assert "Dissertation (Ph.D.)" in caltech
    rows.append(
        {
            "source": "caltech_thesis",
            "source_id": "14622",
            "title": (
                "Quantum Statistical Mechanics, Noncommutative Geometry, "
                "and the Boundary of Modular Curves"
            ),
            "source_year": "2022",
            "genre": "doctoral_thesis",
            "doi": "10.7907/xrba-8471",
            "work_family": "doctoral-thesis",
            "source_url": "https://thesis.caltech.edu/14622/",
        }
    )

    mcgill = (source_dir / "mcgill-masters-index.html").read_text()
    assert "Panangaden, Jane" in mcgill
    assert "2016 — Energy full counting statistics in return-to-equilibrium" in mcgill
    cover = (source_dir / "masters-cover.txt").read_text()
    assert "arXiv:2110.01678v1" in cover
    assert "Jane Panangaden" in cover
    assert "April 2016" in cover and "degree of Master of Science" in cover
    rows.append(
        {
            "source": "mcgill_thesis",
            "source_id": "2016-energy-full-counting-statistics",
            "title": "Energy full counting statistics in return-to-equilibrium",
            "source_year": "2016",
            "genre": "masters_thesis",
            "doi": "",
            "work_family": "masters-energy-fcs",
            "source_url": "https://www.erudit.org/en/theses/mcgill/P/?page=12&sort_by=date_desc",
        }
    )
    magazine = json.loads((source_dir / "magazine-article-metadata.json").read_text())
    assert magazine["headline"] == "Against COVID Austerity"
    assert magazine["byline"] == [
        "Jane Mariam Panangaden",
        "Ashay Naren Patel",
        "Kriti Sharma",
    ]
    assert magazine["publisher_genre"] == "chapter report"
    assert magazine["online_published_at"].startswith("2021-01-25")
    assert re.fullmatch(r"[0-9a-f]{64}", magazine["source_html_sha256"])
    rows.append(
        {
            "source": "science_for_the_people_magazine",
            "source_id": "vol23no3-socialists-of-caltech",
            "title": magazine["headline"],
            "source_year": "2021",
            "genre": "magazine_chapter_report",
            "doi": "",
            "work_family": "against-covid-austerity",
            "source_url": magazine["source_url"],
        }
    )
    assert len(rows) == 15
    counts = Counter(row["work_family"] for row in rows)
    assert len(counts) == 9
    assert sorted(counts.values()) == [1, 1, 1, 2, 2, 2, 2, 2, 2]
    return rows


def validate(source_dir: Path = SOURCE_DIR) -> dict[str, int]:
    manifest = json.loads((source_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "review_only_no_snapshot_no_release"
    assert manifest["career_completeness"] == "unverified"
    assert manifest["person_id"] == "person-janepanangaden-dc0f9aec"
    assert manifest["scholar_account_status"] == "no_verified_account"
    assert manifest["orcid"] == "0000-0002-6214-2130"
    magazine = json.loads((source_dir / "magazine-article-metadata.json").read_text())
    assert (
        magazine["source_url"]
        == manifest["source_urls"]["science_for_the_people_magazine"]
    )
    assert manifest["mcgill_masters_pdf_sha256"] == (
        "62a72594213d3105c16dcb21bd48e46e3fd4cbe5ee7f88eee76ea8f6cf7a0ab6"
    )
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((source_dir / name).read_bytes()).hexdigest() == digest
    cmu = (source_dir / "cmu-profile.html").read_text()
    assert "Jane Panangaden" in cmu
    assert "Assistant Teaching Professor" in cmu
    assert "Department of Mathematical Sciences" in cmu
    assert "jpananga@andrew.cmu.edu" in cmu

    account_review = json.loads(
        (
            source_dir.parents[2]
            / "registry/reviews/2026-10-25-math-scholar-account-coverage.json"
        ).read_text()
    )
    assert any(
        row["person_id"] == manifest["person_id"]
        and row["identity_decision"] == "no_verified_scholar_id"
        for row in account_review["unresolved_people"]
    )

    actual = source_observations(source_dir)
    with (source_dir / "observations-15.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == FIELDNAMES
        expected = list(reader)
    assert expected == actual
    counts = Counter(row["source"] for row in actual)
    assert counts == {
        "orcid": 5,
        "arxiv": 7,
        "caltech_thesis": 1,
        "mcgill_thesis": 1,
        "science_for_the_people_magazine": 1,
    }
    return {"source_observations": 15, "provisional_work_families": 9, **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-observations", action="store_true")
    args = parser.parse_args()
    if args.write_observations:
        with (SOURCE_DIR / "observations-15.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
            writer.writeheader()
            writer.writerows(source_observations())
    print(json.dumps(validate(), sort_keys=True))


if __name__ == "__main__":
    main()
