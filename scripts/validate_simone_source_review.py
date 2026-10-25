"""Validate the bounded Jonathan Simone whole-author-site candidate bundle."""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "data/source_reviews/jonathan-simone-2026-10-25"
ATOM = "{http://www.w3.org/2005/Atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalized(value: str) -> str:
    value = html.unescape(unicodedata.normalize("NFKC", value)).casefold()
    value = re.sub(r"[‐‑‒–—−]", "-", value)
    return " ".join(re.sub(r"[^\w]+", " ", value).split())


def validate_bundle(bundle: Path = DEFAULT_BUNDLE) -> dict[str, int]:
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        _require(actual == expected, f"source changed: {name}")

    candidate = json.loads((bundle / "candidate-11.json").read_text())
    works = candidate["works"]
    _require(candidate["status"] == manifest["status"], "release status drift")
    _require(candidate["person_id"] == manifest["person_id"], "person drift")
    _require(len(works) == 11, "expected 11 public work families")
    _require(
        all(row["person_id"] == candidate["person_id"] for row in works),
        "a work belongs to a different researcher",
    )
    _require(len({row["family_id"] for row in works}) == 11, "duplicate family")
    _require(
        len({row["arxiv_id"] for row in works if "arxiv_id" in row}) == 10,
        "expected ten distinct arXiv families",
    )
    _require(sum("doi" in row for row in works) == 9, "expected nine DOI families")
    _require(
        sum("source_native_id" in row for row in works) == 2,
        "expected two arXiv-only families",
    )
    _require(
        sum(row["genre"] == "doctoral_dissertation" for row in works) == 1,
        "dissertation must remain separate",
    )
    _require(
        not any("in preparation" in row["citation"].casefold() for row in works),
        "an in-preparation work entered the public batch",
    )
    _require(
        "in preparation" in candidate["held_in_preparation"]["reason"],
        "the in-preparation hold is missing",
    )

    cmu = (bundle / "cmu-profile.html").read_text()
    _require(
        "Jonathan Simone" in cmu and "jsimone" in cmu, "CMU identity link is missing"
    )
    site = (bundle / "research.html").read_text()
    _require(
        "Publications and Preprints" in site and "PhD Dissertation" in site,
        "research page headings are missing",
    )
    section = site.split("Publications and Preprints", 1)[1].split(
        "PhD Dissertation", 1
    )[0]
    anchors = re.findall(r'<a href="([^"]+)"[^>]*>(.*?)</a>', section, re.S)
    _require(len(anchors) == 10, "author page no longer has ten public entries")
    article_rows = [row for row in works if "arxiv_id" in row]
    for (url, raw_title), row in zip(anchors, article_rows, strict=True):
        title = re.sub(r"<[^>]*>", "", raw_title)
        _require(url == row["author_page_entry_url"], "author-page URL drift")
        _require(
            _normalized(title) == _normalized(row["title"]), "author-page title drift"
        )
    _require(
        "Cut-and-paste operations and exotic 4-manifolds" in site,
        "author-page dissertation is missing",
    )

    cv = (bundle / "cv-extract.txt").read_text()
    _require(
        "Jonathan Simone" in cv and "jsimone@andrew.cmu.edu" in cv, "CV identity drift"
    )
    cv_publications = cv.split("Publications", 1)[1].split("Preprints", 1)[0]
    _require(
        sum(bool(row["cv_publications_entry"]) for row in article_rows) == 9,
        "expected nine CV publication entries",
    )
    for row in article_rows:
        present = _normalized(row["title"]) in _normalized(cv_publications)
        _require(
            present == row["cv_publications_entry"],
            f"CV publication crosswalk drift: {row['arxiv_id']}",
        )
    _require(
        "in preparation" in cv and "plumbings bounding rational" in cv,
        "CV-only in-preparation work is missing",
    )
    _require(
        "Cut-and-paste operations and exotic 4-manifolds" in cv,
        "CV dissertation is missing",
    )

    arxiv = ET.fromstring((bundle / "arxiv-author-query.xml").read_bytes())
    _require(
        arxiv.findtext(OPENSEARCH + "totalResults") == "10",
        "arXiv exact-author total changed",
    )
    entries = arxiv.findall(ATOM + "entry")
    _require(len(entries) == 10, "arXiv query was truncated")
    arxiv_ids = set()
    for entry in entries:
        match = re.search(r"(\d{4}\.\d{4,5})", entry.findtext(ATOM + "id", ""))
        _require(match is not None, "arXiv entry ID missing")
        arxiv_ids.add(match.group(1))
        authors = [a.findtext(ATOM + "name") for a in entry.findall(ATOM + "author")]
        _require("Jonathan Simone" in authors, "arXiv author mismatch")
    _require(
        arxiv_ids == {row["arxiv_id"] for row in article_rows},
        "author-site/arXiv family coverage mismatch",
    )

    crossref = json.loads((bundle / "crossref-doi-records.json").read_text())
    doi_rows = [row for row in article_rows if "doi" in row]
    _require(len(doi_rows) == 8, "expected eight journal DOI families")
    _require(
        set(crossref["records"]) == {row["doi"] for row in doi_rows},
        "publisher DOI coverage mismatch",
    )
    for row in doi_rows:
        record = crossref["records"][row["doi"]]
        _require(record["DOI"].casefold() == row["doi"], "DOI record mismatch")
        _require(
            any(
                a.get("given") == "Jonathan" and a.get("family") == "Simone"
                for a in record.get("author", [])
            ),
            f"DOI namesake mismatch: {row['doi']}",
        )
        _require(
            _normalized(record["title"][0])[:15] == _normalized(row["title"])[:15],
            f"DOI title mismatch: {row['doi']}",
        )
    pretzel = next(row for row in doi_rows if row["doi"].endswith("109578"))
    pretzel_record = crossref["records"][pretzel["doi"]]
    _require(
        pretzel_record["volume"] == "375" and pretzel["year"] == 2025,
        "pretzel publisher correction lost",
    )
    _require(
        pretzel_record["published-print"]["date-parts"][0][0] == 2025,
        "pretzel publication year drift",
    )

    thesis = json.loads((bundle / "uva-thesis-datacite.json").read_text())["data"][
        "attributes"
    ]
    dissertation = next(row for row in works if row["genre"] == "doctoral_dissertation")
    _require(thesis["doi"] == dissertation["doi"], "thesis DOI mismatch")
    _require(thesis["publicationYear"] == 2018, "thesis year mismatch")
    _require(thesis["types"]["schemaOrg"] == "Thesis", "thesis genre mismatch")
    thesis_creator = thesis["creators"][0]
    _require(
        thesis_creator["givenName"] == "Jonathan"
        and thesis_creator["familyName"] == "Simone",
        "thesis creator mismatch",
    )
    _require(
        _normalized(thesis["titles"][0]["title"]) == _normalized(dissertation["title"]),
        "thesis title mismatch",
    )
    defense = (bundle / "uva-defense.html").read_text()
    _require(
        "Jonathan Simone" in defense and "Cut-and-paste operations" in defense,
        "UVA defense corroboration missing",
    )

    return {
        "public_families": 11,
        "author_page_entries": 10,
        "arxiv_entries": 10,
        "journal_dois": 8,
        "theses": 1,
        "held_in_preparation": 1,
    }


def main() -> None:
    print(json.dumps(validate_bundle(), sort_keys=True))


if __name__ == "__main__":
    main()
