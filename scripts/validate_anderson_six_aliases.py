"""Validate Theresa Anderson's six held source/CV title-alias decisions."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

REVIEW_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/theresa-anderson-six-aliases-2026-10-25"
)
PERSON_ID = "person-theresaanderson-5ed760a9"
ORDINALS = (21, 22, 23, 24, 25, 33)
CV_NUMBERS = (21, 18, 20, 22, 15, 9)
EXPECTED_DOIS = {
    21: "10.1016/j.jfa.2021.109334",
    22: "10.1007/s00025-021-01587-z",
    23: "10.1016/j.jmaa.2021.125620",
    24: "10.1007/s13348-020-00308-z",
    25: "10.1112/blms.12465",
    33: None,
}
ARXIV_IDS = {
    21: "1703.02713",
    22: "2009.11906",
    23: "2012.05778",
    24: "1910.11409",
    25: "1911.00464",
    33: "1606.03461",
}
MATCH_KEYS = {
    21: ("282", "5", "109334"),
    22: ("77", "2", "69"),
    23: ("505", "1", "125620"),
    24: ("73", "1", "75-87"),
    25: ("53", "3", "855-860"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _html(root: Path, name: str) -> bytes:
    return gzip.decompress((root / name).read_bytes())


def validate_review(root: Path = REVIEW_ROOT) -> tuple[dict, list[dict]]:
    """Check original titles, six exact matches, provenance, and release holds."""
    root = Path(root)
    manifest = _json(root / "manifest.json")
    _require(manifest["status"] == "held_source_review", "review must stay held")
    _require(manifest["person_id"] == PERSON_ID, "person ID changed")
    _require(manifest["upstream_source_pr"] == 340, "upstream source changed")
    _require(manifest["verified_google_scholar_id"] is None, "Scholar ID unverified")
    for field in (
        "lifetime_complete",
        "release_eligible",
        "map_eligible",
        "collaboration_eligible",
    ):
        _require(manifest[field] is False, f"{field} must remain false")
    _require(
        manifest["counts"]
        == {
            "source_cv_title_variant_pairs": 6,
            "resolved_single_published_work_aliases": 6,
            "publisher_doied_works": 5,
            "journal_url_without_displayed_doi": 1,
            "arxiv_version_observations": 6,
        },
        "review counts changed",
    )

    files = manifest["files"]
    actual = {path.name for path in root.iterdir() if path.is_file()}
    _require(
        actual == set(files) | {"README.md", "manifest.json"},
        "source file inventory differs from manifest",
    )
    for name, entry in files.items():
        data = (root / name).read_bytes()
        _require(len(data) == entry["bytes"], f"{name} length changed")
        _require(
            hashlib.sha256(data).hexdigest() == entry["sha256"],
            f"{name} hash changed",
        )

    index = _json(root / "source-capture-index.json")
    html = _html(root, "cmu-author-research.html.gz")
    _require(
        hashlib.sha256(html).hexdigest()
        == index["cmu_author_site"]["source_sha256"]
        == "7aa17fd09fbfc9eea7af95766a02f144f014976a8d23343077f817e51417d3b3",
        "CMU author-page capture changed",
    )
    _require(
        index["cmu_cv_pdf"]["source_sha256"]
        == "c70013a8b9a0617ec0ed6b60bf52e66d62bd9e8685babe3e2a7bc6cf60e659bd"
        and index["cmu_cv_pdf"]["content_committed"] is False
        and index["upstream_crosswalk"]["full_42_row_sha256"]
        == "cb151379a15f9a6df6082df7376341dffb2184fc849f46727ae6cda7e1897642",
        "CV or upstream 42-row source provenance changed",
    )

    with (root / "author-site-cv-raw-six.csv").open(newline="") as file:
        raw = list(csv.DictReader(file))
    resolved = _json(root / "alias-resolution-six.json")
    _require(len(raw) == len(resolved) == 6, "six source pairs not preserved")
    _require(
        tuple(int(row["author_site_ordinal"]) for row in raw)
        == tuple(row["source_ordinal"] for row in resolved)
        == ORDINALS,
        "source ordinals changed",
    )
    _require(
        tuple(int(row["cv_number"]) for row in raw)
        == tuple(row["cv_number"] for row in resolved)
        == CV_NUMBERS,
        "CV pairing changed",
    )
    _require(
        len({row["publisher_doi"] for row in resolved}) == 6, "DOI alias collision"
    )

    for source, row in zip(raw, resolved, strict=True):
        n = row["source_ordinal"]
        _require(
            row["source_citation_raw"] == source["source_citation"]
            and row["cv_citation_raw"] == source["cv_citation"]
            and row["site_title_raw"] in source["source_citation"]
            and row["cv_title_raw"] in source["cv_citation"]
            and row["site_title_raw"] != row["cv_title_raw"],
            "raw author-page or CV title changed",
        )
        _require(
            row["decision"] == "confirmed_single_published_work_alias"
            and row["publisher_doi"] == EXPECTED_DOIS[n]
            and row["arxiv_id"] == ARXIV_IDS[n]
            and row["arxiv_url"] == f"https://arxiv.org/abs/{ARXIV_IDS[n]}"
            and row["verified_scholar_id"] is None,
            "publisher or arXiv alias decision changed",
        )
        for field in ("release_eligible", "map_eligible", "collaboration_eligible"):
            _require(row[field] is False, f"row {n} {field} escaped hold")
        _require(
            row["publisher_authors_raw"]
            and row["publisher_record_url"]
            and row["journal_fingerprint"],
            "publisher matching evidence missing",
        )
        arxiv = _html(root, f"arxiv-{n}.html.gz")
        _require(
            f"arXiv:{ARXIV_IDS[n]}".encode() in arxiv
            and b"Theresa C. Anderson" in arxiv,
            "preprint identity changed",
        )

        if n == 33:
            journal = _html(root, "nyjm-33.html.gz")
            _require(
                row["publisher_doi"] is None
                and row["publisher_record_url"]
                == "https://nyjm.albany.edu/j/2018/24-1.html"
                and row["publisher_title_raw"].encode() in journal
                and b"David E. Weirich" in journal
                and b"1-19" in journal,
                "NYJM original record or no-journal-DOI decision changed",
            )
            continue

        deposit = _json(root / f"crossref-{n}.json")["message"]
        volume, issue, page_or_article = MATCH_KEYS[n]
        _require(
            deposit["DOI"].lower() == EXPECTED_DOIS[n]
            and deposit["title"][0] == row["publisher_title_raw"]
            and deposit["volume"] == volume
            and deposit["issue"] == issue
            and (
                deposit.get("article-number") == page_or_article
                or deposit.get("page") == page_or_article
            )
            and row["publisher_record_url"] == deposit["URL"]
            and row["publisher_authors_raw"]
            == [
                f"{author.get('given', '')} {author['family']}".strip()
                for author in deposit["author"]
            ],
            "publisher title, DOI, byline, or issue coordinates changed",
        )
    _require(
        b"10.1112/blms.12465" in _html(root, "arxiv-25.html.gz"),
        "explicit arXiv BLMS DOI relation changed",
    )
    _require(
        resolved[3]["publisher_issue_year"] == 2022
        and resolved[3]["crossref_or_journal_published_year"] == 2021,
        "Collectanea online/print dates changed",
    )
    return manifest, resolved


def load_held_aliases(root: Path = REVIEW_ROOT) -> list[dict]:
    """Expose six aliases without creating canonical works or coauthor links."""
    manifest, rows = validate_review(root)
    return [
        {
            "person_id": manifest["person_id"],
            "source_ordinal": row["source_ordinal"],
            "publisher_doi": row["publisher_doi"],
            "canonical_work_id": None,
            "canonical_author_ids": [],
            "scholar_publication_id": None,
            "release_eligible": False,
            "map_eligible": False,
            "collaboration_eligible": False,
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_ROOT)
    args = parser.parse_args()
    manifest, rows = validate_review(args.review_dir)
    print(
        f"{manifest['person_id']}: {len(rows)} title aliases resolved; "
        "all source-only and release-ineligible"
    )


if __name__ == "__main__":
    main()
