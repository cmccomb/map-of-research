"""Prepare a nonoperational Kim cache and exact-ID release proposal offline.

This uses only the saved complete Scholar browser capture and the checked-in
review ledger. It never fetches Scholar, edits a canonical cache, or publishes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from map_of_research.collector import normalize_publications
from map_of_research.exclusions import load_exclusions
from map_of_research.io import atomic_write_json, sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SCHOLAR_ID = "bC4aq-4AAAAJ"
PERSON_ID = "person-hyungjkim-2fcad292"
SOURCE_SHA256 = "ea04610e120039df7aaf83c402d193021f328b319cd41cfabf541f3942069867"
EXPECTED_CACHE_SHA256 = (
    "f4b7b37efcd3310a05ff82057488ecb286450e3db0dc390b4e26b125fc378b2e"
)
SOURCE_ROWS = 918
REVIEWED_AT = "2026-10-25"
PROFILE_SOURCE = "https://www.cmu.edu/chemistry/people/faculty/kim.html"


def _ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def prepare(capture_path: Path, output_dir: Path) -> dict[str, object]:
    """Write reproducible proposal files without touching pipeline inputs."""

    if sha256_file(capture_path) != SOURCE_SHA256:
        raise ValueError("Kim browser capture hash does not match the reviewed source")
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    works = capture.get("works")
    if (
        capture.get("scholar_user_id") != SCHOLAR_ID
        or capture.get("displayed_count") != SOURCE_ROWS
        or not isinstance(works, list)
        or len(works) != SOURCE_ROWS
        or capture.get("more_remain") is not False
        or capture.get("challenge") is not False
        or capture.get("page_error")
    ):
        raise ValueError("Kim browser capture is not the complete reviewed profile")

    source_publications = []
    for work in works:
        citation_url = str(work["citation_url"])
        citation_ids = parse_qs(urlparse(citation_url).query).get(
            "citation_for_view", []
        )
        if len(citation_ids) != 1 or not citation_ids[0].startswith(f"{SCHOLAR_ID}:"):
            raise ValueError("Kim browser row has an invalid citation identity")
        source_publications.append(
            {
                "author_pub_id": citation_ids[0],
                "bib": {
                    "title": work["title"],
                    "author": work["authors"],
                    "pub_year": work["year"],
                    "venue": work["venue"],
                },
                "num_citations": int(work["citation_count"] or 0),
                "pub_url": citation_url,
            }
        )
    publications = normalize_publications(source_publications)
    publication_ids = {row["author_pub_id"] for row in publications}
    if len(publications) != len(publication_ids) or len(publications) != SOURCE_ROWS:
        raise ValueError("Kim cache would lose or duplicate a captured row")

    cache = {
        "schema_version": 1,
        "scholar_id": SCHOLAR_ID,
        "display_name": "Hyung J. Kim",
        "fetched_at_utc": capture["captured_at"],
        "publication_count": SOURCE_ROWS,
        "publications": publications,
        "browser_capture": {
            "coverage": "complete_visible_profile",
            "more_remain": False,
            "displayed_count": SOURCE_ROWS,
            "source_capture_sha256": SOURCE_SHA256,
            "source_url": capture["url"],
        },
    }
    cache_path = output_dir / f"{SCHOLAR_ID}.json"
    atomic_write_json(cache_path, cache)
    cache_sha256 = sha256_file(cache_path)
    if cache_sha256 != EXPECTED_CACHE_SHA256:
        raise ValueError("Kim candidate cache changed from the reviewed byte sequence")

    with (HERE / "2026-10-24-chem-kim-918-screen.csv").open(newline="") as handle:
        ledger = list(csv.DictReader(handle))
    if (
        len(ledger) != SOURCE_ROWS
        or {row["author_pub_id"] for row in ledger} != publication_ids
    ):
        raise ValueError("Kim review ledger does not cover the complete capture")
    released = sorted(
        row["author_pub_id"]
        for row in ledger
        if row["status"] == "cmu_primary_work_retained"
    )
    excluded = sorted(
        row["author_pub_id"]
        for row in ledger
        if row["status"] == "already_excluded" or row["status"].startswith("proposed_")
    )
    held = sorted(publication_ids - set(released) - set(excluded))
    if (len(released), len(excluded), len(held)) != (121, 396, 401):
        raise ValueError("Kim review partition changed")
    manifest_excluded = {
        author_pub_id
        for scholar_id, author_pub_id in load_exclusions(
            ROOT / "data/curated/scholar-exclusions.json"
        )
        if scholar_id == SCHOLAR_ID
    }
    if set(excluded) != manifest_excluded:
        raise ValueError("Kim exact exclusions differ from the producer manifest")

    release_proposal = {
        "schema_version": 1,
        "profiles": [
            {
                "person_id": PERSON_ID,
                "scholar_id": SCHOLAR_ID,
                "state": "reviewed_subset",
                "expected_source_rows": SOURCE_ROWS,
                "cache_sha256": cache_sha256,
                "reason": (
                    "Proposal only: release 121 exact publisher/CMU-verified works; "
                    "keep Kim off public maps until the full history is reviewed."
                ),
                "reviewed_at": REVIEWED_AT,
                "released_author_pub_ids": released,
                "excluded_author_pub_ids": excluded,
                "held_author_pub_ids": held,
            }
        ],
    }
    confirmed = []
    released_set = set(released)
    for row in ledger:
        if row["author_pub_id"] not in released_set:
            continue
        evidence_url = row["evidence_url"]
        if int(row["capture_index"]) in {725, 778, 817, 888, 897, 903, 904}:
            evidence_basis = (
                f"{row['evidence_source']}; saved Scholar row matched to the "
                "publisher output by title, year, authors, and identifier or pages"
            )
        else:
            evidence_basis = (
                f"{row['evidence_source']}; exact saved Scholar title, year, "
                "and journal/page match reviewed"
            )
        if row["capture_index"] == "104":
            evidence_url = (
                "https://api.crossref.org/works/10.1016%2FS1010-6030%2896%2904543-1"
            )
            evidence_basis = (
                "Elsevier publisher-deposited exact DOI names Hyung J. Kim "
                "and James T. Hynes; coauthor Hynes Colorado CV item 173 "
                "corroborates the 1997 journal volume 105 pages 337–343 "
                "despite a title transcription variant."
            )
        confirmed.append(
            {
                "scholar_id": SCHOLAR_ID,
                "author_pub_id": row["author_pub_id"],
                "bibliography_entry": None,
                "evidence_basis": evidence_basis,
                "evidence_url": evidence_url,
                "verified_at": REVIEWED_AT,
            }
        )
    attribution_proposal = {
        "schema_version": 1,
        "profiles": [
            {
                "scholar_id": SCHOLAR_ID,
                "capture_sha256": SOURCE_SHA256,
                "captured_observations": SOURCE_ROWS,
                "evidence_url": PROFILE_SOURCE,
                "verified_at": REVIEWED_AT,
            }
        ],
        "confirmed_observations": confirmed,
    }
    if {row["author_pub_id"] for row in confirmed} != released_set:
        raise ValueError("Kim confirmed works do not match the released set")
    if any(
        urlparse(row["evidence_url"]).scheme != "https" or not row["evidence_basis"]
        for row in confirmed
    ):
        raise ValueError("Kim confirmed works need specific HTTPS evidence")

    atomic_write_json(
        output_dir / "scholar-profile-releases.kim-proposal.json", release_proposal
    )
    atomic_write_json(
        output_dir / "scholar-attribution-reviews.kim-proposal.json",
        attribution_proposal,
    )
    return {
        "cache_sha256": cache_sha256,
        "captured": SOURCE_ROWS,
        "released": len(released),
        "excluded": len(excluded),
        "held": len(held),
        "released_ids_sha256": _ids_sha256(released),
        "excluded_ids_sha256": _ids_sha256(excluded),
        "held_ids_sha256": _ids_sha256(held),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.capture, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
