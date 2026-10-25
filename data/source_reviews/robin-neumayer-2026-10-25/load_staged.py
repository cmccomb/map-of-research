"""Read Neumayer's whole visible bibliography into release-held source records.

This loader deliberately does not feed the publication snapshot. The current
snapshot source vocabulary cannot represent the CV-only observation honestly.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PERSON_ID = "person-robinneumayer-a5f54a8c"
SITE_URL = "https://www.math.cmu.edu/~rneumaye/"
CV_URL = "https://www.math.cmu.edu/~rneumaye/CVNeumayer2026.pdf"
EXPECTED_GENRES = {
    "research_preprint",
    "journal_article",
    "research_paper_with_claimed_book_placement",
    "book_chapter_survey",
    "oberwolfach_talk_summary",
    "proceedings_survey",
    "doctoral_thesis",
    "expository_lecture_notes",
}
NONCOLLABORATION_GENRES = {
    "book_chapter_survey",
    "oberwolfach_talk_summary",
    "proceedings_survey",
    "doctoral_thesis",
    "expository_lecture_notes",
}
OBERWOLFACH_REPORT_URLS = {
    24: "https://ems.press/content/serial-article-files/46965",
    26: "https://ems.press/content/serial-article-files/46813",
    31: "https://ems.press/content/serial-article-files/47471",
}


def _read_pinned(path: Path, name: str, digest: str) -> bytes:
    source = path / name
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Neumayer source checksum changed: {name}")
    return data


def _csv_rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(data.decode("utf-8").splitlines()))


def _url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"Neumayer evidence URL is invalid: {value}")
    return value


def _year(ledger: dict[str, str]) -> tuple[int | None, str]:
    decision = ledger["ingestion_decision"]
    if decision in {
        "hold_publisher_placement",
        "hold_publication_year",
        "hold_undated_nonpublication",
        "hold_report_extent",
    }:
        return None, "unresolved_publication_year"
    if ledger["registered_doi"]:
        return int(ledger["doi_registered_year"]), "registered_doi_publication_year"
    if ledger["reviewed_genre"] == "research_preprint":
        return int(ledger["arxiv_submitted"][:4]), "arxiv_submission_year"
    if ledger["site_year_as_listed"]:
        return int(ledger["site_year_as_listed"]), "author_site_year_label"
    return None, "not_stated"


def load_staged_neumayer_rows(path: Path) -> list[dict[str, Any]]:
    """Return 31 source-only observations, including the CV-only report.

    The returned records remain release-held, have no inferred Scholar account
    or citation count, and never enter a source snapshot through this function.
    """

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("person_id") != PERSON_ID
        or manifest.get("scholar_account_status") != "no_verified_account"
        or manifest.get("release_eligible") is not False
        or manifest.get("counts", {}).get("union_distinct") != 31
        or manifest.get("counts", {}).get("arxiv_site_research_matches") != 23
        or manifest.get("exact_registered_doi_count") != 18
    ):
        raise ValueError("Neumayer source scope or release hold changed")
    raw_sources = manifest["raw_sources"]
    for source in raw_sources:
        _url(source["url"])
        _read_pinned(path, source["file"], source["sha256"])
    source_urls = {source["kind"]: source["url"] for source in raw_sources}
    if source_urls["author_site"] != SITE_URL or source_urls["researcher_cv"] != CV_URL:
        raise ValueError("Neumayer bibliography sources changed")
    audit = json.loads(
        _read_pinned(
            path, manifest["source_audit_file"], manifest["source_audit_sha256"]
        )
    )
    crosswalk = _csv_rows(
        _read_pinned(
            path, manifest["cv_crosswalk_file"], manifest["cv_crosswalk_sha256"]
        )
    )
    ledger = _csv_rows(
        _read_pinned(
            path, manifest["doi_genre_ledger_file"], manifest["doi_genre_ledger_sha256"]
        )
    )
    doi_evidence = json.loads(
        _read_pinned(
            path,
            manifest["exact_doi_evidence_file"],
            manifest["exact_doi_evidence_sha256"],
        )
    )
    if (
        audit.get("person_id") != PERSON_ID
        or audit.get("scholar_id") is not None
        or audit.get("map_ready") is not False
        or len(audit["observations"]) != 31
        or len(crosswalk) != 31
    ):
        raise ValueError("Neumayer audit identity or row count changed")
    if len(ledger) != 31 or len(doi_evidence) != 18:
        raise ValueError("Neumayer source review is incomplete")
    observations = audit["observations"]
    if len({row["observation_id"] for row in observations}) != 31:
        raise ValueError("Neumayer observation identity is duplicated")
    if {row["cv_item"] for row in crosswalk if row["cv_item"]} != {
        str(index) for index in range(1, 31)
    }:
        raise ValueError("Neumayer CV item coverage changed")

    rows: list[dict[str, Any]] = []
    for number, (observation, relation, review) in enumerate(
        zip(observations, crosswalk, ledger, strict=True), start=1
    ):
        observation_id = observation["observation_id"]
        genre = review["reviewed_genre"]
        if (
            review["observation_row"] != str(number)
            or relation["observation_id"] != observation_id
            or review["observation_id"] != observation_id
            or genre not in EXPECTED_GENRES
            or not any(
                name in review["authors_in_primary_record"]
                for name in ("Robin Neumayer", "R. Neumayer")
            )
            or relation["title_as_listed"] != review["title_as_listed"]
        ):
            raise ValueError(f"Neumayer review or author byline changed: {number}")
        provenance = observation["provenance"]
        if provenance == "researcher_owned_website":
            if relation["source_relation"] not in {"site_and_cv", "site_only"}:
                raise ValueError(f"Neumayer website relation changed: {number}")
            primary_source_kind = "researcher_owned_site"
            primary_source_url = SITE_URL
        elif provenance == "researcher_cv_plus_publisher_report":
            if relation["source_relation"] != "cv_only" or number != 31:
                raise ValueError("Neumayer CV-only observation changed")
            primary_source_kind = "researcher_cv"
            primary_source_url = CV_URL
        else:
            raise ValueError(f"Unknown Neumayer provenance: {number}")
        if relation["source_relation"] == "site_only":
            if number != 30 or relation["cv_item"] or relation["cv_entry_as_extracted"]:
                raise ValueError("Neumayer site-only note changed")
        elif not relation["cv_item"] or not relation["cv_entry_as_extracted"]:
            raise ValueError(f"Neumayer CV transcription is missing: {number}")

        doi = review["registered_doi"]
        if doi:
            evidence = doi_evidence.get(str(number))
            if (
                evidence is None
                or evidence["doi"] != doi
                or evidence["registry_url"] != review["doi_registry_url"]
                or evidence["registry_year"] != int(review["doi_registered_year"])
            ):
                raise ValueError(f"Neumayer DOI evidence changed: {number}")
            _read_pinned(
                path,
                f"doi-registry/row-{number:02}.json",
                evidence["registry_response_sha256"],
            )
        elif str(number) in doi_evidence:
            raise ValueError(f"Unassigned Neumayer DOI evidence: {number}")
        arxiv = observation.get("arxiv_correspondence")
        if arxiv and review["arxiv_id"] != arxiv["arxiv_id"]:
            raise ValueError(f"Neumayer arXiv correspondence changed: {number}")
        if not arxiv and review["arxiv_id"]:
            raise ValueError(f"Unreviewed Neumayer arXiv identity: {number}")

        work_url = _url(review["primary_record_url"])
        verification_urls = [primary_source_url, work_url]
        if relation["source_relation"] == "site_and_cv":
            verification_urls.append(CV_URL)
        if arxiv:
            verification_urls.append(f"https://arxiv.org/abs/{arxiv['arxiv_id']}")
        if doi:
            verification_urls.append(_url(review["doi_registry_url"]))
        if number in OBERWOLFACH_REPORT_URLS:
            verification_urls.append(_url(OBERWOLFACH_REPORT_URLS[number]))
        year, year_basis = _year(review)
        digest = hashlib.sha256(observation_id.encode("utf-8")).hexdigest()[:20]
        rows.append(
            {
                "person_id": PERSON_ID,
                "display_name": "Robin Neumayer",
                "primary_department_id": "math",
                "scholar_id": None,
                "scholar_account_status": "no_verified_account",
                "observation_id": observation_id,
                "source_native_id": f"researcher-bibliography:neumayer:{digest}",
                "source_kind": primary_source_kind,
                "source_url": primary_source_url,
                "source_work_url": work_url,
                "source_verification_urls": list(dict.fromkeys(verification_urls)),
                "title": review["title_as_listed"],
                "authors": review["authors_in_primary_record"],
                "year": year,
                "year_basis": year_basis,
                "doi": doi or None,
                "reviewed_genre": genre,
                "publication_type": genre,
                "map_eligible": False,
                "collaboration_eligible": False,
                "candidate_collaboration_eligible": genre
                in {"research_preprint", "journal_article"},
                "citation_count": None,
                "citation_count_known": False,
                "release_state": "source_only_release_held",
                "row_review_state": review["ingestion_decision"],
                "row_hold": review["version_identity_review"]
                if review["ingestion_decision"].startswith("hold_")
                else None,
                "source_record": {
                    "author_site_observation_or_cv_only_record": observation,
                    "cv_crosswalk": relation,
                    "genre_doi_review": review,
                },
            }
        )
    if (
        sum(row["source_kind"] == "researcher_owned_site" for row in rows) != 30
        or sum(row["source_kind"] == "researcher_cv" for row in rows) != 1
        or sum(row["doi"] is not None for row in rows) != 18
        or sum(row["row_hold"] is not None for row in rows) != 6
        or sum(
            row["source_record"]["author_site_observation_or_cv_only_record"].get(
                "arxiv_correspondence"
            )
            is not None
            for row in rows
        )
        != 23
        or sum(
            row["source_record"]["cv_crosswalk"]["source_relation"] == "site_and_cv"
            for row in rows
        )
        != 29
    ):
        raise ValueError("Neumayer staged source coverage changed")
    if any(
        row["candidate_collaboration_eligible"]
        for row in rows
        if row["reviewed_genre"] in NONCOLLABORATION_GENRES
    ):
        raise ValueError("Nonpublication Neumayer sources cannot create collaborations")
    return rows
