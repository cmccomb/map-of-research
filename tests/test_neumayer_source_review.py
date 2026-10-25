"""Guard the bounded Neumayer source capture against incomplete edits."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/source_reviews/robin-neumayer-2026-10-25"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_neumayer_review_preserves_every_source_observation() -> None:
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((SOURCE / "source-audit.json").read_text(encoding="utf-8"))
    with (SOURCE / "cv-crosswalk.csv").open(newline="", encoding="utf-8") as file:
        crosswalk = list(csv.DictReader(file))
    with (SOURCE / "doi-genre-hold-ledger.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        genre_ledger = list(csv.DictReader(file))
    doi_evidence = json.loads(
        (SOURCE / "exact-doi-evidence.json").read_text(encoding="utf-8")
    )

    assert manifest["release_eligible"] is False
    assert manifest["scholar_account_status"] == "no_verified_account"
    assert audit["person_id"] == "person-robinneumayer-a5f54a8c"
    assert audit["scholar_id"] is None
    assert audit["map_ready"] is False
    assert (
        _sha256(SOURCE / manifest["source_audit_file"])
        == manifest["source_audit_sha256"]
    )
    assert (
        _sha256(SOURCE / manifest["cv_crosswalk_file"])
        == manifest["cv_crosswalk_sha256"]
    )
    for prefix in ("doi_genre_ledger", "exact_doi_evidence"):
        assert (
            _sha256(SOURCE / manifest[f"{prefix}_file"]) == manifest[f"{prefix}_sha256"]
        )
    assert {source["file"] for source in manifest["raw_sources"]} == {
        source["path"] for source in audit["sources"]
    }
    for source in manifest["raw_sources"]:
        assert _sha256(SOURCE / source["file"]) == source["sha256"]

    observations = audit["observations"]
    assert len(observations) == len(crosswalk) == len(genre_ledger) == 31
    assert {item["observation_id"] for item in observations} == {
        row["observation_id"] for row in crosswalk
    }
    assert {item["observation_id"] for item in observations} == {
        row["observation_id"] for row in genre_ledger
    }
    held = sum(row["ingestion_decision"].startswith("hold_") for row in genre_ledger)
    assert held == 6
    assert len(doi_evidence) == manifest["exact_registered_doi_count"] == 18
    assert {int(index) for index in doi_evidence} == {
        int(row["observation_row"]) for row in genre_ledger if row["registered_doi"]
    }
    for index, evidence in doi_evidence.items():
        raw = SOURCE / "doi-registry" / f"row-{int(index):02}.json"
        assert _sha256(raw) == evidence["registry_response_sha256"]
        record = json.loads(raw.read_text(encoding="utf-8"))
        registered = (
            record["data"]["attributes"]["doi"]
            if evidence["registry"] == "datacite"
            else record["message"]["DOI"]
        )
        assert registered == evidence["doi"]
        assert genre_ledger[int(index) - 1]["registered_doi"] == registered
    assert {int(row["cv_item"]) for row in crosswalk if row["cv_item"]} == set(
        range(1, 31)
    )
    assert sum(row["source_relation"] == "site_and_cv" for row in crosswalk) == 29
    assert sum(row["source_relation"] == "cv_only" for row in crosswalk) == 1
    assert sum(row["source_relation"] == "site_only" for row in crosswalk) == 1
    assert all(row["cv_entry_as_extracted"] for row in crosswalk if row["cv_item"])

    site_observations = [
        item
        for item in observations
        if item["provenance"] == "researcher_owned_website"
    ]
    assert len(site_observations) == 30
    assert len(
        re.findall(r"<li(?:\s|>)", (SOURCE / "author-site.html").read_text(), re.I)
    ) == len(site_observations)

    root = ElementTree.parse(SOURCE / "arxiv-author.xml").getroot()
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    arxiv_ids = {
        re.sub(r"v\d+$", "", entry.find("atom:id", namespace).text.rsplit("/", 1)[-1])
        for entry in root.findall("atom:entry", namespace)
    }
    reviewed_ids = {
        item["arxiv_correspondence"]["arxiv_id"]
        for item in observations
        if item["arxiv_correspondence"]
    }
    assert len(arxiv_ids) == len(reviewed_ids) == 23
    assert arxiv_ids == reviewed_ids
