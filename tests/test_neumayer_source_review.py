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
    assert {source["file"] for source in manifest["raw_sources"]} == {
        source["path"] for source in audit["sources"]
    }
    for source in manifest["raw_sources"]:
        assert _sha256(SOURCE / source["file"]) == source["sha256"]

    observations = audit["observations"]
    assert len(observations) == len(crosswalk) == 31
    assert {item["observation_id"] for item in observations} == {
        row["observation_id"] for row in crosswalk
    }
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
