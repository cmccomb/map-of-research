"""Validate the held Ágoston Pisztora source inventory; never publish it."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PERSON_ID = "person-agostonpisztora-810e0d1f"


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", html.unescape(value))
    value = "".join(
        character for character in value if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def validate() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        contents = path.read_bytes()
        assert len(contents) == item["bytes"], item["path"]
        assert hashlib.sha256(contents).hexdigest() == item["sha256"], item["path"]

    observed = rows("observed-11.csv")
    related = rows("related-held-5.csv")
    assert len(observed) == 11
    assert len(related) == 5
    assert len({row["record_id"] for row in observed}) == 11
    assert len({row["observation_id"] for row in related}) == 5
    assert not (
        {row["record_id"] for row in observed}
        & {row["observation_id"] for row in related}
    )
    assert all(row["person_id"] == PERSON_ID for row in observed + related)
    assert all(
        row["release_state"] == "held" and row["graph_eligible"] == "false"
        for row in observed + related
    )
    assert all(not row["scholar_id"] for row in observed)
    assert len([row for row in observed if row["genre"] == "journal_article"]) == 8
    assert (
        len([row for row in observed if row["genre"] == "conference_proceedings"]) == 1
    )
    assert len([row for row in observed if row["genre"] == "doctoral_thesis"]) == 1
    assert len([row for row in observed if row["genre"] == "unpublished_preprint"]) == 1
    assert len([row for row in observed if row["cmu_select"] == "true"]) == 4
    assert len([row for row in observed if row["doi"]]) == 8
    assert all(
        row["candidate_family_id"] in {entry["record_id"] for entry in observed} | {""}
        for row in related
    )
    icm_version = next(
        row for row in related if row["observation_id"] == "pisztora-arxiv-icm-2003"
    )
    assert icm_version["candidate_family_id"] == "pisztora-icm-2002"
    assert icm_version["relationship"] == "proceedings_preprint_version"

    by_doi = {row["doi"].lower(): row for row in observed if row["doi"]}
    for path in (ROOT / "raw").glob("crossref-*.json"):
        message = json.loads(path.read_text(encoding="utf-8"))["message"]
        row = by_doi[message["DOI"].lower()]
        assert normalized(message["title"][0]) == normalized(row["title"]), path
        assert message["published"]["date-parts"][0][0] == int(row["year"]), path
        authors = [
            normalized(f"{author.get('given', '')} {author['family']}")
            for author in message["author"]
        ]
        assert authors == [normalized(name) for name in row["authors"].split("; ")], (
            path
        )
    assert len(list((ROOT / "raw").glob("crossref-*.json"))) == 7

    thesis = json.loads(
        (ROOT / "raw" / "datacite-thesis.json").read_text(encoding="utf-8")
    )["data"]["attributes"]
    thesis_row = by_doi[thesis["doi"].lower()]
    assert normalized(thesis["titles"][0]["title"]) == normalized(thesis_row["title"])
    assert thesis["publicationYear"] == int(thesis_row["year"])
    assert "pisztora" in normalized(thesis["creators"][0]["name"])

    cmu = normalized((ROOT / "raw" / "cmu-profile.html").read_text(encoding="utf-8"))
    assert "select publications" in cmu
    for row in observed:
        if row["cmu_select"] == "true":
            assert normalized(row["title"]) in cmu, row["record_id"]

    cna_search = normalized(
        (ROOT / "raw" / "cna-author-search.html").read_text(encoding="utf-8")
    )
    cna_detail = normalized(
        (ROOT / "raw" / "cna-00-001-detail.html").read_text(encoding="utf-8")
    )
    cna_wulff = normalized(
        (ROOT / "raw" / "cna-99-012-detail.html").read_text(encoding="utf-8")
    )
    preprint = next(
        row for row in observed if row["record_id"] == "pisztora-cna-scaling-2000"
    )
    assert "00 cna 001" in cna_search and "99 cna 012" in cna_search
    assert (
        normalized(preprint["title"]) in cna_detail and "agoston pisztora" in cna_detail
    )
    assert "on the wulff crystal in the ising model" in cna_wulff
    assert "pisztora" in normalized(
        (ROOT / "raw" / "numdam-phase-coexistence.html").read_text(encoding="utf-8")
    )
    assert "math 0304371" in normalized(
        (ROOT / "raw" / "arxiv-icm-version.html").read_text(encoding="utf-8")
    )
    assert "proceedings of the icm beijing 2002 vol 3" in normalized(
        (ROOT / "raw" / "arxiv-icm-version.html").read_text(encoding="utf-8")
    )
    assert "pisztora andrew cmu edu" in normalized(
        (ROOT / "raw" / "jsm-2001-abstract.html").read_text(encoding="utf-8")
    )
    source_hashes = json.loads(
        (ROOT / "cna-preprint-source-hashes.json").read_text(encoding="utf-8")
    )
    assert (
        source_hashes["full_text_pdf_sha256"]
        == "e9913930a8bfe806a569fe2e10377148a5531258cc36c7183e11eedd025488a8"
    )
    assert (
        source_hashes["source_tex_sha256"]
        == "4703d9fbb02b71683ddecb657f0043d79123716cfb00d6645eea622d29324ed9"
    )
    assert manifest["review_state"] == "held_source_only"
    print("Pisztora held review valid: 9 published, 1 thesis, 1 preprint, 5 related")


if __name__ == "__main__":
    validate()
