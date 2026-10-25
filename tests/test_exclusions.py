import csv
import json
from pathlib import Path

import pandas
import pytest

from map_of_research.dataset import build_dataset_tables
from map_of_research.exclusions import (
    DEFAULT_EXCLUSIONS_PATH,
    ReviewedExclusion,
    load_exclusions,
)
from map_of_research.publisher import build_map_artifact
from map_of_research.registry import load_registry
from tests.registry_helpers import write_registry


def _write_manifest(path: Path, entries: list[object], *, version: int = 1) -> Path:
    path.write_text(json.dumps({"schema_version": version, "exclusions": entries}))
    return path


def _entry(**changes: object) -> dict[str, object]:
    return {
        "scholar_id": "oneAAAAJ",
        "author_pub_id": "oneAAAAJ:paper",
        "reason_code": "false_profile_attribution",
        "reason": "Primary author list excludes the profile owner.",
        "evidence_url": "https://example.test/paper",
        "verified_at": "2026-10-24",
        **changes,
    }


def test_checked_in_review_manifest_is_complete_and_keyed() -> None:
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    assert len(exclusions) == 1144
    assert sum(key[0] == "TSkli5EAAAAJ" for key in exclusions) == 139
    assert sum(key[0] == "BWNUifwAAAAJ" for key in exclusions) == 13
    assert sum(key[0] == "ySlRYXoAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "28yZqbcAAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "RPkJ39IAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "n6flaAsAAAAJ" for key in exclusions) == 18
    assert sum(key[0] == "VQfvIfQAAAAJ" for key in exclusions) == 16
    assert sum(key[0] == "bC4aq-4AAAAJ" for key in exclusions) == 415
    assert sum(key[0] == "0e1jwkkAAAAJ" for key in exclusions) == 23
    assert sum(key[0] == "ctYLFRIAAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "5TCNs50AAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "CaQRJ8IAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "7ICaR4EAAAAJ" for key in exclusions) == 14
    assert sum(key[0] == "FmZaK1wAAAAJ" for key in exclusions) == 22
    assert sum(key[0] == "tWQsy4UAAAAJ" for key in exclusions) == 11
    assert sum(key[0] == "CRCzglkAAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "neJ3NIUAAAAJ" for key in exclusions) == 34
    assert sum(key[0] == "JV3ryMgAAAAJ" for key in exclusions) == 21
    assert sum(key[0] == "0tIcPvMAAAAJ" for key in exclusions) == 4
    assert sum(key[0] == "SHxT0AEAAAAJ" for key in exclusions) == 10
    assert sum(key[0] == "FkS0PMEAAAAJ" for key in exclusions) == 8
    assert sum(key[0] == "W9-dYLcAAAAJ" for key in exclusions) == 4
    assert sum(key[0] == "6ArMETUAAAAJ" for key in exclusions) == 18
    assert sum(key[0] == "_IfiZD4AAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "FXqaMl4AAAAJ" for key in exclusions) == 6
    assert sum(key[0] == "T3nx5tgAAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "x_xzlvoAAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "RnA6G0UAAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "qQLlBH4AAAAJ" for key in exclusions) == 37
    assert sum(key[0] == "bKMKO_AAAAAJ" for key in exclusions) == 14
    assert sum(key[0] == "MbRIDjgAAAAJ" for key in exclusions) == 9
    assert sum(key[0] == "wXSgFr4AAAAJ" for key in exclusions) == 16
    assert sum(key[0] == "rQ79IvMAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "wmP8_2AAAAAJ" for key in exclusions) == 9
    assert sum(key[0] == "6VKJW4kAAAAJ" for key in exclusions) == 11
    assert sum(key[0] == "-HzcsYQAAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "IggMbacAAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "xfTeH7YAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "EqKRQ4YAAAAJ" for key in exclusions) == 80
    assert sum(key[0] == "H8qYeA0AAAAJ" for key in exclusions) == 10
    assert sum(key[0] == "tfBFyTMAAAAJ" for key in exclusions) == 25
    assert sum(key[0] == "9lFYMIkAAAAJ" for key in exclusions) == 31
    assert sum(key[0] == "A0ISJPUAAAAJ" for key in exclusions) == 27
    assert sum(key[0] == "P_AmD6MAAAAJ" for key in exclusions) == 14
    assert sum(key[0] == "uHqbF7sAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "eVxdemsAAAAJ" for key in exclusions) == 5
    assert sum(key[0] == "q1g_W5sAAAAJ" for key in exclusions) == 5
    assert sum(key[0] == "0Nj-v7sAAAAJ" for key in exclusions) == 20
    assert sum(key[0] == "uOCVq1cAAAAJ" for key in exclusions) == 1
    assert sum(key[0] == "ZEHOKUkAAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "jrRg4PAAAAAJ" for key in exclusions) == 5
    assert sum(key[0] == "79uTAy8AAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "AcW2EXwAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "kcXOB1MAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "jcb81yAAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "wex9ft0AAAAJ" for key in exclusions) == 5
    assert sum(key[0] == "6pPZKd0AAAAJ" for key in exclusions) == 3
    assert {decision.reason_code for decision in exclusions.values()} == {
        "nonpublication",
        "false_profile_attribution",
        "corrupted_duplicate",
    }


def test_kim_full_capture_screen_only_excludes_exact_reviewed_ids() -> None:
    path = Path("registry/reviews/2026-10-24-chem-kim-918-screen.csv")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 918
    assert len({row["author_pub_id"] for row in rows}) == 918
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    proposed = [row for row in rows if row["status"].startswith("proposed_")]
    assert len(proposed) == 407
    for row in rows:
        key = ("bC4aq-4AAAAJ", row["author_pub_id"])
        if row["status"].startswith("proposed_") or row["status"] == "already_excluded":
            assert key in exclusions
        else:
            assert key not in exclusions
    knu = [
        row
        for row in proposed
        if row["status"].startswith("proposed_other_person_knu_")
    ]
    assert len(knu) == 320
    assert all(
        row["matched_doi"] and "0000-0001-9787-4684" in row["evidence_url"]
        for row in knu
    )
    primary = [
        row for row in proposed if row["status"] == "proposed_other_person_primary_work"
    ]
    assert len(primary) == 65
    assert all(row["evidence_url"].startswith("https://") for row in primary)
    by_index = {int(row["capture_index"]): row for row in rows}
    assert (
        by_index[99]["author_pub_id"] == "bC4aq-4AAAAJ:cB__R-XWw9UC"
        and by_index[99]["status"] == "proposed_other_person_primary_work"
        and "foodsci.oregonstate.edu" in by_index[99]["evidence_url"]
        and by_index[161]["author_pub_id"] == "bC4aq-4AAAAJ:YlPif8NxrbYC"
        and by_index[161]["status"] == "proposed_other_person_primary_work"
        and "dbpia.co.kr" in by_index[161]["evidence_url"]
    )
    assert (
        by_index[142]["author_pub_id"] == "bC4aq-4AAAAJ:VOx2b1Wkg3QC"
        and by_index[142]["status"] == "proposed_other_person_primary_work"
        and "pure.korea.ac.kr" in by_index[142]["evidence_url"]
        and by_index[175]["author_pub_id"] == "bC4aq-4AAAAJ:qsWQJNntlusC"
        and by_index[175]["status"] == "proposed_other_person_primary_work"
        and "research.knu.ac.kr" in by_index[175]["evidence_url"]
    )
    assert (
        by_index[148]["author_pub_id"] == "bC4aq-4AAAAJ:RfUwGJFMQ-0C"
        and by_index[148]["status"] == "proposed_other_person_primary_work"
        and by_index[148]["matched_doi"] == "10.1016/j.jlumin.2015.03.026"
    )
    assert (
        by_index[267]["author_pub_id"] == "bC4aq-4AAAAJ:foquWX3nUaYC"
        and by_index[267]["status"] == "proposed_other_person_primary_work"
        and "research.knu.ac.kr" in by_index[267]["evidence_url"]
    )
    assert (
        by_index[312]["status"] == "cmu_primary_work_retained"
        and by_index[312]["matched_doi"] == "10.1016/j.molstruc.2017.05.138"
        and "scholarscommons.fgcu.edu" in by_index[312]["evidence_url"]
        and by_index[797]["status"] == "proposed_corrupted_duplicate_primary"
        and by_index[797]["author_pub_id"] == "bC4aq-4AAAAJ:9Nmd_mFXekcC"
    )
    assert (
        by_index[140]["status"] == "cmu_primary_work_retained"
        and by_index[140]["matched_doi"]
        == "10.1002/(sici)1522-2594(199902)41:2<299::aid-mrm13>3.0.co;2-r"
        and by_index[470]["status"] == "cmu_primary_work_retained"
        and by_index[470]["matched_doi"] == "10.1002/qua.560382480"
    )
    assert (
        by_index[156]["status"] == "proposed_other_person_primary_work"
        and by_index[795]["status"] == "proposed_corrupted_duplicate_primary"
        and by_index[795]["author_pub_id"] == "bC4aq-4AAAAJ:pUxgyZctzPYC"
    )
    assert all(
        by_index[index]["status"] == "proposed_other_person_primary_work"
        and "healthstat.snu.ac.kr" in by_index[index]["evidence_url"]
        for index in (12, 38, 40)
    )
    assert (
        by_index[106]["status"] == "proposed_other_person_primary_work"
        and "PMC4611728" in by_index[106]["evidence_url"]
    )
    assert (
        by_index[81]["status"] == "proposed_other_person_primary_work"
        and "US20030032293A1" in by_index[81]["evidence_url"]
    )
    assert (
        by_index[85]["status"] == "proposed_other_person_primary_work"
        and "pure.korea.ac.kr" in by_index[85]["evidence_url"]
    )
    assert {
        by_index[index]["author_pub_id"]
        for index in (349, 546, 590, 605, 638, 746)
        if by_index[index]["status"] == "proposed_other_person_primary_work"
        and "research.knu.ac.kr" in by_index[index]["evidence_url"]
    } == {
        "bC4aq-4AAAAJ:pxXbYLTb8EgC",
        "bC4aq-4AAAAJ:D_tqNUsBuKoC",
        "bC4aq-4AAAAJ:2vr6o8x5NLkC",
        "bC4aq-4AAAAJ:4hFrxpcac9AC",
        "bC4aq-4AAAAJ:mJbmKSuM8toC",
        "bC4aq-4AAAAJ:yKzB5RS27GgC",
    }
    assert {
        by_index[index]["author_pub_id"]: by_index[index]["matched_doi"]
        for index in (109, 174, 235, 241)
        if by_index[index]["status"] == "proposed_other_person_primary_work"
        and by_index[index]["evidence_url"].startswith(
            "https://research.knu.ac.kr/en/publications/"
        )
    } == {
        "bC4aq-4AAAAJ:sbeIDTyQOFgC": "10.1103/PhysRevD.90.052006",
        "bC4aq-4AAAAJ:bVQMTfhMCi4C": "10.1088/1748-0221/13/09/P09006",
        "bC4aq-4AAAAJ:M0j1y4EgrScC": "10.1088/1475-7516/2021/02/013",
        "bC4aq-4AAAAJ:cSdaV2aYdYsC": "10.1016/j.nima.2021.165431",
    }
    assert all(
        by_index[index]["status"] == "knu_orcid_exact_title_year_missing_doi_hold"
        for index in (357, 655)
    )
    supporting = next(row for row in rows if row["capture_index"] == "917")
    assert supporting["status"] == "proposed_acs_supporting_information_nonpublication"
    assert (
        exclusions[("bC4aq-4AAAAJ", supporting["author_pub_id"])].reason_code
        == "nonpublication"
    )


@pytest.mark.parametrize(
    ("entries", "version", "message"),
    [
        ([], 2, "Unsupported Scholar-exclusion schema"),
        (None, 1, "Scholar exclusions must be a list"),
        ([None], 1, "must be an object"),
        ([_entry(scholar_id="")], 1, "invalid observation identity"),
        ([_entry(author_pub_id="other:paper")], 1, "invalid observation identity"),
        ([_entry(reason_code="unknown")], 1, "known code"),
        ([_entry(reason="")], 1, "known code"),
        ([_entry(evidence_url="http://example.test")], 1, "HTTPS evidence URL"),
        ([_entry(verified_at="yesterday")], 1, "verified_at"),
        ([_entry(), _entry()], 1, "Duplicate Scholar exclusion"),
    ],
)
def test_review_manifest_rejects_invalid_decisions(
    tmp_path, entries, version, message
) -> None:
    path = _write_manifest(tmp_path / "exclusions.json", entries, version=version)
    with pytest.raises(ValueError, match=message):
        load_exclusions(path)


def test_reviewed_false_attribution_cannot_attach_faculty_to_shared_work(
    tmp_path,
) -> None:
    paths = write_registry(
        tmp_path / "registry",
        people=[
            {
                "person_id": "person-one",
                "display_name": "One",
                "scholar_id": "oneAAAAJ",
            },
            {
                "person_id": "person-two",
                "display_name": "Two",
                "scholar_id": "twoAAAAJ",
            },
        ],
        memberships=[
            {
                "person_id": person_id,
                "department_id": department_id,
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/directory",
                "verified_at": "2026-10-24",
            }
            for person_id, department_id in (
                ("person-one", "ece"),
                ("person-two", "math"),
            )
        ]
        + [
            {
                "person_id": "person-two",
                "department_id": "ece",
                "role": "affiliate",
                "included": "false",
                "source_url": "https://example.test/directory",
                "verified_at": "2026-10-24",
            }
        ],
        departments=[
            {
                "department_id": "ece",
                "school_id": "engineering",
                "title": "ECE",
                "directory_url": "https://example.test/ece",
            },
            {
                "department_id": "math",
                "school_id": "mcs",
                "title": "Mathematics",
                "directory_url": "https://example.test/math",
            },
        ],
        schools=[
            {
                "school_id": "engineering",
                "title": "College of Engineering",
                "directory_url": "https://example.test/engineering",
            },
            {
                "school_id": "mcs",
                "title": "Mellon College of Science",
                "directory_url": "https://example.test/mcs",
            },
        ],
    )
    registry = load_registry(*paths)
    base = {
        "faculty": "",
        "authors": "Two and Collaborator",
        "title": "A Substantive Shared Research Article",
        "year": 2024,
        "venue": "Journal",
        "citation": "Journal 2024",
        "source_record_json": "{}",
        "fetched_at_utc": "2026-10-24T00:00:00+00:00",
        "embedding": [0.1, 0.2],
        "source_type": "scholar",
    }
    bad = {
        **base,
        "scholar_id": "oneAAAAJ",
        "author_pub_id": "oneAAAAJ:paper",
        "person_id": "person-one",
        "display_name": "One",
        "memberships": [{"department_id": "ece", "included": True}],
        "department_ids": ["ece"],
        "authors": "One and Incorrect",
        "venue": "Misattributed Preprint",
        "citation": "Bad venue 2024",
        "citation_count": 900,
        "source_url": "https://example.test/bad-source",
        "external_coauthors": [
            {
                "person_id": "external-bad",
                "display_name": "Bad Link",
                "source_url": "https://example.test/bad-author",
            }
        ],
    }
    good = {
        **base,
        "scholar_id": "twoAAAAJ",
        "author_pub_id": "twoAAAAJ:paper",
        "person_id": "person-two",
        "display_name": "Two",
        "memberships": [
            {"department_id": "math", "included": True},
            {"department_id": "ece", "included": False},
        ],
        "department_ids": ["math"],
        "citation_count": 4,
        "source_url": "https://example.test/good-source",
        "external_coauthors": [],
    }
    nonpublication = {
        **bad,
        "author_pub_id": "oneAAAAJ:award",
        "title": "NSF Award 1234567",
        "source_url": "https://example.test/award",
    }
    decisions = {
        ("oneAAAAJ", "oneAAAAJ:paper"): ReviewedExclusion(
            "false_profile_attribution",
            "The paper's author list excludes One.",
            "https://example.test/paper",
            "2026-10-24",
        ),
        ("oneAAAAJ", "oneAAAAJ:award"): ReviewedExclusion(
            "nonpublication",
            "Grant award, not a publication.",
            "https://example.test/award",
            "2026-10-24",
        ),
    }

    tables = build_dataset_tables(
        pandas.DataFrame([bad, good, nonpublication]), registry, decisions
    )
    raw = tables["profile_publications"].set_index("author_pub_id")
    assert len(raw) == 3
    assert not bool(raw.loc["oneAAAAJ:paper", "map_eligible"])
    assert raw.loc["oneAAAAJ:paper", "review_exclusion_code"] == (
        "false_profile_attribution"
    )
    assert raw.loc["oneAAAAJ:paper", "review_evidence_url"] == (
        "https://example.test/paper"
    )
    assert raw.loc["oneAAAAJ:paper", "review_verified_at"] == "2026-10-24"
    assert raw.loc["oneAAAAJ:award", "review_exclusion_reason"] == (
        "Grant award, not a publication."
    )
    assert raw.loc["twoAAAAJ:paper", "review_exclusion_code"] == ""
    assert tables["authorships"]["person_id"].tolist() == ["person-two"]
    shared = tables["works"].loc[tables["works"]["map_eligible"]].iloc[0]
    assert shared["faculty"] == ["Two"]
    assert shared["person_ids"] == ["person-two"]
    assert shared["department_ids"] == ["math"]
    assert shared["school_ids"] == ["mcs"]
    assert shared["authors"] == "Two and Collaborator"
    assert shared["venue"] == "Journal"
    assert shared["citation"] == "Journal 2024"
    assert shared["citation_count"] == 4
    assert shared["source_urls"] == ["https://example.test/good-source"]
    assert shared["author_pub_ids"] == ["twoAAAAJ:paper"]
    assert shared["observation_count"] == 1
    assert shared["external_coauthor_ids"] == []
    excluded = tables["works"].loc[~tables["works"]["map_eligible"]].iloc[0]
    assert excluded["has_included_source_profile"]
    assert excluded["memberships"] == []
    assert excluded["map_exclusion_reasons"] == ["reviewed_nonpublication"]
    people = tables["people"].set_index("person_id")
    assert people.loc["person-one", "publication_observation_count"] == 2
    assert people.loc["person-one", "unique_work_count"] == 0
    assert people.loc["person-one", "mapped_work_count"] == 0
    assert people.loc["person-two", "mapped_work_count"] == 1

    works = tables["works"].copy()
    works["x"] = 0.1
    works["y"] = 0.2
    works["tsne_x"] = 0.3
    works["tsne_y"] = 0.4
    works["keyword_id"] = "topic-one"
    works["keyword"] = "network theory"
    works["keyword_extracted"] = "network theory"
    works["keyword_label_reviewed"] = False
    works["detail_keyword_id"] = "topic-one-detail"
    works["detail_keyword"] = "random network estimates"
    works["detail_keyword_extracted"] = "random network estimates"
    works["detail_keyword_label_reviewed"] = False
    artifact = build_map_artifact(
        works, registry, generated_at_utc="2026-10-24T12:00:00+00:00"
    )
    assert artifact["point_count"] == 1
    assert artifact["excluded_work_count"] == 1
    assert artifact["points"][0]["faculty_ids"] == ["person-two"]
    assert artifact["points"][0]["department_ids"] == ["math"]
    assert artifact["points"][0]["source_url"] == ("https://example.test/good-source")
