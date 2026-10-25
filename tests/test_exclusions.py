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
    assert len(exclusions) == 683
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
    assert sum(key[0] == "ZEHOKUkAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "jrRg4PAAAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "AcW2EXwAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "kcXOB1MAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "jcb81yAAAAAJ" for key in exclusions) == 2
    assert sum(key[0] == "wex9ft0AAAAJ" for key in exclusions) == 5
    assert sum(key[0] == "6pPZKd0AAAAJ" for key in exclusions) == 3
    assert sum(key[0] == "FrF1YXQAAAAJ" for key in exclusions) == 176
    assert exclusions[("FrF1YXQAAAAJ", "FrF1YXQAAAAJ:HJSXoJQnj-YC")].reason_code == (
        "false_profile_attribution"
    )
    assert exclusions[("FrF1YXQAAAAJ", "FrF1YXQAAAAJ:3bvyWxjaHKcC")].reason_code == (
        "false_profile_attribution"
    )
    assert exclusions[("FrF1YXQAAAAJ", "FrF1YXQAAAAJ:vDijr-p_gm4C")].reason_code == (
        "nonpublication"
    )
    assert {decision.reason_code for decision in exclusions.values()} == {
        "nonpublication",
        "false_profile_attribution",
        "corrupted_duplicate",
    }


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
