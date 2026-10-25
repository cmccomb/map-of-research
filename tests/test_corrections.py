import json
from pathlib import Path

import pandas
import pytest

from map_of_research.corrections import (
    DEFAULT_CORRECTIONS_PATH,
    apply_corrections,
    load_corrections,
)
from map_of_research.dataset import build_dataset_tables
from map_of_research.registry import load_registry
from tests.registry_helpers import write_registry


def _entry(**changes: object) -> dict[str, object]:
    return {
        "scholar_id": "oneAAAAJ",
        "author_pub_id": "oneAAAAJ:paper",
        "expected": {
            "title": "Damaged Title",
            "year": 1712,
            "venue": "Conference, 1712",
        },
        "corrected": {"title": "Verified Title", "year": 2011},
        "basis": "The conference archive gives the publication title and year.",
        "evidence_url": "https://example.test/archive/paper",
        "verified_at": "2026-10-25",
        **changes,
    }


def _manifest(path: Path, entries: object, *, version: int = 1) -> Path:
    path.write_text(json.dumps({"schema_version": version, "corrections": entries}))
    return path


def test_checked_in_corrections_are_exact_and_leave_two_ambiguous_rows_held() -> None:
    corrections = load_corrections(DEFAULT_CORRECTIONS_PATH)
    scholar_id = "8iLLOQgAAAAJ"
    assert {key[1].split(":")[-1] for key in corrections} == {
        "xyvS_IvSCKsC",
        "WJVC3Jt7v1AC",
        "OTTXONDVkokC",
        "lvd772isFD0C",
        "gKiMpY-AVTkC",
        "nRpfm8aw39MC",
        "TesyEGJKHF4C",
    }
    assert all(key[0] == scholar_id for key in corrections)
    for held in ("CdxZDUztZiMC", "sA9dB-pw3HoC"):
        assert (scholar_id, f"{scholar_id}:{held}") not in corrections


def test_correction_changes_effective_work_key_and_preserves_raw_scholar_row(
    tmp_path,
) -> None:
    paths = write_registry(
        tmp_path / "registry",
        people=[
            {
                "person_id": "person-one",
                "display_name": "One Person",
                "scholar_id": "oneAAAAJ",
            }
        ],
        memberships=[
            {
                "person_id": "person-one",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/faculty",
                "verified_at": "2026-10-25",
            }
        ],
    )
    registry = load_registry(*paths)
    raw_json = json.dumps(
        {
            "title": "Damaged Title",
            "pub_year": "1712",
            "venue": "Conference, 1712",
            "author_pub_id": "oneAAAAJ:paper",
        },
        sort_keys=True,
    )
    base = {
        "scholar_id": "oneAAAAJ",
        "person_id": "person-one",
        "display_name": "One Person",
        "faculty": "One Person",
        "department_ids": ["ece"],
        "memberships": [{"department_id": "ece", "role": "faculty", "included": True}],
        "authors": "One Person and Other Person",
        "citation": "",
        "citation_count": 2,
        "source_url": "https://scholar.google.com/citations?user=oneAAAAJ",
        "fetched_at_utc": "2026-10-25T00:00:00+00:00",
        "source_type": "scholar",
        "source_doi": "",
        "source_verified_at": "",
        "source_verification_urls": [],
        "external_coauthors": [],
        "embedding": [0.0] * 768,
    }
    observations = pandas.DataFrame(
        [
            {
                **base,
                "author_pub_id": "oneAAAAJ:paper",
                "title": "Damaged Title",
                "year": 1712,
                "venue": "Conference, 1712",
                "source_record_json": raw_json,
            },
            {
                **base,
                "author_pub_id": "oneAAAAJ:other",
                "title": "Unrelated Paper",
                "year": 2020,
                "venue": "Journal, 2020",
                "source_record_json": '{"title":"Unrelated Paper"}',
            },
        ]
    )
    corrections = load_corrections(_manifest(tmp_path / "corrections.json", [_entry()]))
    before = build_dataset_tables(observations, registry)["profile_publications"]
    effective, count = apply_corrections(observations, corrections)
    after = build_dataset_tables(effective, registry)["profile_publications"]
    assert count == 1
    assert observations.loc[0, "title"] == "Damaged Title"
    assert observations.loc[0, "year"] == 1712
    assert effective.loc[0, "title"] == "Verified Title"
    assert effective.loc[0, "year"] == 2011
    assert effective.loc[0, "source_record_json"] == raw_json
    assert effective.loc[1, "title"] == "Unrelated Paper"
    assert effective.loc[1, "metadata_correction_basis"] == ""
    assert effective.loc[0, "metadata_correction_fields"] == "title,year"
    assert effective.loc[0, "metadata_correction_evidence_url"] == (
        "https://example.test/archive/paper"
    )
    assert effective.loc[0, "metadata_correction_verified_at"] == "2026-10-25"
    assert before.loc[0, "work_id"] != after.loc[0, "work_id"]
    assert before.loc[1, "work_id"] == after.loc[1, "work_id"]
    assert after.loc[0, "source_record_json"] == raw_json
    assert len(build_dataset_tables(effective, registry)["works"]) == 2


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (None, "must be an object"),
        (_entry(scholar_id=""), "invalid observation identity"),
        (_entry(author_pub_id="otherAAAAJ:paper"), "invalid observation identity"),
        (_entry(expected={"title": "A"}), "invalid bibliographic fields"),
        (_entry(corrected={}), "invalid bibliographic fields"),
        (
            _entry(expected={"title": "A", "year": True, "venue": "B"}),
            "invalid bibliographic fields",
        ),
        (_entry(corrected={"doi": "10.1000/test"}), "invalid bibliographic fields"),
        (_entry(corrected={"year": 2101}), "invalid bibliographic fields"),
        (_entry(corrected={"year": 1712}), "must change every listed field"),
        (_entry(basis=""), "needs a specific basis"),
        (_entry(evidence_url="http://example.test"), "HTTPS evidence URL"),
        (_entry(verified_at="not-a-date"), "YYYY-MM-DD"),
    ],
)
def test_manifest_rejects_unsafe_corrections(tmp_path, entry, message) -> None:
    path = _manifest(tmp_path / "corrections.json", [entry])
    with pytest.raises(ValueError, match=message):
        load_corrections(path)


def test_manifest_rejects_bad_schema_list_and_duplicate_identity(tmp_path) -> None:
    with pytest.raises(ValueError, match="schema"):
        load_corrections(_manifest(tmp_path / "corrections.json", [], version=2))
    (tmp_path / "corrections.json").write_text("[]")
    with pytest.raises(ValueError, match="schema"):
        load_corrections(tmp_path / "corrections.json")
    with pytest.raises(ValueError, match="must be a list"):
        load_corrections(_manifest(tmp_path / "corrections.json", {}))
    with pytest.raises(ValueError, match="Duplicate"):
        load_corrections(_manifest(tmp_path / "corrections.json", [_entry(), _entry()]))


def test_correction_fails_closed_on_changed_source_or_non_scholar_row(tmp_path) -> None:
    corrections = load_corrections(_manifest(tmp_path / "corrections.json", [_entry()]))
    row = {
        "scholar_id": "oneAAAAJ",
        "author_pub_id": "oneAAAAJ:paper",
        "title": "Damaged Title",
        "year": 1712,
        "venue": "Conference, 1712",
        "source_type": "scholar",
        "source_record_json": "{}",
    }
    with pytest.raises(ValueError, match="source title changed"):
        apply_corrections(
            pandas.DataFrame([{**row, "title": "New Source Title"}]), corrections
        )
    with pytest.raises(ValueError, match="non-Scholar"):
        apply_corrections(
            pandas.DataFrame([{**row, "source_type": "curated_verified"}]),
            corrections,
        )
