from types import SimpleNamespace

import numpy
import pandas
import pytest

import map_of_research.dataset as dataset
from map_of_research.dataset import build_dataset_tables
from map_of_research.registry import load_registry
from tests.registry_helpers import write_registry


def test_dataset_retains_observations_while_deduplicating_exact_work(tmp_path) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path / "registry",
        people=[
            {
                "person_id": "person-one",
                "display_name": "One Person",
                "scholar_id": "oneAAAAJ",
                "scholar_id_source_url": "https://example.test/one",
                "scholar_id_verified_at": "2026-07-17",
                "orcid": "",
                "homepage_url": "",
                "notes": "",
            },
            {
                "person_id": "person-two",
                "display_name": "Two Person",
                "scholar_id": "twoAAAAJ",
                "orcid": "",
                "homepage_url": "",
                "notes": "",
            },
        ],
        memberships=[
            {
                "person_id": person_id,
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "legacy_label": person_id,
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-07-17",
            }
            for person_id in ("person-one", "person-two")
        ],
    )
    registry = load_registry(people_path, memberships_path, departments_path)
    memberships = [
        {
            "department_id": "ece",
            "role": "faculty",
            "included": True,
            "legacy_label": "Person",
            "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
            "verified_at": "2026-07-17",
        }
    ]
    rows = []
    for person_id, scholar_id, display_name in (
        ("person-one", "oneAAAAJ", "One Person"),
        ("person-two", "twoAAAAJ", "Two Person"),
    ):
        rows.append(
            {
                "scholar_id": scholar_id,
                "person_id": person_id,
                "display_name": display_name,
                "faculty": display_name,
                "department_ids": ["ece"],
                "memberships": memberships,
                "author_pub_id": f"{scholar_id}:paper",
                "title": "The Same Paper!",
                "authors": "One Person and Two Person",
                "year": 2025,
                "venue": "Journal",
                "citation": "Journal, 2025",
                "citation_count": 4,
                "source_url": "",
                "source_record_json": '{"pages":"1-10"}',
                "fetched_at_utc": "2026-01-01T00:00:00+00:00",
                "embedding": [0.0] * 768,
            }
        )

    tables = build_dataset_tables(pandas.DataFrame(rows), registry)

    assert len(tables["profile_publications"]) == 2
    assert len(tables["authorships"]) == 2
    assert len(tables["works"]) == 1
    person = tables["people"].set_index("person_id").loc["person-one"]
    assert person["scholar_id_source_url"] == "https://example.test/one"
    assert person["scholar_id_verified_at"] == "2026-07-17"
    work = tables["works"].iloc[0]
    assert work["observation_count"] == 2
    assert bool(work["map_eligible"])
    assert work["map_exclusion_reasons"] == []
    assert work["faculty"] == ["One Person", "Two Person"]
    assert set(work["author_pub_ids"]) == {
        "oneAAAAJ:paper",
        "twoAAAAJ:paper",
    }
    assert person["mapped_work_count"] == 1


def test_dataset_retains_quality_exclusions_and_prefers_an_eligible_observation(
    tmp_path,
) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path / "registry",
        people=[
            {
                "person_id": "person-one",
                "display_name": "One Person",
                "scholar_id": "oneAAAAJ",
                "orcid": "",
                "homepage_url": "",
                "notes": "",
            }
        ],
        memberships=[
            {
                "person_id": "person-one",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "legacy_label": "One",
                "source_url": "https://example.test/ece",
                "verified_at": "2026-07-17",
            }
        ],
    )
    registry = load_registry(people_path, memberships_path, departments_path)
    membership = {
        "department_id": "ece",
        "role": "faculty",
        "included": True,
    }
    base = {
        "scholar_id": "oneAAAAJ",
        "person_id": "person-one",
        "display_name": "One Person",
        "faculty": "One Person",
        "department_ids": ["ece"],
        "memberships": [membership],
        "authors": "",
        "year": None,
        "venue": "",
        "citation": "",
        "citation_count": 0,
        "source_url": "",
        "fetched_at_utc": "2026-01-01T00:00:00+00:00",
    }
    rows = [
        {
            **base,
            "author_pub_id": "one:affiliation",
            "title": "Department of Biomedical Engineering",
            "source_record_json": "{}",
            "embedding": [1.0, 0.0],
        },
        {
            **base,
            "author_pub_id": "one:editorial",
            "title": "Guest Editorial: A Special Issue",
            "source_record_json": '{"doi":"10.1000/shared"}',
            "embedding": [0.0, 1.0],
        },
        {
            **base,
            "author_pub_id": "one:person-index",
            "title": "One Person",
            "source_record_json": '{"doi":"10.1000/shared"}',
            "embedding": [0.25, 0.75],
        },
        {
            **base,
            "author_pub_id": "one:research",
            "title": "A substantive research paper",
            "year": 2025,
            "venue": "Journal",
            "source_record_json": '{"doi":"10.1000/shared"}',
            "embedding": [0.5, 0.5],
        },
    ]

    tables = build_dataset_tables(pandas.DataFrame(rows), registry)
    observations = tables["profile_publications"].set_index("author_pub_id")
    works = tables["works"].set_index("doi")
    person = tables["people"].iloc[0]

    assert not bool(observations.loc["one:affiliation", "map_eligible"])
    assert observations.loc["one:affiliation", "map_exclusion_reasons"] == [
        "affiliation_or_contact"
    ]
    assert not bool(observations.loc["one:editorial", "map_eligible"])
    assert observations.loc["one:person-index", "map_exclusion_reasons"] == [
        "person_or_citation_index"
    ]
    assert bool(observations.loc["one:research", "map_eligible"])
    assert not bool(works.loc["", "map_eligible"])
    assert works.loc["", "map_exclusion_reasons"] == ["affiliation_or_contact"]
    assert bool(works.loc["10.1000/shared", "map_eligible"])
    assert works.loc["10.1000/shared", "map_exclusion_reasons"] == []
    assert works.loc["10.1000/shared", "title"] == "A substantive research paper"
    assert works.loc["10.1000/shared", "embedding"] == [0.5, 0.5]
    assert person["unique_work_count"] == 2
    assert person["mapped_work_count"] == 1


def test_scalar_normalizers_are_conservative_and_deterministic() -> None:
    assert dataset._as_list(None) == []
    assert dataset._as_list([1]) == [1]
    assert dataset._as_list((1, 2)) == [1, 2]
    assert dataset._as_list(numpy.asarray([1, 2])) == [1, 2]
    assert dataset._as_list(numpy.asarray(1)) == [1]
    assert dataset._as_list(1) == [1]
    assert dataset.normalize_title("  A Study: Of Things! ") == "a study of things"
    assert dataset._clean_text(None) == ""
    assert dataset._clean_text(float("nan")) == ""
    assert dataset._year(None) is None
    assert dataset._year(float("nan")) is None
    assert dataset._year("2025") == 2025
    assert dataset._year("unknown") is None
    assert dataset._doi("prefix 10.1000/ABC.1);", "") == "10.1000/abc.1"
    assert dataset._doi("none") == ""


def test_work_identity_prefers_doi_then_title_year_then_observation() -> None:
    base = {
        "scholar_id": "oneAAAAJ",
        "author_pub_id": "one:paper",
        "title": "A Paper",
        "year": 2025,
        "source_url": "",
        "citation": "",
        "source_record_json": "",
    }
    doi_identity = dataset._work_identity(
        SimpleNamespace(**{**base, "source_url": "https://doi.org/10.1000/ABC"})
    )
    title_identity = dataset._work_identity(SimpleNamespace(**base))
    fallback_identity = dataset._work_identity(
        SimpleNamespace(**{**base, "title": "", "year": None})
    )
    assert doi_identity[1:] == ("doi", "10.1000/abc")
    assert title_identity[1:] == ("normalized_title_year", "")
    assert fallback_identity[1:] == ("profile_publication", "")
    assert len({doi_identity[0], title_identity[0], fallback_identity[0]}) == 3


def test_variant_selection_has_stable_tie_breaks() -> None:
    assert dataset._best_text([]) == ""
    assert dataset._best_text(["Short", "A longer value", "Short"]) == "Short"
    assert dataset._best_text(["beta", "Alpha"]) == "Alpha"
    assert dataset._variants([" beta ", "Alpha", "beta", None]) == [
        "Alpha",
        "beta",
    ]
    assert dataset._mode_year([None, "bad"]) is None
    assert dataset._mode_year([2025, "2024", 2024, 2025]) == 2024


def test_membership_normalization_handles_strings_and_rejects_scalars() -> None:
    normalized = dataset._normalize_memberships(
        [
            {
                "department_id": " ece ",
                "role": " faculty ",
                "included": "TRUE",
            },
            {"department_id": "mse", "included": 0},
        ]
    )
    assert normalized[0]["included"] is True
    assert normalized[0]["department_id"] == "ece"
    assert normalized[1]["included"] is False
    with pytest.raises(ValueError, match="must be an object"):
        dataset._normalize_memberships(["ece"])


def test_dataset_keeps_excluded_memberships_and_people_without_observations(
    tmp_path,
) -> None:
    people_path, memberships_path, departments_path = write_registry(
        tmp_path / "registry",
        people=[
            {
                "person_id": "person-observed",
                "display_name": "Observed Person",
                "scholar_id": "observedAJ",
                "orcid": "",
                "homepage_url": "",
                "notes": "",
            },
            {
                "person_id": "person-unobserved",
                "display_name": "Unobserved Person",
                "scholar_id": "",
                "orcid": "",
                "homepage_url": "",
                "notes": "Retained roster row",
            },
        ],
        memberships=[
            {
                "person_id": "person-observed",
                "department_id": "ece",
                "role": "affiliate",
                "included": "false",
                "legacy_label": "Observed",
                "source_url": "https://example.test/ece",
                "verified_at": "2026-07-17",
            },
            {
                "person_id": "person-unobserved",
                "department_id": "ece",
                "role": "emeritus",
                "included": "true",
                "legacy_label": "Unobserved",
                "source_url": "https://example.test/ece",
                "verified_at": "2026-07-17",
            },
        ],
    )
    registry = load_registry(people_path, memberships_path, departments_path)
    row = {
        "scholar_id": "observedAJ",
        "person_id": "person-observed",
        "display_name": "Observed Person",
        "faculty": "Observed Person",
        "department_ids": [],
        "memberships": [
            {
                "department_id": "ece",
                "role": "affiliate",
                "included": False,
            }
        ],
        "author_pub_id": "observed:paper",
        "title": "Paper",
        "authors": "Observed Person",
        "year": None,
        "venue": "",
        "citation": "",
        "citation_count": 0,
        "source_url": "",
        "source_record_json": "",
        "fetched_at_utc": "2026-01-01T00:00:00+00:00",
        "embedding": [0.0] * 4,
    }
    tables = build_dataset_tables(pandas.DataFrame([row]), registry)
    work = tables["works"].iloc[0]
    people = tables["people"].set_index("person_id")
    assert work["memberships"] == []
    assert work["department_ids"] == []
    assert people.loc["person-unobserved", "publication_observation_count"] == 0
    assert people.loc["person-unobserved", "unique_work_count"] == 0
    assert dataset.table_summary(tables) == (
        '{"authorships": 1, "people": 2, "profile_publications": 1, "works": 1}'
    )
