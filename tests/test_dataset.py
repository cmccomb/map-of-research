from types import SimpleNamespace

import numpy
import pandas
import pytest

import map_of_research.dataset as dataset
from map_of_research.dataset import build_dataset_tables
from map_of_research.doi_aliases import DEFAULT_DOI_ALIASES_PATH, load_doi_aliases
from map_of_research.registry import load_registry
from tests.registry_helpers import write_registry


@pytest.mark.parametrize("include_published_version", [False, True])
def test_reviewed_preprint_alias_keeps_scholar_source_and_one_work(
    tmp_path, include_published_version
) -> None:
    paths = write_registry(
        tmp_path / "registry",
        people=[
            {
                "person_id": "person-walkington",
                "display_name": "Noel J. Walkington",
                "scholar_id": "CGSMDGwAAAAJ",
            }
        ],
        memberships=[
            {
                "person_id": "person-walkington",
                "department_id": "math",
                "role": "faculty",
                "included": "true",
                "source_url": "https://www.cmu.edu/math/people/faculty/walkington.html",
                "verified_at": "2026-10-24",
            }
        ],
        departments=[
            {
                "department_id": "math",
                "school_id": "mcs",
                "title": "Mathematical Sciences",
                "directory_url": "https://www.cmu.edu/math/people/faculty/index.html",
            }
        ],
        schools=[
            {
                "school_id": "mcs",
                "title": "Mellon College of Science",
                "directory_url": "https://www.cmu.edu/mcs/",
            }
        ],
    )
    registry = load_registry(*paths)
    alias = next(iter(load_doi_aliases(DEFAULT_DOI_ALIASES_PATH).values()))
    base = {
        "person_id": "person-walkington",
        "display_name": "Noel J. Walkington",
        "faculty": "Noel J. Walkington",
        "department_ids": ["math"],
        "memberships": [{"department_id": "math", "included": True}],
        "title": alias.observed_title,
        "authors": "GL Miller, NJ Walkington, AL Wang",
        "citation_count": 6,
        "embedding": [0.0, 1.0],
        "external_coauthors": [],
        "source_verified_at": "",
        "source_verification_urls": [],
    }
    scholar = {
        **base,
        "source_type": "scholar",
        "scholar_id": "CGSMDGwAAAAJ",
        "author_pub_id": "CGSMDGwAAAAJ:SP6oXDckpogC",
        "year": 2018,
        "venue": "arXiv preprint arXiv:1812.02841, 2018",
        "citation": "arXiv preprint arXiv:1812.02841, 2018",
        "source_url": "https://scholar.google.com/citations?user=CGSMDGwAAAAJ",
        "source_record_json": '{"pub_year":"2018"}',
        "source_doi": "",
        "fetched_at_utc": "2026-10-24T23:00:00+00:00",
    }
    rows = [scholar]
    if include_published_version:
        rows.append(
            {
                **base,
                "source_type": "curated_verified",
                "scholar_id": "",
                "author_pub_id": "curated-doi:person-walkington:" + alias.doi,
                "year": 2019,
                "venue": "LIPIcs APPROX/RANDOM 2019",
                "citation": "LIPIcs 145, article 8 (2019)",
                "source_url": alias.evidence_urls[1],
                "source_record_json": "{}",
                "source_doi": alias.doi,
                "source_verified_at": alias.verified_at,
                "source_verification_urls": [alias.evidence_urls[1]],
                "fetched_at_utc": "",
            }
        )
    observations = pandas.DataFrame(rows)
    tables = build_dataset_tables(
        observations,
        registry,
        {},
        {("CGSMDGwAAAAJ", "CGSMDGwAAAAJ:SP6oXDckpogC"): alias},
    )
    assert len(tables["profile_publications"]) == len(rows)
    assert len(tables["works"]) == 1
    assert len(tables["authorships"]) == len(rows)
    assert tables["works"].iloc[0]["doi"] == alias.doi
    assert tables["works"].iloc[0]["year"] == 2018
    assert set(tables["works"].iloc[0]["source_verification_urls"]) == set(
        alias.evidence_urls
    )
    normalized = tables["profile_publications"].iloc[0]
    assert normalized["work_match_method"] == "reviewed_version_alias"
    assert normalized["source_record_json"] == scholar["source_record_json"]
    assert normalized["year"] == scholar["year"]
    assert normalized["source_doi"] == ""
    assert observations.iloc[0]["source_verified_at"] == ""


def test_reviewed_preprint_alias_rejects_conflicting_source_doi(tmp_path) -> None:
    paths = write_registry(
        tmp_path / "registry",
        people=[{"person_id": "person-one", "display_name": "One"}],
        memberships=[
            {
                "person_id": "person-one",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/one",
                "verified_at": "2026-10-24",
            }
        ],
    )
    registry = load_registry(*paths)
    alias = next(iter(load_doi_aliases(DEFAULT_DOI_ALIASES_PATH).values()))
    rows = pandas.DataFrame(
        [
            {
                "source_type": "scholar",
                "scholar_id": "CGSMDGwAAAAJ",
                "person_id": "person-one",
                "author_pub_id": "CGSMDGwAAAAJ:SP6oXDckpogC",
                "source_doi": "10.1000/conflicting",
            }
        ]
    )
    with pytest.raises(ValueError, match="conflicts with a source DOI"):
        build_dataset_tables(
            rows,
            registry,
            {},
            {("CGSMDGwAAAAJ", "CGSMDGwAAAAJ:SP6oXDckpogC"): alias},
        )


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
        ]
        + [
            {
                "person_id": "person-one",
                "department_id": "s3d",
                "role": "faculty",
                "included": "true",
                "legacy_label": "One Person",
                "source_url": "https://example.test/s3d",
                "verified_at": "2026-10-24",
            }
        ],
        departments=[
            {
                "department_id": "ece",
                "school_id": "engineering",
                "title": "Electrical & Computer Engineering",
                "directory_url": "https://example.test/ece",
            },
            {
                "department_id": "s3d",
                "school_id": "scs",
                "title": "Software and Societal Systems Department",
                "directory_url": "https://example.test/s3d",
            },
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
                "department_ids": ["ece", "s3d"]
                if person_id == "person-one"
                else ["ece"],
                "memberships": memberships
                + (
                    [
                        {
                            "department_id": "s3d",
                            "role": "faculty",
                            "included": True,
                        }
                    ]
                    if person_id == "person-one"
                    else []
                ),
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
    assert person["included_school_ids"] == ["engineering", "scs"]
    assert person["primary_department_id"] == ""
    assert person["primary_department_basis"] == "unresolved"
    assert person["primary_academic_home_type"] == "unresolved"
    assert person["primary_academic_home_id"] == ""
    assert work["school_ids"] == ["engineering", "scs"]
    assert work["school_titles"] == [
        "College of Engineering",
        "School of Computer Science",
    ]
    authorship = tables["authorships"].set_index("person_id").loc["person-one"]
    assert authorship["school_ids"] == [
        "engineering",
        "scs",
    ]


def test_curated_doi_reconciles_later_scholar_rows_without_losing_provenance(
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
                "verified_at": "2026-10-24",
            }
        ],
    )
    registry = load_registry(*paths)
    membership = {
        "department_id": "ece",
        "role": "faculty",
        "included": True,
    }
    base = {
        "person_id": "person-one",
        "display_name": "One Person",
        "faculty": "One Person",
        "department_ids": ["ece"],
        "memberships": [membership],
        "title": "A Verified Paper",
        "authors": "One Person and External Example",
        "year": 1995,
        "venue": "Journal",
        "citation": "Journal 1995",
        "citation_count": 0,
        "source_url": "",
        "source_record_json": "{}",
        "embedding": [0.0, 1.0],
        "source_doi": "",
        "source_verified_at": "",
        "source_verification_urls": [],
        "external_coauthors": [],
    }
    external = {
        "person_id": "external-example",
        "display_name": "External Example",
        "source_url": "https://example.test/author",
    }
    curated = {
        **base,
        "scholar_id": "",
        "author_pub_id": "curated-doi:10.1000/example",
        "source_type": "curated_verified",
        "source_doi": "10.1000/example",
        "source_url": "https://example.test/publisher",
        "source_verified_at": "2026-10-24",
        "source_verification_urls": ["https://example.test/publisher"],
        "external_coauthors": [external],
        "fetched_at_utc": "",
    }
    scholar_doi = {
        **base,
        "scholar_id": "oneAAAAJ",
        "author_pub_id": "one:doi",
        "source_type": "scholar",
        "source_doi": "10.1000/example",
        "fetched_at_utc": "2026-11-01T12:00:00+00:00",
    }
    scholar_title = {
        **base,
        "scholar_id": "oneAAAAJ",
        "author_pub_id": "one:no-doi",
        "source_type": "scholar",
        "fetched_at_utc": "2026-11-02T12:00:00+00:00",
    }
    conflicting_doi = {
        **scholar_title,
        "author_pub_id": "one:other-doi",
        "source_doi": "10.1000/other",
    }
    tables = build_dataset_tables(
        pandas.DataFrame([curated, scholar_doi, scholar_title, conflicting_doi]),
        registry,
    )
    observations = tables["profile_publications"].set_index("author_pub_id")
    assert observations.loc["one:no-doi", "work_match_method"] == "curated_title_year"
    assert observations.loc["one:other-doi", "doi"] == "10.1000/other"
    works = tables["works"].set_index("doi")
    assert len(works) == 2
    merged = works.loc["10.1000/example"]
    assert merged["observation_count"] == 3
    assert merged["source_types"] == ["curated_verified", "scholar"]
    assert merged["source_verified_at"] == "2026-10-24"
    assert merged["source_verification_urls"] == ["https://example.test/publisher"]
    assert merged["external_coauthor_ids"] == ["external-example"]
    assert merged["last_fetched_at_utc"] == "2026-11-02T12:00:00+00:00"
    assert merged["scholar_ids"] == ["oneAAAAJ"]
    person = tables["people"].iloc[0]
    assert person["scholar_observation_count"] == 3
    assert person["curated_observation_count"] == 1
    assert person["publication_observation_count"] == 4


def test_verified_native_id_reconciles_one_exact_doi_free_scholar_row(
    tmp_path,
) -> None:
    paths = write_registry(
        tmp_path / "registry",
        people=[
            {"person_id": "person-one", "display_name": "One", "scholar_id": "one"}
        ],
        memberships=[
            {
                "person_id": "person-one",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/faculty",
                "verified_at": "2026-10-24",
            }
        ],
    )
    registry = load_registry(*paths)
    base = {
        "person_id": "person-one",
        "display_name": "One",
        "faculty": "One",
        "department_ids": ["ece"],
        "memberships": [{"department_id": "ece", "included": True}],
        "title": "A Conference Abstract",
        "authors": "One and Two",
        "year": 2012,
        "venue": "Meeting",
        "citation": "Meeting 2012",
        "citation_count": 0,
        "source_url": "https://example.test/abstract",
        "source_record_json": "{}",
        "source_doi": "",
        "source_native_id": "",
        "source_verified_at": "",
        "source_verification_urls": [],
        "external_coauthors": [],
        "embedding": [0.0, 1.0],
    }
    curated = {
        **base,
        "scholar_id": "",
        "author_pub_id": "curated-native:person-one:egu2012:abstract-1",
        "source_type": "curated_verified",
        "source_native_id": "egu2012:abstract-1",
        "source_verified_at": "2026-10-24",
        "source_verification_urls": ["https://example.test/abstract"],
        "fetched_at_utc": "",
    }
    scholar = {
        **base,
        "scholar_id": "one",
        "author_pub_id": "one:later",
        "source_type": "scholar",
        "fetched_at_utc": "2026-11-01T12:00:00+00:00",
    }
    tables = build_dataset_tables(pandas.DataFrame([curated, scholar]), registry)
    observations = tables["profile_publications"].set_index("author_pub_id")
    assert observations.loc["one:later", "work_match_method"] == (
        "curated_native_title_year"
    )
    assert (
        observations.loc["one:later", "work_id"]
        == observations.loc[curated["author_pub_id"], "work_id"]
    )
    assert len(tables["works"]) == 1
    work = tables["works"].iloc[0]
    assert work["doi"] == ""
    assert work["source_native_ids"] == ["egu2012:abstract-1"]
    assert work["source_types"] == ["curated_verified", "scholar"]
    assert work["observation_count"] == 2

    # A DOI mentioned elsewhere in the provenance does not rekey a
    # source-native work that has no verified source_doi field.
    incidental_doi = {
        **curated,
        "source_record_json": '{"related_url":"https://example.test/10.1000/other"}',
    }
    incidental = build_dataset_tables(
        pandas.DataFrame([incidental_doi, scholar]), registry
    )
    assert len(incidental["works"]) == 1
    assert incidental["works"].iloc[0]["source_native_ids"] == ["egu2012:abstract-1"]

    other_curated = {
        **curated,
        "author_pub_id": "curated-native:person-one:agu2012:other",
        "source_native_id": "agu2012:other",
    }
    ambiguous = build_dataset_tables(
        pandas.DataFrame([curated, other_curated, scholar]), registry
    )
    assert len(ambiguous["works"]) == 3
    scholar_row = (
        ambiguous["profile_publications"].set_index("author_pub_id").loc["one:later"]
    )
    assert scholar_row["work_match_method"] == "normalized_title_year"

    doi_curated = {
        **curated,
        "author_pub_id": "curated-doi:person-one:10.1000/article",
        "source_native_id": "",
        "source_doi": "10.1000/article",
    }
    mixed_identifiers = build_dataset_tables(
        pandas.DataFrame([curated, doi_curated, scholar]), registry
    )
    assert len(mixed_identifiers["works"]) == 3
    scholar_row = (
        mixed_identifiers["profile_publications"]
        .set_index("author_pub_id")
        .loc["one:later"]
    )
    assert scholar_row["work_match_method"] == "normalized_title_year"


def test_scholar_fields_win_when_a_verified_doi_record_matches(tmp_path) -> None:
    paths = write_registry(
        tmp_path / "registry",
        people=[
            {"person_id": "person-one", "display_name": "One", "scholar_id": "one"}
        ],
        memberships=[
            {
                "person_id": "person-one",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/faculty",
                "verified_at": "2026-10-24",
            }
        ],
    )
    registry = load_registry(*paths)
    base = {
        "person_id": "person-one",
        "display_name": "One",
        "faculty": "One",
        "department_ids": ["ece"],
        "memberships": [{"department_id": "ece", "included": True}],
        "source_doi": "10.1000/example",
        "source_record_json": "{}",
        "embedding": [0.0, 1.0],
        "source_verification_urls": [],
        "external_coauthors": [],
    }
    curated = {
        **base,
        "scholar_id": "",
        "author_pub_id": "curated:example",
        "source_type": "curated_verified",
        "title": "A Longer Publisher Title for Graph Models of Research Collaboration",
        "authors": "One, Two, and Three Researchers",
        "year": 1996,
        "venue": "Publisher Journal of Graph Models",
        "citation": "Publisher Journal of Graph Models, 1996",
        "citation_count": 12,
        "source_url": "https://example.test/publisher",
        "source_verified_at": "2026-10-24",
        "source_verification_urls": ["https://example.test/publisher"],
        "fetched_at_utc": "",
    }
    scholar = {
        **base,
        "scholar_id": "one",
        "author_pub_id": "one:example",
        "source_type": "scholar",
        "title": "Graph Models of Research Collaboration",
        "authors": "One and Two",
        "year": 1995,
        "venue": "Graph Journal",
        "citation": "",
        "citation_count": 4,
        "source_url": "https://scholar.google.com/citations?view_op=view_citation",
        "source_verified_at": "",
        "fetched_at_utc": "2026-10-24T12:00:00+00:00",
    }
    tables = build_dataset_tables(pandas.DataFrame([curated, scholar]), registry)
    work = tables["works"].iloc[0]
    assert work["title"] == scholar["title"]
    assert work["authors"] == scholar["authors"]
    assert work["year"] == scholar["year"]
    assert work["venue"] == scholar["venue"]
    assert work["citation_count"] == scholar["citation_count"]
    assert work["citation"] == curated["citation"]  # Scholar field is empty.
    assert set(work["title_variants"]) == {curated["title"], scholar["title"]}
    assert work["source_types"] == ["curated_verified", "scholar"]
    assert work["source_verification_urls"] == ["https://example.test/publisher"]

    # Pandas uses a nullable scalar for absent Scholar years in a full snapshot.
    scholar_without_year = {**scholar, "year": pandas.NA}
    missing_year_tables = build_dataset_tables(
        pandas.DataFrame([curated, scholar_without_year]), registry
    )
    missing_year_work = missing_year_tables["works"].iloc[0]
    assert missing_year_work["title"] == scholar["title"]
    assert missing_year_work["year"] == curated["year"]


def test_ambiguous_curated_title_year_does_not_guess_a_doi(tmp_path) -> None:
    paths = write_registry(
        tmp_path / "registry",
        people=[
            {"person_id": "person-one", "display_name": "One", "scholar_id": "one"}
        ],
        memberships=[
            {
                "person_id": "person-one",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/faculty",
                "verified_at": "2026-10-24",
            }
        ],
    )
    registry = load_registry(*paths)
    base = {
        "person_id": "person-one",
        "display_name": "One",
        "faculty": "One",
        "department_ids": ["ece"],
        "memberships": [{"department_id": "ece", "included": True}],
        "title": "Same Title",
        "authors": "One",
        "year": 1995,
        "venue": "Journal",
        "citation": "",
        "citation_count": 0,
        "source_url": "",
        "source_record_json": "{}",
        "embedding": [0.0, 1.0],
        "source_verified_at": "",
        "source_verification_urls": [],
        "external_coauthors": [],
        "fetched_at_utc": "",
    }
    rows = [
        {
            **base,
            "scholar_id": "",
            "author_pub_id": f"curated:{doi}",
            "source_type": "curated_verified",
            "source_doi": doi,
        }
        for doi in ("10.1000/one", "10.1000/two")
    ]
    rows.append(
        {
            **base,
            "scholar_id": "one",
            "author_pub_id": "one:no-doi",
            "source_type": "scholar",
            "source_doi": "",
            "fetched_at_utc": "2026-10-24T12:00:00+00:00",
        }
    )
    tables = build_dataset_tables(pandas.DataFrame(rows), registry)
    assert len(tables["works"]) == 3
    scholar = (
        tables["profile_publications"].set_index("author_pub_id").loc["one:no-doi"]
    )
    assert scholar["work_match_method"] == "normalized_title_year"


def test_dataset_rejects_conflicting_external_coauthor_evidence(tmp_path) -> None:
    paths = write_registry(
        tmp_path / "registry",
        people=[
            {"person_id": "person-one", "display_name": "One", "scholar_id": "one"}
        ],
        memberships=[
            {
                "person_id": "person-one",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/faculty",
                "verified_at": "2026-10-24",
            }
        ],
    )
    registry = load_registry(*paths)
    row = {
        "scholar_id": "",
        "person_id": "person-one",
        "display_name": "One",
        "faculty": "One",
        "department_ids": ["ece"],
        "memberships": [{"department_id": "ece", "included": True}],
        "author_pub_id": "curated:one",
        "source_type": "curated_verified",
        "source_doi": "10.1000/one",
        "title": "One Work",
        "authors": "One and External",
        "year": 1995,
        "venue": "Journal",
        "citation": "",
        "citation_count": 0,
        "source_url": "",
        "source_record_json": "{}",
        "embedding": [0.0, 1.0],
        "source_verified_at": "2026-10-24",
        "source_verification_urls": ["https://example.test/work"],
        "external_coauthors": [
            {
                "person_id": "external-one",
                "display_name": "External One",
                "source_url": "https://example.test/author",
            }
        ],
        "fetched_at_utc": "",
    }
    with pytest.raises(ValueError, match="must be an object"):
        build_dataset_tables(
            pandas.DataFrame([{**row, "external_coauthors": ["bad"]}]), registry
        )
    different = {
        **row,
        "author_pub_id": "curated:two",
        "external_coauthors": [
            {**row["external_coauthors"][0], "display_name": "Different Name"}
        ],
    }
    with pytest.raises(ValueError, match="Conflicting external"):
        build_dataset_tables(pandas.DataFrame([row, different]), registry)

    # A malformed curator row cannot serve as an exact title/year DOI anchor.
    no_doi = {**row, "source_doi": "", "author_pub_id": "curated:no-doi"}
    result = build_dataset_tables(pandas.DataFrame([no_doi]), registry)
    assert result["works"].iloc[0]["doi"] == ""


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
    assert person["unique_work_count"] == 1
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
        '{"authorships": 0, "people": 2, "profile_publications": 1, "works": 1}'
    )
