import json
from collections import Counter
from pathlib import Path

import pandas
import pytest

from map_of_research.contexts import (
    DEFAULT_CONTEXTS_PATH,
    ReviewedContext,
    load_contexts,
)
from map_of_research.dataset import build_dataset_tables
from map_of_research.exclusions import ReviewedExclusion
from map_of_research.registry import load_registry
from tests.registry_helpers import write_registry


def _entry(**changes: object) -> dict[str, object]:
    return {
        "scholar_id": "oneAAAAJ",
        "author_pub_id": "oneAAAAJ:patent-one",
        "publication_type": "patent",
        "document_id": "US6486123B1",
        "reason": "The source lists inventors and an exact patent publication number.",
        "evidence_url": "https://patents.google.com/patent/US6486123B1/en",
        "verified_at": "2026-10-25",
        **changes,
    }


def _manifest(path: Path, entries: object, version: int = 1) -> Path:
    path.write_text(json.dumps({"schema_version": version, "contexts": entries}))
    return path


def test_checked_in_chien_contexts_are_exact_and_source_backed() -> None:
    contexts = load_contexts(DEFAULT_CONTEXTS_PATH)
    assert len(contexts) == 12
    assert {key[0] for key in contexts} == {"8iLLOQgAAAAJ"}
    assert Counter(item.publication_type for item in contexts.values()) == {
        "patent": 9,
        "edited_book": 1,
        "withdrawn_article": 1,
        "panel_transcript": 1,
    }
    patent_numbers = {
        item.document_id
        for item in contexts.values()
        if item.publication_type == "patent"
    }
    assert len(patent_numbers) == 9
    assert {"US6486123B1", "US20020151468A1"} <= patent_numbers
    assert {key[1] for key, item in contexts.items() if not item.map_eligible} == {
        "8iLLOQgAAAAJ:CaZNVDsoPx4C",
        "8iLLOQgAAAAJ:otzGkya1bYkC",
    }
    editors = contexts[("8iLLOQgAAAAJ", "8iLLOQgAAAAJ:yD5IFk8b50cC")]
    assert editors.display_contributors == (
        "C. Ho; W. A. Eaton; J. P. Collman; Q. H. Gibson; J. S. Leigh, Jr.; "
        "E. Margoliash; K. Moffat; W. R. Scheidt"
    )


def test_context_manifest_validates_identity_type_and_provenance(
    tmp_path: Path,
) -> None:
    path = _manifest(tmp_path / "contexts.json", [_entry(document_id="us6486123b1")])
    assert load_contexts(path)[("oneAAAAJ", "oneAAAAJ:patent-one")].document_id == (
        "US6486123B1"
    )
    bad_entries = [
        ("bad schema", [_entry()], 2, "Unsupported Scholar-context schema"),
        ("not a list", {}, 1, "Scholar contexts must be a list"),
        ("not an object", [1], 1, "Scholar context must be an object"),
        (
            "bad identity",
            [_entry(author_pub_id="other:paper")],
            1,
            "invalid observation identity",
        ),
        ("bad type", [_entry(publication_type="article")], 1, "known publication type"),
        ("missing number", [_entry(document_id="")], 1, "publication number"),
        (
            "nonpatent number",
            [_entry(publication_type="edited_book")],
            1,
            "Only patent contexts",
        ),
        ("missing reason", [_entry(reason="")], 1, "specific reason"),
        (
            "nonbook contributor override",
            [_entry(display_contributors="C. Ho; W. A. Eaton")],
            1,
            "Only edited-book contexts",
        ),
        (
            "nonboolean map eligibility",
            [_entry(map_eligible="false")],
            1,
            "map_eligible must be boolean",
        ),
        (
            "bad URL",
            [_entry(evidence_url="http://example.test")],
            1,
            "HTTPS evidence URL",
        ),
        ("bad date", [_entry(verified_at="tomorrow")], 1, "YYYY-MM-DD"),
        ("duplicate", [_entry(), _entry()], 1, "Duplicate Scholar context"),
    ]
    for _, entries, version, message in bad_entries:
        _manifest(path, entries, version)
        with pytest.raises(ValueError, match=message):
            load_contexts(path)
    _manifest(path, [_entry(publication_type="edited_book", document_id="")])
    assert load_contexts(path)[
        ("oneAAAAJ", "oneAAAAJ:patent-one")
    ].publication_type == ("edited_book")


def test_contexts_preserve_patent_documents_and_withdrawn_raw_observations(
    tmp_path: Path,
) -> None:
    people, memberships, departments = write_registry(
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
                "source_url": "https://example.test/ece",
                "verified_at": "2026-10-25",
            }
        ],
        departments=[
            {
                "department_id": "ece",
                "school_id": "engineering",
                "title": "Electrical and Computer Engineering",
                "directory_url": "https://example.test/ece",
            }
        ],
    )
    registry = load_registry(people, memberships, departments)
    base = {
        "scholar_id": "oneAAAAJ",
        "person_id": "person-one",
        "display_name": "One Person",
        "faculty": "One Person",
        "memberships": [{"department_id": "ece", "role": "faculty", "included": True}],
        "authors": "One Person; Second Person",
        "year": 2002,
        "venue": "Journal",
        "citation": "",
        "citation_count": 0,
        "source_url": "",
        "source_type": "scholar",
        "source_doi": "",
        "external_coauthors": [],
        "source_record_json": '{"original":"Scholar observation"}',
        "fetched_at_utc": "2026-10-25T00:00:00+00:00",
        "embedding": [0.1, 0.2],
    }
    rows = [
        {
            **base,
            "author_pub_id": "oneAAAAJ:p1",
            "title": "Low oxygen affinity mutant hemoglobins",
        },
        {
            **base,
            "author_pub_id": "oneAAAAJ:p2",
            "title": "Low oxygen affinity mutant hemoglobins",
        },
        {
            **base,
            "author_pub_id": "oneAAAAJ:book",
            "title": "Hemoglobin and oxygen binding",
        },
        {
            **base,
            "author_pub_id": "oneAAAAJ:panel",
            "title": "After Dinner Program To Honor Max Perutz",
        },
        {
            **base,
            "author_pub_id": "oneAAAAJ:panel-copy",
            "title": "After Dinner Program To Honor Max Perutz",
        },
        {
            **base,
            "author_pub_id": "oneAAAAJ:panel-allowed",
            "title": "A Different Panel Transcript",
        },
        {
            **base,
            "author_pub_id": "oneAAAAJ:withdrawn",
            "title": "Withdrawn hemoglobin research article",
        },
        {
            **base,
            "author_pub_id": "oneAAAAJ:other-copy",
            "title": "Withdrawn hemoglobin research article",
        },
        {
            **base,
            "author_pub_id": "oneAAAAJ:paper",
            "title": "Published hemoglobin research article",
        },
    ]

    def context(
        kind: str, document_id: str = "", *, map_eligible: bool = True
    ) -> ReviewedContext:
        return ReviewedContext(
            kind,
            document_id,
            "Primary source confirms type",
            "https://example.test/source",
            "2026-10-25",
            map_eligible=map_eligible,
        )

    contexts = {
        ("oneAAAAJ", "oneAAAAJ:p1"): context("patent", "US6486123B1"),
        ("oneAAAAJ", "oneAAAAJ:p2"): context("patent", "US20020151468A1"),
        ("oneAAAAJ", "oneAAAAJ:book"): ReviewedContext(
            "edited_book",
            "",
            "Primary source confirms editors",
            "https://example.test/book",
            "2026-10-25",
            "One Person; Second Person",
        ),
        ("oneAAAAJ", "oneAAAAJ:panel"): context("panel_transcript", map_eligible=False),
        ("oneAAAAJ", "oneAAAAJ:panel-allowed"): context("panel_transcript"),
        ("oneAAAAJ", "oneAAAAJ:withdrawn"): context("withdrawn_article"),
    }
    tables = build_dataset_tables(pandas.DataFrame(rows), registry, contexts=contexts)
    observations = tables["profile_publications"].set_index("author_pub_id")
    works = tables["works"].set_index("work_id")
    assert len(observations) == 9
    assert observations["source_record_json"].eq(base["source_record_json"]).all()
    assert observations.loc["oneAAAAJ:book", "authors"] == base["authors"]
    assert observations.loc["oneAAAAJ:book", "context_display_contributors"] == (
        "One Person; Second Person"
    )
    assert (
        observations.loc["oneAAAAJ:p1", "work_id"]
        != observations.loc["oneAAAAJ:p2", "work_id"]
    )
    assert observations.loc["oneAAAAJ:p1", "work_match_method"] == "patent_document"
    assert observations.loc["oneAAAAJ:p2", "work_match_method"] == "patent_document"
    assert len(works) == 7
    assert (
        observations.loc["oneAAAAJ:other-copy", "work_id"]
        == observations.loc["oneAAAAJ:withdrawn", "work_id"]
    )
    assert not bool(observations.loc["oneAAAAJ:withdrawn", "map_eligible"])
    assert not bool(observations.loc["oneAAAAJ:other-copy", "map_eligible"])
    assert not bool(observations.loc["oneAAAAJ:panel", "map_eligible"])
    assert not bool(observations.loc["oneAAAAJ:panel-copy", "map_eligible"])
    assert bool(observations.loc["oneAAAAJ:panel-allowed", "map_eligible"])
    assert (
        observations.loc["oneAAAAJ:panel", "work_id"]
        == observations.loc["oneAAAAJ:panel-copy", "work_id"]
    )
    assert (
        "reviewed_context_not_mapped"
        in observations.loc["oneAAAAJ:panel-copy", "map_exclusion_reasons"]
    )
    assert (
        "reviewed_withdrawn_article"
        in observations.loc["oneAAAAJ:other-copy", "map_exclusion_reasons"]
    )
    assert tables["authorships"]["contributor_role"].tolist() == [
        "inventor",
        "inventor",
        "editor",
        "participant",
        "author",
    ]
    assert len(tables["authorships"]) == 5
    assert sum(bool(value) for value in works["collaboration_eligible"]) == 1
    assert (
        works.loc[observations.loc["oneAAAAJ:book", "work_id"], "authors"]
        == (base["authors"])
    )
    assert (
        works.loc[observations.loc["oneAAAAJ:book", "work_id"], "display_contributors"]
        == "One Person; Second Person"
    )
    assert not bool(
        works.loc[observations.loc["oneAAAAJ:withdrawn", "work_id"], "map_eligible"]
    )
    assert not bool(
        works.loc[observations.loc["oneAAAAJ:panel", "work_id"], "map_eligible"]
    )
    assert (
        works.loc[observations.loc["oneAAAAJ:p1", "work_id"], "source_document_id"]
        == "US6486123B1"
    )

    # A reviewed non-article type still blocks coauthor links when that exact
    # observation is excluded and a second, unreviewed copy remains mapped.
    copy = {
        **base,
        "author_pub_id": "oneAAAAJ:book-copy",
        "title": "Hemoglobin and oxygen binding",
        "source_type": "curated_verified",
        "source_doi": "10.1234/hemoglobin",
        "external_coauthors": [
            {
                "person_id": "external-editor",
                "display_name": "Second Person",
                "source_url": "https://example.test/editor",
            }
        ],
    }
    reviewed_exclusion = ReviewedExclusion(
        "corrupted_duplicate",
        "Duplicate record",
        "https://example.test/review",
        "2026-10-25",
    )
    with_copy = build_dataset_tables(
        pandas.DataFrame([*rows, copy]),
        registry,
        exclusions={("oneAAAAJ", "oneAAAAJ:book"): reviewed_exclusion},
        contexts=contexts,
    )
    copy_observations = with_copy["profile_publications"].set_index("author_pub_id")
    book_work = (
        with_copy["works"]
        .set_index("work_id")
        .loc[copy_observations.loc["oneAAAAJ:book-copy", "work_id"]]
    )
    assert bool(book_work["map_eligible"])
    assert book_work["publication_type"] == "edited_book"
    assert not bool(book_work["collaboration_eligible"])
    assert book_work["external_coauthor_ids"] == []
    assert book_work["external_coauthors"] == []


def test_conflicting_reviewed_types_for_one_work_fail_closed(tmp_path: Path) -> None:
    people, memberships, departments = write_registry(
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
                "source_url": "https://example.test/ece",
                "verified_at": "2026-10-25",
            }
        ],
        departments=[
            {
                "department_id": "ece",
                "school_id": "engineering",
                "title": "ECE",
                "directory_url": "https://example.test/ece",
            }
        ],
    )
    base = {
        "scholar_id": "oneAAAAJ",
        "person_id": "person-one",
        "display_name": "One Person",
        "faculty": "One Person",
        "memberships": [{"department_id": "ece", "included": True}],
        "title": "The Same Published Volume",
        "authors": "One Person",
        "year": 1982,
        "venue": "Book",
        "citation": "",
        "citation_count": 0,
        "source_url": "",
        "source_record_json": "{}",
        "fetched_at_utc": "2026-10-25T00:00:00+00:00",
        "embedding": [0.1, 0.2],
    }
    contexts = {
        ("oneAAAAJ", "oneAAAAJ:book"): ReviewedContext(
            "edited_book", "", "Source", "https://example.test/book", "2026-10-25"
        ),
        ("oneAAAAJ", "oneAAAAJ:panel"): ReviewedContext(
            "panel_transcript", "", "Source", "https://example.test/panel", "2026-10-25"
        ),
    }
    rows = [
        {**base, "author_pub_id": "oneAAAAJ:book"},
        {**base, "author_pub_id": "oneAAAAJ:panel"},
    ]
    with pytest.raises(ValueError, match="Conflicting reviewed publication types"):
        build_dataset_tables(
            pandas.DataFrame(rows),
            load_registry(people, memberships, departments),
            contexts=contexts,
        )
