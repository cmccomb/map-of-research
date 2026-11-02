import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas
import pytest

from map_of_research.contexts import (
    DEFAULT_CONTEXTS_PATH,
    ReviewedContext,
    load_contexts,
)
from map_of_research.dataset import build_dataset_tables
from map_of_research.doi_aliases import ReviewedDoiAlias
from map_of_research.exclusions import ReviewedExclusion
from map_of_research.registry import load_registry
from tests.registry_helpers import write_registry

FROZEN_CHEM_CONTEXTS = Path("tests/fixtures/chem_approved_contexts.json")


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


def _authored_entry(**changes: object) -> dict[str, object]:
    return {
        "scholar_id": "oneAAAAJ",
        "author_pub_id": "oneAAAAJ:chapter",
        "publication_type": "book_chapter",
        "document_id": "",
        "reason": "The publisher names both chapter authors and its DOI.",
        "evidence_url": "https://publisher.example.test/book/chapter",
        "verified_at": "2026-10-30",
        "expected_title": "An Authored Book Chapter",
        "expected_year": 2026,
        "verified_doi": "10.1234/chapter.2026",
        "collaboration_eligible": False,
        **changes,
    }


def _scholar_profile_entry(**changes: object) -> dict[str, object]:
    return {
        "scholar_id": "oneAAAAJ",
        "author_pub_id": "oneAAAAJ:scholar-first",
        "publication_type": "publication",
        "document_id": "",
        "reason": (
            "The exact owner-profile citation is substantive; "
            "item type is unclassified."
        ),
        "evidence_url": (
            "https://scholar.google.com/citations?user=oneAAAAJ"
            "&citation_for_view=oneAAAAJ:scholar-first"
        ),
        "verified_at": "2026-10-31",
        "expected_title": "A Substantive Scholarly Work",
        "expected_year": 2025,
        "collaboration_eligible": False,
        **changes,
    }


def test_checked_in_chien_contexts_are_exact_and_source_backed() -> None:
    contexts = {
        key: item
        for key, item in load_contexts(DEFAULT_CONTEXTS_PATH).items()
        if key[0] in {"8iLLOQgAAAAJ", "ILambBMAAAAJ"}
    }
    assert contexts == load_contexts(FROZEN_CHEM_CONTEXTS)
    assert len(contexts) == 14
    assert {key[0] for key in contexts} == {"8iLLOQgAAAAJ", "ILambBMAAAAJ"}
    assert Counter(item.publication_type for item in contexts.values()) == {
        "patent": 9,
        "edited_book": 1,
        "withdrawn_article": 1,
        "panel_transcript": 1,
        "conference_presentation": 2,
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
    _manifest(
        path,
        [_entry(publication_type="conference_presentation", document_id="")],
    )
    assert (
        load_contexts(path)[("oneAAAAJ", "oneAAAAJ:patent-one")].publication_type
        == "conference_presentation"
    )


def test_authored_type_doi_and_collaboration_need_exact_review(tmp_path: Path) -> None:
    path = _manifest(tmp_path / "contexts.json", [_authored_entry()])
    context = load_contexts(path)[("oneAAAAJ", "oneAAAAJ:chapter")]
    assert context.publication_type == "book_chapter"
    assert context.verified_doi == "10.1234/chapter.2026"
    assert context.collaboration_eligible is False
    for kind in ("conference_paper", "journal_article", "book_chapter", "preprint"):
        _manifest(path, [_authored_entry(publication_type=kind)])
        context = load_contexts(path)[("oneAAAAJ", "oneAAAAJ:chapter")]
        assert context.publication_type == kind
    bad = [
        ({"expected_title": ""}, "exact expected title and year"),
        ({"expected_title": 123}, "exact expected title and year"),
        ({"expected_year": True}, "exact expected title and year"),
        ({"expected_year": None}, "exact expected title and year"),
        ({"verified_doi": "not-a-doi"}, "valid verified DOI"),
        ({"verified_doi": None}, "valid verified DOI"),
        ({"collaboration_eligible": "true"}, "explicit collaboration eligibility"),
        (
            {"collaboration_eligible": True, "map_eligible": False},
            "cannot permit collaboration",
        ),
    ]
    for changes, message in bad:
        _manifest(path, [_authored_entry(**changes)])
        with pytest.raises(ValueError, match=message):
            load_contexts(path)
    for missing, message in (
        ("verified_doi", "valid verified DOI"),
        ("collaboration_eligible", "explicit collaboration eligibility"),
    ):
        entry = _authored_entry()
        del entry[missing]
        _manifest(path, [entry])
        with pytest.raises(ValueError, match=message):
            load_contexts(path)
    _manifest(path, [_entry(verified_doi="10.1234/not-a-patent")])
    with pytest.raises(ValueError, match="Only authored Scholar contexts"):
        load_contexts(path)


def test_scholar_profile_generic_context_requires_exact_no_edge_decision(
    tmp_path: Path,
) -> None:
    path = _manifest(tmp_path / "contexts.json", [_scholar_profile_entry()])
    context = load_contexts(path)[("oneAAAAJ", "oneAAAAJ:scholar-first")]
    assert context.publication_type == "publication"
    assert context.collaboration_eligible is False
    assert context.verified_doi == ""
    for changes, error in (
        ({"expected_title": ""}, "exact title and year"),
        ({"expected_year": None}, "exact title and year"),
        ({"verified_doi": ""}, "cannot claim a verified DOI"),
        ({"collaboration_eligible": True}, "must hold collaboration"),
        ({"map_eligible": False}, "remain source eligible"),
    ):
        _manifest(path, [_scholar_profile_entry(**changes)])
        with pytest.raises(ValueError, match=error):
            load_contexts(path)


def test_scholar_profile_generic_context_holds_only_reviewed_new_row(
    tmp_path: Path,
) -> None:
    people, memberships, departments = write_registry(
        tmp_path / "registry",
        people=[
            {"person_id": "person-one", "display_name": "One", "scholar_id": "oneAAAAJ"}
        ],
        memberships=[
            {
                "person_id": "person-one",
                "department_id": "s3d",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/s3d",
                "verified_at": "2026-10-31",
            }
        ],
        departments=[
            {
                "department_id": "s3d",
                "school_id": "scs",
                "title": "Software and Societal Systems",
                "directory_url": "https://example.test/s3d",
            }
        ],
    )
    registry = load_registry(people, memberships, departments)
    source_url = _scholar_profile_entry()["evidence_url"]
    captured = {
        "author_pub_id": "oneAAAAJ:scholar-first",
        "title": "A Substantive Scholarly Work",
        "pub_year": "2025",
        "venue": "An Unclassified Research Venue",
        "pub_url": source_url,
    }
    row = {
        "scholar_id": "oneAAAAJ",
        "person_id": "person-one",
        "display_name": "One",
        "faculty": "One",
        "memberships": [{"department_id": "s3d", "role": "faculty", "included": True}],
        "author_pub_id": "oneAAAAJ:scholar-first",
        "title": "A Substantive Scholarly Work",
        "authors": "One; Second Person",
        "year": 2025,
        "venue": "An Unclassified Research Venue",
        "citation": "",
        "citation_count": 0,
        "source_url": source_url,
        "source_type": "scholar",
        "source_doi": "",
        "source_record_json": json.dumps(captured),
        "fetched_at_utc": "2026-10-31T00:00:00+00:00",
        "embedding": [0.1, 0.2],
    }
    path = _manifest(tmp_path / "contexts.json", [_scholar_profile_entry()])
    contexts = load_contexts(path)
    old = {
        **row,
        "author_pub_id": "oneAAAAJ:old",
        "title": "An Older Reviewed Work",
        "year": 2024,
        "source_url": "https://scholar.google.com/citations?user=oneAAAAJ&citation_for_view=oneAAAAJ:old",
    }
    tables = build_dataset_tables(
        pandas.DataFrame([old, row]), registry, contexts=contexts
    )
    profiles = tables["profile_publications"].set_index("author_pub_id")
    assert bool(profiles.loc["oneAAAAJ:old", "collaboration_eligible"])
    assert not bool(profiles.loc["oneAAAAJ:scholar-first", "collaboration_eligible"])
    assert not bool(
        tables["works"]
        .set_index("work_id")
        .loc[
            profiles.loc["oneAAAAJ:scholar-first", "work_id"],
            "collaboration_eligible",
        ]
    )
    assert profiles.loc["oneAAAAJ:scholar-first", "publication_type"] == "publication"
    assert bool(profiles.loc["oneAAAAJ:scholar-first", "map_eligible"])
    assert (
        profiles.loc["oneAAAAJ:scholar-first", "source_record_json"]
        == row["source_record_json"]
    )
    generic_identity = profiles.loc["oneAAAAJ:scholar-first", "work_id"]
    for doi_looking_venue in (
        "Data Science Foundation. https://doi. org/10.5281/zenodo",
        "Journal of Examples. https://doi.org/10.1234/full-record",
    ):
        changed = {
            **row,
            "venue": doi_looking_venue,
            "source_record_json": json.dumps({**captured, "venue": doi_looking_venue}),
        }
        held = build_dataset_tables(
            pandas.DataFrame([changed]), registry, contexts=contexts
        )
        observation = held["profile_publications"].iloc[0]
        assert observation["doi"] == ""
        assert observation["work_match_method"] == "profile_publication"
        assert observation["work_id"] == generic_identity
        assert not bool(observation["collaboration_eligible"])
        work = held["works"].iloc[0]
        assert work["doi"] == ""
        assert not bool(work["collaboration_eligible"])
    for regional_host in ("scholar.google.ca", "scholar.google.co.jp"):
        regional_url = str(source_url).replace("scholar.google.com", regional_host)
        _manifest(
            path,
            [_scholar_profile_entry(evidence_url=regional_url)],
        )
        regional_tables = build_dataset_tables(
            pandas.DataFrame(
                [
                    {
                        **row,
                        "source_url": regional_url,
                        "source_record_json": json.dumps(
                            {**captured, "pub_url": regional_url}
                        ),
                    }
                ]
            ),
            registry,
            contexts=load_contexts(path),
        )
        assert not bool(
            regional_tables["profile_publications"].iloc[0]["collaboration_eligible"]
        )
    japan_url = str(source_url).replace("scholar.google.com", "scholar.google.co.jp")
    for bad_url in (
        str(source_url) + "&user=",
        str(source_url) + "&citation_for_view=",
        japan_url.replace("/citations?", ".attacker.example/citations?"),
        japan_url.replace("/citations?", ":443/citations?"),
        japan_url.replace("https://", "https://name@", 1),
        japan_url.replace("/citations?", "/not-citations?"),
        japan_url.replace("user=oneAAAAJ", "user=otherAAAAJ"),
        japan_url.replace(
            "citation_for_view=oneAAAAJ:scholar-first",
            "citation_for_view=oneAAAAJ:wrong",
        ),
    ):
        with pytest.raises(ValueError, match="exact source citation"):
            build_dataset_tables(
                pandas.DataFrame(
                    [
                        {
                            **row,
                            "source_url": bad_url,
                            "source_record_json": json.dumps(
                                {**captured, "pub_url": bad_url}
                            ),
                        }
                    ]
                ),
                registry,
                contexts=load_contexts(
                    _manifest(
                        path,
                        [_scholar_profile_entry(evidence_url=bad_url)],
                    )
                ),
            )
    contexts = load_contexts(_manifest(path, [_scholar_profile_entry()]))
    mistyped_owner_context = load_contexts(
        _manifest(
            path,
            [
                _scholar_profile_entry(
                    scholar_id="wrongAAAAJ",
                    author_pub_id="wrongAAAAJ:scholar-first",
                    evidence_url="https://scholar.google.com/citations?user=wrongAAAAJ&citation_for_view=wrongAAAAJ:scholar-first",
                )
            ],
        )
    )
    with pytest.raises(ValueError, match="ID must occur exactly once"):
        build_dataset_tables(
            pandas.DataFrame([row]), registry, contexts=mistyped_owner_context
        )
    for bad_row in (
        {**row, "title": "Another Work"},
        {**row, "year": 2024},
        {**row, "year": 2025.9},
        {**row, "source_record_json": "{invalid"},
        {
            **row,
            "source_record_json": json.dumps(
                {**captured, "title": "Different Raw Title"}
            ),
        },
        {
            **row,
            "source_record_json": json.dumps({**captured, "pub_year": "2025.9"}),
        },
        {
            **row,
            "source_record_json": json.dumps(
                {**captured, "venue": "Different Raw Venue"}
            ),
        },
        {
            **row,
            "source_url": "https://scholar.google.com/citations?user=oneAAAAJ&citation_for_view=oneAAAAJ:other",
        },
        {
            **row,
            "source_url": "https://scholar.google.com.evil.test/citations?user=oneAAAAJ&citation_for_view=oneAAAAJ:scholar-first",
        },
        {**row, "citation": "10.1234/one and 10.1234/two"},
        {**row, "citation": "10.1234%2Fone and 10.1234%2Ftwo"},
        {**row, "citation": "10.1000/alpha;10.2000/beta"},
        {**row, "citation": "10.1000/alpha%3B10.2000/beta"},
    ):
        with pytest.raises(
            ValueError,
            match="exact source citation|exact captured fields|ambiguous source DOI",
        ):
            build_dataset_tables(
                pandas.DataFrame([bad_row]), registry, contexts=contexts
            )
    wrong_context = _scholar_profile_entry(
        author_pub_id="oneAAAAJ:missing",
        evidence_url="https://scholar.google.com/citations?user=oneAAAAJ&citation_for_view=oneAAAAJ:missing",
    )
    with pytest.raises(ValueError, match="ID must occur exactly once"):
        build_dataset_tables(
            pandas.DataFrame([row]),
            registry,
            contexts=load_contexts(_manifest(path, [wrong_context])),
        )


@pytest.mark.parametrize(
    "kind", ["conference_paper", "journal_article", "book_chapter", "preprint"]
)
def test_authored_type_doi_and_eligibility_are_explicit(
    tmp_path: Path, kind: str
) -> None:
    people, memberships, departments = write_registry(
        tmp_path / "registry",
        people=[
            {"person_id": "person-one", "display_name": "One", "scholar_id": "oneAAAAJ"}
        ],
        memberships=[
            {
                "person_id": "person-one",
                "department_id": "s3d",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/s3d",
                "verified_at": "2026-10-30",
            }
        ],
        departments=[
            {
                "department_id": "s3d",
                "school_id": "scs",
                "title": "Software and Societal Systems",
                "directory_url": "https://example.test/s3d",
            }
        ],
    )
    registry = load_registry(people, memberships, departments)
    source_record = '{"title":"An Authored Book Chapter","pub_year":"2026"}'
    row = {
        "scholar_id": "oneAAAAJ",
        "person_id": "person-one",
        "display_name": "One",
        "faculty": "One",
        "memberships": [{"department_id": "s3d", "role": "faculty", "included": True}],
        "author_pub_id": "oneAAAAJ:chapter",
        "title": "An Authored Book Chapter",
        "authors": "One; Second Person",
        "year": 2026,
        "venue": "A Reviewed Book",
        "citation": "",
        "citation_count": 0,
        "source_url": "https://scholar.google.com/citations?user=oneAAAAJ",
        "source_type": "scholar",
        "source_doi": "",
        "source_record_json": source_record,
        "fetched_at_utc": "2026-10-30T00:00:00+00:00",
        "embedding": [0.1, 0.2],
    }
    path = _manifest(
        tmp_path / "contexts.json", [_authored_entry(publication_type=kind)]
    )
    reviewed = load_contexts(path)
    tables = build_dataset_tables(pandas.DataFrame([row]), registry, contexts=reviewed)
    observation = tables["profile_publications"].iloc[0]
    work = tables["works"].iloc[0]
    assert observation["source_record_json"] == source_record
    assert observation["source_doi"] == ""
    assert observation["doi"] == "10.1234/chapter.2026"
    assert observation["publication_type"] == kind
    assert observation["work_match_method"] == "reviewed_context_doi"
    assert observation["source_verified_at"] == "2026-10-30"
    assert observation["source_verification_urls"] == [
        "https://publisher.example.test/book/chapter"
    ]
    assert not bool(observation["collaboration_eligible"])
    assert (
        work["work_id"]
        == "work-" + hashlib.sha256(b"doi:10.1234/chapter.2026").hexdigest()
    )
    assert work["publication_type"] == kind
    assert work["doi"] == "10.1234/chapter.2026"
    assert not bool(work["collaboration_eligible"])
    assert tables["authorships"].iloc[0]["contributor_role"] == "author"

    _manifest(
        path,
        [_authored_entry(publication_type=kind, collaboration_eligible=True)],
    )
    allowed = build_dataset_tables(
        pandas.DataFrame([row]), registry, contexts=load_contexts(path)
    )
    assert bool(allowed["profile_publications"].iloc[0]["collaboration_eligible"])
    assert bool(allowed["works"].iloc[0]["collaboration_eligible"])

    for bad_row in (
        {**row, "title": "A Different Chapter"},
        {**row, "year": 2025},
        {**row, "source_doi": "10.1234/a-different-doi"},
    ):
        with pytest.raises(
            ValueError, match="citation title or year changed|DOI conflicts"
        ):
            build_dataset_tables(
                pandas.DataFrame([bad_row]), registry, contexts=load_contexts(path)
            )

    matching_source_doi = {**row, "source_doi": "10.1234/chapter.2026"}
    for field, conflicting_value in (
        ("source_url", "https://publisher.example.test/10.1234/other-doi"),
        ("citation", "A citation with 10.1234/other-doi"),
        ("source_record_json", '{"other_doi":"10.1234/other-doi"}'),
    ):
        conflicting = {**matching_source_doi, field: conflicting_value}
        with pytest.raises(ValueError, match="DOI conflicts with source"):
            build_dataset_tables(
                pandas.DataFrame([conflicting]),
                registry,
                contexts=load_contexts(path),
            )
    with pytest.raises(ValueError, match="DOI conflicts with source"):
        build_dataset_tables(
            pandas.DataFrame(
                [
                    {
                        **row,
                        "source_record_json": (
                            '{"citations":"10.1234/chapter.2026, 10.1234/other-doi"}'
                        ),
                    }
                ]
            ),
            registry,
            contexts=load_contexts(path),
        )
    repeated_verified_doi = {
        **matching_source_doi,
        "citation": "The verified DOI is 10.1234/chapter.2026.",
    }
    assert (
        build_dataset_tables(
            pandas.DataFrame([repeated_verified_doi]),
            registry,
            contexts=load_contexts(path),
        )["profile_publications"].iloc[0]["doi"]
        == "10.1234/chapter.2026"
    )
    fully_encoded_conflict = "".join(
        f"%{ord(character):02X}" for character in "10.1234/other-doi"
    )
    for field, conflicting_value in (
        ("source_url", "https://doi.org/10.1234%2Fother-doi"),
        ("source_url", "https://doi.org/10%2E1234%2Fother-doi"),
        ("source_url", f"https://doi.org/{fully_encoded_conflict}"),
        ("source_url", "https://publisher.example.test/?doi=10.1234%2Fother-doi"),
        ("source_url", "https://doi.org/10.1234%252Fother-doi"),
        ("citation", "Cites 10.1234&#47;other-doi"),
        ("citation", "Cites 10.1234&sol;other-doi"),
    ):
        with pytest.raises(ValueError, match="DOI conflicts with source"):
            build_dataset_tables(
                pandas.DataFrame([{**matching_source_doi, field: conflicting_value}]),
                registry,
                contexts=load_contexts(path),
            )
    encoded_same_url = "https://doi.org/10.1234%2FCHAPTER.2026."
    encoded_same = build_dataset_tables(
        pandas.DataFrame([{**row, "source_url": encoded_same_url}]),
        registry,
        contexts=load_contexts(path),
    )["profile_publications"].iloc[0]
    assert encoded_same["doi"] == "10.1234/chapter.2026"
    assert encoded_same["source_url"] == encoded_same_url
    deeply_encoded = "10.1234%2Fchapter.2026"
    for _ in range(8):
        deeply_encoded = deeply_encoded.replace("%", "%25")
    with pytest.raises(ValueError, match="DOI encoding exceeds review limit"):
        build_dataset_tables(
            pandas.DataFrame(
                [{**row, "source_url": f"https://doi.org/{deeply_encoded}"}]
            ),
            registry,
            contexts=load_contexts(path),
        )

    conflict = ReviewedDoiAlias(
        doi="10.1234/another-version",
        observed_title=row["title"],
        observed_year=row["year"],
        relationship="version_of_record",
        evidence_urls=(
            "https://publisher.example.test/version",
            "https://doi.org/10.1234/another-version",
        ),
        verified_at="2026-10-30",
    )
    with pytest.raises(ValueError, match="DOI conflicts with reviewed alias"):
        build_dataset_tables(
            pandas.DataFrame([row]),
            registry,
            contexts=load_contexts(path),
            doi_aliases={("oneAAAAJ", "oneAAAAJ:chapter"): conflict},
        )

    _manifest(path, [_authored_entry(publication_type=kind, verified_doi="")])
    without_doi = build_dataset_tables(
        pandas.DataFrame([row]), registry, contexts=load_contexts(path)
    )
    assert without_doi["profile_publications"].iloc[0]["doi"] == ""
    assert without_doi["profile_publications"].iloc[0]["work_match_method"] == (
        "normalized_title_year"
    )
    assert without_doi["works"].iloc[0]["publication_type"] == kind
    for field, observed_value in (
        ("source_doi", "10.1234/unreviewed"),
        ("source_url", "https://publisher.example.test/10.1234/unreviewed"),
        ("source_url", "https://doi.org/10.1234%2Funreviewed"),
        ("source_url", f"https://doi.org/{fully_encoded_conflict}"),
        ("citation", "Unreviewed DOI 10.1234/unreviewed"),
        ("citation", "Unreviewed DOI 10.1234&sol;unreviewed"),
        ("source_record_json", '{"doi":"10.1234/unreviewed"}'),
    ):
        with pytest.raises(ValueError, match="DOI conflicts with source"):
            build_dataset_tables(
                pandas.DataFrame([{**row, field: observed_value}]),
                registry,
                contexts=load_contexts(path),
            )

    _manifest(
        path,
        [
            _authored_entry(
                publication_type=kind,
                author_pub_id="oneAAAAJ:unseen-chapter",
            )
        ],
    )
    with pytest.raises(ValueError, match="ID must occur exactly once"):
        build_dataset_tables(
            pandas.DataFrame([row]), registry, contexts=load_contexts(path)
        )
    with pytest.raises(ValueError, match="ID must occur exactly once"):
        build_dataset_tables(
            pandas.DataFrame([row, row]),
            registry,
            contexts={
                ("oneAAAAJ", "oneAAAAJ:chapter"): ReviewedContext(
                    kind,
                    "",
                    "The publisher confirms this authored work and DOI.",
                    "https://publisher.example.test/book/chapter",
                    "2026-10-30",
                    expected_title=row["title"],
                    expected_year=row["year"],
                    verified_doi="10.1234/chapter.2026",
                    collaboration_eligible=False,
                )
            },
        )

    _manifest(path, [_authored_entry(publication_type=kind)])
    valid_context = load_contexts(path)[("oneAAAAJ", "oneAAAAJ:chapter")]
    bad_contexts = (
        (replace(valid_context, collaboration_eligible=None), "collaboration status"),
        (
            replace(valid_context, map_eligible=False, collaboration_eligible=True),
            "collaboration status",
        ),
        (replace(valid_context, verified_doi="invalid-doi"), "invalid verified DOI"),
        (replace(valid_context, verified_doi=None), "invalid verified DOI"),
        (
            replace(valid_context, verified_doi="10.1234/UPPERCASE"),
            "invalid verified DOI",
        ),
        (replace(valid_context, reason=""), "primary-source evidence"),
        (
            replace(valid_context, evidence_url="http://example.test/not-primary"),
            "primary-source evidence",
        ),
    )
    for bad_context, message in bad_contexts:
        with pytest.raises(ValueError, match=message):
            build_dataset_tables(
                pandas.DataFrame([row]),
                registry,
                contexts={("oneAAAAJ", "oneAAAAJ:chapter"): bad_context},
            )


def test_source_backed_conference_presentation_keeps_authored_link(
    tmp_path: Path,
) -> None:
    people, memberships, departments = write_registry(
        tmp_path / "registry",
        people=[
            {"person_id": "person-one", "display_name": "One", "scholar_id": "oneAAAAJ"}
        ],
        memberships=[
            {
                "person_id": "person-one",
                "department_id": "chem",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/chem",
                "verified_at": "2026-10-25",
            }
        ],
        departments=[
            {
                "department_id": "chem",
                "school_id": "engineering",
                "title": "Chemistry",
                "directory_url": "https://example.test/chem",
            }
        ],
    )
    registry = load_registry(people, memberships, departments)
    row = {
        "scholar_id": "oneAAAAJ",
        "person_id": "person-one",
        "display_name": "One",
        "faculty": "One",
        "memberships": [{"department_id": "chem", "role": "faculty", "included": True}],
        "author_pub_id": "oneAAAAJ:meeting",
        "title": "An AGU presentation",
        "authors": "One; Second Person",
        "year": 2009,
        "venue": "AGU Fall Meeting",
        "citation": "",
        "citation_count": 0,
        "source_url": "https://scholar.google.com/citations?user=oneAAAAJ",
        "source_type": "scholar",
        "source_doi": "",
        "external_coauthors": [],
        "source_record_json": '{"title":"An AGU presentation"}',
        "fetched_at_utc": "2026-10-25T00:00:00+00:00",
        "embedding": [0.1, 0.2],
    }
    context = ReviewedContext(
        "conference_presentation",
        "",
        "Organizer program identifies the oral presentation and authors",
        "https://example.test/agu-program",
        "2026-10-25",
    )
    tables = build_dataset_tables(
        pandas.DataFrame([row]),
        registry,
        contexts={("oneAAAAJ", "oneAAAAJ:meeting"): context},
    )
    observation = tables["profile_publications"].iloc[0]
    work = tables["works"].iloc[0]
    assert observation["publication_type"] == "conference_presentation"
    assert bool(observation["map_eligible"])
    assert bool(observation["collaboration_eligible"])
    assert work["publication_type"] == "conference_presentation"
    assert bool(work["collaboration_eligible"])
    assert len(tables["authorships"]) == 1


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
