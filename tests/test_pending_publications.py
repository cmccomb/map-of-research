"""Pending rows are selected public works with no reviewed map coordinates."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pandas
import pytest

from map_of_research.pending_publications import build_pending_publications
from map_of_research.public_subsets import ids_sha256

ROW_ID = "ScholarOne:paper1"
SOURCE_URL = (
    "https://scholar.google.com/citations?user=ScholarOne"
    "&citation_for_view=ScholarOne:paper1"
)


class RegistryStub:
    def __init__(self) -> None:
        self.memberships = [
            SimpleNamespace(person_id="person-one", department_id="ece", included=True)
        ]
        self.home = SimpleNamespace(
            home_type="department",
            home_id="ece",
            source_url="https://cmu.edu/ece/person-one",
            verified_at="2026-10-30",
            basis="reviewed_primary",
        )

    def primary_affiliation(self, person_id: str) -> SimpleNamespace:
        assert person_id == "person-one"
        return self.home


def inputs():
    works = pandas.DataFrame(
        [
            {
                "work_id": "work-one",
                "title": "A reviewed paper",
                "year": 2026.0,
                "map_eligible": False,
                "map_exclusion_reasons": ["pending_placement_review"],
                "collaboration_eligible": True,
                "memberships": [{"person_id": "person-one", "department_id": "ece"}],
                "author_pub_ids": [ROW_ID],
                "source_urls": [SOURCE_URL],
                "x": None,
                "y": None,
                "tsne_x": None,
                "tsne_y": None,
            }
        ]
    )
    profiles = pandas.DataFrame(
        [
            {
                "work_id": "work-one",
                "person_id": "person-one",
                "scholar_id": "ScholarOne",
                "author_pub_id": ROW_ID,
                "source_url": SOURCE_URL,
                "collaboration_eligible": True,
            }
        ]
    )
    receipt = [
        {
            "person_id": "person-one",
            "scholar_id": "ScholarOne",
            "cache_scope": "reviewed_public_subset",
            "selected_author_pub_ids": [ROW_ID],
            "selected_ids_sha256": ids_sha256({ROW_ID}),
        }
    ]
    return works, profiles, receipt, RegistryStub(), ["work-one"]


def test_pending_publication_is_selected_and_nonspatial() -> None:
    works, profiles, receipt, registry, held = inputs()
    assert build_pending_publications(works, profiles, receipt, registry, held) == [
        {
            "work_id": "work-one",
            "title": "A reviewed paper",
            "year": 2026,
            "faculty_ids": ["person-one"],
            "source_url": SOURCE_URL,
            "release_decision": "selected",
            "map_hold_reason": "pending_topic_review",
            "collaboration_eligible": True,
        }
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("x", 0.1, "unreviewed coordinates"),
        ("map_eligible", True, "exclusive placement hold"),
        ("map_exclusion_reasons", ["some_other_hold"], "exclusive placement hold"),
        ("author_pub_ids", ["unselected:row"], "exact public source rows"),
        ("source_urls", ["http://example.com"], "not selected HTTPS"),
        ("title", "", "lacks title"),
        ("year", 2026.5, "invalid year"),
    ],
)
def test_pending_publication_rejects_invalid_work(field, value, message) -> None:
    works, profiles, receipt, registry, held = inputs()
    works.at[0, field] = value
    with pytest.raises(ValueError, match=message):
        build_pending_publications(works, profiles, receipt, registry, held)


def test_pending_publication_rejects_unselected_profile() -> None:
    works, profiles, receipt, registry, held = inputs()
    receipt[0]["selected_author_pub_ids"] = ["ScholarOne:other"]
    receipt[0]["selected_ids_sha256"] = ids_sha256({"ScholarOne:other"})
    with pytest.raises(ValueError, match="unselected source row"):
        build_pending_publications(works, profiles, receipt, registry, held)


def test_pending_publication_rejects_unevidenced_home() -> None:
    works, profiles, receipt, registry, held = inputs()
    registry.home = deepcopy(registry.home)
    registry.home.source_url = ""
    with pytest.raises(ValueError, match="evidenced typed home"):
        build_pending_publications(works, profiles, receipt, registry, held)


def test_pending_publication_rejects_excluded_membership() -> None:
    works, profiles, receipt, registry, held = inputs()
    registry.memberships[0].included = False
    with pytest.raises(ValueError, match="excluded faculty membership"):
        build_pending_publications(works, profiles, receipt, registry, held)


def test_pending_publication_rejects_duplicate_or_missing_review_ids() -> None:
    works, profiles, receipt, registry, _ = inputs()
    with pytest.raises(ValueError, match="missing or duplicated"):
        build_pending_publications(
            works, profiles, receipt, registry, ["work-one", "work-one"]
        )
    with pytest.raises(ValueError, match="missing or duplicate works"):
        build_pending_publications(works, profiles, receipt, registry, ["work-missing"])


def test_no_pending_rows_need_no_subset_receipt() -> None:
    works, profiles, _, registry, _ = inputs()
    assert build_pending_publications(works, profiles, [], registry, []) == []


@pytest.mark.parametrize("table", ["works", "profiles"])
def test_pending_publication_requires_public_columns(table) -> None:
    works, profiles, receipt, registry, held = inputs()
    if table == "works":
        works = works.drop(columns=["title"])
    else:
        profiles = profiles.drop(columns=["person_id"])
    with pytest.raises(ValueError, match="required public table columns"):
        build_pending_publications(works, profiles, receipt, registry, held)


def test_pending_publication_rejects_duplicate_work_table_id() -> None:
    works, profiles, receipt, registry, held = inputs()
    works = pandas.concat([works, works], ignore_index=True)
    with pytest.raises(ValueError, match="missing or duplicate works"):
        build_pending_publications(works, profiles, receipt, registry, held)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r[0].update(cache_scope="other"), "reviewed selected subsets"),
        (lambda r: r[0].update(selected_author_pub_ids="bad"), "invalid selected IDs"),
        (
            lambda r: r[0].update(selected_author_pub_ids=[ROW_ID, ROW_ID]),
            "invalid selected IDs",
        ),
        (
            lambda r: r[0].update(selected_ids_sha256="0" * 64),
            "selected digest changed",
        ),
        (lambda r: r[0].update(person_id=""), "lacks owner identity"),
        (lambda r: r.append(deepcopy(r[0])), "repeats across receipts"),
    ],
)
def test_pending_publication_rejects_invalid_subset_receipt(change, message) -> None:
    works, profiles, receipt, registry, held = inputs()
    change(receipt)
    with pytest.raises(ValueError, match=message):
        build_pending_publications(works, profiles, receipt, registry, held)


def test_unrelated_public_profile_is_ignored() -> None:
    works, profiles, receipt, registry, held = inputs()
    unrelated = profiles.copy()
    unrelated.at[0, "work_id"] = "another-work"
    profiles = pandas.concat([profiles, unrelated], ignore_index=True)
    assert (
        len(build_pending_publications(works, profiles, receipt, registry, held)) == 1
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("has_included_source_profile", False, "included source profile"),
        ("memberships", [], "included faculty"),
        ("memberships", "bad", "included faculty"),
        ("memberships", ["bad"], "invalid faculty membership"),
        ("author_pub_ids", [ROW_ID, ROW_ID], "exact public source rows"),
        ("author_pub_ids", "bad", "exact public source rows"),
        ("source_urls", [], "public source URL"),
        ("source_urls", "bad", "public source URL"),
        ("year", "invalid", "invalid year"),
        ("year", 1700, "invalid year"),
        ("year", float("inf"), "invalid year"),
    ],
)
def test_pending_publication_rejects_other_work_leaks(field, value, message) -> None:
    works, profiles, receipt, registry, held = inputs()
    if field == "has_included_source_profile":
        works[field] = value
    else:
        if field == "year" and isinstance(value, str):
            works["year"] = works["year"].astype(object)
        works.at[0, field] = value
    with pytest.raises(ValueError, match=message):
        build_pending_publications(works, profiles, receipt, registry, held)


def test_pending_publication_rejects_missing_or_duplicate_profile_rows() -> None:
    works, profiles, receipt, registry, held = inputs()
    empty = profiles.iloc[:0]
    with pytest.raises(ValueError, match="exact public source rows"):
        build_pending_publications(works, empty, receipt, registry, held)
    duplicate = pandas.concat([profiles, profiles], ignore_index=True)
    with pytest.raises(ValueError, match="exact public source rows"):
        build_pending_publications(works, duplicate, receipt, registry, held)


def test_pending_publication_rejects_source_owner_outside_included_memberships() -> (
    None
):
    works, profiles, receipt, registry, held = inputs()
    profiles.at[0, "person_id"] = "other-person"
    receipt[0]["person_id"] = "other-person"
    with pytest.raises(ValueError, match="source owner lacks included membership"):
        build_pending_publications(works, profiles, receipt, registry, held)


@pytest.mark.parametrize(
    "field", ["home_type", "home_id", "source_url", "verified_at", "basis"]
)
def test_pending_publication_requires_each_typed_home_evidence_field(field) -> None:
    works, profiles, receipt, registry, held = inputs()
    setattr(registry.home, field, "")
    with pytest.raises(ValueError, match="evidenced typed home"):
        build_pending_publications(works, profiles, receipt, registry, held)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test/paper",
        "https:///missing-host",
        "https://user:pass@example.test/paper",
    ],
)
def test_pending_publication_requires_selected_safe_https_source(url) -> None:
    works, profiles, receipt, registry, held = inputs()
    works.at[0, "source_urls"] = [url]
    profiles.at[0, "source_url"] = url
    with pytest.raises(ValueError, match="not selected HTTPS"):
        build_pending_publications(works, profiles, receipt, registry, held)


def test_pending_publication_requires_selected_source_link() -> None:
    works, profiles, receipt, registry, held = inputs()
    works.at[0, "source_urls"] = ["https://example.test/unselected"]
    with pytest.raises(ValueError, match="not selected HTTPS"):
        build_pending_publications(works, profiles, receipt, registry, held)


def test_pending_publication_accepts_unknown_year() -> None:
    works, profiles, receipt, registry, held = inputs()
    works.at[0, "year"] = None
    assert (
        build_pending_publications(works, profiles, receipt, registry, held)[0]["year"]
        is None
    )


def test_pending_ineligible_work_is_visible_but_cannot_invent_an_edge() -> None:
    works, profiles, receipt, registry, held = inputs()
    works.at[0, "collaboration_eligible"] = False
    profiles.at[0, "collaboration_eligible"] = False
    result = build_pending_publications(works, profiles, receipt, registry, held)
    assert result[0]["collaboration_eligible"] is False


@pytest.mark.parametrize("table", ["works", "profiles"])
def test_pending_collaboration_eligibility_must_match_selected_source(table) -> None:
    works, profiles, receipt, registry, held = inputs()
    if table == "works":
        works.at[0, "collaboration_eligible"] = False
    else:
        profiles.at[0, "collaboration_eligible"] = False
    with pytest.raises(ValueError, match="collaboration eligibility disagrees"):
        build_pending_publications(works, profiles, receipt, registry, held)


@pytest.mark.parametrize("table", ["works", "profiles"])
def test_pending_collaboration_eligibility_must_be_exact_boolean(table) -> None:
    works, profiles, receipt, registry, held = inputs()
    if table == "works":
        works["collaboration_eligible"] = works["collaboration_eligible"].astype(
            object
        )
        works.at[0, "collaboration_eligible"] = "true"
    else:
        profiles["collaboration_eligible"] = profiles[
            "collaboration_eligible"
        ].astype(object)
        profiles.at[0, "collaboration_eligible"] = "true"
    with pytest.raises(ValueError, match="collaboration eligibility disagrees"):
        build_pending_publications(works, profiles, receipt, registry, held)
