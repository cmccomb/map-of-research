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
