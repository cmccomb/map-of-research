"""Bind the complete Das capture to one exact, disjoint release partition."""

import csv
from pathlib import Path

from map_of_research.attribution import (
    DEFAULT_ATTRIBUTION_REVIEWS_PATH,
    load_attribution_reviews,
)
from map_of_research.exclusions import DEFAULT_EXCLUSIONS_PATH, load_exclusions
from map_of_research.profile_releases import (
    DEFAULT_PROFILE_RELEASES_PATH,
    load_profile_releases,
)
from map_of_research.provisional_holds import (
    DEFAULT_PROVISIONAL_HOLDS_PATH,
    hold_ids_sha256,
    load_provisional_holds,
)

SCHOLAR = "FrF1YXQAAAAJ"
LEDGER = Path("registry/reviews/2026-10-25-chem-das-complete-profile-ledger.csv")
CACHE_SHA256 = "85700e7ee25079ed30633df91137e84cc0ef6c0e6628166f80acf4c133f4c1df"


def test_das_release_partitions_every_saved_citation_exactly_once() -> None:
    with LEDGER.open(newline="", encoding="utf-8") as handle:
        ledger = list(csv.DictReader(handle))
    assert len(ledger) == 670
    by_id = {row["author_pub_id"]: row for row in ledger}
    assert len(by_id) == 670
    assert [int(row["capture_row_index"]) for row in ledger] == list(range(1, 671))

    confirmations = load_attribution_reviews(DEFAULT_ATTRIBUTION_REVIEWS_PATH)
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    hold_profiles = load_provisional_holds(DEFAULT_PROVISIONAL_HOLDS_PATH)
    releases = load_profile_releases(DEFAULT_PROFILE_RELEASES_PATH)
    release = next(item for item in releases if item.scholar_id == SCHOLAR)
    hold = next(item for item in hold_profiles if item.scholar_id == SCHOLAR)

    positive_ids = {
        pub for scholar, pub in confirmations.confirmed if scholar == SCHOLAR
    }
    excluded_ids = {pub for scholar, pub in exclusions if scholar == SCHOLAR}
    held_ids = {item.author_pub_id for item in hold.held_observations}
    assert (len(positive_ids), len(excluded_ids), len(held_ids)) == (68, 486, 116)
    assert positive_ids.isdisjoint(excluded_ids)
    assert positive_ids.isdisjoint(held_ids)
    assert excluded_ids.isdisjoint(held_ids)
    assert positive_ids | excluded_ids | held_ids == set(by_id)
    assert positive_ids == {
        row["author_pub_id"]
        for row in ledger
        if row["decision"] == "positive_primary_source_reviewed_subset"
    }
    assert excluded_ids == {
        row["author_pub_id"] for row in ledger if row["decision"] == "reviewed_excluded"
    }
    assert held_ids == {
        row["author_pub_id"] for row in ledger if row["decision"] == "unresolved_held"
    }

    assert release.state == "released"
    assert release.person_id == hold.person_id == "person-subhardas-42cf6c1e"
    assert release.expected_source_rows == hold.expected_source_rows == 670
    assert release.cache_sha256 == hold.cache_sha256 == CACHE_SHA256
    assert release.expected_provisional_hold_rows == 116
    assert release.provisional_hold_ids_sha256 == hold_ids_sha256(held_ids)
    assert hold.expected_reviewed_exclusion_rows == 486
    assert hold.reviewed_exclusion_ids_sha256 == hold_ids_sha256(excluded_ids)
    assert all(
        by_id[item.author_pub_id]["capture_row_index"] == str(item.capture_row_index)
        and item.evidence_url.startswith("https://scholar.google.com/citations?")
        and f"citation_for_view={item.author_pub_id}" in item.evidence_url
        for item in hold.held_observations
    )
    initials_only_sources = {
        "FrF1YXQAAAAJ:QoJ_w57xiyAC": "https://api.crossref.org/works/10.3997/2214-4609.202321038",
        "FrF1YXQAAAAJ:QVtou7C4vgoC": "https://api.crossref.org/works/10.3997/2214-4609.2022616012",
    }
    for item in hold.held_observations:
        if item.author_pub_id in initials_only_sources:
            assert (
                item.primary_evidence_url == initials_only_sources[item.author_pub_id]
            )
            assert "Identity unresolved" in item.reason
            assert (
                by_id[item.author_pub_id]["evidence_url"] == item.primary_evidence_url
            )
