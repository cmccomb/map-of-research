"""Keep the complete Kim review bounded to its saved Scholar citation IDs."""

import csv
import json
from pathlib import Path

from map_of_research.exclusions import DEFAULT_EXCLUSIONS_PATH, load_exclusions
from registry.reviews.rebuild_chem_kim_918_review import EXACT_FOLLOWUP_CAPTURE_GUARDS

REVIEWS = Path("registry/reviews")
SCHOLAR_ID = "bC4aq-4AAAAJ"


def test_kim_review_partitions_all_918_observations_without_public_release() -> None:
    with (REVIEWS / "2026-10-24-chem-kim-918-screen.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["author_pub_id"]: row for row in rows}
    assert len(rows) == len(by_id) == 918
    assert {int(row["capture_index"]) for row in rows} == set(range(1, 919))

    proposal = json.loads(
        (REVIEWS / "proposals/scholar-profile-releases.kim-proposal.json").read_text()
    )["profiles"][0]
    assert proposal["scholar_id"] == SCHOLAR_ID
    assert proposal["expected_source_rows"] == 918
    confirmed = set(proposal["released_author_pub_ids"])
    excluded = set(proposal["excluded_author_pub_ids"])
    held = set(proposal["held_author_pub_ids"])
    assert (len(confirmed), len(excluded), len(held)) == (123, 492, 303)
    assert confirmed | excluded | held == set(by_id)
    assert not (confirmed & excluded or confirmed & held or excluded & held)
    assert confirmed == {
        row["author_pub_id"]
        for row in rows
        if row["status"] == "cmu_primary_work_retained"
    }
    assert excluded == {
        row["author_pub_id"]
        for row in rows
        if row["status"] == "already_excluded" or row["status"].startswith("proposed_")
    }

    manifest = load_exclusions(
        REVIEWS / "proposals/scholar-exclusions.kim-proposal.json"
    )
    assert {
        author_pub_id
        for scholar_id, author_pub_id in manifest
        if scholar_id == SCHOLAR_ID
    } == excluded
    assert all(scholar_id == SCHOLAR_ID for scholar_id, _ in manifest)
    assert not any(
        scholar_id == SCHOLAR_ID
        for scholar_id, _ in load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    )
    assert all(
        by_id[author_pub_id]["evidence_url"].startswith("https://")
        for author_pub_id in excluded
        - {row["author_pub_id"] for row in rows if row["status"] == "already_excluded"}
    )

    active_releases = json.loads(
        Path("data/curated/scholar-profile-releases.json").read_text()
    )
    active_reviews = json.loads(
        Path("data/curated/scholar-attribution-reviews.json").read_text()
    )
    assert SCHOLAR_ID not in json.dumps(active_releases)
    assert SCHOLAR_ID not in json.dumps(active_reviews)
    assert not Path(f"data/authors/{SCHOLAR_ID}.json").exists()


def test_kim_exact_follow_up_decisions_do_not_expand_to_neighboring_works() -> None:
    with (REVIEWS / "2026-10-24-chem-kim-918-screen.csv").open(newline="") as handle:
        by_index = {int(row["capture_index"]): row for row in csv.DictReader(handle)}
    assert {
        index: (by_index[index]["author_pub_id"], by_index[index]["status"])
        for index in (511, 637, 717, 775, 809)
    } == {
        511: ("bC4aq-4AAAAJ:KbBQZpvPDL4C", "proposed_other_person_primary_work"),
        637: ("bC4aq-4AAAAJ:E8ajGqO0XoUC", "proposed_no_target_author_primary_work"),
        717: ("bC4aq-4AAAAJ:iKswqCX-FLkC", "proposed_other_person_primary_work"),
        775: ("bC4aq-4AAAAJ:V3AGJWp-ZtQC", "proposed_no_target_author_primary_work"),
        809: ("bC4aq-4AAAAJ:OU6Ihb5iCvQC", "unresolved_attribution_hold"),
    }
    for index, fingerprint in EXACT_FOLLOWUP_CAPTURE_GUARDS.items():
        row = by_index[index]
        assert fingerprint == (
            row["author_pub_id"],
            row["source_title"],
            row["source_authors"],
            row["source_year"],
            row["source_venue"],
        )


def test_amore_i_author_orcid_excludes_only_exact_kim_citation() -> None:
    with (REVIEWS / "2026-10-24-chem-kim-918-screen.csv").open(newline="") as handle:
        rows = {int(row["capture_index"]): row for row in csv.DictReader(handle)}
    row = rows[162]
    assert row["author_pub_id"] == "bC4aq-4AAAAJ:qSd0DAb9jMoC"
    assert row["status"] == "proposed_other_person_primary_work"
    assert row["matched_doi"] == "10.1103/PhysRevLett.134.082501"
    assert row["evidence_url"] == (
        "https://api.crossref.org/works/10.1103%2FPhysRevLett.134.082501"
    )
    manifest = json.loads(
        (REVIEWS / "proposals/scholar-exclusions.kim-proposal.json").read_text()
    )["exclusions"]
    proposed = next(
        item for item in manifest if item["author_pub_id"] == row["author_pub_id"]
    )
    assert proposed["reason_code"] == "false_profile_attribution"
    assert "0000-0001-9787-4684" in proposed["reason"]
    assert all(
        rows[index]["status"] == "unresolved_attribution_hold"
        for index in (135, 138, 159, 172, 182, 222)
    )
