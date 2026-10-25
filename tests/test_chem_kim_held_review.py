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
    assert (len(confirmed), len(excluded), len(held)) == (123, 527, 268)
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


def test_three_hongjoo_links_leave_neighboring_held_rows_untouched() -> None:
    with (REVIEWS / "2026-10-24-chem-kim-918-screen.csv").open(newline="") as handle:
        rows = {int(row["capture_index"]): row for row in csv.DictReader(handle)}
    assert {
        index: (rows[index]["author_pub_id"], rows[index]["matched_doi"])
        for index in (374, 379, 390)
    } == {
        374: ("bC4aq-4AAAAJ:R22Rs3tN8aoC", "10.1109/TNS.2016.2530828"),
        379: ("bC4aq-4AAAAJ:dMpQl7XwOw4C", "10.1140/epjc/s10052-024-13516-9"),
        390: ("bC4aq-4AAAAJ:mel-f30kHHgC", "10.1016/j.nima.2005.06.080"),
    }
    assert all(
        rows[index]["status"] == "proposed_other_person_primary_work"
        for index in (374, 379, 390)
    )
    assert all(
        rows[index]["status"] == "unresolved_attribution_hold"
        for index in (364, 365, 373, 376, 385, 398, 399, 402, 406)
    )


def test_four_more_exact_author_links_leave_adjacent_rows_held() -> None:
    with (REVIEWS / "2026-10-24-chem-kim-918-screen.csv").open(newline="") as handle:
        rows = {int(row["capture_index"]): row for row in csv.DictReader(handle)}
    assert {
        index: (rows[index]["author_pub_id"], rows[index]["matched_doi"])
        for index in (420, 430, 432, 435)
    } == {
        420: ("bC4aq-4AAAAJ:aIdbFUkbNIkC", "10.1140/epjc/s10052-018-6462-0"),
        430: ("bC4aq-4AAAAJ:anf4URPfarAC", "10.3938/jkps.53.1575"),
        432: ("bC4aq-4AAAAJ:edDO8Oi4QzsC", "10.1080/00223131.2008.10875862"),
        435: ("bC4aq-4AAAAJ:F9HO9s0W2bwC", "10.1016/j.astropartphys.2024.102991"),
    }
    assert all(
        rows[index]["status"] == "proposed_other_person_primary_work"
        for index in (420, 430, 432, 435)
    )
    assert all(
        rows[index]["status"] == "unresolved_attribution_hold"
        for index in (413, 427, 429, 445, 454, 456, 458, 459)
    )


def test_six_exact_author_resolutions_leave_initials_only_neighbors_held() -> None:
    with (REVIEWS / "2026-10-24-chem-kim-918-screen.csv").open(newline="") as handle:
        rows = {int(row["capture_index"]): row for row in csv.DictReader(handle)}
    assert {
        index: (rows[index]["author_pub_id"], rows[index]["status"])
        for index in (467, 469, 476, 490, 491, 494)
    } == {
        467: ("bC4aq-4AAAAJ:Ej9njvOgR2oC", "proposed_other_person_primary_work"),
        469: (
            "bC4aq-4AAAAJ:b1wdh0AR-JQC",
            "proposed_no_target_author_primary_work",
        ),
        476: ("bC4aq-4AAAAJ:LPtt_HFRSbwC", "proposed_other_person_primary_work"),
        490: ("bC4aq-4AAAAJ:lbI08cpqPnQC", "proposed_other_person_primary_work"),
        491: ("bC4aq-4AAAAJ:U5uP8zs9lfgC", "proposed_other_person_primary_work"),
        494: (
            "bC4aq-4AAAAJ:_Ybze24A_UAC",
            "proposed_no_target_author_primary_work",
        ),
    }
    assert all(
        rows[index]["status"] == "unresolved_attribution_hold"
        for index in (461, 477, 487, 493)
    )


def test_nine_knu_person_links_exclude_only_exact_citations() -> None:
    with (REVIEWS / "2026-10-24-chem-kim-918-screen.csv").open(newline="") as handle:
        rows = {int(row["capture_index"]): row for row in csv.DictReader(handle)}
    expected = {
        508: ("bC4aq-4AAAAJ:owLR8QvbtFgC", "10.1016/j.nima.2015.02.011"),
        512: (
            "bC4aq-4AAAAJ:buQ7SEKw-1sC",
            "10.1016/j.astropartphys.2012.03.005",
        ),
        516: ("bC4aq-4AAAAJ:1lhNe0rCu4AC", ""),
        529: ("bC4aq-4AAAAJ:s9ia6_kGH2AC", "10.1088/1748-0221/9/11/P11015"),
        533: ("bC4aq-4AAAAJ:CaZNVDsoPx4C", "10.1109/NSSMIC.2012.6551459"),
        571: ("bC4aq-4AAAAJ:6yz0xqPARnAC", "10.1109/TNS.2012.2189019"),
        572: ("bC4aq-4AAAAJ:ALROH1vI_8AC", "10.1109/TNS.2010.2041789"),
        573: ("bC4aq-4AAAAJ:BrOSOlqYqPUC", "10.1109/TNS.2009.2014063"),
        647: ("bC4aq-4AAAAJ:U3qCfcK-7lkC", "10.1109/NSSMIC.2008.4774976"),
    }
    assert {
        index: (rows[index]["author_pub_id"], rows[index]["matched_doi"])
        for index in expected
    } == expected
    assert all(
        rows[index]["status"] == "proposed_other_person_primary_work"
        and "research.knu.ac.kr/en/publications/" in rows[index]["evidence_url"]
        for index in expected
    )
    assert all(
        rows[index]["status"] == "unresolved_attribution_hold"
        for index in (514, 523, 532, 549, 557, 558, 598, 615)
    )


def test_late_nine_namesake_ids_preserve_initials_only_holds() -> None:
    with (REVIEWS / "2026-10-24-chem-kim-918-screen.csv").open(newline="") as handle:
        rows = {int(row["capture_index"]): row for row in csv.DictReader(handle)}
    expected = {
        661: "bC4aq-4AAAAJ:gFrPXmx1TSsC",
        662: "bC4aq-4AAAAJ:naSTrk-c4S8C",
        669: "bC4aq-4AAAAJ:SAZ1SQo2q1kC",
        672: "bC4aq-4AAAAJ:dFKc6_kCK1wC",
        680: "bC4aq-4AAAAJ:OcBU2YAGkTUC",
        685: "bC4aq-4AAAAJ:ce2CqMG-AY4C",
        692: "bC4aq-4AAAAJ:_9Xh93LWpsYC",
        694: "bC4aq-4AAAAJ:ovGv7akYl-cC",
        697: "bC4aq-4AAAAJ:PVjk1bu6vJQC",
    }
    assert {index: rows[index]["author_pub_id"] for index in expected} == expected
    assert all(
        rows[index]["status"] == "proposed_other_person_primary_work"
        and rows[index]["evidence_url"].startswith("https://")
        for index in expected
    )
    assert all(
        rows[index]["status"] == "unresolved_attribution_hold"
        for index in (650, 659, 674, 682, 689, 693, 696, 710)
    )


def test_late_four_person_links_preserve_initials_only_holds() -> None:
    with (REVIEWS / "2026-10-24-chem-kim-918-screen.csv").open(newline="") as handle:
        rows = {int(row["capture_index"]): row for row in csv.DictReader(handle)}
    expected = {
        731: ("bC4aq-4AAAAJ:3_LpOwP6eMYC", "10.1016/j.apradiso.2013.03.057"),
        753: ("bC4aq-4AAAAJ:6ZzL7HXColQC", "10.3938/NPSM.66.417"),
        757: ("bC4aq-4AAAAJ:BmWJbWwHJAwC", "10.48550/arXiv.1501.02944"),
        766: ("bC4aq-4AAAAJ:zdX0sdgBH_kC", ""),
    }
    assert {
        index: (rows[index]["author_pub_id"], rows[index]["matched_doi"])
        for index in expected
    } == expected
    assert all(
        rows[index]["status"] == "proposed_other_person_primary_work"
        and rows[index]["evidence_url"].startswith("https://")
        for index in expected
    )
    assert all(
        rows[index]["status"] == "unresolved_attribution_hold"
        for index in (
            713, 730, 738, 744, 747, 748, 758, 764, 769, 774, 779,
            780, 781, 782, 783, 784, 785, 788,
        )
    )
