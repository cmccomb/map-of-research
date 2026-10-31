"""Release gates for corpus-dependent weak-region classifications."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy
import pandas
import pytest

from map_of_research.composite_region import BLANK_TOPICS, COORDINATE_FIELDS
from map_of_research.mccomb_continuity import ExactMcCombContinuity
from map_of_research.pending_publications import carry_forward_parent_pending_works
from map_of_research.region_continuity import (
    RegionReleaseDiffError,
    _fingerprint,
    _indexed_observations,
    _indexed_works,
    _value,
    _verification_url_identity,
    reconcile_region_decisions,
    require_revision,
)
from map_of_research.sullivan_continuity_addendum import (
    SOURCE_COMMIT as SULLIVAN_SOURCE_COMMIT,
)
from map_of_research.sullivan_continuity_addendum import (
    SOURCE_SHA256 as SULLIVAN_SOURCE_SHA256,
)
from map_of_research.sullivan_continuity_addendum import (
    validate_sullivan_continuity_addendum,
)
from map_of_research.sullivan_loss_approval import (
    APPROVAL_SHA256,
    PRIOR_REVISION,
    ExactSullivanLossApproval,
)

REVISION = "a" * 40


@pytest.mark.parametrize(
    ("url", "identity"),
    [
        (
            "https://PRIMARY.TEST:443/one?utm_source=mail&item=5#section",
            "https://primary.test/one?item=5",
        ),
        ("https://primary.test:8443/one", "https://primary.test:8443/one"),
        ("https://[2001:db8::1]/one", "https://[2001:db8::1]/one"),
        ("https://primary.test/%6fne", "https://primary.test/one"),
        ("https://primary.test/./one", "https://primary.test/one"),
        ("https://primary.test/one/.", "https://primary.test/one/"),
        ("https://primary.test/a/../one", "https://primary.test/one"),
        ("https://primary.test/one/..", "https://primary.test/"),
        ("https://primary.test/../one", "https://primary.test/one"),
        ("https://primary.test/%2e/one", "https://primary.test/one"),
        ("https://primary.test/one%2fpart", "https://primary.test/one%2Fpart"),
        ("https://primary.test/one?item=6", "https://primary.test/one?item=6"),
        ("  ", ""),
        ("https://primary.test:invalid/one", "https://primary.test:invalid/one"),
    ],
)
def test_verification_url_identity_preserves_document_location(url, identity):
    assert _verification_url_identity(url) == identity


def test_verification_url_identity_keeps_substantive_path_and_query_distinctions():
    assert _verification_url_identity(
        "https://primary.test/one?item=5"
    ) != _verification_url_identity("https://primary.test/one?item=6")
    assert _verification_url_identity(
        "https://primary.test/one%2Fpart"
    ) != _verification_url_identity("https://primary.test/one/part")


def observation(
    person: str,
    work_id: str,
    *,
    eligible: bool = True,
    source: str = '{"title":"A paper"}',
) -> dict:
    return {
        "person_id": person,
        "scholar_id": "profile",
        "author_pub_id": "profile:shared-row",
        "source_type": "scholar",
        "work_id": work_id,
        "map_eligible": eligible,
        "map_exclusion_reasons": [],
        "source_record_json": source,
        "title": "A paper",
        "authors": "A Author",
        "year": 2020,
        "venue": "Journal",
        "citation": "Journal 1, 2020",
        "citation_count": 2,
        "source_url": "https://scholar.google.com/citations?x=1",
        "doi": "",
        "source_doi": "",
        "source_native_id": "",
        "source_verified_at": "",
        "source_verification_urls": [],
        "metadata_correction_fields": "",
        "metadata_correction_basis": "",
        "metadata_correction_evidence_url": "",
        "metadata_correction_verified_at": "",
        "publication_type": "publication",
        "source_document_id": "",
        "quality_assessment_version": "quality-v1",
        "review_exclusion_code": "",
        "context_verified_at": "",
        "work_match_method": "normalized_title_year",
    }


def work(
    work_id: str,
    *,
    eligible: bool,
    outlier: bool,
    department: bool = True,
) -> dict:
    return {
        "work_id": work_id,
        "title": "A paper",
        "authors": "A Author",
        "year": 2020,
        "venue": "Journal",
        "map_eligible": eligible,
        "map_region_outlier": outlier,
        "map_region_documented_override": False,
        "map_exclusion_reasons": ["low_information_region"] if outlier else [],
        "department_ids": ["bio"] if department else [],
        "quality_assessment_version": "quality-v1",
        "source_types": ["scholar"],
        "source_verified_at": "",
        "source_verification_urls": [],
        "source_native_ids": [],
        "doi": "",
    }


def reconcile(current_works, current_rows, prior_works, prior_rows):
    return reconcile_region_decisions(
        pandas.DataFrame(current_works),
        pandas.DataFrame(current_rows),
        prior_works,
        prior_rows,
        prior_revision=REVISION,
    )


def test_unchanged_source_preserves_both_region_decisions_and_person_identity():
    prior_rows = [observation("person-one", "one"), observation("person-two", "two")]
    current_rows = copy.deepcopy(prior_rows)
    prior_works = [
        work("one", eligible=True, outlier=False),
        work("two", eligible=False, outlier=True),
    ]
    current_works = [
        work("one", eligible=False, outlier=True),
        work("two", eligible=True, outlier=False),
    ]
    result, receipt = reconcile(current_works, current_rows, prior_works, prior_rows)
    assert result.map_eligible.tolist() == [True, False]
    assert result.map_region_outlier.tolist() == [False, True]
    assert result.map_exclusion_reasons.tolist() == [[], ["low_information_region"]]
    assert receipt["stable_works"] == 2
    assert receipt["prior_eligible_overrides"] == 1
    assert receipt["prior_excluded_overrides"] == 1
    assert receipt["prior_visible_observations"] == 1
    assert receipt["current_visible_observations"] == 1
    assert receipt["prior_revision"] == REVISION
    assert receipt["unexplained_losses"] == []
    assert receipt["unexplained_gains"] == []


def test_parent_pending_carryforward_leaves_unrelated_gain_region_gated():
    pending = work("pending", eligible=False, outlier=False)
    pending.update(
        {
            "map_exclusion_reasons": ["pending_placement_review"],
            "citation": "Journal 1, 2020",
            "citation_count": 2,
            "publication_type": "publication",
            "source_document_id": "",
            "source_urls": ["https://scholar.google.com/citations?x=1"],
            **dict.fromkeys(COORDINATE_FIELDS),
            **BLANK_TOPICS,
        }
    )
    other = work("other", eligible=False, outlier=False)
    other.update(
        {
            "citation": "Journal 1, 2020",
            "citation_count": 2,
            "publication_type": "publication",
            "source_document_id": "",
            "source_urls": ["https://scholar.google.com/citations?x=2"],
            **dict.fromkeys(COORDINATE_FIELDS),
            **BLANK_TOPICS,
        }
    )
    prior_works = [pending, other]
    prior_pending = observation("person-one", "pending")
    prior_other = {
        **observation("person-two", "other"),
        "author_pub_id": "profile:other-row",
    }
    prior_profiles = [prior_pending, prior_other]
    candidate = copy.deepcopy(prior_works)
    for row in candidate:
        row["map_eligible"] = True
        row["map_exclusion_reasons"] = []
        row["x"] = 0.2
    carried, ids = carry_forward_parent_pending_works(
        pandas.DataFrame(candidate),
        pandas.DataFrame(prior_profiles),
        prior_works,
        prior_profiles,
    )
    assert ids == ["pending"]
    assert carried.loc[carried.work_id == "pending", "map_eligible"].eq(False).all()
    assert carried.loc[carried.work_id == "other", "map_eligible"].eq(True).all()
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained gains"):
        reconcile_region_decisions(
            carried,
            pandas.DataFrame(prior_profiles),
            prior_works,
            prior_profiles,
            prior_revision=REVISION,
        )


def test_visible_observation_cannot_change_work_without_exact_review():
    prior_rows = [observation("person-one", "one"), observation("person-two", "two")]
    current_rows = [{**prior_rows[0], "work_id": "two"}, prior_rows[1]]
    works = [
        work("one", eligible=True, outlier=False),
        work("two", eligible=True, outlier=False),
    ]
    with pytest.raises(RegionReleaseDiffError, match="mapped binding changes") as error:
        reconcile(works, current_rows, works, prior_rows)
    assert error.value.receipt["unapproved_mapped_binding_changes"] == [
        {
            "observation": "person-one/profile/profile:shared-row/scholar",
            "prior_work_id": "one",
            "current_work_id": "two",
            "prior_collaboration_eligible": False,
            "current_collaboration_eligible": False,
        }
    ]


def test_visible_observation_cannot_change_collaboration_eligibility_silently():
    prior = {**observation("person-one", "one"), "collaboration_eligible": True}
    current = {**prior, "collaboration_eligible": False}
    works = [work("one", eligible=True, outlier=False)]
    with pytest.raises(RegionReleaseDiffError, match="mapped binding changes") as error:
        reconcile(works, [current], works, [prior])
    assert error.value.receipt["unapproved_mapped_binding_changes"] == [
        {
            "observation": "person-one/profile/profile:shared-row/scholar",
            "prior_work_id": "one",
            "current_work_id": "one",
            "prior_collaboration_eligible": True,
            "current_collaboration_eligible": False,
        }
    ]


def test_exact_person_hold_loss_approval_preserves_other_rows() -> None:
    held = observation("person-held", "one")
    coauthor = observation("person-coauthor", "one")
    works = [work("one", eligible=True, outlier=False)]
    approved = frozenset({("person-held", "profile", "profile:shared-row", "scholar")})
    _, receipt = reconcile_region_decisions(
        pandas.DataFrame(works),
        pandas.DataFrame([coauthor]),
        works,
        [held, coauthor],
        prior_revision=REVISION,
        approved_person_hold_losses=approved,
    )
    assert receipt["approved_person_hold_losses"] == [
        "person-held/profile/profile:shared-row/scholar"
    ]
    assert receipt["unexplained_losses"] == []
    assert receipt["unused_person_hold_approvals"] == []
    with pytest.raises(RegionReleaseDiffError) as error:
        reconcile_region_decisions(
            pandas.DataFrame(works),
            pandas.DataFrame([coauthor]),
            works,
            [held, coauthor],
            prior_revision=REVISION,
            approved_person_hold_losses=frozenset(
                {("person-other", "profile", "profile:shared-row", "scholar")}
            ),
        )
    assert error.value.receipt["unexplained_losses"] == [
        "person-held/profile/profile:shared-row/scholar"
    ]
    assert error.value.receipt["unused_person_hold_approvals"] == [
        "person-other/profile/profile:shared-row/scholar"
    ]


def test_sullivan_approval_rejects_wrong_object_type():
    rows = [observation("person-one", "one")]
    works = [work("one", eligible=True, outlier=False)]
    with pytest.raises(ValueError, match="wrong type"):
        reconcile_region_decisions(
            pandas.DataFrame(works),
            pandas.DataFrame(rows),
            works,
            rows,
            prior_revision=REVISION,
            sullivan_loss_approval=object(),
        )


def test_current_exact_exclusion_wins_and_is_receipted():
    old = observation("person", "one")
    current = {
        **old,
        "map_eligible": False,
        "review_exclusion_code": "duplicate",
        "map_exclusion_reasons": ["reviewed_duplicate"],
    }
    result, receipt = reconcile(
        [work("one", eligible=False, outlier=False)],
        [current],
        [work("one", eligible=True, outlier=False)],
        [old],
    )
    assert result.map_eligible.tolist() == [False]
    assert receipt["reviewed_losses"] == ["person/profile/profile:shared-row/scholar"]
    assert receipt["stable_works"] == 0


def test_current_context_review_can_explain_loss():
    old = observation("person", "one")
    current = {
        **old,
        "map_eligible": False,
        "context_verified_at": "2026-10-25",
        "map_exclusion_reasons": ["reviewed_context_not_mapped"],
    }
    _, receipt = reconcile(
        [work("one", eligible=False, outlier=False)],
        [current],
        [work("one", eligible=True, outlier=False)],
        [old],
    )
    assert len(receipt["reviewed_losses"]) == 1


def test_provisional_hold_alone_cannot_remove_prior_visible_observation():
    old = observation("person", "one")
    current = {
        **old,
        "map_eligible": False,
        "map_exclusion_reasons": ["attribution_provisional_hold"],
        "review_attribution_status": "held",
        "review_attribution_evidence_url": "https://scholar.google.com/citations?user=profile",
    }
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained losses"):
        reconcile(
            [work("one", eligible=False, outlier=False)],
            [current],
            [work("one", eligible=True, outlier=False)],
            [old],
        )


def _sullivan_exact_loss_fixture():
    path = (
        Path(__file__).resolve().parents[1]
        / "data/curated/sullivan-exact-losses-bb88.json"
    )
    entries = json.loads(path.read_text(encoding="utf-8"))["approved_losses"]
    identities = [
        tuple(
            entry[field]
            for field in ("person_id", "scholar_id", "author_pub_id", "source_type")
        )
        for entry in entries
    ]
    held = frozenset(
        identity
        for entry, identity in zip(entries, identities, strict=True)
        if entry["decision"] == "provisional_hold"
    )
    rekeys = frozenset(set(identities) - held)
    approval = ExactSullivanLossApproval(
        identities=frozenset(identities),
        manifest_sha256=APPROVAL_SHA256,
        held_identities=held,
        rekey_identities=rekeys,
    )
    prior_rows = []
    current_rows = []
    prior_works = []
    current_works = []
    for index, (entry, identity) in enumerate(zip(entries, identities, strict=True)):
        work_id = f"sullivan-work-{index}"
        row = observation(identity[0], work_id)
        row.update(
            scholar_id=identity[1],
            author_pub_id=identity[2],
            source_type=identity[3],
        )
        prior_rows.append(row)
        prior_works.append(work(work_id, eligible=True, outlier=False))
        current_works.append(work(work_id, eligible=False, outlier=False))
        if entry["decision"] == "provisional_hold":
            current_rows.append(
                {
                    **row,
                    "map_eligible": False,
                    "map_exclusion_reasons": ["attribution_provisional_hold"],
                    "review_attribution_status": "held",
                }
            )
    return approval, prior_rows, current_rows, prior_works, current_works


def test_exact_sullivan_loss_set_receipts_18_holds_and_four_rekeys():
    approval, prior_rows, current_rows, prior_works, current_works = (
        _sullivan_exact_loss_fixture()
    )
    _, receipt = reconcile_region_decisions(
        pandas.DataFrame(current_works),
        pandas.DataFrame(current_rows),
        prior_works,
        prior_rows,
        prior_revision=PRIOR_REVISION,
        sullivan_loss_approval=approval,
    )
    assert len(receipt["approved_sullivan_held_losses"]) == 18
    assert len(receipt["approved_sullivan_rekey_losses"]) == 4
    assert receipt["sullivan_loss_approval_sha256"] == APPROVAL_SHA256
    assert receipt["unexplained_losses"] == []
    assert receipt["unused_sullivan_loss_approvals"] == []


def test_person_hold_cannot_reapprove_a_sullivan_loss() -> None:
    approval, prior_rows, current_rows, prior_works, current_works = (
        _sullivan_exact_loss_fixture()
    )
    with pytest.raises(ValueError, match="overlaps another loss approval"):
        reconcile_region_decisions(
            pandas.DataFrame(current_works),
            pandas.DataFrame(current_rows),
            prior_works,
            prior_rows,
            prior_revision=PRIOR_REVISION,
            sullivan_loss_approval=approval,
            approved_person_hold_losses=frozenset({next(iter(approval.identities))}),
        )


def test_forged_sullivan_loss_approval_cannot_waive_generic_loss():
    old = observation("person-one", "one")
    identity = ("person-one", "profile", "profile:shared-row", "scholar")
    forged = ExactSullivanLossApproval(
        identities=frozenset({identity}),
        manifest_sha256=APPROVAL_SHA256,
        held_identities=frozenset({identity}),
        rekey_identities=frozenset(),
    )
    with pytest.raises(ValueError, match="not the reviewed set"):
        reconcile_region_decisions(
            pandas.DataFrame([work("one", eligible=False, outlier=False)]),
            pandas.DataFrame([{**old, "map_eligible": False}]),
            [work("one", eligible=True, outlier=False)],
            [old],
            prior_revision=PRIOR_REVISION,
            sullivan_loss_approval=forged,
        )


def test_exact_sullivan_approval_fails_for_unapproved_loss_and_unused_id():
    approval, prior_rows, current_rows, prior_works, current_works = (
        _sullivan_exact_loss_fixture()
    )
    extra = observation("person-extra", "extra")
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained losses"):
        reconcile_region_decisions(
            pandas.DataFrame(
                [*current_works, work("extra", eligible=False, outlier=False)]
            ),
            pandas.DataFrame([*current_rows, {**extra, "map_eligible": False}]),
            [*prior_works, work("extra", eligible=True, outlier=False)],
            [*prior_rows, extra],
            prior_revision=PRIOR_REVISION,
            sullivan_loss_approval=approval,
        )

    prior_rows = prior_rows[1:]
    prior_works = prior_works[1:]
    with pytest.raises(RegionReleaseDiffError, match="1 unused Sullivan approvals"):
        reconcile_region_decisions(
            pandas.DataFrame(current_works[1:]),
            pandas.DataFrame(current_rows[1:]),
            prior_works,
            prior_rows,
            prior_revision=PRIOR_REVISION,
            sullivan_loss_approval=approval,
        )


def test_exact_sullivan_approval_rejects_extra_generic_reviewed_loss():
    approval, prior_rows, current_rows, prior_works, current_works = (
        _sullivan_exact_loss_fixture()
    )
    extra = observation("person-extra", "extra")
    reviewed_extra = {
        **extra,
        "map_eligible": False,
        "review_exclusion_code": "duplicate",
        "map_exclusion_reasons": ["reviewed_duplicate"],
    }
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained losses"):
        reconcile_region_decisions(
            pandas.DataFrame(
                [*current_works, work("extra", eligible=False, outlier=False)]
            ),
            pandas.DataFrame([*current_rows, reviewed_extra]),
            [*prior_works, work("extra", eligible=True, outlier=False)],
            [*prior_rows, extra],
            prior_revision=PRIOR_REVISION,
            sullivan_loss_approval=approval,
        )


def _sullivan_addendum_fixture():
    approval, prior_rows, current_rows, prior_works, current_works = (
        _sullivan_exact_loss_fixture()
    )
    root = Path(__file__).resolve().parents[1]
    additions = json.loads(
        (root / "tests/fixtures/sullivan_continuity_addendum_rows.json").read_text()
    )
    addendum = validate_sullivan_continuity_addendum(
        addendum_path=root / "data/curated/sullivan-continuity-addendum-bb88.json",
        prior_revision=PRIOR_REVISION,
        source_snapshot_commit=SULLIVAN_SOURCE_COMMIT,
        source_snapshot_sha256=SULLIVAN_SOURCE_SHA256,
        **additions,
    )
    prior_rows.extend(additions["prior_profiles"])
    current_rows.extend(additions["current_profiles"])
    prior_works.extend(additions["prior_works"])
    for entry in additions["current_works"]:
        current_works.append(
            {
                **work(entry["work_id"], eligible=entry["map_eligible"], outlier=False),
                "collaboration_eligible": entry["collaboration_eligible"],
                "person_ids": entry["person_ids"],
            }
        )
    return approval, addendum, prior_rows, current_rows, prior_works, current_works


def test_exact_sullivan_addendum_allows_only_five_losses_and_three_work_rekeys():
    approval, addendum, prior_rows, current_rows, prior_works, current_works = (
        _sullivan_addendum_fixture()
    )
    _, receipt = reconcile_region_decisions(
        pandas.DataFrame(current_works),
        pandas.DataFrame(current_rows),
        prior_works,
        prior_rows,
        prior_revision=PRIOR_REVISION,
        sullivan_loss_approval=approval,
        sullivan_continuity_addendum=addendum,
    )
    assert len(receipt["approved_sullivan_held_losses"]) == 18
    assert len(receipt["approved_sullivan_rekey_losses"]) == 4
    assert len(receipt["approved_sullivan_additional_losses"]) == 5
    assert len(receipt["approved_sullivan_binding_changes"]) == 3
    assert receipt["unexplained_losses"] == []
    assert receipt["unapproved_mapped_binding_changes"] == []
    assert receipt["unused_sullivan_loss_approvals"] == []
    assert receipt["unused_sullivan_binding_approvals"] == []
    assert receipt["missing_sullivan_curated_targets"] == []


def test_exact_sullivan_addendum_rejects_ninth_binding_change():
    approval, addendum, prior_rows, current_rows, prior_works, current_works = (
        _sullivan_addendum_fixture()
    )
    extra = observation("person-extra", "extra-before")
    prior_rows.append(extra)
    current_rows.append({**extra, "work_id": "extra-after"})
    prior_works.append(work("extra-before", eligible=True, outlier=False))
    current_works.append(work("extra-after", eligible=True, outlier=False))
    with pytest.raises(RegionReleaseDiffError, match="1 mapped binding changes"):
        reconcile_region_decisions(
            pandas.DataFrame(current_works),
            pandas.DataFrame(current_rows),
            prior_works,
            prior_rows,
            prior_revision=PRIOR_REVISION,
            sullivan_loss_approval=approval,
            sullivan_continuity_addendum=addendum,
        )


def test_exact_sullivan_addendum_requires_both_curated_work_targets():
    approval, addendum, prior_rows, current_rows, prior_works, current_works = (
        _sullivan_addendum_fixture()
    )
    target = next(
        row
        for row in current_works
        if row["work_id"] in addendum.required_curated_work_ids
    )
    target["collaboration_eligible"] = False
    with pytest.raises(RegionReleaseDiffError) as error:
        reconcile_region_decisions(
            pandas.DataFrame(current_works),
            pandas.DataFrame(current_rows),
            prior_works,
            prior_rows,
            prior_revision=PRIOR_REVISION,
            sullivan_loss_approval=approval,
            sullivan_continuity_addendum=addendum,
        )
    assert error.value.receipt["missing_sullivan_curated_targets"] == [
        target["work_id"]
    ]


def test_forged_sullivan_addendum_cannot_waive_another_loss():
    approval, addendum, prior_rows, current_rows, prior_works, current_works = (
        _sullivan_addendum_fixture()
    )
    forged = replace(
        addendum,
        loss_identities=frozenset(
            {
                *list(addendum.loss_identities)[:-1],
                ("other", "profile", "row", "scholar"),
            }
        ),
    )
    with pytest.raises(ValueError, match="not the reviewed set"):
        reconcile_region_decisions(
            pandas.DataFrame(current_works),
            pandas.DataFrame(current_rows),
            prior_works,
            prior_rows,
            prior_revision=PRIOR_REVISION,
            sullivan_loss_approval=approval,
            sullivan_continuity_addendum=forged,
        )


def test_unused_sullivan_binding_approval_blocks_release():
    approval, addendum, prior_rows, current_rows, prior_works, current_works = (
        _sullivan_addendum_fixture()
    )
    target = next(
        row
        for row in current_rows
        if row["author_pub_id"] == "wXSgFr4AAAAJ:NJ774b8OgUMC"
    )
    prior = next(
        row for row in prior_rows if row["author_pub_id"] == "wXSgFr4AAAAJ:NJ774b8OgUMC"
    )
    target["work_id"] = prior["work_id"]
    current_works.append(work(prior["work_id"], eligible=True, outlier=False))
    with pytest.raises(RegionReleaseDiffError) as error:
        reconcile_region_decisions(
            pandas.DataFrame(current_works),
            pandas.DataFrame(current_rows),
            prior_works,
            prior_rows,
            prior_revision=PRIOR_REVISION,
            sullivan_loss_approval=approval,
            sullivan_continuity_addendum=addendum,
        )
    assert len(error.value.receipt["unused_sullivan_binding_approvals"]) == 1


@pytest.mark.parametrize(
    "review_field",
    [
        "metadata_correction_evidence_url",
        "context_verified_at",
        "review_exclusion_code",
    ],
)
def test_unrelated_review_metadata_cannot_explain_loss(review_field):
    old = observation("person", "one")
    current = {**old, "map_eligible": False, review_field: "review present"}
    current["map_exclusion_reasons"] = ["underspecified_title"]
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained losses"):
        reconcile(
            [work("one", eligible=False, outlier=False)],
            [current],
            [work("one", eligible=True, outlier=False)],
            [old],
        )


def test_changed_raw_source_does_not_inherit_prior_inclusion():
    old = observation("person", "one")
    current = {**old, "source_record_json": '{"title":"Different capture"}'}
    with pytest.raises(ValueError, match="1 unexplained losses"):
        reconcile(
            [work("one", eligible=False, outlier=True)],
            [current],
            [work("one", eligible=True, outlier=False)],
            [old],
        )


def test_changed_canonical_work_metadata_does_not_inherit_prior_region_decision():
    row = observation("person", "one")
    changed = {**work("one", eligible=False, outlier=True), "authors": "Another Author"}
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained losses"):
        reconcile(
            [changed],
            [row],
            [work("one", eligible=True, outlier=False)],
            [row],
        )


def test_new_eligible_member_does_not_inherit_prior_work_decision():
    old = observation("person-one", "one")
    new = observation("person-two", "one")
    with pytest.raises(ValueError, match="1 unexplained losses"):
        reconcile(
            [work("one", eligible=False, outlier=True)],
            [old, new],
            [work("one", eligible=True, outlier=False)],
            [old],
        )


def test_unverified_prior_outlier_cannot_be_promoted_by_cluster_movement():
    old = observation("person", "one")
    changed = {**old, "venue": "New venue"}
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained gains") as error:
        reconcile(
            [work("one", eligible=True, outlier=False)],
            [changed],
            [work("one", eligible=False, outlier=True)],
            [old],
        )
    assert len(error.value.receipt["unexplained_gains"]) == 1


def test_reviewed_doi_alias_can_promote_prior_outlier_after_rekey():
    old = observation("person", "old")
    current = {
        **old,
        "work_id": "new",
        "work_match_method": "reviewed_version_alias",
        "doi": "10.1000/verified",
        "source_verified_at": "2026-10-25",
        "source_verification_urls": ["https://primary.test/doi"],
    }
    new_work = {
        **work("new", eligible=True, outlier=False),
        "doi": "10.1000/verified",
        "source_verified_at": "2026-10-25",
        "source_verification_urls": ["https://primary.test/doi"],
    }
    _, receipt = reconcile(
        [new_work], [current], [work("old", eligible=False, outlier=True)], [old]
    )
    assert receipt["verified_gains"] == ["person/profile/profile:shared-row/scholar"]
    assert receipt["stable_works"] == 0


def test_native_curated_evidence_can_promote_prior_outlier():
    old = observation("person", "one")
    current = {**old, "source_native_id": "report:one"}
    new_work = {
        **work("one", eligible=True, outlier=False),
        "source_types": ["scholar", "curated_verified"],
        "source_native_ids": ["report:one"],
        "source_verified_at": "2026-10-25",
        "source_verification_urls": ["https://primary.test/report"],
    }
    _, receipt = reconcile(
        [new_work], [current], [work("one", eligible=False, outlier=True)], [old]
    )
    assert len(receipt["verified_gains"]) == 1


def test_documented_rule_cannot_promote_exact_prior_outlier_without_new_evidence():
    row = {
        **observation("person", "one"),
        "work_match_method": "reviewed_version_alias",
        "doi": "10.1000/verified",
        "source_verified_at": "2026-10-25",
        "source_verification_urls": ["https://primary.test/doi"],
    }
    documented_work = {
        **work("one", eligible=True, outlier=False),
        "map_region_documented_override": True,
        "doi": "10.1000/verified",
        "source_verified_at": "2026-10-25",
        "source_verification_urls": ["https://primary.test/doi"],
    }
    prior_work = {**documented_work, "map_eligible": False, "map_region_outlier": True}
    result, receipt = reconcile([documented_work], [row], [prior_work], [row])
    assert not bool(result.iloc[0].map_eligible)
    assert receipt["prior_excluded_overrides"] == 1
    assert receipt["verified_gains"] == []


def test_unchanged_evidence_cannot_explain_gain_when_quality_version_changes():
    row = {
        **observation("person", "one"),
        "work_match_method": "reviewed_version_alias",
        "doi": "10.1000/verified",
        "source_verified_at": "2026-10-25",
        "source_verification_urls": ["https://primary.test/doi"],
    }
    mapped = {
        **work("one", eligible=True, outlier=False),
        "doi": "10.1000/verified",
        "source_verified_at": "2026-10-25",
        "source_verification_urls": ["https://primary.test/doi"],
        "quality_assessment_version": "quality-v2",
    }
    prior = {**mapped, "quality_assessment_version": "quality-v1"}
    prior["map_eligible"] = False
    prior["map_region_outlier"] = True
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained gains"):
        reconcile([mapped], [row], [prior], [row])


@pytest.mark.parametrize(
    ("row_updates", "work_updates"),
    [
        ({"citation_count": 3, "source_record_json": '{"cited_by":3}'}, {}),
        ({}, {"citation_count": 3}),
        ({}, {"title": "A corrected paper title", "authors": "Another Author"}),
        ({"source_url": "https://scholar.google.com/citations?x=2"}, {}),
        (
            {
                "source_verification_urls": [
                    "https://primary.test/two",
                    "https://primary.test/one",
                ]
            },
            {},
        ),
        (
            {},
            {
                "source_verification_urls": [
                    "https://primary.test/two",
                    "https://primary.test/one",
                ]
            },
        ),
        ({}, {"source_types": ["curated_verified", "scholar"]}),
        (
            {},
            {
                "source_urls": [
                    "https://source.test/two",
                    "https://source.test/one",
                ]
            },
        ),
        (
            {"source_verification_urls": ["https://primary.test/one"]},
            {"source_verification_urls": ["https://primary.test/one"]},
        ),
        (
            {},
            {
                "source_verification_urls": [
                    "https://primary.test/one",
                    "https://primary.test/two",
                    "https://scholar.google.com/citations?x=2",
                ]
            },
        ),
        (
            {"source_verification_urls": ["https://primary.test/one#section"]},
            {},
        ),
        (
            {},
            {
                "source_verification_urls": [
                    "https://PRIMARY.TEST:443/one?utm_source=newsletter#section",
                    "https://primary.test/two",
                ]
            },
        ),
        (
            {},
            {
                "source_verification_urls": [
                    "https://primary.test/one?fbclid=tracking",
                    "https://primary.test/two",
                ]
            },
        ),
        ({"source_verification_urls": ["https://primary.test/%6fne"]}, {}),
        (
            {},
            {
                "source_verification_urls": [
                    "https://primary.test/./one",
                    "https://primary.test/two",
                ]
            },
        ),
        ({"source_verified_at": "2026-10-26"}, {}),
        ({}, {"source_verified_at": "2026-10-26"}),
    ],
)
def test_non_evidence_changes_cannot_promote_prior_outlier(row_updates, work_updates):
    urls = ["https://primary.test/one", "https://primary.test/two"]
    row = {
        **observation("person", "one"),
        "work_match_method": "reviewed_version_alias",
        "doi": "10.1000/verified",
        "source_verified_at": "2026-10-25",
        "source_verification_urls": urls,
    }
    prior_work = {
        **work("one", eligible=False, outlier=True),
        "doi": "10.1000/verified",
        "source_types": ["scholar", "curated_verified"],
        "source_urls": ["https://source.test/one", "https://source.test/two"],
        "source_verified_at": "2026-10-25",
        "source_verification_urls": urls,
    }
    current_work = {
        **prior_work,
        "map_eligible": True,
        "map_region_outlier": False,
        "map_exclusion_reasons": [],
        **work_updates,
    }
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained gains"):
        reconcile([current_work], [{**row, **row_updates}], [prior_work], [row])


def test_new_independent_verification_url_can_promote_prior_outlier():
    row = {
        **observation("person", "one"),
        "work_match_method": "reviewed_version_alias",
        "doi": "10.1000/verified",
        "source_verified_at": "2026-10-25",
        "source_verification_urls": ["https://primary.test/one"],
    }
    prior_work = {
        **work("one", eligible=False, outlier=True),
        "doi": "10.1000/verified",
        "source_verified_at": "2026-10-25",
        "source_verification_urls": ["https://primary.test/one"],
    }
    current_work = {
        **prior_work,
        "map_eligible": True,
        "map_region_outlier": False,
        "map_exclusion_reasons": [],
        "source_verification_urls": [
            "https://primary.test/one",
            "https://primary.test/two",
        ],
    }
    _, receipt = reconcile([current_work], [row], [prior_work], [row])
    assert receipt["verified_gains"] == ["person/profile/profile:shared-row/scholar"]


@pytest.mark.parametrize(
    ("bad_work", "bad_row"),
    [
        ({"doi": "not-a-doi"}, {}),
        ({"source_verified_at": "2026-02-31"}, {}),
        (
            {"source_verification_urls": ["https://scholar.google.com/citations?x=1"]},
            {},
        ),
        ({"authors": ""}, {}),
        (
            {},
            {"source_verification_urls": ["https://scholar.google.com/citations?x=1"]},
        ),
    ],
)
def test_prior_outlier_gain_requires_valid_independent_evidence(bad_work, bad_row):
    old = observation("person", "one")
    current = {
        **old,
        "work_match_method": "reviewed_version_alias",
        "doi": "10.1000/verified",
        "source_verified_at": "2026-10-25",
        "source_verification_urls": ["https://primary.test/doi"],
        **bad_row,
    }
    new_work = {
        **work("one", eligible=True, outlier=False),
        "doi": "10.1000/verified",
        "source_verified_at": "2026-10-25",
        "source_verification_urls": ["https://primary.test/doi"],
        **bad_work,
    }
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained gains"):
        reconcile(
            [new_work], [current], [work("one", eligible=False, outlier=True)], [old]
        )


def test_unreviewed_prior_quality_exclusion_cannot_be_promoted():
    old = observation("person", "one", eligible=False)
    current = {**old, "map_eligible": True}
    with pytest.raises(ValueError, match="1 unexplained gains"):
        reconcile(
            [work("one", eligible=True, outlier=False)],
            [current],
            [work("one", eligible=False, outlier=False)],
            [old],
        )


def test_new_work_quality_exclusion_does_not_inherit_prior_region_decision():
    old = observation("person", "one")
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained losses"):
        reconcile(
            [work("one", eligible=False, outlier=False)],
            [old],
            [work("one", eligible=True, outlier=False)],
            [old],
        )


def test_new_source_row_is_reported_without_inheriting_old_decision():
    old = observation("person-one", "one")
    new = observation("person-two", "two")
    _, receipt = reconcile(
        [
            work("one", eligible=True, outlier=False),
            work("two", eligible=True, outlier=False),
        ],
        [old, new],
        [work("one", eligible=True, outlier=False)],
        [old],
    )
    assert receipt["new_source_gains"] == 1


def test_quality_version_change_prevents_continuity_and_requires_review():
    old = observation("person", "one")
    changed = work("one", eligible=False, outlier=True)
    changed["quality_assessment_version"] = "quality-v2"
    with pytest.raises(ValueError, match="1 unexplained losses"):
        reconcile([changed], [old], [work("one", eligible=True, outlier=False)], [old])


def test_department_change_cannot_inherit_prior_region_decision():
    old = observation("person", "one")
    current = work("one", eligible=False, outlier=True)
    current["department_ids"] = ["bme"]
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained losses"):
        reconcile([current], [old], [work("one", eligible=True, outlier=False)], [old])


def test_department_order_does_not_change_region_identity():
    old = observation("person", "one")
    prior = work("one", eligible=True, outlier=False)
    prior["department_ids"] = ["bio", "bme"]
    current = work("one", eligible=False, outlier=True)
    current["department_ids"] = ["bme", "bio"]
    result, receipt = reconcile([current], [old], [prior], [old])
    assert result.map_eligible.tolist() == [True]
    assert receipt["prior_eligible_overrides"] == 1


def test_published_authorship_is_guarded_even_without_included_department():
    old = observation("person", "one")
    _, receipt = reconcile(
        [work("one", eligible=False, outlier=True, department=False)],
        [old],
        [work("one", eligible=True, outlier=False, department=False)],
        [old],
    )
    assert receipt["prior_visible_observations"] == 1
    assert receipt["current_visible_observations"] == 1


def test_identifiers_and_source_fingerprint_are_strict():
    with pytest.raises(ValueError, match="immutable"):
        require_revision("main")
    assert require_revision(REVISION) == REVISION
    row = observation("person", "one")
    assert len(_fingerprint(row)) == 64
    assert _fingerprint({**row, "year": 2020.0}) == _fingerprint(row)
    assert _fingerprint({**row, "year": float("nan")}) != _fingerprint(row)
    assert _value(None) is None
    assert _value(numpy.asarray(["one", "two"])) == ["one", "two"]
    with pytest.raises(ValueError, match="Duplicate profile observation"):
        _indexed_observations([row, row])
    with pytest.raises(ValueError, match="duplicate work identity"):
        _indexed_works([work("one", eligible=True, outlier=False)] * 2)
    with pytest.raises(ValueError, match="Missing or duplicate work identity"):
        _indexed_works([work("", eligible=True, outlier=False)])


def test_exact_mccomb_continuity_keeps_unrelated_loss_gate():
    def row(author_pub_id: str, work_id: str) -> dict:
        return {
            **observation("person-one", work_id),
            "author_pub_id": author_pub_id,
            "collaboration_eligible": True,
        }

    old_loss = row("profile:old", "old-work")
    new_target = row("profile:new", "new-work")
    old_binding = row("profile:retained", "old-binding")
    new_binding = row("profile:retained", "new-binding")
    prior_rows = [old_loss, old_binding]
    current_rows = [new_target, new_binding]
    prior_works = [
        work("old-work", eligible=True, outlier=False),
        work("old-binding", eligible=True, outlier=False),
    ]
    current_works = [
        work("new-work", eligible=True, outlier=False),
        work("new-binding", eligible=True, outlier=False),
    ]
    approval = ExactMcCombContinuity(
        manifest_sha256="reviewed-sha",
        loss_identities=frozenset(
            {("person-one", "profile", "profile:old", "scholar")}
        ),
        binding_changes=frozenset(
            {
                (
                    ("person-one", "profile", "profile:retained", "scholar"),
                    "old-binding",
                    "new-binding",
                    True,
                    True,
                )
            }
        ),
    )
    _, receipt = reconcile_region_decisions(
        pandas.DataFrame(current_works),
        pandas.DataFrame(current_rows),
        prior_works,
        prior_rows,
        prior_revision=REVISION,
        mccomb_continuity=approval,
    )
    assert len(receipt["approved_mccomb_losses"]) == 1
    assert len(receipt["approved_mccomb_binding_changes"]) == 1
    assert receipt["mccomb_continuity_sha256"] == "reviewed-sha"
    assert receipt["unexplained_losses"] == []
    assert receipt["unapproved_mapped_binding_changes"] == []

    unrelated = row("profile:unrelated", "unrelated-work")
    with pytest.raises(RegionReleaseDiffError) as error:
        reconcile_region_decisions(
            pandas.DataFrame(current_works),
            pandas.DataFrame(current_rows),
            [*prior_works, work("unrelated-work", eligible=True, outlier=False)],
            [*prior_rows, unrelated],
            prior_revision=REVISION,
            mccomb_continuity=approval,
        )
    assert error.value.receipt["unexplained_losses"] == [
        "person-one/profile/profile:unrelated/scholar"
    ]

    with pytest.raises(RegionReleaseDiffError) as error:
        reconcile_region_decisions(
            pandas.DataFrame(
                [*current_works, work("old-work", eligible=True, outlier=False)]
            ),
            pandas.DataFrame([old_loss, new_binding]),
            prior_works,
            prior_rows,
            prior_revision=REVISION,
            mccomb_continuity=approval,
        )
    assert error.value.receipt["unused_mccomb_loss_approvals"] == [
        "person-one/profile/profile:old/scholar"
    ]
