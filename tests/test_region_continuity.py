"""Release gates for corpus-dependent weak-region classifications."""

from __future__ import annotations

import copy

import numpy
import pandas
import pytest

from map_of_research.provisional_holds import ProvisionalHold
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


def reconcile(
    current_works,
    current_rows,
    prior_works,
    prior_rows,
    *,
    provisional_holds=None,
):
    return reconcile_region_decisions(
        pandas.DataFrame(current_works),
        pandas.DataFrame(current_rows),
        prior_works,
        prior_rows,
        prior_revision=REVISION,
        provisional_holds=provisional_holds,
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


def test_validated_provisional_hold_can_explain_exact_observation_loss():
    old = observation("person", "one")
    current = {
        **old,
        "map_eligible": False,
        "map_exclusion_reasons": ["attribution_provisional_hold"],
        "review_attribution_status": "held",
        "review_attribution_evidence_url": "https://scholar.google.com/citations?user=profile",
    }
    hold = ProvisionalHold(
        "profile",
        "profile:shared-row",
        1,
        "Original byline unavailable",
        current["review_attribution_evidence_url"],
        "2026-10-25",
    )
    _, receipt = reconcile(
        [work("one", eligible=False, outlier=False)],
        [current],
        [work("one", eligible=True, outlier=False)],
        [old],
        provisional_holds={(hold.scholar_id, hold.author_pub_id): hold},
    )
    assert receipt["reviewed_losses"] == ["person/profile/profile:shared-row/scholar"]


def test_unapproved_provisional_hold_cannot_explain_loss():
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


@pytest.mark.parametrize(
    "change",
    [
        {"map_exclusion_reasons": []},
        {"review_attribution_status": ""},
        {"review_attribution_evidence_url": ""},
        {"review_attribution_evidence_url": "https://other.test/poster"},
    ],
)
def test_incomplete_provisional_hold_cannot_explain_loss(change):
    old = observation("person", "one")
    current = {
        **old,
        "map_eligible": False,
        "map_exclusion_reasons": ["attribution_provisional_hold"],
        "review_attribution_status": "held",
        "review_attribution_evidence_url": "https://scholar.google.com/citations?user=profile",
        **change,
    }
    hold = ProvisionalHold(
        "profile",
        "profile:shared-row",
        1,
        "Original byline unavailable",
        "https://scholar.google.com/citations?user=profile",
        "2026-10-25",
    )
    with pytest.raises(RegionReleaseDiffError, match="1 unexplained losses"):
        reconcile(
            [work("one", eligible=False, outlier=False)],
            [current],
            [work("one", eligible=True, outlier=False)],
            [old],
            provisional_holds={(hold.scholar_id, hold.author_pub_id): hold},
        )


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
