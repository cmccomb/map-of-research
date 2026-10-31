"""Adversarial coverage for exact-parent additive person preservation."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy
import pandas
import pyarrow.parquet as parquet
import pytest

from map_of_research.prior_person_continuity import (
    COUNT_FIELDS,
    HODA_PERSON_ID,
    HODA_PROVENANCE_CHANGES,
    REQUIRED_PEOPLE_FIELDS,
    _indexed_people,
    _new_selected_profiles,
    _normal,
    _same,
    read_pinned_prior_people,
    require_prior_person_continuity,
    require_staged_people_equal,
)

PARENT = "9" * 40
SOURCE = "c" * 64
PRODUCER = "e" * 40
REGISTRY = "b" * 64
PEOPLE_SHA = "a" * 64


def _person(person_id: str) -> dict:
    row = dict.fromkeys(REQUIRED_PEOPLE_FIELDS, "")
    row.update(
        person_id=person_id,
        primary_academic_home_type="department",
        primary_academic_home_id="mld",
        primary_department_id="mld",
        memberships=[{"department_id": "mld", "included": True}],
        included_department_ids=["mld"],
        included_school_ids=["scs"],
    )
    row.update(dict.fromkeys(COUNT_FIELDS, 0))
    for field, (before, _) in HODA_PROVENANCE_CHANGES.items():
        row[field] = before
    return row


def _review(tmp_path: Path, **edits) -> Path:
    review = {
        "schema_version": 1,
        "decision": "allow_exact_hoda_provenance",
        "prior_revision": PARENT,
        "parent_people_sha256": PEOPLE_SHA,
        "source_snapshot_sha256": SOURCE,
        "producer_commit": PRODUCER,
        "producer_people_registry_sha256": REGISTRY,
        "person_id": HODA_PERSON_ID,
        "reviewed_at": "2026-10-31",
        "rationale": "The owner CV resolves the reviewed MLD tenure home.",
        "changes": {
            field: {"before": before, "after": after}
            for field, (before, after) in HODA_PROVENANCE_CHANGES.items()
        },
    }
    review.update(edits)
    path = tmp_path / "scholar-prior-person-continuity-review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    return path


def _check(
    prior,
    current,
    *,
    review_path=None,
    prior_profiles=(),
    current_profiles=(),
    current_authorships=(),
    current_works=(),
    release_receipt=None,
    **changes,
):
    kwargs = {
        "prior_revision": PARENT,
        "parent_people_sha256": PEOPLE_SHA,
        "source_snapshot_sha256": SOURCE,
        "producer_commit": PRODUCER,
        "producer_people_registry_sha256": REGISTRY,
        "review_path": review_path,
    }
    kwargs.update(changes)
    return require_prior_person_continuity(
        prior,
        current,
        prior_profiles,
        current_profiles,
        current_authorships,
        current_works,
        release_receipt,
        **kwargs,
    )


def test_unchanged_schema7_and_schema8_people_pass_without_review():
    parent = [_person(HODA_PERSON_ID), _person("person-other")]
    for schema in (7, 8):
        candidate = copy.deepcopy(parent)
        if schema == 8:
            for row in parent + candidate:
                row["future_schema_marker"] = None
        receipt = _check(parent, candidate)
        assert receipt["prior_person_count"] == 2
        assert receipt["review_sha256"] is None
        assert receipt["changed_people"] == {}


def test_arrow_pandas_null_list_and_scalar_wrappers_are_only_physical_changes():
    parent = [_person(HODA_PERSON_ID)]
    candidate = copy.deepcopy(parent)
    parent[0]["notes"] = None
    candidate[0]["notes"] = float("nan")
    candidate[0]["included_department_ids"] = numpy.array(["mld"])
    candidate[0]["memberships"] = tuple(parent[0]["memberships"])
    candidate[0]["publication_observation_count"] = numpy.int64(0)
    assert _normal(numpy.int64(2)) == 2
    assert _normal(float("nan")) is None
    assert _same(parent[0]["memberships"], candidate[0]["memberships"])
    assert _check(parent, candidate)["changed_people"] == {}
    assert not _same(True, 1)


def test_prior_selected_identity_multiplicity_is_preserved():
    person = _person("person-other")
    profile = {
        "person_id": "person-other",
        "scholar_id": "scholar-other",
        "author_pub_id": "scholar-other:one",
        "source_type": "scholar",
        "work_id": "work-one",
    }
    assert (
        _check(
            [person],
            [person],
            prior_profiles=[profile, profile],
            current_profiles=[profile, profile],
        )["status"]
        == "pass"
    )
    with pytest.raises(ValueError, match="loses published profiles"):
        _check(
            [person],
            [person],
            prior_profiles=[profile, profile],
            current_profiles=[profile],
        )
    with pytest.raises(ValueError, match="duplicate source identities"):
        _check(
            [person],
            [person],
            prior_profiles=[profile, profile],
            current_profiles=[profile, profile, profile],
        )
    selected_receipt = [
        {
            "cache_scope": "reviewed_public_subset",
            "person_id": "person-other",
            "scholar_id": "scholar-other",
            "selected_author_pub_ids": ["scholar-other:one"],
        }
    ]
    with pytest.raises(ValueError, match="duplicate source identities"):
        _check(
            [person],
            [person],
            current_profiles=[profile, profile],
            release_receipt=selected_receipt,
        )


@pytest.mark.parametrize("rows", [[], [None], [{"person_id": ""}]])
def test_invalid_or_empty_parent_people_fail(rows):
    with pytest.raises(ValueError, match="rows|schema|IDs"):
        _indexed_people(rows, label="Pinned parent")


def test_exact_hoda_four_field_review_passes_without_hoda_observations(tmp_path):
    prior = [_person(HODA_PERSON_ID), _person("person-other")]
    current = copy.deepcopy(prior)
    for field, (_, after) in HODA_PROVENANCE_CHANGES.items():
        current[0][field] = after
    receipt = _check(prior, current, review_path=_review(tmp_path))
    assert set(receipt["changed_people"][HODA_PERSON_ID]) == set(
        HODA_PROVENANCE_CHANGES
    )
    assert len(receipt["review_sha256"]) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(display_name="Wrong name"),
        lambda row: row.update(primary_academic_home_id="s3d"),
        lambda row: row.update(included_department_ids=["s3d"]),
        lambda row: row.update(memberships=[]),
        lambda row: row.pop("primary_academic_home_basis"),
        lambda row: row.update(extra_unreviewed_field="leak"),
    ],
)
def test_unrelated_or_missing_prior_person_field_fails(mutation):
    prior = [_person(HODA_PERSON_ID), _person("person-other")]
    current = copy.deepcopy(prior)
    mutation(current[1])
    with pytest.raises(ValueError, match="people schema|field changed|columns"):
        _check(prior, current)


def test_missing_duplicate_and_changed_parent_ids_fail():
    prior = [_person(HODA_PERSON_ID), _person("person-other")]
    with pytest.raises(ValueError, match="identities"):
        _check(prior, prior[:1])
    with pytest.raises(ValueError, match="duplicated"):
        _check(prior, [*prior, prior[0]])
    prior[0].pop("primary_department_source_url")
    with pytest.raises(ValueError, match="schema"):
        _check(prior, prior)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("prior_revision", "f" * 40),
        ("parent_people_sha256", "f" * 64),
        ("source_snapshot_sha256", "f" * 64),
        ("producer_commit", "f" * 40),
        ("producer_people_registry_sha256", "f" * 64),
        ("person_id", "person-other"),
        ("schema_version", 2),
    ],
)
def test_stale_review_binding_fails(tmp_path, field, bad):
    prior = [_person(HODA_PERSON_ID)]
    current = copy.deepcopy(prior)
    for key, (_, after) in HODA_PROVENANCE_CHANGES.items():
        current[0][key] = after
    with pytest.raises(ValueError, match="stale|binding"):
        _check(prior, current, review_path=_review(tmp_path, **{field: bad}))


def test_hoda_missing_partial_wrong_and_unused_review_fail(tmp_path):
    prior = [_person(HODA_PERSON_ID)]
    current = copy.deepcopy(prior)
    for field, (_, after) in HODA_PROVENANCE_CHANGES.items():
        current[0][field] = after
    with pytest.raises(ValueError, match="without review"):
        _check(prior, current)
    wrong = _review(tmp_path)
    payload = json.loads(wrong.read_text())
    payload["changes"].pop("primary_department_verified_at")
    wrong.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="four Hoda"):
        _check(prior, current, review_path=wrong)
    current[0]["primary_department_verified_at"] = "wrong"
    with pytest.raises(ValueError, match="without review"):
        _check(prior, current, review_path=_review(tmp_path))
    with pytest.raises(ValueError, match="unused"):
        _check(prior, copy.deepcopy(prior), review_path=_review(tmp_path))
    changed = json.loads(_review(tmp_path).read_text())
    changed["changes"]["primary_department_verified_at"]["after"] = "2026-11-01"
    path = _review(tmp_path)
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="evidence values"):
        _check(prior, current, review_path=path)


def test_review_rejects_bad_date_rationale_extra_and_missing_file(tmp_path):
    prior = [_person(HODA_PERSON_ID)]
    for edits, message in (
        ({"reviewed_at": "yesterday"}, "date"),
        ({"rationale": "short"}, "rationale"),
        ({"extra": "unreviewed"}, "binding"),
    ):
        with pytest.raises(ValueError, match=message):
            _check(prior, prior, review_path=_review(tmp_path, **edits))
    with pytest.raises(ValueError, match="cannot be read"):
        _check(prior, prior, review_path=tmp_path / "missing.json")
    path = _review(tmp_path)
    path.write_text("{invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be read"):
        _check(prior, prior, review_path=path)
    for bad in ({"producer_commit": "short"}, {"source_snapshot_sha256": "short"}):
        with pytest.raises(ValueError, match="binding"):
            _check(prior, prior, review_path=_review(tmp_path), **bad)


@pytest.mark.parametrize(
    "bad_receipt",
    [
        {"cache_scope": "reviewed_public_subset", "person_id": None},
        {
            "cache_scope": "reviewed_public_subset",
            "person_id": "person-new",
            "scholar_id": "s",
            "selected_author_pub_ids": "not-a-list",
        },
        {
            "cache_scope": "reviewed_public_subset",
            "person_id": "person-new",
            "scholar_id": "s",
            "selected_author_pub_ids": ["s:a", "s:a"],
        },
        {
            "cache_scope": "reviewed_public_subset",
            "person_id": "person-new",
            "scholar_id": "s",
            "selected_author_pub_ids": [1],
        },
    ],
)
def test_selected_receipt_malformed_values_fail(bad_receipt):
    with pytest.raises(ValueError, match="receipt"):
        _new_selected_profiles([], [], [bad_receipt])


def test_selected_receipt_duplicate_person_and_nonselected_state():
    item = {
        "cache_scope": "reviewed_public_subset",
        "person_id": "person-new",
        "scholar_id": "s",
        "selected_author_pub_ids": ["s:a"],
    }
    with pytest.raises(ValueError, match="receipt"):
        _new_selected_profiles([], [], [item, item])
    assert _new_selected_profiles([], [], [{"cache_scope": "full"}]) == {}


def test_new_source_row_requires_selected_scholar_receipt_and_complete_columns():
    person = _person("person-new")
    profile = {
        "person_id": "person-new",
        "scholar_id": "s",
        "author_pub_id": "s:a",
        "source_type": "scholar",
        "work_id": "w",
    }
    with pytest.raises(ValueError, match="Prior profile rows"):
        _check([person], [person], prior_profiles=[{"person_id": "person-new"}])
    with pytest.raises(ValueError, match="Candidate profile rows"):
        _check([person], [person], current_profiles=[{"person_id": "person-new"}])
    with pytest.raises(ValueError, match="selected-profile receipt"):
        _check([person], [person], current_profiles=[profile])
    receipt = [
        {
            "cache_scope": "reviewed_public_subset",
            "person_id": "person-new",
            "scholar_id": "s",
            "selected_author_pub_ids": ["s:a"],
        }
    ]
    curated = dict(profile, source_type="curated_verified")
    with pytest.raises(ValueError, match="selected-profile receipt"):
        _check([person], [person], current_profiles=[curated], release_receipt=receipt)


def test_selected_profile_count_and_binding_growth_must_match_receipt():
    prior = [_person("person-new")]
    current = copy.deepcopy(prior)
    current[0].update(
        publication_observation_count=1,
        scholar_observation_count=1,
        unique_work_count=1,
        mapped_work_count=1,
    )
    profile = {
        "person_id": "person-new",
        "scholar_id": "scholar-new",
        "author_pub_id": "scholar-new:one",
        "source_type": "scholar",
        "work_id": "work-one",
    }
    receipt = [
        {
            "cache_scope": "reviewed_public_subset",
            "person_id": "person-new",
            "scholar_id": "scholar-new",
            "selected_author_pub_ids": ["scholar-new:one"],
        }
    ]
    authorships = [{"person_id": "person-new", "work_id": "work-one"}]
    works = [
        {
            "work_id": "work-one",
            "map_eligible": True,
            "memberships": [{"person_id": "person-new"}],
        }
    ]

    def check(
        *, candidate=current, reviews=receipt, auth=authorships, pubs=None, graph=works
    ):
        return require_prior_person_continuity(
            prior,
            candidate,
            [],
            [profile] if pubs is None else pubs,
            auth,
            graph,
            reviews,
            prior_revision=PARENT,
            parent_people_sha256=PEOPLE_SHA,
            source_snapshot_sha256=SOURCE,
            producer_commit=PRODUCER,
            producer_people_registry_sha256=REGISTRY,
            review_path=None,
        )

    assert check()["changed_people"]["person-new"] == [
        "mapped_work_count",
        "publication_observation_count",
        "scholar_observation_count",
        "unique_work_count",
    ]
    with pytest.raises(ValueError, match="selected-profile"):
        check(reviews=[])
    bad = copy.deepcopy(current)
    bad[0]["publication_observation_count"] = 2
    with pytest.raises(ValueError, match="count growth"):
        check(candidate=bad)
    bad = copy.deepcopy(current)
    bad[0]["unique_work_count"] = 2
    with pytest.raises(ValueError, match="count growth"):
        check(candidate=bad)
    with pytest.raises(ValueError, match="work growth"):
        check(auth=[*authorships, {"person_id": "person-new", "work_id": "other"}])
    extra_mapped = {
        "work_id": "other",
        "map_eligible": True,
        "memberships": [{"person_id": "person-new"}],
    }
    with pytest.raises(ValueError, match="mapped growth"):
        check(graph=[*works, extra_mapped])
    bad = copy.deepcopy(current)
    bad[0]["mapped_work_count"] = 2
    with pytest.raises(ValueError, match="count growth"):
        check(candidate=bad)
    bad = copy.deepcopy(current)
    bad[0]["scholar_observation_count"] = True
    with pytest.raises(ValueError, match="counts decrease or change type"):
        check(candidate=bad)
    bad = copy.deepcopy(current)
    bad[0]["publication_observation_count"] = -1
    with pytest.raises(ValueError, match="counts decrease"):
        check(candidate=bad)
    with pytest.raises(ValueError, match="loses published profiles"):
        require_prior_person_continuity(
            prior,
            current,
            [profile],
            [],
            authorships,
            works,
            receipt,
            prior_revision=PARENT,
            parent_people_sha256=PEOPLE_SHA,
            source_snapshot_sha256=SOURCE,
            producer_commit=PRODUCER,
            producer_people_registry_sha256=REGISTRY,
            review_path=None,
        )


def test_unselected_person_count_change_and_bad_candidate_relationships_fail():
    prior = [_person("person-other")]
    current = copy.deepcopy(prior)
    current[0]["unique_work_count"] = 1
    with pytest.raises(ValueError, match="selected profile"):
        _check(prior, current)
    with pytest.raises(ValueError, match="authorships"):
        _check(prior, prior, current_authorships=[{"person_id": "person-other"}])
    with pytest.raises(ValueError, match="works lack"):
        _check(prior, prior, current_works=[{"work_id": "w"}])
    with pytest.raises(ValueError, match="invalid membership"):
        _check(
            prior,
            prior,
            current_works=[{"work_id": "w", "map_eligible": True, "memberships": [{}]}],
        )
    assert (
        _check(
            prior,
            prior,
            current_works=[{"work_id": "w", "map_eligible": False, "memberships": []}],
        )["status"]
        == "pass"
    )


def test_staged_parquet_all_columns_equal_or_fail(tmp_path):
    expected = [_person(HODA_PERSON_ID), _person("person-other")]
    path = tmp_path / "people.parquet"
    pandas.DataFrame(expected).to_parquet(path, index=False)
    require_staged_people_equal(path, expected)
    changed = copy.deepcopy(expected)
    changed[1]["memberships"] = []
    pandas.DataFrame(changed).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="differ from prewrite"):
        require_staged_people_equal(path, expected)
    pandas.DataFrame(expected).drop(columns="notes").to_parquet(path, index=False)
    with pytest.raises(ValueError, match="schema"):
        require_staged_people_equal(path, expected)
    pandas.DataFrame(expected[:1]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="IDs"):
        require_staged_people_equal(path, expected)


def test_pinned_reader_requires_full_people_schema_and_immutable_revision(
    tmp_path, monkeypatch
):
    path = tmp_path / "people/train-00000-of-00001.parquet"
    path.parent.mkdir()
    pandas.DataFrame([_person(HODA_PERSON_ID)]).to_parquet(path, index=False)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda *args, **kwargs: str(tmp_path)),
    )
    rows, digest = read_pinned_prior_people(
        repo_id="repo", prior_revision=PARENT, hf_token=""
    )
    assert rows[0]["person_id"] == HODA_PERSON_ID
    assert len(digest) == 64
    with pytest.raises(ValueError, match="immutable"):
        read_pinned_prior_people(repo_id="repo", prior_revision="main", hf_token="")
    parquet.write_table(
        parquet.read_table(path).drop(["primary_department_verified_at"]), path
    )
    with pytest.raises(ValueError, match="schema"):
        read_pinned_prior_people(repo_id="repo", prior_revision=PARENT, hf_token="")
    valid = [_person(HODA_PERSON_ID)]
    monkeypatch.setattr(
        parquet,
        "read_table",
        lambda path: SimpleNamespace(
            to_pylist=lambda: valid,
            schema=SimpleNamespace(names=["person_id"]),
        ),
    )
    with pytest.raises(ValueError, match="Parquet schema"):
        read_pinned_prior_people(repo_id="repo", prior_revision=PARENT, hf_token="")
    (path.parent / "train-00001-of-00002.parquet").write_bytes(b"extra")
    with pytest.raises(ValueError, match="one reviewed shard"):
        read_pinned_prior_people(repo_id="repo", prior_revision=PARENT, hf_token="")


def test_publisher_rejects_out_of_source_prior_person_review_path(tmp_path):
    from map_of_research import publisher

    with pytest.raises(ValueError, match="source-relative"):
        publisher.publish_snapshot(
            tmp_path / "snapshots/snapshot.parquet",
            tmp_path / "snapshots/manifest.json",
            hf_token="",
            prior_revision=PARENT,
            prior_person_continuity_review_path=tmp_path / "elsewhere/review.json",
        )
    with pytest.raises(ValueError, match="source-relative"):
        publisher.publish_snapshot(
            tmp_path / "snapshots/snapshot.parquet",
            tmp_path / "snapshots/manifest.json",
            hf_token="",
            prior_revision=PARENT,
            additive_placement_review_path=tmp_path / "placement.json",
            prior_person_continuity_review_path=tmp_path / "elsewhere/review.json",
        )
