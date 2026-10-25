"""Complete Scholar sources can hold exact uncertain rows without deleting them."""

import json
from dataclasses import replace

import pandas
import pytest

from map_of_research.contexts import ReviewedContext
from map_of_research.dataset import build_dataset_tables
from map_of_research.io import sha256_file
from map_of_research.profile_releases import ProfileRelease
from map_of_research.provisional_holds import (
    DEFAULT_PROVISIONAL_HOLDS_PATH,
    ProvisionalHold,
    ProvisionalHoldProfile,
    hold_ids_sha256,
    load_provisional_holds,
    require_no_provisional_hold_contributions,
    validate_provisional_holds,
)
from map_of_research.registry import load_registry
from tests.registry_helpers import write_registry

SCHOLAR = "scholar-one"
PERSON = "person-one"
HELD = f"{SCHOLAR}:held"
RETAINED = f"{SCHOLAR}:retained"
URL = f"https://scholar.google.com/citations?user={SCHOLAR}&citation_for_view={HELD}"


def _hold(**changes):
    return replace(
        ProvisionalHold(
            scholar_id=SCHOLAR,
            author_pub_id=HELD,
            capture_row_index=1,
            reason="Full primary byline is unavailable.",
            evidence_url=URL,
            reviewed_at="2026-10-25",
        ),
        **changes,
    )


def _source(tmp_path, *, rows=None):
    cache_dir = tmp_path / "data" / "authors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{SCHOLAR}.json"
    path.write_text(
        json.dumps(
            {
                "scholar_id": SCHOLAR,
                "publication_count": 2,
                "browser_capture": {
                    "coverage": "complete_visible_profile",
                    "more_remain": False,
                },
                "publications": rows
                if rows is not None
                else [
                    {
                        "capture_row_index": 1,
                        "author_pub_id": HELD,
                        "pub_url": URL,
                    },
                    {
                        "capture_row_index": 2,
                        "author_pub_id": RETAINED,
                        "pub_url": "https://scholar.google.com/retained",
                    },
                ],
            }
        )
    )
    digest = sha256_file(path)
    profile = ProvisionalHoldProfile(PERSON, SCHOLAR, 2, digest, (_hold(),))
    decision = ProfileRelease(
        PERSON,
        SCHOLAR,
        "released",
        2,
        digest,
        "Release all but a source-bound hold.",
        "2026-10-25",
        1,
        hold_ids_sha256({HELD}),
    )
    receipt = [
        {
            "scholar_id": SCHOLAR,
            "state": "released",
            "source_observation_rows": 2,
            "cache_sha256": digest,
        }
    ]
    return cache_dir, profile, decision, receipt


def _validate(cache_dir, profile, decision, receipt, **kwargs):
    return validate_provisional_holds(
        (profile,),
        (decision,),
        receipt,
        cache_dir=cache_dir,
        excluded_ids=kwargs.get("excluded_ids", set()),
        confirmed_ids=kwargs.get("confirmed_ids", set()),
    )


def test_checked_in_collins_hold_is_exact_and_separate_from_exclusions():
    (profile,) = load_provisional_holds(DEFAULT_PROVISIONAL_HOLDS_PATH)
    assert profile.person_id == "person-terrencejcollins-61cf6ed9"
    assert profile.scholar_id == "0e1jwkkAAAAJ"
    assert profile.expected_source_rows == 361
    assert profile.cache_sha256 == (
        "71ad4333c8f43adb4b2438be05ee3d8365c46a6da3f67b71b61ed740300bc83c"
    )
    assert profile.expected_reviewed_exclusion_rows == 31
    assert profile.reviewed_exclusion_ids_sha256 == (
        "2834d16fcfa1a5da902b925b6f2d3b2cdc3d7e1c3e974b873370b74805baf979"
    )
    assert [
        (item.capture_row_index, item.author_pub_id)
        for item in profile.held_observations
    ] == [
        (214, "0e1jwkkAAAAJ:vRqMK49ujn8C"),
        (274, "0e1jwkkAAAAJ:cK4Rrx0J3m0C"),
        (299, "0e1jwkkAAAAJ:BJbdYPG6LGMC"),
        (355, "0e1jwkkAAAAJ:yqoGN6RLRZoC"),
    ]
    assert hold_ids_sha256(
        {item.author_pub_id for item in profile.held_observations}
    ) == ("78b7f2f4e7a7beb6ecdd9e67e2720135ef5225d6c4b9443a248a621359e55269")


def _manifest(tmp_path, profiles):
    path = tmp_path / "holds.json"
    path.write_text(json.dumps({"schema_version": 1, "profiles": profiles}))
    return path


def _entry():
    return {
        "person_id": PERSON,
        "scholar_id": SCHOLAR,
        "expected_source_rows": 2,
        "cache_sha256": "a" * 64,
        "expected_reviewed_exclusion_rows": 0,
        "reviewed_exclusion_ids_sha256": "",
        "held_observations": [
            {
                "capture_row_index": 1,
                "author_pub_id": HELD,
                "reason": "Exact author identity unresolved.",
                "evidence_url": URL,
                "reviewed_at": "2026-10-25",
            }
        ],
    }


def test_loader_accepts_empty_manifest_for_future_retirement(tmp_path):
    assert load_provisional_holds(_manifest(tmp_path, [])) == ()


@pytest.mark.parametrize(
    "body,message",
    [
        ([], "schema"),
        ({"schema_version": 2, "profiles": []}, "schema"),
        ({"schema_version": 1}, "profile list"),
        ({"schema_version": 1, "profiles": [None]}, "must be an object"),
        ({"schema_version": 1, "profiles": [_entry(), _entry()]}, "source fields"),
    ],
)
def test_loader_rejects_structure(tmp_path, body, message):
    path = tmp_path / "holds.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match=message):
        load_provisional_holds(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("person_id", "invalid"),
        ("scholar_id", ""),
        ("expected_source_rows", True),
        ("expected_source_rows", 0),
        ("cache_sha256", "bad"),
        ("expected_reviewed_exclusion_rows", None),
        ("expected_reviewed_exclusion_rows", True),
        ("expected_reviewed_exclusion_rows", -1),
        ("expected_reviewed_exclusion_rows", 1),
        ("expected_reviewed_exclusion_rows", 2),
        ("reviewed_exclusion_ids_sha256", None),
        ("reviewed_exclusion_ids_sha256", "bad"),
        ("held_observations", []),
        ("held_observations", None),
    ],
)
def test_loader_rejects_bad_profile_field(tmp_path, field, value):
    entry = _entry()
    entry[field] = value
    with pytest.raises(ValueError, match="source fields"):
        load_provisional_holds(_manifest(tmp_path, [entry]))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("capture_row_index", 0, "invalid identity"),
        ("capture_row_index", 3, "invalid identity"),
        ("capture_row_index", True, "invalid identity"),
        ("author_pub_id", "other:held", "invalid identity"),
        ("author_pub_id", f"{SCHOLAR}:", "invalid identity"),
        ("reason", " ", "invalid identity"),
        ("evidence_url", "http://example.test", "dated HTTPS"),
        ("reviewed_at", "tomorrow", "dated HTTPS"),
    ],
)
def test_loader_rejects_bad_hold_field(tmp_path, field, value, message):
    entry = _entry()
    entry["held_observations"][0][field] = value
    with pytest.raises(ValueError, match=message):
        load_provisional_holds(_manifest(tmp_path, [entry]))


def test_loader_rejects_duplicate_held_ids_and_indices(tmp_path):
    entry = _entry()
    entry["held_observations"].append(entry["held_observations"][0])
    with pytest.raises(ValueError, match="invalid identity"):
        load_provisional_holds(_manifest(tmp_path, [entry]))
    entry["held_observations"][1] = {
        **entry["held_observations"][0],
        "author_pub_id": RETAINED,
    }
    with pytest.raises(ValueError, match="invalid identity"):
        load_provisional_holds(_manifest(tmp_path, [entry]))
    entry["held_observations"][1]["capture_row_index"] = 2
    assert (
        len(load_provisional_holds(_manifest(tmp_path, [entry]))[0].held_observations)
        == 2
    )


def test_loader_rejects_nonobject_held_observation(tmp_path):
    entry = _entry()
    entry["held_observations"] = [None]
    with pytest.raises(ValueError, match="must be an object"):
        load_provisional_holds(_manifest(tmp_path, [entry]))


def test_complete_source_and_exact_hold_validate(tmp_path):
    cache_dir, profile, decision, receipt = _source(tmp_path)
    holds, audit = _validate(cache_dir, profile, decision, receipt)
    assert holds == {(SCHOLAR, HELD): _hold()}
    assert audit[0]["held_observation_rows"] == 1
    assert audit[0]["held_ids_sha256"] == hold_ids_sha256({HELD})
    assert audit[0]["reviewed_exclusion_rows"] == 0
    assert audit[0]["reviewed_exclusion_ids_sha256"] == ""


@pytest.mark.parametrize("change", ["missing", "extra", "different"])
def test_reviewed_exclusion_pin_fails_on_drift(tmp_path, change):
    cache_dir, profile, decision, receipt = _source(tmp_path)
    reviewed_id = RETAINED
    profile = replace(
        profile,
        expected_reviewed_exclusion_rows=1,
        reviewed_exclusion_ids_sha256=hold_ids_sha256({reviewed_id}),
    )
    excluded = {(SCHOLAR, reviewed_id)}
    holds, audit = _validate(
        cache_dir, profile, decision, receipt, excluded_ids=excluded
    )
    assert set(holds) == {(SCHOLAR, HELD)}
    assert audit[0]["reviewed_exclusion_rows"] == 1
    if change == "missing":
        excluded.clear()
    elif change == "extra":
        excluded.add((SCHOLAR, f"{SCHOLAR}:other"))
    else:
        excluded = {(SCHOLAR, f"{SCHOLAR}:different")}
    with pytest.raises(ValueError, match="reviewed exclusions differ"):
        _validate(cache_dir, profile, decision, receipt, excluded_ids=excluded)


def test_reviewed_exclusion_must_belong_to_pinned_source(tmp_path):
    cache_dir, profile, decision, receipt = _source(tmp_path)
    absent_id = f"{SCHOLAR}:absent"
    profile = replace(
        profile,
        expected_reviewed_exclusion_rows=1,
        reviewed_exclusion_ids_sha256=hold_ids_sha256({absent_id}),
    )
    with pytest.raises(ValueError, match="exclusions lack source IDs"):
        _validate(
            cache_dir,
            profile,
            decision,
            receipt,
            excluded_ids={(SCHOLAR, absent_id)},
        )


@pytest.mark.parametrize(
    "change",
    [
        "missing_profile",
        "state",
        "person",
        "count",
        "cache",
        "hold_count",
        "hold_ids",
        "receipt_missing",
        "receipt_state",
        "receipt_count",
        "receipt_cache",
    ],
)
def test_release_contract_rejects_drift(tmp_path, change):
    cache_dir, profile, decision, receipt = _source(tmp_path)
    if change == "missing_profile":
        with pytest.raises(ValueError, match="profiles differ"):
            validate_provisional_holds(
                (),
                (decision,),
                receipt,
                cache_dir=cache_dir,
                excluded_ids=set(),
                confirmed_ids=set(),
            )
        return
    if change == "state":
        decision = replace(decision, state="held")
    elif change == "person":
        decision = replace(decision, person_id="person-other")
    elif change == "count":
        decision = replace(decision, expected_source_rows=3)
    elif change == "cache":
        decision = replace(decision, cache_sha256="0" * 64)
    elif change == "hold_count":
        decision = replace(decision, expected_provisional_hold_rows=2)
    elif change == "hold_ids":
        decision = replace(decision, provisional_hold_ids_sha256="0" * 64)
    elif change == "receipt_missing":
        receipt = []
    elif change == "receipt_state":
        receipt[0]["state"] = "held"
    elif change == "receipt_count":
        receipt[0]["source_observation_rows"] = 1
    else:
        receipt[0]["cache_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="release source differs"):
        _validate(cache_dir, profile, decision, receipt)


def test_unexpected_hold_profile_rejected(tmp_path):
    cache_dir, profile, decision, receipt = _source(tmp_path)
    with pytest.raises(ValueError, match="profiles differ"):
        _validate(
            cache_dir,
            profile,
            replace(decision, expected_provisional_hold_rows=0),
            receipt,
        )


@pytest.mark.parametrize("change", ["missing", "hash", "count", "indices", "id", "url"])
def test_cache_or_index_drift_fails_before_upload(tmp_path, change):
    cache_dir, profile, decision, receipt = _source(tmp_path)
    path = cache_dir / f"{SCHOLAR}.json"
    if change == "missing":
        path.unlink()
    elif change == "hash":
        path.write_text(path.read_text() + " ")
    else:
        cache = json.loads(path.read_text())
        if change == "count":
            cache["publications"].pop()
        elif change == "indices":
            cache["publications"][1]["capture_row_index"] = 1
        elif change == "id":
            cache["publications"][0]["author_pub_id"] = f"{SCHOLAR}:changed"
        else:
            cache["publications"][0]["pub_url"] = "https://example.test/different"
        path.write_text(json.dumps(cache))
        digest = sha256_file(path)
        profile = replace(profile, cache_sha256=digest)
        decision = replace(decision, cache_sha256=digest)
        receipt[0]["cache_sha256"] = digest
    message = (
        "cache is missing or changed"
        if change in {"missing", "hash"}
        else "cache row count"
        if change == "count"
        else "cache indices"
        if change == "indices"
        else "ID, evidence"
    )
    with pytest.raises(ValueError, match=message):
        _validate(cache_dir, profile, decision, receipt)


@pytest.mark.parametrize("conflict", ["excluded", "confirmed"])
def test_hold_cannot_reuse_exact_exclusion_or_positive_attribution(tmp_path, conflict):
    cache_dir, profile, decision, receipt = _source(tmp_path)
    if conflict == "excluded":
        profile = replace(
            profile,
            expected_reviewed_exclusion_rows=1,
            reviewed_exclusion_ids_sha256=hold_ids_sha256({HELD}),
        )
    with pytest.raises(ValueError, match="review conflicts"):
        _validate(
            cache_dir,
            profile,
            decision,
            receipt,
            **{f"{conflict}_ids": {(SCHOLAR, HELD)}},
        )


def test_dataset_retains_raw_hold_but_omits_its_map_contribution(tmp_path):
    people = [
        {
            "person_id": person_id,
            "display_name": name,
            "scholar_id": scholar_id,
            "scholar_id_source_url": "https://example.test/scholar",
            "scholar_id_verified_at": "2026-10-25",
            "orcid": "",
            "homepage_url": "",
            "notes": "",
        }
        for person_id, name, scholar_id in (
            (PERSON, "One Person", SCHOLAR),
            ("person-two", "Two Person", "scholar-two"),
        )
    ]
    memberships = [
        {
            "person_id": person["person_id"],
            "department_id": "ece",
            "role": "faculty",
            "included": "true",
            "legacy_label": "",
            "source_url": "https://example.test/roster",
            "verified_at": "2026-10-25",
        }
        for person in people
    ]
    registry = load_registry(
        *write_registry(tmp_path / "registry", people=people, memberships=memberships)
    )
    rows = []
    for scholar_id, person_id, author_pub_id in (
        (SCHOLAR, PERSON, HELD),
        ("scholar-two", "person-two", "scholar-two:retained"),
    ):
        rows.append(
            {
                "scholar_id": scholar_id,
                "person_id": person_id,
                "display_name": registry.people_by_id[person_id].display_name,
                "faculty": registry.people_by_id[person_id].display_name,
                "department_ids": ["ece"],
                "memberships": [
                    {"department_id": "ece", "role": "faculty", "included": True}
                ],
                "author_pub_id": author_pub_id,
                "title": "Catalytic oxidation of water using iron complexes",
                "authors": "One Person, Two Person",
                "year": 2020,
                "venue": "Journal 1, 1-4, 2020",
                "citation": "Journal 1, 1-4, 2020",
                "citation_count": 1,
                "source_url": "",
                "source_record_json": "{}",
                "fetched_at_utc": "2026-10-25T00:00:00+00:00",
                "embedding": [0.0, 1.0],
            }
        )
    tables = build_dataset_tables(
        pandas.DataFrame(rows), registry, provisional_holds={(SCHOLAR, HELD): _hold()}
    )
    raw = tables["profile_publications"].set_index("author_pub_id")
    assert len(raw) == 2
    assert raw.loc[HELD, "review_attribution_status"] == "held"
    assert raw.loc[HELD, "review_attribution_evidence_url"] == URL
    assert not bool(raw.loc[HELD, "map_eligible"])
    assert "attribution_provisional_hold" in raw.loc[HELD, "map_exclusion_reasons"]
    assert tables["authorships"]["person_id"].tolist() == ["person-two"]
    (work,) = list(tables["works"].itertuples(index=False))
    assert work.map_eligible
    assert work.person_ids == ["person-two"]
    assert raw.loc[HELD, "observation_id"] not in work.observation_ids
    require_no_provisional_hold_contributions(tables, {(SCHOLAR, HELD): _hold()})

    # Context attached only to the held row cannot change the mapped work type.
    contextual_tables = build_dataset_tables(
        pandas.DataFrame(rows),
        registry,
        provisional_holds={(SCHOLAR, HELD): _hold()},
        contexts={
            (SCHOLAR, HELD): ReviewedContext(
                "edited_book",
                "",
                "Context requiring review",
                "https://example.test/context",
                "2026-10-25",
                "Held contributor",
            )
        },
    )
    contextual_work = contextual_tables["works"].iloc[0]
    assert contextual_work["publication_type"] == "publication"
    assert contextual_work["display_contributors"] == ""
    assert contextual_work["context_evidence_urls"] == []
    require_no_provisional_hold_contributions(
        contextual_tables, {(SCHOLAR, HELD): _hold()}
    )

    # A held CMU observation cannot make an otherwise external work map-related.
    rows[1]["memberships"][0]["included"] = False
    external_tables = build_dataset_tables(
        pandas.DataFrame(rows), registry, provisional_holds={(SCHOLAR, HELD): _hold()}
    )
    assert not bool(external_tables["works"].iloc[0]["has_included_source_profile"])
    require_no_provisional_hold_contributions(
        external_tables, {(SCHOLAR, HELD): _hold()}
    )


def _audit_tables():
    observation_id = "observation-held"
    return {
        "profile_publications": pandas.DataFrame(
            [
                {
                    "scholar_id": SCHOLAR,
                    "author_pub_id": HELD,
                    "observation_id": observation_id,
                    "map_eligible": False,
                    "map_exclusion_reasons": ["attribution_provisional_hold"],
                    "review_attribution_status": "held",
                }
            ]
        ),
        "authorships": pandas.DataFrame(
            columns=["scholar_id", "author_pub_id", "observation_id"]
        ),
        "works": pandas.DataFrame(
            [
                {
                    "map_eligible": False,
                    "observation_ids": [observation_id],
                    "author_pub_ids": [HELD],
                }
            ]
        ),
    }


def test_preupload_guard_allows_raw_only_and_noop():
    tables = _audit_tables()
    require_no_provisional_hold_contributions(tables, {(SCHOLAR, HELD): _hold()})
    require_no_provisional_hold_contributions({}, {})


@pytest.mark.parametrize(
    "change,message",
    [
        ("raw_missing_column", "lack hold audit"),
        ("relationship_missing_column", "lack hold provenance"),
        ("raw_missing_row", "escaped raw map gate"),
        ("raw_mapped", "escaped raw map gate"),
        ("raw_missing_reason", "escaped raw map gate"),
        ("raw_wrong_status", "escaped raw map gate"),
        ("authorship_id", "contributed to a map link"),
        ("authorship_observation", "contributed to a map link"),
        ("mapped_work", "contributed to a map link"),
        ("mapped_work_id", "contributed to a map link"),
    ],
)
def test_preupload_guard_rejects_held_contributions(change, message):
    tables = _audit_tables()
    if change == "raw_missing_column":
        tables["profile_publications"] = tables["profile_publications"].drop(
            columns="map_eligible"
        )
    elif change == "relationship_missing_column":
        tables["authorships"] = tables["authorships"].drop(columns="observation_id")
    elif change == "raw_missing_row":
        tables["profile_publications"] = tables["profile_publications"].iloc[0:0]
    elif change == "raw_mapped":
        tables["profile_publications"].at[0, "map_eligible"] = True
    elif change == "raw_missing_reason":
        tables["profile_publications"].at[0, "map_exclusion_reasons"] = []
    elif change == "raw_wrong_status":
        tables["profile_publications"].at[0, "review_attribution_status"] = ""
    elif change == "authorship_id":
        tables["authorships"] = pandas.DataFrame(
            [{"scholar_id": SCHOLAR, "author_pub_id": HELD, "observation_id": "other"}]
        )
    elif change == "authorship_observation":
        tables["authorships"] = pandas.DataFrame(
            [
                {
                    "scholar_id": "other",
                    "author_pub_id": "other:row",
                    "observation_id": "observation-held",
                }
            ]
        )
    elif change == "mapped_work_id":
        tables["works"].at[0, "map_eligible"] = True
        tables["works"].at[0, "observation_ids"] = []
    else:
        tables["works"].at[0, "map_eligible"] = True
    with pytest.raises(ValueError, match=message):
        require_no_provisional_hold_contributions(tables, {(SCHOLAR, HELD): _hold()})


def test_upload_workflow_passes_trusted_hold_manifest():
    from pathlib import Path

    workflow = Path(".github/workflows/upload.yml").read_text()
    assert (
        "--provisional-holds source/data/curated/scholar-provisional-holds.json"
        in workflow
    )
    assert (
        "--profile-releases source/data/curated/scholar-profile-releases.json"
        in workflow
    )
