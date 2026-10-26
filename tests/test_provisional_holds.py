"""Complete Scholar sources can hold exact uncertain rows without deleting them."""

import json
from dataclasses import replace

import pandas
import pytest

from map_of_research.contexts import ReviewedContext
from map_of_research.dataset import build_dataset_tables
from map_of_research.exclusions import DEFAULT_EXCLUSIONS_PATH, load_exclusions
from map_of_research.io import sha256_file
from map_of_research.profile_releases import (
    DEFAULT_PROFILE_RELEASES_PATH,
    ProfileRelease,
    load_profile_releases,
)
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
    profiles = load_provisional_holds(DEFAULT_PROVISIONAL_HOLDS_PATH)
    profile = next(item for item in profiles if item.scholar_id == "0e1jwkkAAAAJ")
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
        (141, "0e1jwkkAAAAJ:70eg2SAEIzsC"),
        (214, "0e1jwkkAAAAJ:vRqMK49ujn8C"),
        (274, "0e1jwkkAAAAJ:cK4Rrx0J3m0C"),
        (299, "0e1jwkkAAAAJ:BJbdYPG6LGMC"),
        (350, "0e1jwkkAAAAJ:hsZV8lGYWTMC"),
        (355, "0e1jwkkAAAAJ:yqoGN6RLRZoC"),
        (356, "0e1jwkkAAAAJ:yFnVuubrUp4C"),
        (282, "0e1jwkkAAAAJ:Y5dfb0dijaUC"),
        (278, "0e1jwkkAAAAJ:u-coK7KVo8oC"),
        (247, "0e1jwkkAAAAJ:MhiOAD_qIWkC"),
    ]
    assert hold_ids_sha256(
        {item.author_pub_id for item in profile.held_observations}
    ) == ("b48946eaecc8fdf09d32c892cf4862f69eda4109983d6839dc7cae95c82c9180")


def test_checked_in_sullivan_hold_is_source_bound_and_separate_from_exclusions():
    profile = next(
        item
        for item in load_provisional_holds(DEFAULT_PROVISIONAL_HOLDS_PATH)
        if item.scholar_id == "wXSgFr4AAAAJ"
    )
    assert profile.person_id == "person-sullivan-589c9545"
    assert profile.expected_source_rows == 250
    assert profile.cache_sha256 == (
        "960b296af5cf0c9e3aab6be32844ec2fe2ad7f2a2b3e331cfc4c857f4df57fd7"
    )
    assert profile.expected_reviewed_exclusion_rows == 56
    assert profile.reviewed_exclusion_ids_sha256 == (
        "bc82650651779bda8ea25c141d14af00633ec8795ffd3e2f5c2aef956a0a0e21"
    )
    held_ids = {item.author_pub_id for item in profile.held_observations}
    assert len(held_ids) == 35
    assert hold_ids_sha256(held_ids) == (
        "78fe11269cdda8e270e1c4ec98e1cb24ddec53b5c0a942e79acff00cc9f677ae"
    )
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    exclusion_ids = {
        author_pub_id
        for scholar_id, author_pub_id in exclusions
        if scholar_id == profile.scholar_id
    }
    assert len(exclusion_ids) == profile.expected_reviewed_exclusion_rows
    assert hold_ids_sha256(exclusion_ids) == profile.reviewed_exclusion_ids_sha256
    assert held_ids.isdisjoint(exclusion_ids)


def test_physics_preview_citations_are_bound_to_exact_source_holds():
    profiles = {
        item.scholar_id: item
        for item in load_provisional_holds(DEFAULT_PROVISIONAL_HOLDS_PATH)
    }
    decisions = {
        item.scholar_id: item
        for item in load_profile_releases(DEFAULT_PROFILE_RELEASES_PATH)
    }
    expected = {
        "11EoZncAAAAJ": {
            (89, "11EoZncAAAAJ:hFOr9nPyWt4C"),
        },
        "1J28ExgAAAAJ": {
            (10, "1J28ExgAAAAJ:r0BpntZqJG4C"),
            (152, "1J28ExgAAAAJ:rmuvC79q63oC"),
            (175, "1J28ExgAAAAJ:vDijr-p_gm4C"),
            (188, "1J28ExgAAAAJ:HtEfBTGE9r8C"),
        },
        "p1aFEDYAAAAJ": {
            (246, "p1aFEDYAAAAJ:uJ-U7cs_P_0C"),
            (261, "p1aFEDYAAAAJ:-_dYPAW6P2MC"),
            (392, "p1aFEDYAAAAJ:nPTYJWkExTIC"),
            (403, "p1aFEDYAAAAJ:Vr2j17o0sqMC"),
            (410, "p1aFEDYAAAAJ:rCNdntzdTkkC"),
        },
    }
    expected_counts = {"11EoZncAAAAJ": 44, "1J28ExgAAAAJ": 38, "p1aFEDYAAAAJ": 56}
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    for scholar_id, rows in expected.items():
        profile = profiles[scholar_id]
        decision = decisions[scholar_id]
        held = {item.author_pub_id: item for item in profile.held_observations}
        assert decision.expected_provisional_hold_rows == len(held)
        assert decision.provisional_hold_ids_sha256 == hold_ids_sha256(set(held))
        assert len(held) == expected_counts[scholar_id]
        for index, author_pub_id in rows:
            item = held[author_pub_id]
            assert item.capture_row_index == index
            assert item.evidence_url.endswith("citation_for_view=" + author_pub_id)
            assert (scholar_id, author_pub_id) not in exclusions


def test_breivik_release_accounts_for_every_source_row_without_hold_exclusion_overlap():
    scholar_id = "AcW2EXwAAAAJ"
    profile = next(
        item
        for item in load_provisional_holds(DEFAULT_PROVISIONAL_HOLDS_PATH)
        if item.scholar_id == scholar_id
    )
    decision = next(
        item
        for item in load_profile_releases(DEFAULT_PROFILE_RELEASES_PATH)
        if item.scholar_id == scholar_id
    )
    held_ids = {item.author_pub_id for item in profile.held_observations}
    excluded_ids = {
        author_pub_id
        for sid, author_pub_id in load_exclusions(DEFAULT_EXCLUSIONS_PATH)
        if sid == scholar_id
    }
    assert decision.state == "released"
    assert profile.expected_source_rows == decision.expected_source_rows == 131
    assert len(excluded_ids) == profile.expected_reviewed_exclusion_rows == 9
    assert len(held_ids) == decision.expected_provisional_hold_rows == 52
    assert held_ids.isdisjoint(excluded_ids)
    assert hold_ids_sha256(excluded_ids) == profile.reviewed_exclusion_ids_sha256
    assert hold_ids_sha256(held_ids) == decision.provisional_hold_ids_sha256
    assert profile.expected_source_rows - len(held_ids) - len(excluded_ids) == 70


def test_checked_in_bernhard_hold_targets_only_unresolved_gottingen_row():
    profiles = load_provisional_holds(DEFAULT_PROVISIONAL_HOLDS_PATH)
    profile = next(item for item in profiles if item.scholar_id == "hfFKwRcAAAAJ")
    assert profile.person_id == "person-stefanbernhard-7f8d72cd"
    assert profile.expected_source_rows == 228
    assert profile.cache_sha256 == (
        "45ee0c659f02e036f9a742ffe87ec22ee7827922451eb4fd7af472f5bac5ced7"
    )
    assert profile.expected_reviewed_exclusion_rows == 43
    assert profile.reviewed_exclusion_ids_sha256 == (
        "f7aaee154375508ffa1bed85d0acce78fe6f3c59d20e4dcf541ede3a3a8447f6"
    )
    assert [
        (item.capture_row_index, item.author_pub_id)
        for item in profile.held_observations
    ] == [(212, "hfFKwRcAAAAJ:2KloaMYe4IUC")]
    assert (
        hold_ids_sha256({item.author_pub_id for item in profile.held_observations})
        == "36a9ab8afe9e53304f3689464d2fce5234362b08df9c393e90de7e3d31e3a388"
    )


def test_checked_in_mcmanus_hold_targets_only_unverified_poster_byline():
    profiles = load_provisional_holds(DEFAULT_PROVISIONAL_HOLDS_PATH)
    profile = next(item for item in profiles if item.scholar_id == "28yZqbcAAAAJ")
    assert profile.person_id == "person-cjoelmcmanus-f28918ce"
    assert profile.expected_source_rows == 62
    assert profile.cache_sha256 == (
        "012435c098eddc0ad41b27ce214ad0f00ea89b8311ac9edbe5b7148900f29096"
    )
    assert profile.expected_reviewed_exclusion_rows == 3
    assert profile.reviewed_exclusion_ids_sha256 == (
        "3316309a95705c86671fc2144795b95a22296fc4df3c358334290e9fe002bc84"
    )
    assert [
        (item.capture_row_index, item.author_pub_id)
        for item in profile.held_observations
    ] == [(60, "28yZqbcAAAAJ:HKviVsUxM5wC")]
    assert (
        hold_ids_sha256({item.author_pub_id for item in profile.held_observations})
        == "5009f431f17b2eaecdd281dc6ed51e01f8a4914074e85d7f7ab4e3273ca0e2aa"
    )


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
        ("primary_evidence_url", "http://example.test", "primary evidence"),
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
    primary = "https://example.org/primary-record"
    primary_tables = build_dataset_tables(
        pandas.DataFrame(rows),
        registry,
        provisional_holds={(SCHOLAR, HELD): _hold(primary_evidence_url=primary)},
    )
    assert (
        primary_tables["profile_publications"]
        .set_index("author_pub_id")
        .loc[HELD, "review_attribution_evidence_url"]
        == primary
    )
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


def test_preupload_guard_checks_all_held_rows_in_one_work():
    tables = _audit_tables()
    second_id = f"{SCHOLAR}:held-two"
    second = _hold(
        author_pub_id=second_id,
        capture_row_index=2,
        evidence_url=f"https://scholar.google.com/citations?citation_for_view={second_id}",
    )
    tables["profile_publications"] = pandas.concat(
        [
            tables["profile_publications"],
            pandas.DataFrame(
                [
                    {
                        "scholar_id": SCHOLAR,
                        "author_pub_id": second_id,
                        "observation_id": "observation-held-two",
                        "map_eligible": False,
                        "map_exclusion_reasons": ["attribution_provisional_hold"],
                        "review_attribution_status": "held",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    tables["works"].at[0, "map_eligible"] = True
    tables["works"].at[0, "observation_ids"] = ["observation-held-two"]
    tables["works"].at[0, "author_pub_ids"] = [second_id]
    with pytest.raises(ValueError, match="contributed to a map link"):
        require_no_provisional_hold_contributions(
            tables, {(SCHOLAR, HELD): _hold(), (SCHOLAR, second_id): second}
        )


@pytest.mark.parametrize(
    "change,message",
    [
        ("raw_missing_column", "lack hold audit"),
        ("relationship_missing_column", "lack hold provenance"),
        ("raw_missing_row", "escaped raw map gate"),
        ("raw_duplicate", "escaped raw map gate"),
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
    elif change == "raw_duplicate":
        tables["profile_publications"] = pandas.concat(
            [tables["profile_publications"], tables["profile_publications"]],
            ignore_index=True,
        )
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


def test_croft_release_accounts_for_source_rows_and_unresolved_bristol_records():
    scholar_id = "wex9ft0AAAAJ"
    profile = next(
        item
        for item in load_provisional_holds(DEFAULT_PROVISIONAL_HOLDS_PATH)
        if item.scholar_id == scholar_id
    )
    decision = next(
        item
        for item in load_profile_releases(DEFAULT_PROFILE_RELEASES_PATH)
        if item.scholar_id == scholar_id
    )
    held_ids = {item.author_pub_id for item in profile.held_observations}
    held_rows = {item.capture_row_index for item in profile.held_observations}
    excluded_ids = {
        author_pub_id
        for sid, author_pub_id in load_exclusions(DEFAULT_EXCLUSIONS_PATH)
        if sid == scholar_id
    }
    assert decision.state == "released"
    assert profile.expected_source_rows == decision.expected_source_rows == 267
    assert len(excluded_ids) == profile.expected_reviewed_exclusion_rows == 12
    assert len(held_ids) == decision.expected_provisional_hold_rows == 83
    assert {250, 264, 265, 266} <= held_rows
    assert held_ids.isdisjoint(excluded_ids)
    assert hold_ids_sha256(excluded_ids) == profile.reviewed_exclusion_ids_sha256
    assert hold_ids_sha256(held_ids) == decision.provisional_hold_ids_sha256
    assert profile.expected_source_rows - len(held_ids) - len(excluded_ids) == 172


def test_dodelson_release_accounts_for_exact_awards_and_unresolved_rows():
    scholar_id = "kcXOB1MAAAAJ"
    profile = next(
        item
        for item in load_provisional_holds(DEFAULT_PROVISIONAL_HOLDS_PATH)
        if item.scholar_id == scholar_id
    )
    decision = next(
        item
        for item in load_profile_releases(DEFAULT_PROFILE_RELEASES_PATH)
        if item.scholar_id == scholar_id
    )
    held_ids = {item.author_pub_id for item in profile.held_observations}
    held_rows = {item.capture_row_index for item in profile.held_observations}
    excluded_ids = {
        author_pub_id
        for sid, author_pub_id in load_exclusions(DEFAULT_EXCLUSIONS_PATH)
        if sid == scholar_id
    }
    assert profile.person_id == decision.person_id == "person-scottdodelson-73513ee9"
    assert decision.state == "released"
    assert profile.expected_source_rows == decision.expected_source_rows == 443
    assert len(excluded_ids) == profile.expected_reviewed_exclusion_rows == 6
    assert len(held_ids) == decision.expected_provisional_hold_rows == 161
    assert {247, 295, 321, 341, 437} <= held_rows
    assert held_ids.isdisjoint(excluded_ids)
    assert hold_ids_sha256(excluded_ids) == profile.reviewed_exclusion_ids_sha256
    assert hold_ids_sha256(held_ids) == decision.provisional_hold_ids_sha256
    assert profile.expected_source_rows - len(held_ids) - len(excluded_ids) == 276
    assert {
        f"{scholar_id}:hHIA4WEVY-EC",
        f"{scholar_id}:FsLZdJ3BAzkC",
        f"{scholar_id}:cBPnxVikjH8C",
        f"{scholar_id}:rCNdntzdTkkC",
    } <= excluded_ids


def test_schumacher_release_preserves_uncertain_name_matches():
    scholar_id = "LibSWRMAAAAJ"
    profile = next(
        item
        for item in load_provisional_holds(DEFAULT_PROVISIONAL_HOLDS_PATH)
        if item.scholar_id == scholar_id
    )
    decision = next(
        item
        for item in load_profile_releases(DEFAULT_PROFILE_RELEASES_PATH)
        if item.scholar_id == scholar_id
    )
    held_ids = {item.author_pub_id for item in profile.held_observations}
    held_rows = {item.capture_row_index for item in profile.held_observations}
    excluded_ids = {
        author_pub_id
        for sid, author_pub_id in load_exclusions(DEFAULT_EXCLUSIONS_PATH)
        if sid == scholar_id
    }
    assert (
        profile.person_id
        == decision.person_id
        == ("person-reinhardschumacher-35cd3c6f")
    )
    assert decision.state == "released"
    assert profile.expected_source_rows == decision.expected_source_rows == 543
    assert len(excluded_ids) == profile.expected_reviewed_exclusion_rows == 26
    assert len(held_ids) == decision.expected_provisional_hold_rows == 258
    assert {21, 513, 514, 521, 527, 528, 529} <= held_rows
    assert held_ids.isdisjoint(excluded_ids)
    assert hold_ids_sha256(excluded_ids) == profile.reviewed_exclusion_ids_sha256
    assert hold_ids_sha256(held_ids) == decision.provisional_hold_ids_sha256
    assert profile.expected_source_rows - len(held_ids) - len(excluded_ids) == 259


def test_cremonesi_release_is_bound_to_complete_source_review():
    scholar_id = "wJ_gQLoAAAAJ"
    profile = next(
        item
        for item in load_provisional_holds(DEFAULT_PROVISIONAL_HOLDS_PATH)
        if item.scholar_id == scholar_id
    )
    decision = next(
        item
        for item in load_profile_releases(DEFAULT_PROFILE_RELEASES_PATH)
        if item.scholar_id == scholar_id
    )
    held_ids = {item.author_pub_id for item in profile.held_observations}
    held_rows = {item.capture_row_index for item in profile.held_observations}
    excluded_ids = {
        author_pub_id
        for sid, author_pub_id in load_exclusions(DEFAULT_EXCLUSIONS_PATH)
        if sid == scholar_id
    }
    assert (
        profile.person_id == decision.person_id == ("person-matteocremonesi-b2aba740")
    )
    assert decision.state == "released"
    assert profile.expected_source_rows == decision.expected_source_rows == 1963
    assert len(excluded_ids) == profile.expected_reviewed_exclusion_rows == 80
    assert len(held_ids) == decision.expected_provisional_hold_rows == 1087
    assert {68, 1430, 1926, 1962, 183, 801} <= held_rows
    assert held_ids.isdisjoint(excluded_ids)
    assert hold_ids_sha256(excluded_ids) == profile.reviewed_exclusion_ids_sha256
    assert hold_ids_sha256(held_ids) == decision.provisional_hold_ids_sha256
    assert decision.provisional_hold_ids_sha256 == (
        "2ef585a2854d142f72fa9ee01ed2e7624300305335b6d706a5395ebb58012c60"
    )
    assert profile.expected_source_rows - len(held_ids) - len(excluded_ids) == 796


def test_palmese_release_holds_unresolved_output_types_and_name_matches():
    scholar_id = "6pPZKd0AAAAJ"
    profile = next(
        item
        for item in load_provisional_holds(DEFAULT_PROVISIONAL_HOLDS_PATH)
        if item.scholar_id == scholar_id
    )
    decision = next(
        item
        for item in load_profile_releases(DEFAULT_PROFILE_RELEASES_PATH)
        if item.scholar_id == scholar_id
    )
    held_ids = {item.author_pub_id for item in profile.held_observations}
    held_rows = {item.capture_row_index for item in profile.held_observations}
    excluded_ids = {
        author_pub_id
        for sid, author_pub_id in load_exclusions(DEFAULT_EXCLUSIONS_PATH)
        if sid == scholar_id
    }
    assert profile.person_id == decision.person_id == "person-antonellapalmese-5378b53d"
    assert decision.state == "released"
    assert profile.expected_source_rows == decision.expected_source_rows == 624
    assert len(excluded_ids) == profile.expected_reviewed_exclusion_rows == 3
    assert len(held_ids) == decision.expected_provisional_hold_rows == 346
    assert {27, 43, 177, 539, 585, 586, 615, 616, 618, 619, 624} <= held_rows
    assert held_ids.isdisjoint(excluded_ids)
    assert hold_ids_sha256(excluded_ids) == profile.reviewed_exclusion_ids_sha256
    assert hold_ids_sha256(held_ids) == decision.provisional_hold_ids_sha256
    assert profile.expected_source_rows - len(held_ids) - len(excluded_ids) == 275
