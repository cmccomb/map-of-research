import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pandas
import pytest

from map_of_research.io import sha256_file
from map_of_research.profile_releases import (
    DEFAULT_PROFILE_RELEASES_PATH,
    ProfileRelease,
    load_profile_releases,
    require_no_held_contributions,
    select_released_observations,
)

PERSON = "person-held"
SCHOLAR = "held-scholar"
OTHER = "person-other"


def decision(**changes):
    return replace(
        ProfileRelease(
            person_id=PERSON,
            scholar_id=SCHOLAR,
            state="held",
            expected_source_rows=2,
            cache_sha256="a" * 64,
            reason="Exact attribution review pending.",
            reviewed_at="2026-10-25",
        ),
        **changes,
    )


def source(tmp_path, **changes):
    cache_dir = tmp_path / "data" / "authors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = {
        "scholar_id": SCHOLAR,
        "publication_count": 2,
        "publications": [
            {"author_pub_id": f"{SCHOLAR}:one"},
            {"author_pub_id": f"{SCHOLAR}:two"},
        ],
        "browser_capture": {
            "coverage": "complete_visible_profile",
            "more_remain": False,
        },
    }
    cache.update(changes)
    path = cache_dir / f"{SCHOLAR}.json"
    path.write_text(json.dumps(cache), encoding="utf-8")
    return cache_dir, decision(cache_sha256=sha256_file(path))


def frame():
    return pandas.DataFrame(
        [
            {
                "person_id": PERSON,
                "scholar_id": SCHOLAR,
                "author_pub_id": f"{SCHOLAR}:one",
                "source_type": "scholar",
                "authors": "Held Person and Other Person",
            },
            {
                "person_id": PERSON,
                "scholar_id": SCHOLAR,
                "author_pub_id": f"{SCHOLAR}:two",
                "source_type": "scholar",
                "authors": "Held Person",
            },
            {
                "person_id": PERSON,
                "scholar_id": "",
                "author_pub_id": "curated:held",
                "source_type": "curated_verified",
                "authors": "Held Person",
            },
            {
                "person_id": OTHER,
                "scholar_id": "other-scholar",
                "author_pub_id": "other-scholar:one",
                "source_type": "scholar",
                "authors": "Other Person and Held Person",
            },
        ]
    )


def registry(scholar_id=SCHOLAR, person_id=PERSON):
    return SimpleNamespace(
        people_by_id={person_id: SimpleNamespace(scholar_id=scholar_id)}
    )


def test_checked_in_chien_release_is_source_bound() -> None:
    items = load_profile_releases(DEFAULT_PROFILE_RELEASES_PATH)
    item = next(item for item in items if item.scholar_id == "8iLLOQgAAAAJ")
    assert item.person_id == "person-chienho-da71f01b"
    assert item.scholar_id == "8iLLOQgAAAAJ"
    assert item.state == "released"
    assert item.expected_source_rows == 582
    assert len(item.cache_sha256) == 64
    collins = next(item for item in items if item.scholar_id == "0e1jwkkAAAAJ")
    assert collins.person_id == "person-terrencejcollins-61cf6ed9"
    assert collins.state == "released"
    assert collins.expected_source_rows == 361
    assert collins.expected_provisional_hold_rows == 10
    assert collins.provisional_hold_ids_sha256 == (
        "b48946eaecc8fdf09d32c892cf4862f69eda4109983d6839dc7cae95c82c9180"
    )


def test_missing_trusted_profile_release_manifest_fails_closed(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_profile_releases(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "body,error",
    [
        ([], "schema"),
        ({"schema_version": 2, "profiles": []}, "schema"),
        ({"schema_version": 1}, "profile decisions"),
        ({"schema_version": 1, "profiles": []}, "profile decisions"),
        ({"schema_version": 1, "profiles": [None]}, "must be an object"),
    ],
)
def test_profile_release_manifest_rejects_structure(tmp_path, body, error) -> None:
    path = tmp_path / "releases.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match=error):
        load_profile_releases(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("person_id", None),
        ("person_id", "invalid"),
        ("scholar_id", None),
        ("scholar_id", ""),
        ("state", "pending"),
        ("expected_source_rows", True),
        ("expected_source_rows", 0),
        ("cache_sha256", None),
        ("cache_sha256", "wrong"),
        ("reason", None),
        ("reason", "   "),
        ("expected_provisional_hold_rows", True),
        ("expected_provisional_hold_rows", -1),
        ("expected_provisional_hold_rows", 2),
        ("provisional_hold_ids_sha256", "bad"),
    ],
)
def test_profile_release_manifest_rejects_bad_field(tmp_path, field, value) -> None:
    item = vars(decision()).copy()
    item[field] = value
    path = tmp_path / "releases.json"
    path.write_text(json.dumps({"schema_version": 1, "profiles": [item]}))
    with pytest.raises(ValueError, match="invalid fields"):
        load_profile_releases(path)


def test_profile_release_manifest_requires_date_and_unique_ids(tmp_path) -> None:
    item = vars(decision()).copy()
    path = tmp_path / "releases.json"
    item["reviewed_at"] = "tomorrow"
    path.write_text(json.dumps({"schema_version": 1, "profiles": [item]}))
    with pytest.raises(ValueError, match="needs a date"):
        load_profile_releases(path)

    item["reviewed_at"] = "2026-10-25"
    for second in (
        {**item, "scholar_id": "second"},
        {**item, "person_id": "person-second"},
    ):
        path.write_text(json.dumps({"schema_version": 1, "profiles": [item, second]}))
        with pytest.raises(ValueError, match="invalid fields"):
            load_profile_releases(path)
    second = {**item, "person_id": "person-second", "scholar_id": "second"}
    path.write_text(json.dumps({"schema_version": 1, "profiles": [item, second]}))
    assert len(load_profile_releases(path)) == 2


def test_held_profile_is_removed_only_from_published_observations(tmp_path) -> None:
    cache_dir, hold = source(tmp_path)
    original = frame()
    released, receipt, held_ids = select_released_observations(
        original, registry(), (hold,), cache_dir=cache_dir
    )
    assert len(original) == 4
    assert released["person_id"].tolist() == [OTHER]
    assert released["authors"].tolist() == ["Other Person and Held Person"]
    assert receipt[0]["state"] == "held"
    assert receipt[0]["source_observation_rows"] == 2
    assert receipt[0]["person_observation_rows"] == 3
    assert receipt[0]["cache_sha256"] == hold.cache_sha256
    assert len(held_ids) == 3


def test_explicit_release_restores_source_after_same_verification(tmp_path) -> None:
    cache_dir, hold = source(tmp_path)
    released, receipt, held_ids = select_released_observations(
        frame(), registry(), (replace(hold, state="released"),), cache_dir=cache_dir
    )
    assert len(released) == 4
    assert receipt[0]["state"] == "released"
    assert held_ids == set()


@pytest.mark.parametrize(
    "person_id,scholar_id", [("absent", SCHOLAR), (PERSON, "other")]
)
def test_profile_release_rejects_registry_identity(
    tmp_path, person_id, scholar_id
) -> None:
    cache_dir, hold = source(tmp_path)
    with pytest.raises(ValueError, match="identity differs"):
        select_released_observations(
            frame(), registry(scholar_id, person_id), (hold,), cache_dir=cache_dir
        )


def test_profile_release_rejects_missing_or_changed_cache(tmp_path) -> None:
    cache_dir, hold = source(tmp_path)
    with pytest.raises(ValueError, match="missing or changed"):
        select_released_observations(
            frame(), registry(), (hold,), cache_dir=tmp_path / "missing"
        )
    with pytest.raises(ValueError, match="missing or changed"):
        select_released_observations(
            frame(),
            registry(),
            (replace(hold, cache_sha256="0" * 64),),
            cache_dir=cache_dir,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"scholar_id": "wrong"},
        {"publications": None},
        {"publications": []},
        {"publication_count": 1},
        {"browser_capture": None},
        {"browser_capture": {"coverage": "capped_partial", "more_remain": False}},
        {
            "browser_capture": {
                "coverage": "complete_visible_profile",
                "more_remain": True,
            }
        },
    ],
)
def test_profile_release_rejects_incomplete_cache(tmp_path, change) -> None:
    cache_dir, hold = source(tmp_path, **change)
    with pytest.raises(ValueError, match="source cache is incomplete"):
        select_released_observations(frame(), registry(), (hold,), cache_dir=cache_dir)


@pytest.mark.parametrize(
    "rows",
    [
        [None, {"author_pub_id": f"{SCHOLAR}:two"}],
        [{"author_pub_id": "wrong:one"}, {"author_pub_id": f"{SCHOLAR}:two"}],
        [{"author_pub_id": f"{SCHOLAR}:one"}] * 2,
    ],
)
def test_profile_release_rejects_bad_cache_ids(tmp_path, rows) -> None:
    cache_dir, hold = source(tmp_path, publications=rows)
    with pytest.raises(ValueError, match="source IDs are invalid"):
        select_released_observations(frame(), registry(), (hold,), cache_dir=cache_dir)


@pytest.mark.parametrize(
    "change", ["missing", "wrong_person", "wrong_id", "other_person"]
)
def test_profile_release_rejects_snapshot_source_mismatch(tmp_path, change) -> None:
    cache_dir, hold = source(tmp_path)
    observations = frame()
    if change == "missing":
        observations = observations.drop(index=1)
    elif change == "wrong_person":
        observations.loc[1, "person_id"] = OTHER
    elif change == "wrong_id":
        observations.loc[1, "author_pub_id"] = f"{SCHOLAR}:wrong"
    else:
        observations.loc[2, "scholar_id"] = SCHOLAR
        observations.loc[2, "person_id"] = OTHER
    with pytest.raises(ValueError, match="snapshot differs from source"):
        select_released_observations(
            observations, registry(), (hold,), cache_dir=cache_dir
        )


def tables():
    return {
        "profile_publications": pandas.DataFrame(
            [{"person_id": OTHER, "scholar_id": "other-scholar"}]
        ),
        "authorships": pandas.DataFrame(
            [{"person_id": OTHER, "scholar_id": "other-scholar"}]
        ),
        "works": pandas.DataFrame(
            [{"observation_ids": ["observation-other"], "authors": "Held Person"}]
        ),
        "people": pandas.DataFrame(
            [
                {
                    "person_id": PERSON,
                    "publication_observation_count": 0,
                    "scholar_observation_count": 0,
                    "curated_observation_count": 0,
                    "unique_work_count": 0,
                    "mapped_work_count": 0,
                }
            ]
        ),
    }


def test_public_tables_preserve_other_faculty_coauthored_work() -> None:
    output = tables()
    require_no_held_contributions(output, (decision(),), {"observation-held"})
    require_no_held_contributions(
        output, (replace(decision(), state="released"),), set()
    )


@pytest.mark.parametrize(
    "config,field,value,error",
    [
        ("profile_publications", "person_id", PERSON, "published profile_publications"),
        ("authorships", "scholar_id", SCHOLAR, "published authorships"),
        ("works", "observation_ids", ["observation-held"], "published works"),
        ("people", "mapped_work_count", 1, "zero own works"),
    ],
)
def test_public_tables_reject_held_contribution(config, field, value, error) -> None:
    output = tables()
    output[config].at[0, field] = value
    with pytest.raises(ValueError, match=error):
        require_no_held_contributions(output, (decision(),), {"observation-held"})


@pytest.mark.parametrize(
    "config,field,error",
    [
        ("profile_publications", "scholar_id", "lacks profile identity"),
        ("authorships", "person_id", "lacks profile identity"),
        ("works", "observation_ids", "lack source observation"),
        ("people", "mapped_work_count", "lack held-profile count"),
    ],
)
def test_public_tables_require_auditable_columns(config, field, error) -> None:
    output = tables()
    output[config] = output[config].drop(columns=[field])
    with pytest.raises(ValueError, match=error):
        require_no_held_contributions(output, (decision(),), {"observation-held"})


def test_public_tables_require_one_zero_count_roster_person() -> None:
    output = tables()
    output["people"] = output["people"].iloc[0:0]
    with pytest.raises(ValueError, match="zero own works"):
        require_no_held_contributions(output, (decision(),), {"observation-held"})


def test_observation_identity_matches_dataset_hash(tmp_path) -> None:
    author_pub_id = f"{SCHOLAR}:one"
    expected = (
        "observation-"
        + hashlib.sha256(f"{SCHOLAR}\0{author_pub_id}".encode()).hexdigest()
    )
    cache_dir, hold = source(tmp_path)
    _, _, held_ids = select_released_observations(
        frame(), registry(), (hold,), cache_dir=cache_dir
    )
    assert expected in held_ids
