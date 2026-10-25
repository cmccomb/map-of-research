import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas
import pytest

from map_of_research.attribution import (
    AttributionReviews,
    ConfirmedAttribution,
    load_attribution_reviews,
)
from map_of_research.exclusions import DEFAULT_EXCLUSIONS_PATH, load_exclusions
from map_of_research.io import sha256_file
from map_of_research.profile_releases import (
    DEFAULT_PROFILE_RELEASES_PATH,
    ProfileRelease,
    load_profile_releases,
    require_no_held_contributions,
    select_released_observations,
    validate_reviewed_subset_decisions,
)

PERSON = "person-held"
SCHOLAR = "held-scholar"
OTHER = "person-other"
PROPOSAL = Path("registry/reviews/2026-10-25-chem-das-reviewed-subset-proposal.json")


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
    assert collins.expected_provisional_hold_rows == 7
    assert collins.provisional_hold_ids_sha256 == (
        "92683d4192656f159ccdebf5ed2cc07ff586798df08a1293f5f56767259f66b0"
    )


def test_das_reviewed_subset_is_exhaustive_and_source_bound() -> None:
    active = load_profile_releases(DEFAULT_PROFILE_RELEASES_PATH)
    active_das = next(item for item in active if item.scholar_id == "FrF1YXQAAAAJ")
    assert active_das.state == "held"
    releases = load_profile_releases(PROPOSAL)
    item = next(item for item in releases if item.scholar_id == "FrF1YXQAAAAJ")
    assert item.person_id == "person-subhardas-42cf6c1e"
    assert item.state == "reviewed_subset"
    assert item.expected_source_rows == 670
    assert item.cache_sha256 == (
        "85700e7ee25079ed30633df91137e84cc0ef6c0e6628166f80acf4c133f4c1df"
    )
    with Path("registry/reviews/2026-10-25-chem-das-complete-profile-ledger.csv").open(
        newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len({row["author_pub_id"] for row in rows}) == 670
    assert [int(row["capture_row_index"]) for row in rows] == list(range(1, 671))
    assert set(item.released_author_pub_ids) == {
        row["author_pub_id"]
        for row in rows
        if row["decision"] == "positive_primary_source_reviewed_subset"
    }
    assert set(item.excluded_author_pub_ids) == {
        row["author_pub_id"] for row in rows if row["decision"] == "reviewed_excluded"
    }
    assert set(item.held_author_pub_ids) == {
        row["author_pub_id"]
        for row in rows
        if row["decision"]
        in {
            "unresolved_held",
            "positive_orcid_title_year_signal_held",
            "positive_identity_metadata_held",
        }
    }
    assert tuple(
        map(
            len,
            (
                item.released_author_pub_ids,
                item.excluded_author_pub_ids,
                item.held_author_pub_ids,
            ),
        )
    ) == (65, 236, 369)
    validate_reviewed_subset_decisions(
        releases,
        load_attribution_reviews(Path("data/curated/scholar-attribution-reviews.json")),
        load_exclusions(DEFAULT_EXCLUSIONS_PATH),
    )


@pytest.mark.parametrize(
    "index,pub_id,year,title,authors,decision,code,evidence_url,basis_marker",
    [
        (
            13,
            "FrF1YXQAAAAJ:ziOE8S1-AIUC",
            "2010",
            "Colorectal carcinoma in young adults: a retrospective study on "
            "Indian patients: 2000–2008",
            "S Gupta, D Bhattacharya, AN Acharya, S Majumdar, P Ranjan, S Das",
            "reviewed_excluded",
            "false_profile_attribution",
            "https://onlinelibrary.wiley.com/doi/10.1111/j.1463-1318.2010.02223.x",
            None,
        ),
        (
            136,
            "FrF1YXQAAAAJ:0aBXIfxlw9sC",
            "1998",
            "Sasaki YF, Rothe T, Premkumar LS, Takasu M, Crandall JE, Dikkes P, "
            "Conner DA, Rayudu PV, Cheung W, Chen HS, Lipton SA, Nakanishi N",
            "S Das",
            "reviewed_excluded",
            "false_profile_attribution",
            "https://www.nature.com/articles/30748",
            None,
        ),
        (
            157,
            "FrF1YXQAAAAJ:X5YyAB84Iw4C",
            "1945",
            "Diels L. Pant D. Patil SA Ghangrekar MM J",
            "S Das",
            "reviewed_excluded",
            "false_profile_attribution",
            "https://pub.vito.be/openaccess/2020-Das%20et%20al.%2C%20J.%20Electrochem.%20Soc._MES.pdf",
            None,
        ),
        (
            281,
            "FrF1YXQAAAAJ:c59VksA5Vz4C",
            "2018",
            "Compact Tunable Hafnia Bismuth Erbium Co-Doped Fiber Laser",
            "U Zakaria, S Harun, P Reddy, D Dutta, D Das, S Das, M Yasin",
            "reviewed_excluded",
            "false_profile_attribution",
            "https://chalcogen.ro/77_ZakariaUN.pdf",
            None,
        ),
        (
            354,
            "FrF1YXQAAAAJ:tS2w5q8j5-wC",
            "2010",
            "Dual-wavelength Erbium-doped fiber laser based on Erbium-doped "
            "Zirconia fiber",
            "A Hamzah, MC Paul, SW Harun, NAD Huri, A Lokman, R Sen, M Pal, ...",
            "reviewed_excluded",
            "false_profile_attribution",
            "https://oam-rc.inoe.ro/articles/dual-wavelength-erbium-doped-fiber-laser-based-on-erbium-doped-zirconia-fiber/",
            None,
        ),
        (
            470,
            "FrF1YXQAAAAJ:4OULZ7Gr8RgC",
            "2016",
            "Synthesis of backbone branched RNA and the biochemical investigation "
            "of lariat debranching enzyme",
            "S Mack, S Dey, S Das",
            "positive_primary_source_reviewed_subset",
            "exact_original_meeting_abstract_with_cmu_identity",
            "https://www.acs.org/content/dam/acsorg/meetings/national-meetings/program-book/2016-philadelphia.pdf",
            "published meeting abstract",
        ),
        (
            544,
            "FrF1YXQAAAAJ:Wp0gIr-vW9MC",
            "2010",
            "Five years in review",
            "SR Das, JA Piccirilli",
            "reviewed_excluded",
            "false_profile_attribution",
            "https://www.nature.com/articles/nchembio.379",
            None,
        ),
        (
            601,
            "FrF1YXQAAAAJ:hC7cP41nSMkC",
            "2000",
            "Synthesis of carbocyclic 4'-beta-thio nucleosides: A new class of "
            "carbocyclic 5'-nornucleosides based on the biochemistry of "
            "S-adenosylmethionine",
            "SR Das",
            "positive_primary_source_reviewed_subset",
            "exact_institutional_thesis_record",
            "https://auburn.edu/academic/provost/commencement/archives/_assets/programs/2000s/2000%20Commencement%20Program_001-compressed.pdf",
            "doctoral dissertation",
        ),
    ],
)
def test_das_next_source_decisions_pin_exact_captured_rows(
    index, pub_id, year, title, authors, decision, code, evidence_url, basis_marker
) -> None:
    with Path("registry/reviews/2026-10-25-chem-das-complete-profile-ledger.csv").open(
        newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    row = rows[index - 1]
    assert (
        row["capture_row_index"],
        row["author_pub_id"],
        row["scholar_year"],
        row["title"],
        row["visible_authors"],
        row["decision"],
        row["reason_code"],
        row["evidence_url"],
    ) == (str(index), pub_id, year, title, authors, decision, code, evidence_url)
    key = ("FrF1YXQAAAAJ", pub_id)
    if decision == "reviewed_excluded":
        exclusion = load_exclusions(DEFAULT_EXCLUSIONS_PATH)[key]
        assert (exclusion.reason_code, exclusion.evidence_url) == (
            code,
            evidence_url,
        )
    else:
        review = load_attribution_reviews(
            Path("data/curated/scholar-attribution-reviews.json")
        ).confirmed[key]
        assert review.evidence_url == evidence_url
        assert basis_marker in review.evidence_basis


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
    item = {
        key: value
        for key, value in vars(decision()).items()
        if not key.endswith("_author_pub_ids")
    }
    item[field] = value
    path = tmp_path / "releases.json"
    path.write_text(json.dumps({"schema_version": 1, "profiles": [item]}))
    with pytest.raises(ValueError, match="invalid fields"):
        load_profile_releases(path)


def test_profile_release_manifest_requires_date_and_unique_ids(tmp_path) -> None:
    item = {
        key: value
        for key, value in vars(decision()).items()
        if not key.endswith("_author_pub_ids")
    }
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


@pytest.mark.parametrize(
    "change,error",
    [
        ({"released_author_pub_ids": "held:one"}, "unique citation IDs"),
        ({"released_author_pub_ids": ["wrong:one"]}, "unique citation IDs"),
        (
            {"released_author_pub_ids": [f"{SCHOLAR}:one", f"{SCHOLAR}:one"]},
            "unique citation IDs",
        ),
        ({"released_author_pub_ids": []}, "exact row partition"),
        ({"held_author_pub_ids": [f"{SCHOLAR}:one"]}, "exact row partition"),
        ({"held_author_pub_ids": []}, "exact row partition"),
    ],
)
def test_reviewed_subset_manifest_requires_exact_unique_partition(
    tmp_path, change, error
) -> None:
    item = {
        key: value
        for key, value in vars(decision()).items()
        if not key.endswith("_author_pub_ids")
    }
    item.update(
        state="reviewed_subset",
        released_author_pub_ids=[f"{SCHOLAR}:one"],
        excluded_author_pub_ids=[],
        held_author_pub_ids=[f"{SCHOLAR}:two"],
    )
    item.update(change)
    path = tmp_path / "releases.json"
    path.write_text(json.dumps({"schema_version": 1, "profiles": [item]}))
    with pytest.raises(ValueError, match=error):
        load_profile_releases(path)


def test_full_hold_manifest_rejects_subset_arrays(tmp_path) -> None:
    item = {
        key: value
        for key, value in vars(decision()).items()
        if not key.endswith("_author_pub_ids")
    }
    item["released_author_pub_ids"] = [f"{SCHOLAR}:one"]
    path = tmp_path / "releases.json"
    path.write_text(json.dumps({"schema_version": 1, "profiles": [item]}))
    with pytest.raises(ValueError, match="requires reviewed_subset"):
        load_profile_releases(path)


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


def test_reviewed_subset_keeps_only_exact_positive_source_rows(tmp_path) -> None:
    cache_dir, hold = source(tmp_path)
    subset = replace(
        hold,
        state="reviewed_subset",
        released_author_pub_ids=(f"{SCHOLAR}:one",),
        excluded_author_pub_ids=(),
        held_author_pub_ids=(f"{SCHOLAR}:two",),
    )
    released, receipt, held_ids = select_released_observations(
        frame(), registry(), (subset,), cache_dir=cache_dir
    )
    assert released["person_id"].tolist() == [PERSON, OTHER]
    assert released["author_pub_id"].tolist() == [f"{SCHOLAR}:one", "other-scholar:one"]
    assert receipt[0]["source_observation_rows"] == 2
    assert receipt[0]["released_source_observation_rows"] == 1
    assert receipt[0]["held_source_observation_rows"] == 1
    assert receipt[0]["held_person_observation_rows"] == 2
    assert len(held_ids) == 2


def test_reviewed_subset_rejects_changed_cache_or_partition(tmp_path) -> None:
    cache_dir, hold = source(tmp_path)
    subset = replace(
        hold,
        state="reviewed_subset",
        released_author_pub_ids=(f"{SCHOLAR}:one",),
        held_author_pub_ids=(f"{SCHOLAR}:two",),
    )
    with pytest.raises(ValueError, match="missing or changed"):
        select_released_observations(
            frame(),
            registry(),
            (replace(subset, cache_sha256="0" * 64),),
            cache_dir=cache_dir,
        )
    with pytest.raises(ValueError, match="row partition differs from source"):
        select_released_observations(
            frame(),
            registry(),
            (replace(subset, held_author_pub_ids=(f"{SCHOLAR}:wrong",)),),
            cache_dir=cache_dir,
        )


def test_reviewed_subset_rejects_unconfirmed_positive_or_exclusion() -> None:
    subset = replace(
        decision(),
        state="reviewed_subset",
        released_author_pub_ids=(f"{SCHOLAR}:one",),
        excluded_author_pub_ids=(),
        held_author_pub_ids=(f"{SCHOLAR}:two",),
    )
    confirmed = AttributionReviews(
        frozenset({SCHOLAR}),
        {
            (SCHOLAR, f"{SCHOLAR}:one"): ConfirmedAttribution(
                "https://example.org/one", None, "2026-10-25"
            )
        },
    )
    validate_reviewed_subset_decisions((subset,), confirmed, {})
    with pytest.raises(ValueError, match="differs from exact reviews"):
        validate_reviewed_subset_decisions(
            (replace(subset, released_author_pub_ids=(f"{SCHOLAR}:two",)),),
            confirmed,
            {},
        )
    with pytest.raises(ValueError, match="differs from exact reviews"):
        validate_reviewed_subset_decisions(
            (subset,), confirmed, {(SCHOLAR, f"{SCHOLAR}:two"): object()}
        )


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
            [
                {
                    "person_id": OTHER,
                    "scholar_id": "other-scholar",
                    "observation_id": "observation-other",
                }
            ]
        ),
        "authorships": pandas.DataFrame(
            [
                {
                    "person_id": OTHER,
                    "scholar_id": "other-scholar",
                    "observation_id": "observation-other",
                }
            ]
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


@pytest.mark.parametrize("config", ["profile_publications", "authorships", "works"])
def test_reviewed_subset_rejects_held_observation_in_public_tables(config) -> None:
    output = tables()
    if config == "works":
        output[config].at[0, "observation_ids"] = ["observation-held"]
    else:
        output[config].at[0, "observation_id"] = "observation-held"
    with pytest.raises(ValueError, match=f"published {config}"):
        require_no_held_contributions(
            output,
            (replace(decision(), state="reviewed_subset"),),
            {"observation-held"},
        )


def test_reviewed_subset_requires_public_observation_identity() -> None:
    output = tables()
    output["authorships"] = output["authorships"].drop(columns=["observation_id"])
    with pytest.raises(ValueError, match="lacks profile identity"):
        require_no_held_contributions(
            output,
            (replace(decision(), state="reviewed_subset"),),
            {"observation-held"},
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
