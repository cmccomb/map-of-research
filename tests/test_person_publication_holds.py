"""A partial Math source can be audited without publishing a partial profile."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pandas
import pytest

from map_of_research.dataset import build_dataset_tables
from map_of_research.person_publication_holds import (
    BIO_HOLD_RELEASE_REVISION,
    DEFAULT_PERSON_PUBLICATION_HOLDS_PATH,
    load_person_publication_holds,
    require_no_held_person_contributions,
    select_released_person_observations,
    validate_person_hold_losses,
)
from map_of_research.publisher import build_map_artifact
from map_of_research.registry import load_registry

TAASAN = "person-shlomotaasan-403c9cf9"
TETALI = "person-prasadtetali-bf494329"
KINDERLEHRER = "person-davidkinderlehrer-a58d3c4e"
ROLLETT = "person-rollett-553e9d0d"
BIO_SUCCESSOR = "211cdbcde850867ef9b5a81f7001f3b5b27425cc"


def hold():
    return load_person_publication_holds(DEFAULT_PERSON_PUBLICATION_HOLDS_PATH)[0]


def source_rows():
    selected = hold()
    rows = [
        {
            "person_id": TAASAN,
            "scholar_id": "",
            "source_type": "curated_verified",
            "source_doi": doi,
            "author_pub_id": author_pub_id,
        }
        for doi, author_pub_id, _ in selected.source_observations
    ]
    rows.extend(
        {
            "person_id": person_id,
            "scholar_id": "coauthor-scholar",
            "source_type": "scholar",
            "source_doi": doi,
            "author_pub_id": f"coauthor-scholar:{person_id}:{index}",
        }
        for person_id in (KINDERLEHRER, ROLLETT)
        for index, (doi, _, _) in enumerate(selected.source_observations)
    )
    rows.extend(
        {
            "person_id": TETALI,
            "scholar_id": "FmZaK1wAAAAJ",
            "source_type": "scholar",
            "source_doi": "",
            "author_pub_id": f"FmZaK1wAAAAJ:{index}",
        }
        for index in range(222)
    )
    rows.extend(
        {
            "person_id": TETALI,
            "scholar_id": "",
            "source_type": "curated_verified",
            "source_doi": doi,
            "author_pub_id": f"curated-doi:{TETALI}:{doi}",
        }
        for doi in ("10.1002/rsa.3240010302", "10.4064/aa-69-3-243-258")
    )
    return pandas.DataFrame(rows)


def registry(*, scholar_id=""):
    return SimpleNamespace(
        people_by_id={TAASAN: SimpleNamespace(scholar_id=scholar_id)}
    )


def test_checked_in_hold_binds_three_exact_dois_and_raw_observations() -> None:
    selected = hold()
    assert selected.person_id == TAASAN
    assert selected.expected_source_rows == 3
    assert selected.source_snapshot_sha256 == (
        "d08fb3a422df8dee3b0d0f75a29d68fef9b7121229a9cf6d418917e1d4a19550"
    )
    assert selected.prior_revision == "dcbdfacd7e8cfeaefd4c91081c4b347b7c690061"
    assert {row[0] for row in selected.source_observations} == {
        "10.1016/s1359-6462(97)00530-7",
        "10.4028/www.scientific.net/msf.715-716.473",
        "10.1016/j.pmatsci.2013.03.004",
    }
    assert selected.display_reason == "Publication history pending full capture"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ([], "schema"),
        ({"schema_version": 2, "holds": []}, "schema"),
        ({"schema_version": 1}, "need a list"),
        ({"schema_version": 1, "holds": [None]}, "must be an object"),
    ],
)
def test_manifest_structure_fails_closed(tmp_path, body, message) -> None:
    path = tmp_path / "holds.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_person_publication_holds(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("person_id", "wrong", "observation identity differs"),
        ("source_snapshot_sha256", "bad", "invalid fields"),
        ("prior_revision", "bad", "invalid fields"),
        ("expected_source_rows", 2, "invalid fields"),
        ("expected_source_rows", True, "invalid fields"),
        ("source_observations", "bad", "exact observations"),
        ("source_observations", [None], "must be an object"),
        ("display_reason", "Complete", "invalid fields"),
        ("reason", "", "invalid fields"),
        ("evidence_url", "http://example.test", "invalid fields"),
        ("reviewed_at", "invalid", "review date"),
    ],
)
def test_manifest_field_drift_fails_closed(tmp_path, field, value, message) -> None:
    body = json.loads(DEFAULT_PERSON_PUBLICATION_HOLDS_PATH.read_text())
    body["holds"][0][field] = value
    path = tmp_path / "holds.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_person_publication_holds(path)


def test_manifest_rejects_observation_drift_and_duplicate_people(tmp_path) -> None:
    body = json.loads(DEFAULT_PERSON_PUBLICATION_HOLDS_PATH.read_text())
    body["holds"][0]["source_observations"][0]["observation_id"] = "observation-x"
    path = tmp_path / "holds.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="observation identity differs"):
        load_person_publication_holds(path)
    body = json.loads(DEFAULT_PERSON_PUBLICATION_HOLDS_PATH.read_text())
    body["holds"].append(body["holds"][0])
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid fields"):
        load_person_publication_holds(path)


def test_manifest_rejects_prior_work_or_eligibility_drift(tmp_path) -> None:
    for field, value in (
        ("prior_work_id", "work-wrong"),
        ("prior_collaboration_eligible", "true"),
    ):
        body = json.loads(DEFAULT_PERSON_PUBLICATION_HOLDS_PATH.read_text())
        body["holds"][0]["source_observations"][0][field] = value
        path = tmp_path / "holds.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        with pytest.raises(ValueError, match="observation identity differs"):
            load_person_publication_holds(path)


def prior_hold_tables():
    selected = hold()
    prior_profiles = [
        {
            "person_id": TAASAN,
            "scholar_id": "",
            "author_pub_id": author_pub_id,
            "source_type": "curated_verified",
            "source_doi": doi,
            "work_id": work_id,
            "map_eligible": True,
            "collaboration_eligible": eligible,
        }
        for doi, author_pub_id, _, work_id, eligible in selected.prior_observations
    ]
    prior_works = [
        {"work_id": work_id, "doi": doi, "map_eligible": True}
        for doi, _, _, work_id, _ in selected.prior_observations
    ]
    return prior_works, prior_profiles


def test_exact_held_losses_require_current_parent_and_prior_bindings() -> None:
    prior_works, prior_profiles = prior_hold_tables()
    selected = hold()
    approved = validate_person_hold_losses(
        (selected,),
        prior_revision=selected.prior_revision,
        prior_works=prior_works,
        prior_profiles=prior_profiles,
        current_profiles=[],
    )
    assert len(approved) == 3
    assert approved == frozenset(
        (TAASAN, "", author_pub_id, "curated_verified")
        for _, author_pub_id, _ in selected.source_observations
    )
    assert (
        validate_person_hold_losses(
            (),
            prior_revision=selected.prior_revision,
            prior_works=prior_works,
            prior_profiles=prior_profiles,
            current_profiles=[],
        )
        == frozenset()
    )
    with pytest.raises(ValueError, match="prior or current rows"):
        validate_person_hold_losses(
            (selected,),
            prior_revision="0" * 40,
            prior_works=prior_works,
            prior_profiles=prior_profiles,
            current_profiles=[],
        )

    with pytest.raises(ValueError, match="prior or current rows"):
        validate_person_hold_losses(
            (selected,),
            prior_revision=selected.prior_revision,
            prior_works=prior_works,
            prior_profiles=prior_profiles,
            current_profiles=prior_profiles,
        )
    with pytest.raises(ValueError, match="prior or current rows"):
        validate_person_hold_losses(
            (selected,),
            prior_revision=selected.prior_revision,
            prior_works=prior_works,
            prior_profiles=prior_profiles[:2],
            current_profiles=[],
        )
    with pytest.raises(ValueError, match="not unique"):
        validate_person_hold_losses(
            (selected,),
            prior_revision=selected.prior_revision,
            prior_works=prior_works,
            prior_profiles=[prior_profiles[0], prior_profiles[0], prior_profiles[2]],
            current_profiles=[],
        )
    for change in (
        {"source_doi": "10.1000/wrong"},
        {"work_id": "work-" + "0" * 64},
        {"collaboration_eligible": False},
        {"map_eligible": False},
    ):
        changed = [{**prior_profiles[0], **change}, *prior_profiles[1:]]
        with pytest.raises(ValueError, match="prior binding"):
            validate_person_hold_losses(
                (selected,),
                prior_revision=selected.prior_revision,
                prior_works=prior_works,
                prior_profiles=changed,
                current_profiles=[],
            )
    with pytest.raises(ValueError, match="prior binding"):
        validate_person_hold_losses(
            (selected,),
            prior_revision=selected.prior_revision,
            prior_works=[{**prior_works[0], "map_eligible": False}, *prior_works[1:]],
            prior_profiles=prior_profiles,
            current_profiles=[],
        )
    with pytest.raises(ValueError, match="prior binding"):
        validate_person_hold_losses(
            (
                replace(
                    selected,
                    prior_observations=(
                        (
                            selected.prior_observations[0][0],
                            selected.prior_observations[0][1],
                            "observation-wrong",
                            selected.prior_observations[0][3],
                            True,
                        ),
                        *selected.prior_observations[1:],
                    ),
                ),
            ),
            prior_revision=selected.prior_revision,
            prior_works=prior_works,
            prior_profiles=prior_profiles,
            current_profiles=[],
        )


def test_published_bio_parent_preserves_absent_held_person_links() -> None:
    selected = hold()
    prior_works, prior_profiles = prior_hold_tables()
    assert (
        validate_person_hold_losses(
            (selected,),
            prior_revision=BIO_HOLD_RELEASE_REVISION,
            prior_works=prior_works,
            prior_profiles=[],
            current_profiles=[],
        )
        == frozenset()
    )
    for previous, current in ((prior_profiles, []), ([], prior_profiles)):
        with pytest.raises(ValueError, match="prior or current rows"):
            validate_person_hold_losses(
                (selected,),
                prior_revision=BIO_HOLD_RELEASE_REVISION,
                prior_works=prior_works,
                prior_profiles=previous,
                current_profiles=current,
            )


def test_successor_release_keeps_exact_hold_without_reapproving_losses() -> None:
    prior_works, prior_profiles = prior_hold_tables()
    selected = hold()
    approved = validate_person_hold_losses(
        (selected,),
        prior_revision=BIO_SUCCESSOR,
        prior_works=prior_works,
        prior_profiles=[],
        current_profiles=[],
    )
    assert approved == frozenset()
    published, receipt, held_ids = select_released_person_observations(
        source_rows(),
        registry(),
        (selected,),
        source_snapshot_sha256="9" * 64,
        prior_revision=BIO_SUCCESSOR,
    )
    assert not published.person_id.eq(TAASAN).any()
    assert receipt[0]["held_source_rows"] == 3
    assert held_ids == {row[2] for row in selected.source_observations}
    with pytest.raises(ValueError, match="parent revision differs"):
        validate_person_hold_losses(
            (replace(selected, source_snapshot_sha256="f" * 64),),
            prior_revision=BIO_SUCCESSOR,
            prior_works=prior_works,
            prior_profiles=[],
            current_profiles=[],
        )
    with pytest.raises(ValueError, match="successor works differ"):
        validate_person_hold_losses(
            (selected,),
            prior_revision=BIO_SUCCESSOR,
            prior_works=[{**prior_works[0], "doi": "10.1000/wrong"}, *prior_works[1:]],
            prior_profiles=[],
            current_profiles=[],
        )
    with pytest.raises(ValueError, match="prior or current rows"):
        validate_person_hold_losses(
            (selected,),
            prior_revision=BIO_SUCCESSOR,
            prior_works=prior_works,
            prior_profiles=prior_profiles,
            current_profiles=[],
        )
    changed = source_rows()
    changed.loc[changed.person_id == TAASAN, "source_doi"] = "10.1000/wrong"
    with pytest.raises(ValueError, match="source rows differ"):
        select_released_person_observations(
            changed,
            registry(),
            (selected,),
            source_snapshot_sha256="9" * 64,
            prior_revision=BIO_SUCCESSOR,
        )


def test_only_partial_person_rows_are_held_and_coauthors_remain() -> None:
    raw = source_rows()
    published, receipt, observation_ids = select_released_person_observations(
        raw,
        registry(),
        (hold(),),
        source_snapshot_sha256=hold().source_snapshot_sha256,
        prior_revision=hold().prior_revision,
    )
    assert len(raw) == 233
    assert len(published) == 230
    assert raw.loc[raw.person_id == TAASAN].shape[0] == 3
    assert not published.person_id.eq(TAASAN).any()
    tetali_counts = (
        published.loc[published.person_id == TETALI, "source_type"]
        .value_counts()
        .to_dict()
    )
    assert tetali_counts == {
        "scholar": 222,
        "curated_verified": 2,
    }
    for doi, _, _ in hold().source_observations:
        coauthors = set(published.loc[published.source_doi == doi, "person_id"])
        assert coauthors == {KINDERLEHRER, ROLLETT}
    assert receipt[0]["held_source_rows"] == 3
    assert receipt[0]["display_reason"] == "Publication history pending full capture"
    assert observation_ids == {row[2] for row in hold().source_observations}


@pytest.mark.parametrize("change", ["snapshot", "missing", "new_doi", "scholar"])
def test_source_or_identity_drift_blocks_release(change) -> None:
    raw = source_rows()
    tested_registry = registry(scholar_id="new-profile" if change == "scholar" else "")
    snapshot_sha = "0" * 64 if change == "snapshot" else hold().source_snapshot_sha256
    if change == "missing":
        raw = raw.drop(raw.index[0])
    if change == "new_doi":
        raw.loc[0, "source_doi"] = "10.1000/changed"
    with pytest.raises(ValueError, match="hold (source rows differ|identity differs)"):
        select_released_person_observations(
            raw,
            tested_registry,
            (hold(),),
            source_snapshot_sha256=snapshot_sha,
            prior_revision=hold().prior_revision,
        )


def test_published_bio_hold_survives_new_snapshot_with_identical_owner_rows() -> None:
    raw = source_rows()
    published, receipt, ids = select_released_person_observations(
        raw,
        registry(),
        (hold(),),
        source_snapshot_sha256="0" * 64,
        prior_revision=BIO_HOLD_RELEASE_REVISION,
    )
    assert not published.person_id.eq(TAASAN).any()
    assert receipt[0]["held_source_rows"] == 3
    assert ids == {item[2] for item in hold().source_observations}
    raw.loc[0, "source_doi"] = "10.1000/changed"
    with pytest.raises(ValueError, match="source rows differ"):
        select_released_person_observations(
            raw,
            registry(),
            (hold(),),
            source_snapshot_sha256="0" * 64,
            prior_revision=BIO_HOLD_RELEASE_REVISION,
        )
    with pytest.raises(ValueError, match="source rows differ"):
        select_released_person_observations(
            source_rows(),
            registry(),
            (replace(hold(), source_snapshot_sha256="f" * 64),),
            source_snapshot_sha256="0" * 64,
            prior_revision=BIO_HOLD_RELEASE_REVISION,
        )


def test_missing_held_person_in_registry_fails_even_without_source_rows() -> None:
    raw = source_rows().loc[lambda frame: frame.person_id != TAASAN]
    with pytest.raises(ValueError, match="identity differs"):
        select_released_person_observations(
            raw,
            SimpleNamespace(people_by_id={}),
            (hold(),),
            source_snapshot_sha256=hold().source_snapshot_sha256,
            prior_revision=hold().prior_revision,
        )


def test_missing_registry_person_with_held_rows_fails_closed() -> None:
    with pytest.raises(ValueError, match="identity differs"):
        select_released_person_observations(
            source_rows(),
            SimpleNamespace(people_by_id={}),
            (hold(),),
            source_snapshot_sha256=hold().source_snapshot_sha256,
            prior_revision=hold().prior_revision,
        )


def test_missing_person_id_column_fails_if_held_person_is_in_registry() -> None:
    frame = pandas.DataFrame({"scholar_id": ["synthetic"]})
    with pytest.raises(ValueError, match="source lacks person IDs"):
        select_released_person_observations(
            frame,
            registry(),
            (hold(),),
            source_snapshot_sha256=hold().source_snapshot_sha256,
            prior_revision=hold().prior_revision,
        )
    published, receipt, ids = select_released_person_observations(
        frame, registry(), (), source_snapshot_sha256="other", prior_revision="other"
    )
    assert published.equals(frame)
    assert receipt == []
    assert ids == set()


def clean_tables():
    return {
        "profile_publications": pandas.DataFrame({"person_id": [KINDERLEHRER]}),
        "authorships": pandas.DataFrame({"person_id": [ROLLETT]}),
        "works": pandas.DataFrame(
            {
                "observation_ids": [["coauthor-observation"]],
                "person_ids": [[KINDERLEHRER, ROLLETT]],
            }
        ),
        "people": pandas.DataFrame(
            {
                "person_id": [TAASAN, TETALI],
                "publication_observation_count": [0, 224],
                "scholar_observation_count": [0, 222],
                "curated_observation_count": [0, 2],
                "unique_work_count": [0, 224],
                "mapped_work_count": [0, 185],
            }
        ),
    }


def test_prewrite_gate_preserves_roster_but_rejects_held_authorship() -> None:
    ids = {row[2] for row in hold().source_observations}
    require_no_held_person_contributions(clean_tables(), {TAASAN}, ids)
    for table in ("profile_publications", "authorships"):
        tables = clean_tables()
        tables[table].loc[len(tables[table])] = [TAASAN]
        with pytest.raises(ValueError, match=f"published {table}"):
            require_no_held_person_contributions(tables, {TAASAN}, ids)
    tables = clean_tables()
    tables["works"].at[0, "observation_ids"] = [next(iter(ids))]
    with pytest.raises(ValueError, match="published works"):
        require_no_held_person_contributions(tables, {TAASAN}, ids)
    tables = clean_tables()
    tables["works"].at[0, "person_ids"] = [KINDERLEHRER, TAASAN]
    with pytest.raises(ValueError, match="assigned to published work"):
        require_no_held_person_contributions(tables, {TAASAN}, ids)
    tables = clean_tables()
    tables["people"].loc[0, "mapped_work_count"] = 3
    with pytest.raises(ValueError, match="zero own works"):
        require_no_held_person_contributions(tables, {TAASAN}, ids)


def test_hold_can_be_cleared_only_by_reviewing_manifest(tmp_path) -> None:
    body = {"schema_version": 1, "holds": []}
    path = tmp_path / "holds.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    assert load_person_publication_holds(path) == ()
    raw = source_rows()
    published, receipt, ids = select_released_person_observations(
        raw,
        registry(),
        (),
        source_snapshot_sha256="0" * 64,
        prior_revision=hold().prior_revision,
    )
    assert published.equals(raw)
    assert receipt == []
    assert ids == set()


def test_three_canonical_works_and_tetali_erdos_survive_person_hold() -> None:
    roster = load_registry()
    selected = hold()
    external = {
        "person_id": "external-paul-erdos",
        "display_name": "Paul Erdős",
        "source_url": "https://www.impan.pl/en/publishing-house/journals-and-series/acta-arithmetica/all/69/3/108504/independence-of-solution-sets-and-minimal-asymptotic-bases",
    }

    def make_row(
        person_id,
        *,
        title,
        doi="",
        author_pub_id,
        source_type="scholar",
        year=2000,
        external_coauthors=None,
    ):
        person = roster.people_by_id[person_id]
        memberships = [
            {
                "department_id": membership.department_id,
                "role": membership.role,
                "included": membership.included,
            }
            for membership in roster.memberships
            if membership.person_id == person_id
        ]
        departments = sorted(
            item["department_id"] for item in memberships if item["included"]
        )
        return {
            "person_id": person_id,
            "display_name": person.display_name,
            "faculty": person.display_name,
            "department_ids": departments,
            "memberships": memberships,
            "title": title,
            "authors": "Shlomo Ta'asan, David Kinderlehrer, Anthony Rollett",
            "year": year,
            "venue": "Research journal",
            "citation": "Research journal",
            "citation_count": 1,
            "embedding": [0.0, 1.0],
            "source_type": source_type,
            "scholar_id": person.scholar_id if source_type == "scholar" else "",
            "author_pub_id": author_pub_id,
            "source_doi": doi,
            "source_url": f"https://doi.org/{doi}"
            if doi
            else "https://scholar.google.com/",
            "source_record_json": "{}",
            "source_native_id": "",
            "source_verified_at": "2026-10-25" if source_type != "scholar" else "",
            "source_verification_urls": [f"https://doi.org/{doi}"] if doi else [],
            "external_coauthors": external_coauthors or [],
            "fetched_at_utc": "2026-10-25T00:00:00+00:00"
            if source_type == "scholar"
            else "",
        }

    rows = []
    for index, (doi, author_pub_id, _) in enumerate(selected.source_observations):
        title = f"Verified microstructure research paper {index}"
        rows.append(
            make_row(
                TAASAN,
                title=title,
                doi=doi,
                author_pub_id=author_pub_id,
                source_type="curated_verified",
                year=1998 + index,
            )
        )
        for coauthor in (KINDERLEHRER, ROLLETT):
            rows.append(
                make_row(
                    coauthor,
                    title=title,
                    doi=doi,
                    author_pub_id=f"{roster.people_by_id[coauthor].scholar_id}:paper-{index}",
                    year=1998 + index,
                )
            )
    rows.extend(
        make_row(
            TETALI,
            title=f"Mathematics research contribution number {index}",
            author_pub_id=f"FmZaK1wAAAAJ:paper-{index}",
        )
        for index in range(222)
    )
    for index, doi in enumerate(("10.1002/rsa.3240010302", "10.4064/aa-69-3-243-258")):
        rows.append(
            make_row(
                TETALI,
                title=f"Erdős and Tetali research paper {index}",
                doi=doi,
                author_pub_id=f"curated-doi:{TETALI}:{doi}",
                source_type="curated_verified",
                year=1990 + index,
                external_coauthors=[external],
            )
        )
    raw = pandas.DataFrame(rows)
    published, receipt, held_ids = select_released_person_observations(
        raw,
        roster,
        (selected,),
        source_snapshot_sha256=selected.source_snapshot_sha256,
        prior_revision=selected.prior_revision,
    )
    assert len(raw) == 233
    assert len(published) == 230
    tables = build_dataset_tables(published, roster)
    require_no_held_person_contributions(tables, {TAASAN}, held_ids)
    taasan = tables["people"].set_index("person_id").loc[TAASAN]
    tetali = tables["people"].set_index("person_id").loc[TETALI]
    assert taasan["publication_observation_count"] == 0
    assert tetali["scholar_observation_count"] == 222
    assert tetali["curated_observation_count"] == 2
    doi_works = tables["works"].set_index("doi")
    for doi, _, _ in selected.source_observations:
        work = doi_works.loc[doi]
        assert work["map_eligible"]
        assert set(work["person_ids"]) == {KINDERLEHRER, ROLLETT}
        assert not set(work["observation_ids"]) & held_ids
    works = tables["works"].copy()
    for coordinate in ("x", "y", "tsne_x", "tsne_y"):
        works[coordinate] = 0.0
    works["keyword_id"] = "keyword-math"
    works["keyword"] = "mathematics"
    works["keyword_extracted"] = "mathematics"
    works["keyword_label_reviewed"] = True
    works["detail_keyword_id"] = "keyword-math-research"
    works["detail_keyword"] = "research"
    works["detail_keyword_extracted"] = "research"
    works["detail_keyword_label_reviewed"] = True
    artifact = build_map_artifact(
        works,
        roster,
        generated_at_utc="2026-10-25T00:00:00+00:00",
        publication_hold_reasons={TAASAN: receipt[0]["display_reason"]},
    )
    points_by_doi = {point["doi"]: point for point in artifact["points"]}
    for doi, _, _ in selected.source_observations:
        assert points_by_doi[doi]["faculty_ids"] == [KINDERLEHRER, ROLLETT]
    erdos_points = [
        point
        for point in artifact["points"]
        if "external-paul-erdos" in point["external_coauthor_ids"]
    ]
    assert len(erdos_points) == 2
    assert all(point["faculty_ids"] == [TETALI] for point in erdos_points)
    assert artifact["catalogs"]["external_coauthors"] == [external]
    taasan_catalog = next(
        item for item in artifact["catalogs"]["faculty"] if item["person_id"] == TAASAN
    )
    assert taasan_catalog["publication_count"] == 0
    assert taasan_catalog["publication_hold_reason"] == (
        "Publication history pending full capture"
    )
