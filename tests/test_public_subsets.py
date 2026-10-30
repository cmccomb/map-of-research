"""Public subsets must not disclose privately held capture rows."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pandas
import pytest

from map_of_research import private_subset_preflight as private_preflight
from map_of_research.coverage import validate_complete_receipt
from map_of_research.io import sha256_file
from map_of_research.private_subset_preflight import (
    require_private_preflight_receipt,
    validate_private_public_subsets,
)
from map_of_research.profile_releases import (
    ProfileRelease,
    load_profile_releases,
    select_released_observations,
)
from map_of_research.provisional_holds import (
    ProvisionalHoldProfile,
    validate_provisional_holds,
)
from map_of_research.public_subsets import (
    ids_sha256,
    parse_public_subset,
    require_public_subset_boundary,
    validate_public_subset_cache,
)

SCHOLAR = "subsetAAAAJ"
PERSON = "person-subset"
PRIOR = "person-prior"
SELECTED = [f"{SCHOLAR}:{suffix}" for suffix in ("one", "three", "five")]
WITHHELD = {f"{SCHOLAR}:{suffix}" for suffix in ("two", "four")}


def _metadata() -> dict:
    return {
        "observed_visible_rows": 5,
        "private_capture_sha256": "a" * 64,
        "private_review_sha256": "b" * 64,
        "selected_ids_sha256": ids_sha256(set(SELECTED)),
        "withheld_rows": 2,
        "withheld_hold_rows": 1,
        "withheld_excluded_rows": 1,
        "withheld_ids_sha256": ids_sha256(WITHHELD),
    }


def _cache() -> dict:
    rows = []
    for index, row_id in ((1, SELECTED[2]), (3, SELECTED[0]), (5, SELECTED[1])):
        rows.append(
            {
                "author_pub_id": row_id,
                "capture_row_index": index,
                "title": f"Public work {index}",
                "author": "Subset Owner",
                # Selected rows may still fail ordinary year/venue map quality.
                "pub_year": "" if index == 3 else "2026",
                "venue": "" if index == 5 else "Proceedings",
                "num_citations": 0,
                "citation_count_display": "",
                "pub_url": (
                    "https://scholar.google.com/citations?user="
                    f"{SCHOLAR}&citation_for_view={row_id}"
                ),
            }
        )
    return {
        "schema_version": 1,
        "scholar_id": SCHOLAR,
        "display_name": "Subset Owner",
        "fetched_at_utc": "2026-10-30T00:00:00+00:00",
        "publication_count": 3,
        "publications": rows,
        "browser_capture": {
            "method": "user_chrome_visible_profile",
            "profile_url": f"https://scholar.google.com/citations?user={SCHOLAR}",
            "coverage": "complete_visible_profile",
            "cache_scope": "reviewed_public_subset",
            "displayed_count": 5,
            "limit": 5,
            "more_remain": False,
            "selected_author_pub_ids": sorted(SELECTED),
            **_metadata(),
        },
    }


def _release(tmp_path):
    cache_dir = tmp_path / "authors"
    cache_dir.mkdir()
    path = cache_dir / f"{SCHOLAR}.json"
    path.write_text(json.dumps(_cache()))
    decision = ProfileRelease(
        person_id=PERSON,
        scholar_id=SCHOLAR,
        state="released",
        expected_source_rows=3,
        cache_sha256=sha256_file(path),
        reason="Independent row review selected three of five visible citations.",
        reviewed_at="2026-10-30",
        public_subset=parse_public_subset(_metadata()),
    )
    registry = SimpleNamespace(
        people_by_id={PERSON: SimpleNamespace(scholar_id=SCHOLAR)}
    )
    frame = pandas.DataFrame(
        [
            {
                "person_id": PERSON,
                "scholar_id": SCHOLAR,
                "source_type": "scholar",
                "author_pub_id": row_id,
                "title": f"Public work {index}",
            }
            for index, row_id in enumerate(SELECTED, 1)
        ]
        + [
            {
                "person_id": PRIOR,
                "scholar_id": "priorAAAAJ",
                "source_type": "scholar",
                "author_pub_id": "priorAAAAJ:kept",
                "title": "Previously public work",
            }
        ]
    )
    return cache_dir, decision, registry, frame


def _tables():
    profiles = pandas.DataFrame(
        [
            {
                "person_id": PERSON,
                "scholar_id": SCHOLAR,
                "author_pub_id": row_id,
                "work_id": f"work-{index}",
                "map_eligible": index == 1,
            }
            for index, row_id in enumerate(SELECTED, 1)
        ]
        + [
            {
                "person_id": PRIOR,
                "scholar_id": "priorAAAAJ",
                "author_pub_id": "priorAAAAJ:kept",
                "work_id": "work-prior",
                "map_eligible": True,
            }
        ]
    )
    works = pandas.DataFrame(
        [
            {
                "work_id": f"work-{index}",
                "scholar_ids": [SCHOLAR],
                "author_pub_ids": [row_id],
                "map_eligible": index == 1,
            }
            for index, row_id in enumerate(SELECTED, 1)
        ]
        + [
            {
                "work_id": "work-prior",
                "scholar_ids": ["priorAAAAJ"],
                "author_pub_ids": ["priorAAAAJ:kept"],
                "map_eligible": True,
            }
        ]
    )
    authorships = pandas.DataFrame(
        [
            {"scholar_id": SCHOLAR, "author_pub_id": SELECTED[0], "work_id": "work-1"},
            {
                "scholar_id": "priorAAAAJ",
                "author_pub_id": "priorAAAAJ:kept",
                "work_id": "work-prior",
            },
        ]
    )
    receipt = [
        {
            "person_id": PERSON,
            "scholar_id": SCHOLAR,
            "cache_scope": "reviewed_public_subset",
            "selected_author_pub_ids": sorted(SELECTED),
            "selected_ids_sha256": ids_sha256(set(SELECTED)),
        }
    ]
    map_artifact = {
        "points": [
            {
                "work_id": "work-1",
                "faculty_ids": [PERSON],
                "source_url": (
                    "https://scholar.google.com/citations?user="
                    f"{SCHOLAR}&citation_for_view={SELECTED[0]}"
                ),
            },
            {
                "work_id": "work-prior",
                "faculty_ids": [PRIOR],
                "source_url": "",
            },
        ]
    }
    return (
        {"profile_publications": profiles, "authorships": authorships, "works": works},
        receipt,
        map_artifact,
    )


def test_complete_receipt_accepts_only_selected_rows_without_public_capture_files(
    tmp_path,
):
    cache = _cache()
    validate_complete_receipt(SCHOLAR, cache, tmp_path / "empty-private-captures")
    assert validate_public_subset_cache(SCHOLAR, cache).withheld_rows == 2
    assert cache["publications"][0]["capture_row_index"] == 1
    assert cache["publications"][1]["capture_row_index"] == 3
    assert cache["publications"][2]["capture_row_index"] == 5


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cache: cache["browser_capture"].update(observed_visible_rows=4),
        lambda cache: cache["browser_capture"].update(withheld_rows=1),
        lambda cache: cache["browser_capture"].update(selected_ids_sha256="0" * 64),
        lambda cache: cache["browser_capture"].update(private_capture_sha256="bad"),
        lambda cache: cache["browser_capture"].update(
            held_observations=["private row"]
        ),
        lambda cache: cache["browser_capture"].update(
            source_html_filename="private.html"
        ),
        lambda cache: cache["publications"][0].update(author_pub_id=f"{SCHOLAR}:two"),
        lambda cache: cache["publications"][0].update(capture_row_index=3),
        lambda cache: cache["publications"][0].update(
            pub_url="https://example.org/wrong"
        ),
    ],
)
def test_subset_receipt_rejects_private_payloads_and_drift(mutation):
    cache = _cache()
    mutation(cache)
    with pytest.raises(ValueError):
        validate_public_subset_cache(SCHOLAR, cache)


def test_release_manifest_binds_private_hashes_and_selected_count(tmp_path):
    cache_dir, decision, registry, frame = _release(tmp_path)
    release = {**vars(decision), "public_subset": _metadata()}
    path = tmp_path / "releases.json"
    path.write_text(json.dumps({"schema_version": 1, "profiles": [release]}))
    loaded = load_profile_releases(path)
    assert loaded == (decision,)
    result, receipt, held_ids = select_released_observations(
        frame, registry, loaded, cache_dir=cache_dir
    )
    assert result.to_dict("records") == frame.to_dict("records")
    assert result.iloc[-1]["author_pub_id"] == "priorAAAAJ:kept"
    assert held_ids == set()
    assert receipt[0]["selected_author_pub_ids"] == sorted(SELECTED)
    assert receipt[0]["withheld_rows"] == 2

    for changed in (
        replace(decision, expected_source_rows=4),
        replace(
            decision,
            public_subset=replace(
                decision.public_subset, private_review_sha256="0" * 64
            ),
        ),
    ):
        with pytest.raises(ValueError):
            select_released_observations(
                frame, registry, (changed,), cache_dir=cache_dir
            )


def test_subset_omissions_do_not_enter_legacy_public_hold_or_exclusion_manifests(
    tmp_path,
):
    cache_dir, decision, _, _ = _release(tmp_path)
    holds, receipt = validate_provisional_holds(
        (),
        (decision,),
        [],
        cache_dir=cache_dir,
        excluded_ids=set(),
        confirmed_ids=set(),
    )
    assert holds == {} and receipt == []
    with pytest.raises(ValueError, match="Private subset exclusions"):
        validate_provisional_holds(
            (),
            (decision,),
            [],
            cache_dir=cache_dir,
            excluded_ids={(SCHOLAR, f"{SCHOLAR}:four")},
            confirmed_ids=set(),
        )


def test_public_tables_and_map_accept_selected_rows_and_preserve_prior_rows():
    tables, receipt, artifact = _tables()
    # The subset owner can legitimately coauthor a prior mapped work whose
    # source observation belongs to another Scholar profile.
    artifact["points"][1]["faculty_ids"].append(PERSON)
    require_public_subset_boundary(tables, receipt, artifact=artifact)
    assert tables["profile_publications"].iloc[-1]["author_pub_id"] == "priorAAAAJ:kept"
    assert tables["works"].iloc[-1]["author_pub_ids"] == ["priorAAAAJ:kept"]


@pytest.mark.parametrize(
    "surface", ["profiles", "authorships", "works", "map", "map_url"]
)
def test_public_boundary_rejects_unselected_row_on_every_surface(surface):
    tables, receipt, artifact = _tables()
    if surface == "profiles":
        tables["profile_publications"] = pandas.concat(
            [
                tables["profile_publications"],
                pandas.DataFrame(
                    [
                        {
                            "person_id": PERSON,
                            "scholar_id": SCHOLAR,
                            "author_pub_id": f"{SCHOLAR}:two",
                            "work_id": "work-private",
                            "map_eligible": False,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    elif surface == "authorships":
        tables["authorships"] = pandas.concat(
            [
                tables["authorships"],
                pandas.DataFrame(
                    [
                        {
                            "scholar_id": SCHOLAR,
                            "author_pub_id": f"{SCHOLAR}:two",
                            "work_id": "work-1",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    elif surface == "works":
        tables["works"].at[0, "author_pub_ids"] = [SELECTED[0], f"{SCHOLAR}:two"]
    elif surface == "map":
        artifact["points"].append(
            {"work_id": "work-private", "faculty_ids": [PERSON], "source_url": ""}
        )
    else:
        artifact["points"][0]["source_url"] = (
            "https://scholar.google.com/citations?user="
            f"{SCHOLAR}&citation_for_view={SCHOLAR}:two"
        )
    with pytest.raises(ValueError, match="leaked"):
        require_public_subset_boundary(tables, receipt, artifact=artifact)


@pytest.mark.parametrize("surface", ["native_id", "verification_url", "map_metadata"])
def test_public_boundary_scans_other_public_metadata_fields(surface):
    tables, receipt, artifact = _tables()
    withheld = f"{SCHOLAR}:two"
    if surface == "native_id":
        tables["works"]["source_native_ids"] = [[] for _ in range(len(tables["works"]))]
        tables["works"].at[0, "source_native_ids"] = [withheld]
    elif surface == "verification_url":
        tables["works"]["source_verification_urls"] = [
            [] for _ in range(len(tables["works"]))
        ]
        tables["works"].at[0, "source_verification_urls"] = [
            f"https://scholar.google.com/citations?citation_for_view={SCHOLAR}%3Atwo"
        ]
    else:
        artifact["points"][0]["source_native_ids"] = [withheld]
    with pytest.raises(ValueError, match="leaked"):
        require_public_subset_boundary(tables, receipt, artifact=artifact)


def _private_subset_case(tmp_path):
    source = tmp_path / "source"
    private = tmp_path / "private"
    cache_dir = source / "data" / "authors"
    cache_dir.mkdir(parents=True)
    private.mkdir()
    slug = "subset-owner"
    rows_by_index = {
        item["capture_row_index"]: item for item in _cache()["publications"]
    }
    raw_rows = []
    review_rows = []
    for index in range(1, 6):
        cache_row = rows_by_index.get(index)
        row_id = (
            cache_row["author_pub_id"]
            if cache_row is not None
            else f"{SCHOLAR}:{'two' if index == 2 else 'four'}"
        )
        title = cache_row["title"] if cache_row is not None else f"Private row {index}"
        raw_rows.append(
            {
                "index": index,
                "author_pub_id": row_id,
                "citation_url": (
                    "https://scholar.google.com/citations?user="
                    f"{SCHOLAR}&citation_for_view={row_id}"
                ),
                "title": title,
                "authors": "Subset Owner",
                "venue": cache_row["venue"] if cache_row else "Unreleased",
                "year": cache_row["pub_year"] if cache_row else "2026",
                "citations": "",
            }
        )
        review_rows.append(
            {
                "index": index,
                "author_pub_id": row_id,
                "title": title,
                "disposition": (
                    "retain_candidate"
                    if cache_row is not None
                    else "hold"
                    if index == 2
                    else "exclude_from_release"
                ),
            }
        )
    raw_path = private / f"{slug}-chrome-visible-5-2026-10-30.raw.json"
    review_path = private / f"{slug}-row-review-2026-10-30.json"
    html_path = private / f"{slug}-chrome-rendered-2026-10-30.private.html"
    raw_path.write_text(
        json.dumps(
            {
                "displayed_count": 5,
                "more_remain": False,
                "show_more_disabled": True,
                "challenge": False,
                "page_error": None,
                "rows": raw_rows,
            }
        )
    )
    review_path.write_text(
        json.dumps({"person_id": PERSON, "scholar_id": SCHOLAR, "rows": review_rows})
    )
    html_path.write_text("<html><body>Private full profile</body></html>")
    capture_receipt_path = private / f"{slug}-chrome-capture-receipt-2026-10-30.json"
    capture_receipt_path.write_text(
        json.dumps(
            {
                "person_id": PERSON,
                "scholar_id": SCHOLAR,
                "observed_rows": 5,
                "unique_citation_ids": 5,
                "raw_json": {
                    "sha256": sha256_file(raw_path),
                    "bytes": raw_path.stat().st_size,
                },
            }
        )
    )
    metadata = {
        **_metadata(),
        "private_capture_sha256": sha256_file(raw_path),
        "private_review_sha256": sha256_file(review_path),
        "private_rendered_html_sha256": sha256_file(html_path),
    }
    cache = _cache()
    cache["browser_capture"].update(metadata)
    cache_path = cache_dir / f"{SCHOLAR}.json"
    cache_path.write_text(json.dumps(cache))
    selection_manifest_path = source / "status" / "selection.json"
    selection_manifest_path.parent.mkdir()
    selection_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "local_source_candidate_only",
                "base_source_status_commit": "a" * 40,
                "registry_producer_commit": "b" * 40,
                "private_capture_files_or_html_copied": False,
                "profiles": [
                    {
                        "slug": slug,
                        "person_id": PERSON,
                        "scholar_id": SCHOLAR,
                        "expected_source_rows": 3,
                        "cache_sha256": sha256_file(cache_path),
                        "selected_author_pub_ids": sorted(SELECTED),
                        "public_subset": metadata,
                    }
                ],
                "selected_ids_sha256": ids_sha256(set(SELECTED)),
                "withheld_ids_sha256": ids_sha256(WITHHELD),
                "totals": {
                    "observed_visible_rows": 5,
                    "selected_rows": 3,
                    "withheld_rows": 2,
                    "withheld_hold_rows": 1,
                    "withheld_excluded_rows": 1,
                },
            }
        )
    )
    snapshot_path = source / "snapshots" / "subset.parquet"
    snapshot_path.parent.mkdir()
    pandas.DataFrame(
        [
            {
                "scholar_id": SCHOLAR,
                "author_pub_id": row["author_pub_id"],
                "source_record_json": json.dumps(row),
            }
            for row in cache["publications"]
        ]
    ).to_parquet(snapshot_path)
    receipt = validate_private_public_subsets(
        source_root=source,
        private_outputs=private,
        selection_manifest_path=selection_manifest_path,
        snapshot_path=snapshot_path,
    )
    receipt_path = private / "controlled-preflight.json"
    receipt_path.write_text(json.dumps(receipt))
    decision = ProfileRelease(
        person_id=PERSON,
        scholar_id=SCHOLAR,
        state="released",
        expected_source_rows=3,
        cache_sha256=sha256_file(cache_path),
        reason="Selected three reviewed rows",
        reviewed_at="2026-10-30",
        public_subset=parse_public_subset(metadata),
    )
    kwargs = {
        "receipt_path": receipt_path,
        "selection_manifest_path": selection_manifest_path,
        "snapshot_path": snapshot_path,
        "cache_dir": cache_dir,
        "release_decisions": (decision,),
    }
    return SimpleNamespace(
        source=source,
        private=private,
        cache_path=cache_path,
        raw_path=raw_path,
        review_path=review_path,
        html_path=html_path,
        capture_receipt_path=capture_receipt_path,
        selection_manifest_path=selection_manifest_path,
        snapshot_path=snapshot_path,
        receipt_path=receipt_path,
        decision=decision,
        kwargs=kwargs,
    )


def test_controlled_private_preflight_binds_full_review_to_public_subset(tmp_path):
    case = _private_subset_case(tmp_path)
    kwargs = case.kwargs
    receipt_path = case.receipt_path
    selection_manifest_path = case.selection_manifest_path
    raw_path = case.raw_path
    source = case.source
    private = case.private
    snapshot_path = case.snapshot_path
    assert require_private_preflight_receipt(**kwargs)["selected_rows"] == 3
    with pytest.raises(ValueError, match="requires a private local preflight"):
        require_private_preflight_receipt(**{**kwargs, "receipt_path": None})

    receipt_text = receipt_path.read_text()
    receipt_with_private_row = json.loads(receipt_text)
    receipt_with_private_row["held_author_pub_ids"] = [f"{SCHOLAR}:two"]
    receipt_path.write_text(json.dumps(receipt_with_private_row))
    with pytest.raises(ValueError, match="differs from public source"):
        require_private_preflight_receipt(**kwargs)
    receipt_path.write_text(receipt_text)

    selection_text = selection_manifest_path.read_text()
    selection_with_private_row = json.loads(selection_text)
    selection_with_private_row["held_author_pub_ids"] = [f"{SCHOLAR}:two"]
    selection_manifest_path.write_text(json.dumps(selection_with_private_row))
    with pytest.raises(ValueError, match="invalid or private fields"):
        require_private_preflight_receipt(**kwargs)
    selection_manifest_path.write_text(selection_text)

    raw_path.write_text(raw_path.read_text().replace("Private row 2", "Changed row 2"))
    with pytest.raises(ValueError, match="source binding differs"):
        validate_private_public_subsets(
            source_root=source,
            private_outputs=private,
            selection_manifest_path=selection_manifest_path,
            snapshot_path=snapshot_path,
        )
    selection_manifest_path.write_text(selection_manifest_path.read_text() + " ")
    with pytest.raises(ValueError, match="differs from public source"):
        require_private_preflight_receipt(**kwargs)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda value: value.pop("private_review_sha256"), "incomplete"),
        (lambda value: value.update(held_author_pub_ids=["private"]), "unknown"),
        (lambda value: value.update(observed_visible_rows=1), "counts"),
        (lambda value: value.update(withheld_hold_rows=2), "counts"),
        (lambda value: value.update(private_rendered_html_sha256="bad"), "HTML"),
    ],
)
def test_public_subset_metadata_rejects_incomplete_or_inconsistent_fields(
    mutation, error
):
    metadata = _metadata()
    mutation(metadata)
    with pytest.raises(ValueError, match=error):
        parse_public_subset(metadata)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda cache: cache.pop("publications"), "incomplete"),
        (lambda cache: cache["browser_capture"].update(cache_scope="full"), "scope"),
        (lambda cache: cache["publications"].__setitem__(0, None), "row is invalid"),
        (
            lambda cache: cache["publications"][0].update(private_note="held row"),
            "payload field",
        ),
        (
            lambda cache: cache["publications"][0].update(
                author_pub_id=SELECTED[1],
                pub_url=(
                    "https://scholar.google.com/citations?user="
                    f"{SCHOLAR}&citation_for_view={SELECTED[1]}"
                ),
            ),
            "Unselected citation",
        ),
    ],
)
def test_public_subset_cache_rejects_malformed_selected_rows(mutation, error):
    cache = deepcopy(_cache())
    mutation(cache)
    with pytest.raises(ValueError, match=error):
        validate_public_subset_cache(SCHOLAR, cache)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda tables, receipt, artifact: tables["works"].drop(
                columns=["author_pub_ids"], inplace=True
            ),
            "relationship columns",
        ),
        (
            lambda tables, receipt, artifact: tables["works"].at.__setitem__(
                (1, "work_id"), "work-1"
            ),
            "duplicate work IDs",
        ),
        (lambda tables, receipt, artifact: artifact.update(points={}), "lacks points"),
        (
            lambda tables, receipt, artifact: receipt[0].update(
                selected_ids_sha256="0" * 64
            ),
            "receipt changed",
        ),
        (
            lambda tables, receipt, artifact: artifact["points"].append(None),
            "invalid point",
        ),
        (
            lambda tables, receipt, artifact: artifact["points"][0].pop("faculty_ids"),
            "lacks faculty IDs",
        ),
        (
            lambda tables, receipt, artifact: tables[
                "profile_publications"
            ].at.__setitem__((0, "map_eligible"), False),
            "Unselected work leaked",
        ),
        (
            lambda tables, receipt, artifact: tables.update(other=42),
            "lacks tabular",
        ),
    ],
)
def test_public_boundary_rejects_bad_relationships_and_map_shapes(mutation, error):
    tables, receipt, artifact = _tables()
    mutation(tables, receipt, artifact)
    with pytest.raises(ValueError, match=error):
        require_public_subset_boundary(tables, receipt, artifact=artifact)


def test_public_boundary_accepts_tables_without_a_map_artifact():
    tables, receipt, _ = _tables()
    require_public_subset_boundary(tables, receipt)


def test_subset_release_manifest_rejects_private_row_fields(tmp_path):
    cache_dir, decision, _, _ = _release(tmp_path)
    release = {**vars(decision), "public_subset": _metadata()}
    release["held_author_pub_ids"] = [f"{SCHOLAR}:two"]
    path = tmp_path / "releases.json"
    path.write_text(json.dumps({"schema_version": 1, "profiles": [release]}))
    with pytest.raises(ValueError, match="Private row fields"):
        load_profile_releases(path)
    assert cache_dir.is_dir()


def test_reviewed_subset_cache_requires_a_matching_manifest(tmp_path):
    cache_dir, decision, registry, frame = _release(tmp_path)
    with pytest.raises(ValueError, match="lacks manifest"):
        select_released_observations(
            frame,
            registry,
            (replace(decision, public_subset=None),),
            cache_dir=cache_dir,
        )


def test_private_subset_holds_cannot_enter_public_hold_manifest(tmp_path):
    cache_dir, decision, _, _ = _release(tmp_path)
    profile = ProvisionalHoldProfile(
        person_id=PERSON,
        scholar_id=SCHOLAR,
        expected_source_rows=3,
        cache_sha256=decision.cache_sha256,
        held_observations=(),
    )
    with pytest.raises(ValueError, match="Private subset holds"):
        validate_provisional_holds(
            (profile,),
            (decision,),
            [],
            cache_dir=cache_dir,
            excluded_ids=set(),
            confirmed_ids=set(),
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda manifest: manifest.update(base_source_status_commit="wrong"),
            "base_source_status_commit",
        ),
        (
            lambda manifest: manifest["profiles"][0].update(private_row="held"),
            "invalid or private fields",
        ),
        (
            lambda manifest: manifest["profiles"][0].update(
                selected_author_pub_ids=sorted(SELECTED)[:-1]
            ),
            "selected IDs differ",
        ),
        (
            lambda manifest: manifest.update(selected_ids_sha256="0" * 64),
            "selection aggregate differs",
        ),
        (
            lambda manifest: manifest["totals"].update(withheld_rows=1),
            "selection aggregate differs",
        ),
    ],
)
def test_selection_manifest_rejects_unsafe_or_inconsistent_selection(
    tmp_path, mutation, error
):
    case = _private_subset_case(tmp_path)
    manifest = json.loads(case.selection_manifest_path.read_text())
    mutation(manifest)
    with pytest.raises(ValueError, match=error):
        private_preflight._validate_selection_manifest(manifest)


@pytest.mark.parametrize(
    ("review", "expected"),
    [
        ({"release_disposition": "retain_candidate_after_release_gates"}, "selected"),
        ({"release_disposition": "hold_questionable_attribution"}, "hold"),
        ({"map_disposition": "candidate_after_normal_gates"}, "selected"),
        ({"map_disposition": "hold_uncertain"}, "hold"),
        ({"category": "hard_exclusion"}, "exclude"),
    ],
)
def test_private_review_vocabulary_normalization(review, expected):
    assert private_preflight._decision(review) == expected


def test_unknown_review_disposition_and_nonnumeric_citations_fail_closed():
    for review in (
        {"category": "guess"},
        {"release_disposition": "guess"},
        {"map_disposition": "guess"},
        {},
    ):
        with pytest.raises(ValueError, match="Unknown private row-review"):
            private_preflight._decision(review)
    with pytest.raises(ValueError, match="citation count is not numeric"):
        private_preflight._publication({"citations": "many"})


def _rebind_changed_private_evidence(case):
    """Keep byte digests current so a mutation reaches the intended row gate."""

    cache = json.loads(case.cache_path.read_text())
    subset = cache["browser_capture"]
    subset["private_capture_sha256"] = sha256_file(case.raw_path)
    subset["private_review_sha256"] = sha256_file(case.review_path)
    case.cache_path.write_text(json.dumps(cache))
    capture_receipt = json.loads(case.capture_receipt_path.read_text())
    capture_receipt["raw_json"] = {
        "sha256": sha256_file(case.raw_path),
        "bytes": case.raw_path.stat().st_size,
    }
    case.capture_receipt_path.write_text(json.dumps(capture_receipt))
    selection = json.loads(case.selection_manifest_path.read_text())
    selection["profiles"][0]["cache_sha256"] = sha256_file(case.cache_path)
    selection["profiles"][0]["public_subset"] = {
        key: value
        for key, value in subset.items()
        if key in _metadata() | {"private_rendered_html_sha256": ""}
    }
    case.selection_manifest_path.write_text(json.dumps(selection))


@pytest.mark.parametrize(
    ("which", "error"),
    [
        ("private_inside_source", "must remain outside"),
        ("manifest_inside_private", "must be in the source tree"),
        ("manifest_outside_source", "differs from source tree"),
        ("snapshot_missing_provenance", "lacks source-row provenance"),
        ("missing_private_capture", "needs one"),
        ("private_capture_in_source", "Private full capture entered"),
    ],
)
def test_private_preflight_rejects_unsafe_file_boundaries(tmp_path, which, error):
    case = _private_subset_case(tmp_path)
    kwargs = {
        "source_root": case.source,
        "private_outputs": case.private,
        "selection_manifest_path": case.selection_manifest_path,
        "snapshot_path": case.snapshot_path,
    }
    if which == "private_inside_source":
        kwargs["private_outputs"] = case.source
    elif which == "manifest_inside_private":
        target = case.private / "selection.json"
        target.write_text(case.selection_manifest_path.read_text())
        kwargs["selection_manifest_path"] = target
    elif which == "manifest_outside_source":
        target = tmp_path / "selection.json"
        target.write_text(case.selection_manifest_path.read_text())
        kwargs["selection_manifest_path"] = target
    elif which == "snapshot_missing_provenance":
        pandas.DataFrame({"scholar_id": [SCHOLAR]}).to_parquet(case.snapshot_path)
    elif which == "missing_private_capture":
        case.raw_path.unlink()
    else:
        target = case.source / "data" / "captures"
        target.mkdir(parents=True)
        (target / "subset-owner-full.json").write_text("private")
    with pytest.raises(ValueError, match=error):
        validate_private_public_subsets(**kwargs)


@pytest.mark.parametrize(
    ("which", "error"),
    [
        ("missing_rows", "capture/review rows differ"),
        ("invalid_row", "review row is invalid"),
        ("row_title", "capture/review row differs"),
        ("citation_url", "citation identity differs"),
        ("repeated_selected", "selected ID repeats"),
        ("review_partition", "review partition differs"),
        ("snapshot_row", "snapshot differs from selected cache"),
        ("withheld_aggregate", "Aggregate private/public selection IDs differ"),
    ],
)
def test_private_preflight_rejects_row_and_snapshot_drift(tmp_path, which, error):
    case = _private_subset_case(tmp_path)
    if which == "snapshot_row":
        frame = pandas.read_parquet(case.snapshot_path)
        frame.at[0, "source_record_json"] = json.dumps({"changed": True})
        frame.to_parquet(case.snapshot_path)
    elif which == "withheld_aggregate":
        selection = json.loads(case.selection_manifest_path.read_text())
        selection["withheld_ids_sha256"] = "0" * 64
        case.selection_manifest_path.write_text(json.dumps(selection))
    else:
        raw = json.loads(case.raw_path.read_text())
        review = json.loads(case.review_path.read_text())
        if which == "missing_rows":
            raw["rows"] = []
        elif which == "invalid_row":
            raw["rows"][0] = None
        elif which == "row_title":
            raw["rows"][0]["title"] = "Different private title"
        elif which == "citation_url":
            raw["rows"][0]["citation_url"] = "https://example.org/not-scholar"
        elif which == "repeated_selected":
            raw["rows"][2]["author_pub_id"] = raw["rows"][0]["author_pub_id"]
            raw["rows"][2]["citation_url"] = raw["rows"][0]["citation_url"]
            review["rows"][2]["author_pub_id"] = raw["rows"][0]["author_pub_id"]
        else:
            review["rows"][1]["disposition"] = "exclude_from_release"
        case.raw_path.write_text(json.dumps(raw))
        case.review_path.write_text(json.dumps(review))
        _rebind_changed_private_evidence(case)
    with pytest.raises(ValueError, match=error):
        validate_private_public_subsets(
            source_root=case.source,
            private_outputs=case.private,
            selection_manifest_path=case.selection_manifest_path,
            snapshot_path=case.snapshot_path,
        )


@pytest.mark.parametrize(
    ("which", "error"),
    [
        ("unexpected_profile_field", "unexpected public fields"),
        ("wrong_profile_identity", "profiles differ from release manifest"),
        ("wrong_person_binding", "row binding differs"),
    ],
)
def test_private_preflight_receipt_rejects_profile_drift(tmp_path, which, error):
    case = _private_subset_case(tmp_path)
    receipt = json.loads(case.receipt_path.read_text())
    audit = receipt["profiles"][0]
    if which == "unexpected_profile_field":
        audit["private_title"] = "held"
    elif which == "wrong_profile_identity":
        audit["scholar_id"] = "wrongAAAAJ"
    else:
        audit["person_id"] = "person-wrong"
    case.receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=error):
        require_private_preflight_receipt(**case.kwargs)


def test_private_preflight_is_unneeded_when_no_subset_is_released(tmp_path):
    assert (
        require_private_preflight_receipt(
            receipt_path=None,
            selection_manifest_path=None,
            snapshot_path=tmp_path / "not-read.parquet",
            cache_dir=tmp_path / "not-read",
            release_decisions=(),
        )
        is None
    )
