"""Public subsets must not disclose privately held capture rows."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pandas
import pytest

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
from map_of_research.provisional_holds import validate_provisional_holds
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


def test_controlled_private_preflight_binds_full_review_to_public_subset(tmp_path):
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
