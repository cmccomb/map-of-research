import json
from html import escape
from pathlib import Path

import pandas
import pytest

from map_of_research import publisher
from map_of_research.coverage import (
    main,
    validate_complete_receipt,
    validate_coverage,
)
from map_of_research.io import atomic_write_json, sha256_file
from map_of_research.snapshot import build_snapshot
from tests.registry_helpers import write_registry


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    people, memberships, departments = write_registry(
        tmp_path / "registry",
        people=[
            {
                "person_id": "person-alpha",
                "display_name": "Alpha Person",
                "scholar_id": "alphaAAAAJ",
            }
        ],
        memberships=[
            {
                "person_id": "person-alpha",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://www.ece.cmu.edu/directory/faculty.html",
                "verified_at": "2026-10-24",
            }
        ],
    )
    cache_dir = tmp_path / "authors"
    capture_dir = tmp_path / "captures"
    baseline = tmp_path / "baseline.json"
    cache_dir.mkdir()
    capture_dir.mkdir()
    atomic_write_json(
        baseline,
        {
            "schema_version": 1,
            "frozen_snapshot_commit": "a" * 40,
            "frozen_snapshot_sha256": "b" * 64,
            "profiles": {},
        },
    )
    return people, memberships, departments, cache_dir, capture_dir, baseline


def _options(paths: tuple[Path, Path, Path, Path, Path, Path]) -> dict:
    people, memberships, departments, cache_dir, capture_dir, baseline = paths
    return {
        "people_path": people,
        "memberships_path": memberships,
        "departments_path": departments,
        "cache_dir": cache_dir,
        "captures_dir": capture_dir,
        "baseline_path": baseline,
        "prior_id_crosswalk_path": people.parent.parent / "absent-crosswalk.json",
    }


def _cache(paths: tuple, *, complete: bool) -> dict:
    cache_dir, capture_dir = paths[3:5]
    citation_url = (
        "https://scholar.google.com/citations?user=alphaAAAAJ"
        "&citation_for_view=alphaAAAAJ:one"
    )
    row = {
        "author_pub_id": "alphaAAAAJ:one",
        "title": "One",
        "author": "A Alpha",
        "venue": "Journal",
        "pub_year": "2025",
        "num_citations": 2,
        "pub_url": citation_url,
        "capture_row_index": 1,
    }
    cache = {
        "schema_version": 1,
        "scholar_id": "alphaAAAAJ",
        "display_name": "Alpha Person",
        "fetched_at_utc": "2026-10-25T00:00:00+00:00",
        "publication_count": 1,
        "publications": [row],
    }
    if complete:
        source_json = capture_dir / "alpha.json"
        source_html = capture_dir / "alpha.html"
        atomic_write_json(
            source_json,
            {
                "scholar_user_id": "alphaAAAAJ",
                "displayed_count": 1,
                "more_remain": False,
                "challenge": False,
                "page_error": "",
                "works": [
                    {
                        "index": 1,
                        "citation_url": citation_url,
                        "title": "One",
                        "authors": "A Alpha",
                        "venue": "Journal",
                        "year": "2025",
                        "citation_count": "2",
                    }
                ],
            },
        )
        source_html.write_text(
            '<table><tr class="gsc_a_tr"><td class="gsc_a_t"><a class="gsc_a_at" href="'
            + citation_url
            + '">One</a><div class="gs_gray">A Alpha</div>'
            + '<div class="gs_gray">Journal</div></td>'
            + '<td class="gsc_a_c"><a class="gsc_a_ac">2</a></td>'
            + '<td class="gsc_a_y"><span class="gsc_a_h">2025</span></td>'
            + "</tr></table>"
            + '<button id="gsc_bpf_more" disabled="">Show more</button>'
        )
        cache["browser_capture"] = {
            "coverage": "complete_visible_profile",
            "profile_url": "https://scholar.google.com/citations?user=alphaAAAAJ",
            "displayed_count": 1,
            "more_remain": False,
            "limit": 5000,
            "source_json_filename": source_json.name,
            "source_json_sha256": sha256_file(source_json),
            "source_html_filename": source_html.name,
            "source_html_sha256": sha256_file(source_html),
        }
    atomic_write_json(cache_dir / "alphaAAAAJ.json", cache)
    return cache


def _baseline(paths: tuple, *, included: bool = True) -> None:
    baseline = json.loads(paths[5].read_text())
    baseline["profiles"]["alphaAAAAJ"] = {
        "person_id": "person-alpha",
        "cache_sha256": sha256_file(paths[3] / "alphaAAAAJ.json"),
        "included": included,
        "author_pub_ids": ["alphaAAAAJ:one"],
    }
    atomic_write_json(paths[5], baseline)


def _prior_id_review(paths: tuple, kind: str = "current_scholar") -> tuple[dict, Path]:
    """Freeze one old ID and review its exact replacement in a new capture."""

    _cache(paths, complete=True)
    _baseline(paths)
    baseline = json.loads(paths[5].read_text())
    baseline["profiles"]["alphaAAAAJ"]["author_pub_ids"] = ["alphaAAAAJ:old"]
    atomic_write_json(paths[5], baseline)
    evidence_url = "https://example.org/verified-work"
    target_id = "alphaAAAAJ:one"
    target_title = "One"
    target_year = "2025"
    classification = "duplicate_rekeyed"
    if kind == "curated_doi":
        target_id = "10.1000/reviewed"
        classification = "supporting_material"
    elif kind == "curated_native":
        target_id = "conference:reviewed"
        classification = "historical_conference_abstract"
    elif kind == "nonpublication":
        kind = "current_scholar"
        classification = "nonpublication_rekeyed"
    curated_works = paths[0].parent.parent / "works.json"
    exclusions = paths[0].parent.parent / "exclusions.json"
    works = []
    if kind.startswith("curated_"):
        key = "doi" if kind == "curated_doi" else "source_native_id"
        works.append(
            {
                "person_id": "person-alpha",
                key: target_id,
                "title": target_title,
                "year": int(target_year),
                "source_url": evidence_url,
            }
        )
    atomic_write_json(curated_works, {"works": works})
    exclusions_rows = []
    if classification == "nonpublication_rekeyed":
        exclusions_rows.append(
            {
                "scholar_id": "alphaAAAAJ",
                "author_pub_id": target_id,
                "reason_code": "nonpublication",
                "evidence_url": evidence_url,
            }
        )
    atomic_write_json(exclusions, {"exclusions": exclusions_rows})
    crosswalk = paths[0].parent.parent / "crosswalk.json"
    atomic_write_json(
        crosswalk,
        {
            "schema_version": 1,
            "scholar_id": "alphaAAAAJ",
            "person_id": "person-alpha",
            "frozen_snapshot_commit": baseline["frozen_snapshot_commit"],
            "frozen_snapshot_sha256": baseline["frozen_snapshot_sha256"],
            "prior_cache_sha256": baseline["profiles"]["alphaAAAAJ"]["cache_sha256"],
            "replacement_cache_sha256": sha256_file(paths[3] / "alphaAAAAJ.json"),
            "reviewed_at": "2026-10-25",
            "entries": [
                {
                    "prior_author_pub_id": "alphaAAAAJ:old",
                    "legacy_title": "Old title",
                    "legacy_year": "2024",
                    "legacy_row_sha256": "a" * 64,
                    "classification": classification,
                    "target_kind": kind,
                    "target_id": target_id,
                    "target_title": target_title,
                    "target_year": target_year,
                    "evidence_url": evidence_url,
                    "review_note": "Reviewed historical record.",
                }
            ],
        },
    )
    options = {
        **_options(paths),
        "prior_id_crosswalk_path": crosswalk,
        "curated_works_path": curated_works,
        "exclusions_path": exclusions,
    }
    return options, crosswalk


@pytest.mark.parametrize(
    "kind", ["current_scholar", "curated_doi", "curated_native", "nonpublication"]
)
def test_reviewed_prior_id_can_be_reconciled_to_exact_surviving_target(
    tmp_path: Path, kind: str
) -> None:
    options, _ = _prior_id_review(_paths(tmp_path), kind)
    assert validate_coverage(**options) == {
        "legacy_profiles": 0,
        "complete_profiles": 1,
        "reconciled_prior_ids": 1,
    }


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("scholar_id", "otherAAAAJ", "does not match"),
        ("person_id", "person-other", "does not match"),
        ("frozen_snapshot_commit", "c" * 40, "does not match"),
        ("frozen_snapshot_sha256", "d" * 64, "does not match"),
        ("prior_cache_sha256", "e" * 64, "does not match"),
        ("replacement_cache_sha256", "f" * 64, "does not match"),
        ("reviewed_at", "yesterday", "does not match"),
    ],
)
def test_prior_id_review_is_bound_to_frozen_and_replacement_sources(
    tmp_path: Path, field: str, value: str, error: str
) -> None:
    options, crosswalk_path = _prior_id_review(_paths(tmp_path))
    crosswalk = json.loads(crosswalk_path.read_text())
    crosswalk[field] = value
    atomic_write_json(crosswalk_path, crosswalk)
    with pytest.raises(ValueError, match=error):
        validate_coverage(**options)


@pytest.mark.parametrize("mutation", ["omit", "extra", "duplicate"])
def test_prior_id_review_rejects_missing_extra_or_duplicate_exceptions(
    tmp_path: Path, mutation: str
) -> None:
    options, crosswalk_path = _prior_id_review(_paths(tmp_path))
    crosswalk = json.loads(crosswalk_path.read_text())
    if mutation == "omit":
        crosswalk["entries"] = []
    elif mutation == "extra":
        extra = dict(crosswalk["entries"][0])
        extra["prior_author_pub_id"] = "alphaAAAAJ:never-published"
        crosswalk["entries"].append(extra)
    else:
        crosswalk["entries"].append(dict(crosswalk["entries"][0]))
    atomic_write_json(crosswalk_path, crosswalk)
    with pytest.raises(ValueError, match="omits or adds prior IDs"):
        validate_coverage(**options)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("classification", "arbitrary", "entry is invalid"),
        ("target_kind", "curated_native", "entry is invalid"),
        ("target_id", "alphaAAAAJ:absent", "target is absent"),
        ("target_title", "Other", "target is absent"),
        ("target_year", "2024", "target is absent"),
        ("evidence_url", "http://example.org/work", "entry is invalid"),
        ("legacy_row_sha256", "bad", "entry is invalid"),
    ],
)
def test_prior_id_review_rejects_unverified_resolution(
    tmp_path: Path, field: str, value: str, error: str
) -> None:
    options, crosswalk_path = _prior_id_review(_paths(tmp_path))
    crosswalk = json.loads(crosswalk_path.read_text())
    crosswalk["entries"][0][field] = value
    atomic_write_json(crosswalk_path, crosswalk)
    with pytest.raises(ValueError, match=error):
        validate_coverage(**options)


def test_prior_id_review_rejects_missing_curated_or_excluded_target(
    tmp_path: Path,
) -> None:
    options, _ = _prior_id_review(_paths(tmp_path), "curated_native")
    atomic_write_json(options["curated_works_path"], {"works": []})
    with pytest.raises(ValueError, match="curated work target is absent"):
        validate_coverage(**options)
    options, _ = _prior_id_review(_paths(tmp_path / "other"), "nonpublication")
    atomic_write_json(options["exclusions_path"], {"exclusions": []})
    with pytest.raises(ValueError, match="nonpublication exclusion is absent"):
        validate_coverage(**options)


def test_prior_id_review_rejects_malformed_crosswalk_and_target_catalogs(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    options, crosswalk_path = _prior_id_review(paths, "curated_native")
    original = json.loads(crosswalk_path.read_text())
    atomic_write_json(crosswalk_path, {"entries": original["entries"]})
    with pytest.raises(ValueError, match="crosswalk is invalid"):
        validate_coverage(**options)
    original["entries"][0].pop("review_note")
    atomic_write_json(crosswalk_path, original)
    with pytest.raises(ValueError, match="invalid entry"):
        validate_coverage(**options)
    _prior_id_review(paths, "curated_native")
    atomic_write_json(options["curated_works_path"], {"works": "invalid"})
    with pytest.raises(ValueError, match="targets are invalid"):
        validate_coverage(**options)


@pytest.mark.parametrize("mutation", ["nonobject", "duplicate"])
def test_prior_id_review_rejects_invalid_current_rows(
    tmp_path: Path, mutation: str
) -> None:
    paths = _paths(tmp_path)
    options, crosswalk_path = _prior_id_review(paths)
    cache_path = paths[3] / "alphaAAAAJ.json"
    cache = json.loads(cache_path.read_text())
    cache["publications"].append(
        "invalid" if mutation == "nonobject" else dict(cache["publications"][0])
    )
    atomic_write_json(cache_path, cache)
    crosswalk = json.loads(crosswalk_path.read_text())
    crosswalk["replacement_cache_sha256"] = sha256_file(cache_path)
    atomic_write_json(crosswalk_path, crosswalk)
    error = "needs valid current rows|duplicate current IDs"
    with pytest.raises(ValueError, match=error):
        validate_coverage(**options)


def test_prior_id_review_rejects_curated_target_without_matching_evidence(
    tmp_path: Path,
) -> None:
    options, _ = _prior_id_review(_paths(tmp_path), "curated_native")
    curated = json.loads(options["curated_works_path"].read_text())
    curated["works"][0]["source_url"] = "https://example.org/other"
    atomic_write_json(options["curated_works_path"], curated)
    with pytest.raises(ValueError, match="curated work evidence does not match"):
        validate_coverage(**options)


def test_prior_id_review_rejects_excluded_academic_replacement(tmp_path: Path) -> None:
    options, _ = _prior_id_review(_paths(tmp_path))
    atomic_write_json(
        options["exclusions_path"],
        {
            "exclusions": [
                {
                    "scholar_id": "alphaAAAAJ",
                    "author_pub_id": "alphaAAAAJ:one",
                    "reason_code": "false_profile_attribution",
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="Scholar replacement is excluded"):
        validate_coverage(**options)


def test_prior_id_review_does_not_replace_complete_receipt_or_snapshot_check(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    options, _ = _prior_id_review(paths)
    snapshot = tmp_path / "snapshot.parquet"
    build_snapshot(
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
        cache_dir=paths[3],
        snapshot_path=snapshot,
        manifest_path=tmp_path / "manifest.json",
    )
    assert (
        validate_coverage(**options, snapshot_path=snapshot)["reconciled_prior_ids"]
        == 1
    )
    frame = pandas.read_parquet(snapshot)
    frame.loc[0, "source_record_json"] = "tampered"
    frame.to_parquet(snapshot, index=False)
    with pytest.raises(ValueError, match="Snapshot rows differ"):
        validate_coverage(**options, snapshot_path=snapshot)
    cache_path = paths[3] / "alphaAAAAJ.json"
    cache = json.loads(cache_path.read_text())
    cache.pop("browser_capture")
    atomic_write_json(cache_path, cache)
    crosswalk = json.loads(options["prior_id_crosswalk_path"].read_text())
    crosswalk["replacement_cache_sha256"] = sha256_file(cache_path)
    atomic_write_json(options["prior_id_crosswalk_path"], crosswalk)
    with pytest.raises(ValueError, match="Complete source receipt is missing"):
        validate_coverage(**options)


def test_new_complete_capture_reconciles_at_publication_boundary(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _cache(paths, complete=True)
    options = _options(paths)
    assert validate_coverage(**options) == {
        "legacy_profiles": 0,
        "complete_profiles": 1,
        "reconciled_prior_ids": 0,
    }
    snapshot = tmp_path / "snapshot.parquet"
    build_snapshot(
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
        cache_dir=paths[3],
        snapshot_path=snapshot,
        manifest_path=tmp_path / "manifest.json",
    )
    result = validate_coverage(**options, snapshot_path=snapshot)
    assert result["complete_profiles"] == 1
    assert (
        main(
            [
                "--people",
                str(paths[0]),
                "--memberships",
                str(paths[1]),
                "--departments",
                str(paths[2]),
                "--cache-dir",
                str(paths[3]),
                "--captures-dir",
                str(paths[4]),
                "--baseline",
                str(paths[5]),
                "--snapshot",
                str(snapshot),
            ]
        )
        == 0
    )


def test_publisher_cli_requires_complete_source_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    cache = _cache(paths, complete=True)
    snapshot = tmp_path / "snapshot.parquet"
    manifest = tmp_path / "manifest.json"
    status = tmp_path / "status.json"
    args = [
        "--snapshot",
        str(snapshot),
        "--manifest",
        str(manifest),
        "--prior-revision",
        "a" * 40,
        "--people",
        str(paths[0]),
        "--memberships",
        str(paths[1]),
        "--departments",
        str(paths[2]),
        "--scholar-cache-dir",
        str(paths[3]),
        "--scholar-captures-dir",
        str(paths[4]),
        "--scholar-baseline",
        str(paths[5]),
        "--status-file",
        str(status),
    ]
    build_snapshot(
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
        cache_dir=paths[3],
        snapshot_path=snapshot,
        manifest_path=manifest,
    )
    monkeypatch.setenv(publisher.HF_TOKEN_ENV, "test-token")
    monkeypatch.setattr(
        publisher,
        "validate_math_coverage",
        lambda *args, **kwargs: {"complete_scholar_profiles": 0},
    )
    uploads: list[str] = []
    monkeypatch.setattr(
        publisher,
        "publish_snapshot",
        lambda *args, **kwargs: uploads.append("upload") or {"works": 1},
    )
    assert publisher.main(args) == 0
    assert uploads == ["upload"]
    assert json.loads(status.read_text())["scholar_coverage"] == {
        "legacy_profiles": 0,
        "complete_profiles": 1,
        "reconciled_prior_ids": 0,
    }

    cache.pop("browser_capture")
    atomic_write_json(paths[3] / "alphaAAAAJ.json", cache)
    build_snapshot(
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
        cache_dir=paths[3],
        snapshot_path=snapshot,
        manifest_path=manifest,
    )
    assert publisher.main(args) == 1
    assert uploads == ["upload"]
    failed = json.loads(status.read_text())
    assert failed["status"] == "failure"
    assert "Complete source receipt is missing" in failed["error_message"]


def test_partial_new_and_changed_caches_cannot_publish(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _cache(paths, complete=False)
    with pytest.raises(ValueError, match="Complete source receipt is missing"):
        validate_coverage(**_options(paths))
    _baseline(paths)
    assert validate_coverage(**_options(paths))["legacy_profiles"] == 1
    cache_path = paths[3] / "alphaAAAAJ.json"
    cache = json.loads(cache_path.read_text())
    cache["publications"][0]["title"] = "Changed"
    atomic_write_json(cache_path, cache)
    with pytest.raises(ValueError, match="Complete source receipt is missing"):
        validate_coverage(**_options(paths))


def test_formerly_excluded_legacy_person_needs_complete_capture(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _cache(paths, complete=False)
    _baseline(paths, included=False)
    with pytest.raises(ValueError, match="Complete source receipt is missing"):
        validate_coverage(**_options(paths))


def test_previously_published_ids_and_cache_cannot_disappear(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _cache(paths, complete=True)
    _baseline(paths)
    baseline = json.loads(paths[5].read_text())
    baseline["profiles"]["alphaAAAAJ"]["author_pub_ids"] = ["alphaAAAAJ:old"]
    atomic_write_json(paths[5], baseline)
    with pytest.raises(ValueError, match="Previously published Scholar IDs"):
        validate_coverage(**_options(paths))
    baseline["profiles"]["alphaAAAAJ"]["author_pub_ids"] = ["alphaAAAAJ:one"]
    atomic_write_json(paths[5], baseline)
    (paths[3] / "alphaAAAAJ.json").unlink()
    with pytest.raises(
        ValueError, match="previously published Scholar cache is missing"
    ):
        validate_coverage(**_options(paths))


@pytest.mark.parametrize(
    "field,value",
    [
        ("coverage", "capped_partial"),
        ("more_remain", True),
        ("displayed_count", 2),
        ("limit", 0),
        ("profile_url", "https://scholar.google.com/citations?user=wrong"),
        ("source_json_sha256", "0" * 64),
        ("source_html_sha256", "0" * 64),
        ("source_json_filename", "../alpha.json"),
        ("source_json_sha256", "invalid"),
    ],
)
def test_receipt_rejects_false_claims_and_tampered_sources(
    tmp_path: Path, field: str, value: object
) -> None:
    paths = _paths(tmp_path)
    cache = _cache(paths, complete=True)
    cache["browser_capture"][field] = value
    with pytest.raises(ValueError):
        validate_complete_receipt("alphaAAAAJ", cache, paths[4])


@pytest.mark.parametrize(
    "field,value",
    [
        ("scholar_user_id", "wrong"),
        ("displayed_count", 2),
        ("more_remain", True),
        ("challenge", True),
        ("page_error", "blocked"),
        ("works", []),
    ],
)
def test_receipt_rejects_bad_saved_json(
    tmp_path: Path, field: str, value: object
) -> None:
    paths = _paths(tmp_path)
    cache = _cache(paths, complete=True)
    source = paths[4] / "alpha.json"
    capture = json.loads(source.read_text())
    capture[field] = value
    atomic_write_json(source, capture)
    cache["browser_capture"]["source_json_sha256"] = sha256_file(source)
    with pytest.raises(ValueError, match="Saved capture is incomplete"):
        validate_complete_receipt("alphaAAAAJ", cache, paths[4])


def test_receipt_rejects_invalid_source_row(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    cache = _cache(paths, complete=True)
    source = paths[4] / "alpha.json"
    capture = json.loads(source.read_text())
    capture["works"] = ["bad"]
    atomic_write_json(source, capture)
    cache["browser_capture"]["source_json_sha256"] = sha256_file(source)
    with pytest.raises(ValueError, match="Invalid source publication row"):
        validate_complete_receipt("alphaAAAAJ", cache, paths[4])


@pytest.mark.parametrize(
    "field,value",
    [("year", "2024"), ("citation_count", "3"), ("citation_count", "unknown")],
)
def test_receipt_reconciles_source_year_and_citations(
    tmp_path: Path, field: str, value: str
) -> None:
    paths = _paths(tmp_path)
    cache = _cache(paths, complete=True)
    source = paths[4] / "alpha.json"
    capture = json.loads(source.read_text())
    capture["works"][0][field] = value
    atomic_write_json(source, capture)
    cache["browser_capture"]["source_json_sha256"] = sha256_file(source)
    with pytest.raises(ValueError, match="Cached row differs"):
        validate_complete_receipt("alphaAAAAJ", cache, paths[4])


@pytest.mark.parametrize(
    "old,new",
    [
        ('disabled=""', ""),
        ('class="gsc_a_tr"', 'class="other"'),
        ("alphaAAAAJ:one", "alphaAAAAJ:other"),
        ("citation_for_view", "other_parameter"),
    ],
)
def test_receipt_requires_disabled_final_control_and_exact_html_rows(
    tmp_path: Path, old: str, new: str
) -> None:
    paths = _paths(tmp_path)
    cache = _cache(paths, complete=True)
    html = paths[4] / "alpha.html"
    html.write_text(html.read_text().replace(old, new))
    cache["browser_capture"]["source_html_sha256"] = sha256_file(html)
    with pytest.raises(ValueError, match="Saved HTML does not show|Cached row differs"):
        validate_complete_receipt("alphaAAAAJ", cache, paths[4])


@pytest.mark.parametrize(
    "old,new",
    [
        (">One</a>", ">Another</a>"),
        (">A Alpha</div>", ">B Beta</div>"),
        (">Journal</div>", ">Other Journal</div>"),
        (">2</a>", ">3</a>"),
        (">2025</span>", ">2024</span>"),
    ],
)
def test_receipt_rejects_html_text_changed_after_source_capture(
    tmp_path: Path, old: str, new: str
) -> None:
    paths = _paths(tmp_path)
    cache = _cache(paths, complete=True)
    html = paths[4] / "alpha.html"
    html.write_text(html.read_text().replace(old, new))
    cache["browser_capture"]["source_html_sha256"] = sha256_file(html)
    with pytest.raises(ValueError, match="Cached row differs from saved source"):
        validate_complete_receipt("alphaAAAAJ", cache, paths[4])


def test_receipt_reads_nested_text_and_only_first_two_gray_fields(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    cache = _cache(paths, complete=True)
    html = paths[4] / "alpha.html"
    content = (
        html.read_text()
        .replace(">A Alpha</div>", "><span>A Alpha</span></div>")
        .replace(
            ">Journal</div></td>",
            '>Journal</div><div class="gs_gray">Extra page metadata</div></td>',
        )
    )
    html.write_text(content)
    cache["browser_capture"]["source_html_sha256"] = sha256_file(html)
    validate_complete_receipt("alphaAAAAJ", cache, paths[4])


def test_receipt_reads_scholar_svg_title_labels(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    cache = _cache(paths, complete=True)
    title = "α-pinene between -50 and +25 ∘C"
    cache["publications"][0]["title"] = title
    source = paths[4] / "alpha.json"
    capture = json.loads(source.read_text())
    capture["works"][0]["title"] = title
    atomic_write_json(source, capture)
    cache["browser_capture"]["source_json_sha256"] = sha256_file(source)
    html = paths[4] / "alpha.html"
    html.write_text(
        html.read_text().replace(
            ">One</a>",
            '><svg class="gs_fsvg" aria-label="    α    "></svg>-pinene between '
            '<svg class="gs_fsvg" aria-label="-50"></svg> and '
            '<svg class="gs_fsvg" aria-label="+25"></svg> '
            '<svg class="gs_fsvg" aria-label="∘"></svg>C</a>',
        )
    )
    cache["browser_capture"]["source_html_sha256"] = sha256_file(html)
    validate_complete_receipt("alphaAAAAJ", cache, paths[4])

    html.write_text(html.read_text().replace('aria-label="+25"', 'aria-label="+26"'))
    cache["browser_capture"]["source_html_sha256"] = sha256_file(html)
    with pytest.raises(ValueError, match="Cached row differs from saved source"):
        validate_complete_receipt("alphaAAAAJ", cache, paths[4])


def _tina_repaired_title_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    captures = tmp_path / "data" / "captures"
    captures.mkdir(parents=True)
    status = tmp_path / "status"
    status.mkdir()
    raw_title = (
        "ity (see Kofman & Starobinsky, Lahav et al. 5). All spectra were normalized"
    )
    repaired_title = (
        "Large Scale Structure Formation in Mixed Dark Matter Models "
        "with a Cosmological Constant"
    )
    raw_rows = []
    cached_rows = []
    html_rows = []
    for index in range(1, 117):
        author_pub_id = (
            "au_cLfoAAAAJ:VL0QpB8kHFEC" if index == 116 else f"au_cLfoAAAAJ:row{index}"
        )
        title = raw_title if index == 116 else f"Work {index}"
        url = (
            "https://scholar.google.com/citations?user=au_cLfoAAAAJ"
            f"&citation_for_view={author_pub_id}"
        )
        raw_rows.append(
            {
                "index": index,
                "citation_url": url,
                "title": title,
                "authors": "A Author",
                "venue": "Proceedings",
                "year": "1998",
                "citation_count": "0",
            }
        )
        cached_rows.append(
            {
                "capture_row_index": index,
                "author_pub_id": author_pub_id,
                "pub_url": url,
                "title": repaired_title if index == 116 else title,
                "author": "A Author",
                "venue": "Proceedings",
                "pub_year": "1998",
                "num_citations": 0,
            }
        )
        html_rows.append(
            '<tr class="gsc_a_tr"><td class="gsc_a_t">'
            f'<a class="gsc_a_at" href="{escape(url)}">{escape(title)}</a>'
            '<div class="gs_gray">A Author</div>'
            '<div class="gs_gray">Proceedings</div></td>'
            '<td class="gsc_a_c"><a class="gsc_a_ac">0</a></td>'
            '<td class="gsc_a_y"><span class="gsc_a_h">1998</span></td></tr>'
        )
    source_json = captures / "tina.json"
    source_html = captures / "tina.html"
    atomic_write_json(
        source_json,
        {
            "scholar_user_id": "au_cLfoAAAAJ",
            "displayed_count": 116,
            "more_remain": False,
            "challenge": False,
            "page_error": "",
            "works": raw_rows,
        },
    )
    source_html.write_text(
        "<table>"
        + "".join(html_rows)
        + '</table><button id="gsc_bpf_more" disabled="">Show more</button>'
    )
    cache = {
        "scholar_id": "au_cLfoAAAAJ",
        "publication_count": 116,
        "publications": cached_rows,
        "browser_capture": {
            "coverage": "complete_visible_profile",
            "profile_url": "https://scholar.google.com/citations?user=au_cLfoAAAAJ",
            "displayed_count": 116,
            "more_remain": False,
            "limit": 5000,
            "source_json_filename": source_json.name,
            "source_json_sha256": sha256_file(source_json),
            "source_html_filename": source_html.name,
            "source_html_sha256": sha256_file(source_html),
        },
    }
    ledger = status / "physics-kahniashvili-cache-repair-2026-10-24.json"
    atomic_write_json(
        ledger,
        {
            "schema_version": 1,
            "review_state": "source_reviewed_cache_repaired",
            "capture_receipt": (
                "status/browser-capture-physics-tina-kahniashvili-2026-10-24.json"
            ),
            "repairs": [
                {
                    "scholar_id": "au_cLfoAAAAJ",
                    "capture_row": 116,
                    "author_pub_id": "au_cLfoAAAAJ:VL0QpB8kHFEC",
                    "cache_field": "title",
                    "captured_value": raw_title,
                    "repaired_value": repaired_title,
                    "evidence_url": (
                        "https://books.google.com/books?id=9sFUDwAAQBAJ&pg=PA269"
                    ),
                    "corroborating_url": (
                        "https://www.mao.kiev.ua/biblio/docs/special/"
                        "2007-04-19_Novosyadlyj_aref.pdf"
                    ),
                }
            ],
        },
    )
    return cache, captures, ledger


def test_only_reviewed_tina_row_116_cache_title_repair_is_accepted(
    tmp_path: Path,
) -> None:
    cache, captures, _ = _tina_repaired_title_fixture(tmp_path)
    validate_complete_receipt("au_cLfoAAAAJ", cache, captures)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_ledger",
        "invalid_ledger_json",
        "invalid_ledger_schema",
        "changed_evidence",
        "changed_repaired_title",
        "changed_row_116_author",
        "changed_other_row_title",
        "changed_row_116_id",
    ],
)
def test_tina_title_repair_does_not_waive_other_checks(
    tmp_path: Path, mutation: str
) -> None:
    cache, captures, ledger = _tina_repaired_title_fixture(tmp_path)
    if mutation == "missing_ledger":
        ledger.unlink()
    elif mutation == "invalid_ledger_json":
        ledger.write_text("{invalid")
    elif mutation == "invalid_ledger_schema":
        review = json.loads(ledger.read_text())
        review["schema_version"] = 2
        atomic_write_json(ledger, review)
    elif mutation == "changed_evidence":
        review = json.loads(ledger.read_text())
        review["repairs"][0]["corroborating_url"] = "https://example.test/unverified"
        atomic_write_json(ledger, review)
    elif mutation == "changed_repaired_title":
        cache["publications"][-1]["title"] = "Unverified replacement"
    elif mutation == "changed_row_116_author":
        cache["publications"][-1]["author"] = "Wrong author"
    elif mutation == "changed_other_row_title":
        cache["publications"][0]["title"] = "Wrong title"
    elif mutation == "changed_row_116_id":
        cache["publications"][-1]["author_pub_id"] = "au_cLfoAAAAJ:wrong"
    with pytest.raises(ValueError, match="Cached row differs from saved source"):
        validate_complete_receipt("au_cLfoAAAAJ", cache, captures)


def test_tina_title_repair_rejects_changed_raw_fragment(tmp_path: Path) -> None:
    cache, captures, _ = _tina_repaired_title_fixture(tmp_path)
    source_json = captures / "tina.json"
    source_html = captures / "tina.html"
    capture = json.loads(source_json.read_text())
    old_title = capture["works"][-1]["title"]
    new_title = "A different damaged title"
    capture["works"][-1]["title"] = new_title
    atomic_write_json(source_json, capture)
    source_html.write_text(
        source_html.read_text().replace(escape(old_title), escape(new_title))
    )
    cache["browser_capture"]["source_json_sha256"] = sha256_file(source_json)
    cache["browser_capture"]["source_html_sha256"] = sha256_file(source_html)
    with pytest.raises(ValueError, match="Cached row differs from saved source"):
        validate_complete_receipt("au_cLfoAAAAJ", cache, captures)


def _breivik_repaired_author_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    captures = tmp_path / "data" / "captures"
    captures.mkdir(parents=True)
    status = tmp_path / "status"
    status.mkdir()
    scholar_id = "AcW2EXwAAAAJ"
    captured = "K Breivik, S Chatterjee, J Andrews"
    repaired = "J Andrews, S Chatterjee, K Breivik"
    raw_rows = []
    cached_rows = []
    html_rows = []
    for index in range(1, 116):
        citation_id = (
            f"{scholar_id}:roLk4NBRz8UC" if index == 115 else f"{scholar_id}:row{index}"
        )
        citation_url = (
            f"https://scholar.google.com/citations?user={scholar_id}"
            f"&citation_for_view={citation_id}"
        )
        authors = captured if index == 115 else "A Author"
        title = f"Work {index}"
        raw_rows.append(
            {
                "index": index,
                "citation_url": citation_url,
                "title": title,
                "authors": authors,
                "venue": "AAS 233",
                "year": "2019",
                "citation_count": "0",
            }
        )
        cached_rows.append(
            {
                "capture_row_index": index,
                "author_pub_id": citation_id,
                "pub_url": citation_url,
                "title": title,
                "author": repaired if index == 115 else authors,
                "venue": "AAS 233",
                "pub_year": "2019",
                "num_citations": 0,
            }
        )
        html_rows.append(
            '<tr class="gsc_a_tr"><td class="gsc_a_t">'
            f'<a class="gsc_a_at" href="{escape(citation_url)}">{title}</a>'
            f'<div class="gs_gray">{authors}</div>'
            '<div class="gs_gray">AAS 233</div></td>'
            '<td class="gsc_a_c"><a class="gsc_a_ac">0</a></td>'
            '<td class="gsc_a_y"><span class="gsc_a_h">2019</span></td></tr>'
        )
    source_json = captures / "breivik.json"
    source_html = captures / "breivik.html"
    atomic_write_json(
        source_json,
        {
            "scholar_user_id": scholar_id,
            "displayed_count": 115,
            "more_remain": False,
            "challenge": False,
            "page_error": "",
            "works": raw_rows,
        },
    )
    source_html.write_text(
        "<table>"
        + "".join(html_rows)
        + '</table><button id="gsc_bpf_more" disabled="">Show more</button>'
    )
    cache = {
        "scholar_id": scholar_id,
        "publication_count": 115,
        "publications": cached_rows,
        "browser_capture": {
            "coverage": "complete_visible_profile",
            "profile_url": f"https://scholar.google.com/citations?user={scholar_id}",
            "displayed_count": 115,
            "more_remain": False,
            "limit": 5000,
            "source_json_filename": source_json.name,
            "source_json_sha256": sha256_file(source_json),
            "source_html_filename": source_html.name,
            "source_html_sha256": sha256_file(source_html),
        },
    }
    ledger = status / "physics-breivik-row115-cache-repair-2026-10-24.json"
    atomic_write_json(
        ledger,
        {
            "schema_version": 1,
            "review_state": "source_reviewed_cache_repaired",
            "capture_receipt": (
                "status/browser-capture-physics-katelyn-breivik-2026-10-24.json"
            ),
            "repairs": [
                {
                    "scholar_id": scholar_id,
                    "capture_row": 115,
                    "author_pub_id": f"{scholar_id}:roLk4NBRz8UC",
                    "cache_field": "author",
                    "captured_value": captured,
                    "repaired_value": repaired,
                    "evidence_url": "https://aas.org/files/aas233_abstracts.pdf",
                }
            ],
        },
    )
    return cache, captures, ledger


def test_only_reviewed_breivik_row_115_author_repair_is_accepted(
    tmp_path: Path,
) -> None:
    cache, captures, _ = _breivik_repaired_author_fixture(tmp_path)
    validate_complete_receipt("AcW2EXwAAAAJ", cache, captures)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_ledger",
        "invalid_ledger_json",
        "invalid_ledger_type",
        "invalid_ledger_schema",
        "invalid_review_state",
        "invalid_receipt",
        "invalid_repairs_type",
        "two_repairs",
        "invalid_repair_entry",
        "changed_evidence",
        "changed_repaired_author",
        "changed_other_row_author",
        "changed_row_115_id",
        "changed_raw_author",
    ],
)
def test_breivik_author_repair_does_not_waive_other_checks(
    tmp_path: Path, mutation: str
) -> None:
    cache, captures, ledger = _breivik_repaired_author_fixture(tmp_path)
    if mutation == "missing_ledger":
        ledger.unlink()
    elif mutation == "invalid_ledger_json":
        ledger.write_text("{invalid")
    elif mutation == "invalid_ledger_type":
        atomic_write_json(ledger, [])
    elif mutation in {
        "invalid_ledger_schema",
        "invalid_review_state",
        "invalid_receipt",
        "invalid_repairs_type",
        "two_repairs",
        "invalid_repair_entry",
    }:
        review = json.loads(ledger.read_text())
        if mutation == "invalid_ledger_schema":
            review["schema_version"] = 2
        elif mutation == "invalid_review_state":
            review["review_state"] = "unreviewed"
        elif mutation == "invalid_receipt":
            review["capture_receipt"] = "status/wrong.json"
        elif mutation == "invalid_repairs_type":
            review["repairs"] = "unreviewed"
        elif mutation == "two_repairs":
            review["repairs"].append(review["repairs"][0])
        elif mutation == "invalid_repair_entry":
            review["repairs"] = ["unreviewed"]
        atomic_write_json(ledger, review)
    elif mutation == "changed_evidence":
        review = json.loads(ledger.read_text())
        review["repairs"][0]["evidence_url"] = "https://example.test/unverified"
        atomic_write_json(ledger, review)
    elif mutation == "changed_repaired_author":
        cache["publications"][-1]["author"] = "Wrong author"
    elif mutation == "changed_other_row_author":
        cache["publications"][0]["author"] = "Wrong author"
    elif mutation == "changed_row_115_id":
        cache["publications"][-1]["author_pub_id"] = "AcW2EXwAAAAJ:wrong"
    elif mutation == "changed_raw_author":
        raw_path = captures / "breivik.json"
        html_path = captures / "breivik.html"
        raw = json.loads(raw_path.read_text())
        raw["works"][-1]["authors"] = "Unverified byline"
        atomic_write_json(raw_path, raw)
        html_path.write_text(
            html_path.read_text().replace(
                "K Breivik, S Chatterjee, J Andrews", "Unverified byline"
            )
        )
        cache["browser_capture"]["source_json_sha256"] = sha256_file(raw_path)
        cache["browser_capture"]["source_html_sha256"] = sha256_file(html_path)
    with pytest.raises(ValueError, match="Cached row differs from saved source"):
        validate_complete_receipt("AcW2EXwAAAAJ", cache, captures)


def _breivik_row116_repaired_author_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    cache, captures, _ = _breivik_repaired_author_fixture(tmp_path)
    scholar_id = "AcW2EXwAAAAJ"
    citation_id = f"{scholar_id}:qjMakFHDy7sC"
    citation_url = (
        f"https://scholar.google.com/citations?user={scholar_id}"
        f"&citation_for_view={citation_id}"
    )
    title = (
        "Exploring the Eclipsing Binary Yield of the Large Synoptic Survey Telescope"
    )
    captured = "A Polzin, A Geller, A Miller, K Breivik"
    repaired = "A Geller, A Polzin, K Breivik, A Miller"
    source_json = captures / "breivik.json"
    source_html = captures / "breivik.html"
    source = json.loads(source_json.read_text())
    source["displayed_count"] = 116
    source["works"].append(
        {
            "index": 116,
            "citation_url": citation_url,
            "title": title,
            "authors": captured,
            "venue": "AAS 233",
            "year": "2019",
            "citation_count": "0",
        }
    )
    atomic_write_json(source_json, source)
    html_row = (
        '<tr class="gsc_a_tr"><td class="gsc_a_t">'
        f'<a class="gsc_a_at" href="{escape(citation_url)}">{escape(title)}</a>'
        f'<div class="gs_gray">{captured}</div>'
        '<div class="gs_gray">AAS 233</div></td>'
        '<td class="gsc_a_c"><a class="gsc_a_ac">0</a></td>'
        '<td class="gsc_a_y"><span class="gsc_a_h">2019</span></td></tr>'
    )
    source_html.write_text(
        source_html.read_text().replace("</table>", html_row + "</table>")
    )
    cache["publications"].append(
        {
            "capture_row_index": 116,
            "author_pub_id": citation_id,
            "pub_url": citation_url,
            "title": title,
            "author": repaired,
            "venue": "AAS 233",
            "pub_year": "2019",
            "num_citations": 0,
        }
    )
    cache["publication_count"] = 116
    cache["browser_capture"]["displayed_count"] = 116
    cache["browser_capture"]["source_json_sha256"] = sha256_file(source_json)
    cache["browser_capture"]["source_html_sha256"] = sha256_file(source_html)
    ledger = tmp_path / "status" / "physics-breivik-row116-cache-repair-2026-10-25.json"
    atomic_write_json(
        ledger,
        {
            "schema_version": 1,
            "review_state": "source_reviewed_cache_repaired",
            "capture_receipt": (
                "status/browser-capture-physics-katelyn-breivik-2026-10-24.json"
            ),
            "repairs": [
                {
                    "scholar_id": scholar_id,
                    "capture_row": 116,
                    "author_pub_id": citation_id,
                    "cache_field": "author",
                    "captured_value": captured,
                    "repaired_value": repaired,
                    "evidence_url": "https://aas.org/files/aas233_abstracts.pdf",
                }
            ],
        },
    )
    return cache, captures, ledger


def test_breivik_row_115_and_116_author_repairs_require_their_exact_ledgers(
    tmp_path: Path,
) -> None:
    cache, captures, _ = _breivik_row116_repaired_author_fixture(tmp_path)
    validate_complete_receipt("AcW2EXwAAAAJ", cache, captures)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_row115_ledger",
        "missing_row116_ledger",
        "invalid_row116_ledger_json",
        "wrong_row116_ledger_id",
        "wrong_row116_ledger_row",
        "wrong_row116_receipt",
        "extra_row116_repair",
        "wrong_row116_evidence",
        "wrong_row116_cache_author",
        "wrong_row116_citation_id",
        "wrong_row116_raw_author",
        "wrong_other_row_author",
    ],
)
def test_breivik_row116_author_repair_does_not_waive_other_checks(
    tmp_path: Path, mutation: str
) -> None:
    cache, captures, ledger = _breivik_row116_repaired_author_fixture(tmp_path)
    if mutation == "missing_row115_ledger":
        row115_ledger = (
            tmp_path / "status" / "physics-breivik-row115-cache-repair-2026-10-24.json"
        )
        row115_ledger.unlink()
    elif mutation == "missing_row116_ledger":
        ledger.unlink()
    elif mutation == "invalid_row116_ledger_json":
        ledger.write_text("{invalid")
    elif mutation in {
        "wrong_row116_ledger_id",
        "wrong_row116_ledger_row",
        "wrong_row116_receipt",
        "extra_row116_repair",
        "wrong_row116_evidence",
    }:
        review = json.loads(ledger.read_text())
        repair = review["repairs"][0]
        if mutation == "wrong_row116_ledger_id":
            repair["author_pub_id"] = "AcW2EXwAAAAJ:wrong"
        elif mutation == "wrong_row116_ledger_row":
            repair["capture_row"] = 115
        elif mutation == "wrong_row116_receipt":
            review["capture_receipt"] = "status/unreviewed.json"
        elif mutation == "extra_row116_repair":
            review["repairs"].append(dict(repair))
        else:
            repair["evidence_url"] = "https://example.test/unverified"
        atomic_write_json(ledger, review)
    elif mutation == "wrong_row116_cache_author":
        cache["publications"][-1]["author"] = "Unverified byline"
    elif mutation == "wrong_row116_citation_id":
        cache["publications"][-1]["author_pub_id"] = "AcW2EXwAAAAJ:wrong"
    elif mutation == "wrong_row116_raw_author":
        source_json = captures / "breivik.json"
        source_html = captures / "breivik.html"
        source = json.loads(source_json.read_text())
        source["works"][-1]["authors"] = "Unverified byline"
        atomic_write_json(source_json, source)
        source_html.write_text(
            source_html.read_text().replace(
                "A Polzin, A Geller, A Miller, K Breivik", "Unverified byline"
            )
        )
        cache["browser_capture"]["source_json_sha256"] = sha256_file(source_json)
        cache["browser_capture"]["source_html_sha256"] = sha256_file(source_html)
    elif mutation == "wrong_other_row_author":
        cache["publications"][0]["author"] = "Unverified byline"
    with pytest.raises(ValueError, match="Cached row differs from saved source"):
        validate_complete_receipt("AcW2EXwAAAAJ", cache, captures)


@pytest.mark.parametrize(
    "field,value",
    [
        ("capture_row_index", 2),
        ("author_pub_id", "alphaAAAAJ:wrong"),
        ("pub_url", "https://example.test/wrong"),
        ("title", "Wrong"),
        ("author", "Wrong"),
        ("venue", "Wrong"),
        ("pub_year", "2024"),
        ("num_citations", 3),
    ],
)
def test_receipt_reconciles_every_cached_source_row(
    tmp_path: Path, field: str, value: object
) -> None:
    paths = _paths(tmp_path)
    cache = _cache(paths, complete=True)
    cache["publications"][0][field] = value
    with pytest.raises(ValueError, match="Capture row indexes|Cached row differs"):
        validate_complete_receipt("alphaAAAAJ", cache, paths[4])


def test_receipt_rejects_nonobject_cache_row(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    cache = _cache(paths, complete=True)
    cache["publications"] = ["bad"]
    with pytest.raises(ValueError, match="Invalid cached publication row"):
        validate_complete_receipt("alphaAAAAJ", cache, paths[4])


@pytest.mark.parametrize(
    "column", ["source_record_json", "person_id", "fetched_at_utc"]
)
def test_upload_rejects_snapshot_rows_not_in_source_cache(
    tmp_path: Path, column: str
) -> None:
    paths = _paths(tmp_path)
    _cache(paths, complete=True)
    snapshot = tmp_path / "snapshot.parquet"
    build_snapshot(
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
        cache_dir=paths[3],
        snapshot_path=snapshot,
        manifest_path=tmp_path / "manifest.json",
    )
    frame = pandas.read_parquet(snapshot)
    frame.loc[0, column] = "tampered"
    frame.to_parquet(snapshot, index=False)
    with pytest.raises(ValueError, match="Snapshot rows differ"):
        validate_coverage(**_options(paths), snapshot_path=snapshot)


def test_upload_rejects_missing_or_extra_snapshot_profile(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _cache(paths, complete=True)
    snapshot = tmp_path / "snapshot.parquet"
    build_snapshot(
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
        cache_dir=paths[3],
        snapshot_path=snapshot,
        manifest_path=tmp_path / "manifest.json",
    )
    frame = pandas.read_parquet(snapshot)
    frame.iloc[0:0].to_parquet(snapshot, index=False)
    with pytest.raises(ValueError, match="Snapshot rows differ"):
        validate_coverage(**_options(paths), snapshot_path=snapshot)
    frame = pandas.read_parquet(tmp_path / "snapshot.parquet")
    original = build_snapshot(
        people_path=paths[0],
        memberships_path=paths[1],
        departments_path=paths[2],
        cache_dir=paths[3],
        snapshot_path=snapshot,
        manifest_path=tmp_path / "manifest.json",
    )
    assert original["publication_rows"] == 1
    frame = pandas.read_parquet(snapshot)
    extra = frame.copy()
    extra.loc[0, "scholar_id"] = "outsiderAAAAJ"
    pandas.concat([frame, extra], ignore_index=True).to_parquet(snapshot, index=False)
    with pytest.raises(ValueError, match="Snapshot Scholar profiles differ"):
        validate_coverage(**_options(paths), snapshot_path=snapshot)


def test_uncached_profile_has_no_coverage_claim(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    assert validate_coverage(**_options(paths)) == {
        "legacy_profiles": 0,
        "complete_profiles": 0,
        "reconciled_prior_ids": 0,
    }


def test_invalid_cache_or_baseline_fails_closed(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _cache(paths, complete=False)
    atomic_write_json(paths[5], {"schema_version": 2})
    with pytest.raises(ValueError, match="baseline is invalid"):
        validate_coverage(**_options(paths))
    atomic_write_json(
        paths[5],
        {
            "schema_version": 1,
            "frozen_snapshot_commit": "a" * 40,
            "frozen_snapshot_sha256": "b" * 64,
            "profiles": {"alphaAAAAJ": {"included": "invalid"}},
        },
    )
    with pytest.raises(ValueError, match="cache entry is invalid"):
        validate_coverage(**_options(paths))
    atomic_write_json(
        paths[5],
        {
            "schema_version": 1,
            "frozen_snapshot_commit": "a" * 40,
            "frozen_snapshot_sha256": "b" * 64,
            "profiles": {},
        },
    )
    atomic_write_json(paths[3] / "alphaAAAAJ.json", {"scholar_id": "wrong"})
    with pytest.raises(ValueError, match="Invalid Scholar cache"):
        validate_coverage(**_options(paths))
