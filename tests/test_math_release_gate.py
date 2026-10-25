import datetime as dt
import json
from dataclasses import replace
from pathlib import Path

import pytest

import map_of_research.publisher as publisher
from map_of_research.io import atomic_write_json, sha256_file
from map_of_research.math_release_gate import main, validate_math_coverage
from map_of_research.registry import load_registry
from map_of_research.snapshot import build_snapshot, validate_snapshot
from tests.registry_helpers import write_registry


def _inputs(tmp_path: Path, *, rows: int = 2, primary: bool = True) -> dict:
    registry_dir = tmp_path / "registry"
    primary_fields = (
        {
            "primary_department_id": "math",
            "primary_department_source_url": "https://example.test/math/alpha",
            "primary_department_verified_at": "2026-10-24",
            "primary_department_basis": "reviewed",
        }
        if primary
        else {}
    )
    people_path, memberships_path, departments_path = write_registry(
        registry_dir,
        people=[
            {
                "person_id": "person-alpha",
                "display_name": "Alpha Mathematician",
                "scholar_id": "alphaAAAAJ",
                "scholar_id_source_url": "https://scholar.google.com/citations?user=alphaAAAAJ",
                "scholar_id_verified_at": "2026-10-24",
                **primary_fields,
            },
            {"person_id": "person-gap", "display_name": "Unresolved Scholar Person"},
            {
                "person_id": "person-engineer",
                "display_name": "Uncached Engineer",
                "scholar_id": "engineerAAAAJ",
                "scholar_id_source_url": "https://scholar.google.com/citations?user=engineerAAAAJ",
                "scholar_id_verified_at": "2026-10-24",
            },
        ],
        memberships=[
            {
                "person_id": "person-alpha",
                "department_id": "math",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/math/alpha",
                "verified_at": "2026-10-24",
            },
            {
                "person_id": "person-alpha",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/ece/alpha",
                "verified_at": "2026-10-24",
            },
            {
                "person_id": "person-gap",
                "department_id": "math",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/math/gap",
                "verified_at": "2026-10-24",
            },
            {
                "person_id": "person-engineer",
                "department_id": "ece",
                "role": "faculty",
                "included": "true",
                "source_url": "https://example.test/ece/engineer",
                "verified_at": "2026-10-24",
            },
        ],
        departments=[
            {
                "department_id": "math",
                "school_id": "mcs",
                "title": "Mathematical Sciences",
                "directory_url": "https://example.test/math",
            },
            {
                "department_id": "ece",
                "school_id": "engineering",
                "title": "ECE",
                "directory_url": "https://example.test/ece",
            },
        ],
        schools=[
            {
                "school_id": "mcs",
                "title": "Mellon College of Science",
                "directory_url": "https://example.test/mcs",
            },
            {
                "school_id": "engineering",
                "title": "College of Engineering",
                "directory_url": "https://example.test/engineering",
            },
        ],
    )
    cache_dir = tmp_path / "data" / "authors"
    capture_dir = tmp_path / "data" / "captures"
    cache_dir.mkdir(parents=True)
    capture_dir.mkdir(parents=True)
    works = [
        {
            "index": index,
            "title": f"Publication {index}",
            "citation_url": (
                "https://scholar.google.com/citations?"
                f"user=alphaAAAAJ&citation_for_view=alphaAAAAJ:row{index}"
            ),
        }
        for index in range(1, rows + 1)
    ]
    raw = {
        "scholar_user_id": "alphaAAAAJ",
        "displayed_count": rows,
        "more_remain": False,
        "challenge": False,
        "page_error": "",
        "works": works,
    }
    json_path = capture_dir / "alpha-visible.json"
    html_path = capture_dir / "alpha-visible.html"
    atomic_write_json(json_path, raw)
    html_path.write_text(
        "<table>" + '<tr class="gsc_a_tr"></tr>' * rows + "</table>"
        '<button id="gsc_bpf_more" disabled></button>',
        encoding="utf-8",
    )
    publications = [
        {
            "title": work["title"],
            "author_pub_id": f"alphaAAAAJ:row{work['index']}",
            "pub_url": work["citation_url"],
            "capture_row_index": work["index"],
        }
        for work in works
    ]
    cache_path = cache_dir / "alphaAAAAJ.json"
    atomic_write_json(
        cache_path,
        {
            "schema_version": 1,
            "scholar_id": "alphaAAAAJ",
            "fetched_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "publication_count": rows,
            "publications": publications,
            "browser_capture": {
                "coverage": "complete_visible_profile",
                "more_remain": False,
                "displayed_count": rows,
                "source_json_filename": json_path.name,
                "source_json_sha256": sha256_file(json_path),
                "source_html_filename": html_path.name,
                "source_html_sha256": sha256_file(html_path),
            },
        },
    )
    snapshot_path = tmp_path / "snapshot.parquet"
    manifest_path = tmp_path / "snapshot.manifest.json"
    build_snapshot(
        people_path=people_path,
        memberships_path=memberships_path,
        departments_path=departments_path,
        cache_dir=cache_dir,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
    )
    frame, _ = validate_snapshot(
        snapshot_path,
        manifest_path,
        people_path=people_path,
        memberships_path=memberships_path,
        departments_path=departments_path,
    )
    return {
        "people_path": people_path,
        "memberships_path": memberships_path,
        "departments_path": departments_path,
        "cache_dir": cache_dir,
        "capture_dir": capture_dir,
        "cache_path": cache_path,
        "json_path": json_path,
        "html_path": html_path,
        "snapshot_path": snapshot_path,
        "manifest_path": manifest_path,
        "frame": frame,
        "registry": load_registry(people_path, memberships_path, departments_path),
    }


def _check(paths: dict) -> dict[str, int]:
    return validate_math_coverage(
        paths["frame"],
        paths["registry"],
        cache_dir=paths["cache_dir"],
        capture_dir=paths["capture_dir"],
    )


def _change_json(path: Path, change) -> None:
    value = json.loads(path.read_text())
    change(value)
    atomic_write_json(path, value)


def _sync_capture_sha(paths: dict, kind: str) -> None:
    _change_json(
        paths["cache_path"],
        lambda cache: cache["browser_capture"].update(
            {f"source_{kind}_sha256": sha256_file(paths[f"{kind}_path"])}
        ),
    )


def _rewrite_snapshot(paths: dict, frame) -> None:
    frame.to_parquet(paths["snapshot_path"], index=False)
    _change_json(
        paths["manifest_path"],
        lambda manifest: manifest.update(
            {
                "snapshot_sha256": sha256_file(paths["snapshot_path"]),
                "publication_rows": len(frame),
            }
        ),
    )
    paths["frame"], _ = validate_snapshot(
        paths["snapshot_path"],
        paths["manifest_path"],
        people_path=paths["people_path"],
        memberships_path=paths["memberships_path"],
        departments_path=paths["departments_path"],
    )


def test_valid_complete_math_and_nonmath_cache_gap(tmp_path: Path, capsys) -> None:
    paths = _inputs(tmp_path)
    assert _check(paths) == {
        "math_people": 2,
        "complete_scholar_profiles": 1,
        "unresolved_scholar_id_people": 1,
        "complete_scholar_rows": 2,
    }
    assert (
        main(
            [
                "--snapshot",
                str(paths["snapshot_path"]),
                "--manifest",
                str(paths["manifest_path"]),
                "--people",
                str(paths["people_path"]),
                "--memberships",
                str(paths["memberships_path"]),
                "--departments",
                str(paths["departments_path"]),
                "--cache-dir",
                str(paths["cache_dir"]),
                "--capture-dir",
                str(paths["capture_dir"]),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["complete_scholar_rows"] == 2


def test_missing_registered_math_cache_fails_with_valid_snapshot(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    paths["cache_path"].unlink()
    with pytest.raises(ValueError, match="Scholar cache is missing"):
        _check(paths)


@pytest.mark.parametrize("rows", [100, 200])
def test_bounded_math_cache_fails_even_with_valid_manifest(
    tmp_path: Path, rows: int
) -> None:
    paths = _inputs(tmp_path, rows=rows)
    _change_json(
        paths["json_path"],
        lambda raw: (
            raw["works"].append(
                {
                    "index": rows + 1,
                    "citation_url": (
                        "https://scholar.google.com/citations?"
                        f"citation_for_view=alphaAAAAJ:row{rows + 1}"
                    ),
                }
            ),
            raw.update({"displayed_count": rows + 1}),
        ),
    )
    _sync_capture_sha(paths, "json")
    with pytest.raises(ValueError, match="saved Scholar JSON is incomplete"):
        _check(paths)


def test_dropped_snapshot_row_fails_despite_valid_manifest(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    _rewrite_snapshot(paths, paths["frame"].iloc[:1])
    with pytest.raises(ValueError, match="snapshot omits"):
        _check(paths)


def test_replaced_snapshot_citation_fails_despite_valid_manifest(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    frame = paths["frame"].copy()
    frame.loc[0, "author_pub_id"] = "alphaAAAAJ:substitute"
    _rewrite_snapshot(paths, frame)
    with pytest.raises(ValueError, match="snapshot omits"):
        _check(paths)


def test_unresolved_math_primary_home_fails(tmp_path: Path) -> None:
    paths = _inputs(tmp_path, primary=False)
    with pytest.raises(ValueError, match="primary academic home is unresolved"):
        _check(paths)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "identity or schema"),
        ("scholar_id", "otherAAAAJ", "identity or schema"),
        ("publications", [], "complete visible-profile"),
        ("browser_capture", {}, "complete visible-profile"),
        ("publication_count", 1, "cache row counts"),
    ],
)
def test_invalid_cache_metadata_fails(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    paths = _inputs(tmp_path)
    _change_json(paths["cache_path"], lambda cache: cache.update({field: value}))
    with pytest.raises(ValueError, match=message):
        _check(paths)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("coverage", "bounded", "complete visible-profile"),
        ("more_remain", True, "complete visible-profile"),
        ("displayed_count", 1, "cache row counts"),
        ("source_json_filename", "../elsewhere.json", "invalid saved json filename"),
        ("source_html_sha256", "bad", "saved html capture checksum"),
    ],
)
def test_invalid_capture_metadata_fails(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    paths = _inputs(tmp_path)
    _change_json(
        paths["cache_path"],
        lambda cache: cache["browser_capture"].update({field: value}),
    )
    with pytest.raises(ValueError, match=message):
        _check(paths)


def test_missing_capture_file_fails(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    paths["json_path"].unlink()
    with pytest.raises(ValueError, match="saved json capture is missing"):
        _check(paths)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scholar_user_id", "otherAAAAJ"),
        ("more_remain", True),
        ("challenge", True),
        ("page_error", "blocked"),
        ("displayed_count", 1),
        ("works", []),
    ],
)
def test_incomplete_raw_json_fails(tmp_path: Path, field: str, value: object) -> None:
    paths = _inputs(tmp_path)
    _change_json(paths["json_path"], lambda raw: raw.update({field: value}))
    _sync_capture_sha(paths, "json")
    with pytest.raises(ValueError, match="saved Scholar JSON is incomplete"):
        _check(paths)


@pytest.mark.parametrize(
    "html",
    [
        '<tr class="gsc_a_tr"></tr><button id="gsc_bpf_more" disabled></button>',
        '<tr class="gsc_a_tr"></tr>' * 2 + '<button id="gsc_bpf_more"></button>',
    ],
)
def test_incomplete_html_fails(tmp_path: Path, html: str) -> None:
    paths = _inputs(tmp_path)
    paths["html_path"].write_text(html)
    _sync_capture_sha(paths, "html")
    with pytest.raises(ValueError, match="HTML rows or Show More"):
        _check(paths)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("author_pub_id", "otherAAAAJ:row1", "valid Scholar row identity"),
        ("pub_url", "https://example.test", "valid Scholar row identity"),
        ("pub_url", None, "valid Scholar row identity"),
        ("capture_row_index", True, "valid Scholar row identity"),
        ("capture_row_index", 2, "indices are missing or duplicated"),
    ],
)
def test_invalid_cached_row_fails(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    paths = _inputs(tmp_path)
    _change_json(
        paths["cache_path"],
        lambda cache: cache["publications"][0].update({field: value}),
    )
    with pytest.raises(ValueError, match=message):
        _check(paths)


def test_nondict_cached_row_fails(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    _change_json(
        paths["cache_path"],
        lambda cache: cache["publications"].__setitem__(0, None),
    )
    with pytest.raises(ValueError, match="invalid cached publication"):
        _check(paths)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("citation_url", "https://example.test", "row identity or index is invalid"),
        ("index", 2, "row indices differ from cache"),
        ("index", [], "row identity or index is invalid"),
    ],
)
def test_invalid_raw_row_fails(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    paths = _inputs(tmp_path)
    _change_json(paths["json_path"], lambda raw: raw["works"][0].update({field: value}))
    _sync_capture_sha(paths, "json")
    with pytest.raises(ValueError, match=message):
        _check(paths)


def test_nondict_raw_row_fails(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    _change_json(paths["json_path"], lambda raw: raw["works"].__setitem__(0, None))
    _sync_capture_sha(paths, "json")
    with pytest.raises(ValueError, match="invalid raw row"):
        _check(paths)


def test_changed_cache_text_fails_snapshot_record_check(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    _change_json(
        paths["cache_path"],
        lambda cache: cache["publications"][0].update({"title": "Changed"}),
    )
    with pytest.raises(ValueError, match="snapshot Scholar record differs"):
        _check(paths)


def test_missing_math_roster_fails(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    registry = paths["registry"]
    with pytest.raises(ValueError, match="Mathematical Sciences is not registered"):
        validate_math_coverage(
            paths["frame"],
            replace(
                registry,
                departments=tuple(
                    item
                    for item in registry.departments
                    if item.department_id != "math"
                ),
            ),
        )
    with pytest.raises(ValueError, match="no included Mathematical Sciences people"):
        validate_math_coverage(
            paths["frame"],
            replace(
                registry,
                memberships=tuple(
                    item
                    for item in registry.memberships
                    if item.department_id != "math"
                ),
            ),
        )


def test_hosted_gate_precedes_hugging_face_authentication() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/upload.yml"
    ).read_text()
    assert "data/authors\n" in workflow
    assert "data/captures\n" in workflow
    gate_step = "name: Require complete Mathematical Sciences Scholar captures"
    assert workflow.index(gate_step) < workflow.index('HF_TOKEN="$(hf auth token)"')


def test_cli_rejects_registry_drift_before_math_check(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    people_path = paths["people_path"]
    people_path.write_text(
        people_path.read_text().replace(
            "Unresolved Scholar Person", "Changed Roster Person"
        )
    )
    with pytest.raises(ValueError, match="Registry files do not match"):
        main(
            [
                "--snapshot",
                str(paths["snapshot_path"]),
                "--manifest",
                str(paths["manifest_path"]),
                "--people",
                str(people_path),
                "--memberships",
                str(paths["memberships_path"]),
                "--departments",
                str(paths["departments_path"]),
                "--cache-dir",
                str(paths["cache_dir"]),
                "--capture-dir",
                str(paths["capture_dir"]),
            ]
        )


def test_publisher_cli_refuses_missing_math_evidence_before_upload(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _inputs(tmp_path)
    paths["cache_path"].unlink()
    monkeypatch.setenv(publisher.HF_TOKEN_ENV, "test-token")
    monkeypatch.setattr(
        publisher,
        "publish_snapshot",
        lambda *args, **kwargs: pytest.fail("publisher ran before Math gate"),
    )
    status_path = tmp_path / "status.json"
    assert (
        publisher.main(
            [
                "--snapshot",
                str(paths["snapshot_path"]),
                "--manifest",
                str(paths["manifest_path"]),
                "--prior-revision",
                "a" * 40,
                "--people",
                str(paths["people_path"]),
                "--memberships",
                str(paths["memberships_path"]),
                "--departments",
                str(paths["departments_path"]),
                "--math-cache-dir",
                str(paths["cache_dir"]),
                "--math-capture-dir",
                str(paths["capture_dir"]),
                "--status-file",
                str(status_path),
            ]
        )
        == 1
    )
    status = json.loads(status_path.read_text())
    assert status["status"] == "failure"
    assert "Scholar cache is missing" in status["error_message"]


def test_publisher_cli_uses_snapshot_checkout_evidence_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _inputs(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    paths["snapshot_path"] = paths["snapshot_path"].rename(
        snapshots / paths["snapshot_path"].name
    )
    paths["manifest_path"] = paths["manifest_path"].rename(
        snapshots / paths["manifest_path"].name
    )
    monkeypatch.setenv(publisher.HF_TOKEN_ENV, "test-token")
    monkeypatch.setattr(
        publisher,
        "validate_coverage",
        lambda **kwargs: {"legacy_profiles": 0, "complete_profiles": 0},
    )
    calls = []
    monkeypatch.setattr(
        publisher,
        "publish_snapshot",
        lambda *args, **kwargs: calls.append("upload") or {"works": 2},
    )
    status_path = tmp_path / "status.json"
    assert (
        publisher.main(
            [
                "--snapshot",
                str(paths["snapshot_path"]),
                "--manifest",
                str(paths["manifest_path"]),
                "--prior-revision",
                "a" * 40,
                "--people",
                str(paths["people_path"]),
                "--memberships",
                str(paths["memberships_path"]),
                "--departments",
                str(paths["departments_path"]),
                "--status-file",
                str(status_path),
            ]
        )
        == 0
    )
    assert calls == ["upload"]
    status = json.loads(status_path.read_text())
    assert status["math_coverage"]["complete_scholar_rows"] == 2
