"""Multiple complete author bibliographies share one offline snapshot safely."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas
import pytest

from map_of_research import snapshot

ROOT = Path(__file__).resolve().parents[1]
IYER = ROOT / "data/author_sites/gautam-iyer-2026-10-25.json"
ANDERSON = ROOT / "data/author_sites/theresa-anderson-2026-10-25.json"
PEOPLE = ROOT / "registry/people.csv"
MEMBERSHIPS = ROOT / "registry/memberships.csv"
DEPARTMENTS = ROOT / "registry/departments.csv"
FIXED_TIME = dt.datetime(2026, 10, 25, 12, 0, tzinfo=dt.UTC)


def _build(
    tmp_path: Path, paths: Path | list[Path] | None, name: str
) -> tuple[Path, Path, dict]:
    snapshot_path = tmp_path / f"{name}.parquet"
    manifest_path = tmp_path / f"{name}.manifest.json"
    manifest = snapshot.build_snapshot(
        people_path=PEOPLE,
        memberships_path=MEMBERSHIPS,
        departments_path=DEPARTMENTS,
        cache_dir=tmp_path / "empty-cache",
        author_site_path=paths,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        now=FIXED_TIME,
    )
    return snapshot_path, manifest_path, manifest


def _validate(
    snapshot_path: Path, manifest_path: Path, paths: Path | list[Path] | None
):
    return snapshot.validate_snapshot(
        snapshot_path,
        manifest_path,
        people_path=PEOPLE,
        memberships_path=MEMBERSHIPS,
        departments_path=DEPARTMENTS,
        author_site_path=paths,
        now=FIXED_TIME,
    )


def test_combined_capture_is_complete_order_independent_and_source_exact(
    tmp_path: Path,
) -> None:
    forward = _build(tmp_path, [IYER, ANDERSON], "forward")
    reverse = _build(tmp_path, [ANDERSON, IYER], "reverse")
    assert forward[2]["author_site_rows"] == reverse[2]["author_site_rows"] == 89
    assert forward[2]["publication_rows"] == 117
    assert forward[2]["author_site_capture_file"] == ""
    assert forward[2]["author_site_capture_sha256"] == ""
    assert forward[2]["author_site_captures"] == reverse[2]["author_site_captures"]
    assert [item["rows"] for item in forward[2]["author_site_captures"]] == [47, 42]
    assert [item["person_id"] for item in forward[2]["author_site_captures"]] == [
        "person-gautamiyer-3894621b",
        "person-theresaanderson-5ed760a9",
    ]
    assert forward[2]["snapshot_sha256"] == reverse[2]["snapshot_sha256"]
    frame, _ = _validate(forward[0], forward[1], [ANDERSON, IYER])
    assert frame["source_type"].value_counts().to_dict() == {
        "author_site": 89,
        "curated_verified": 28,
    }
    assert set(frame.loc[frame["source_type"] == "author_site", "scholar_id"]) == {""}


def test_legacy_single_and_empty_capture_manifests_remain_valid(
    tmp_path: Path,
) -> None:
    for paths, name, expected in [(IYER, "single", 47), (None, "none", 0)]:
        snapshot_path, manifest_path, manifest = _build(tmp_path, paths, name)
        assert manifest["author_site_rows"] == expected
        manifest.pop("author_site_captures")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        frame, _ = _validate(snapshot_path, manifest_path, paths)
        assert int((frame["source_type"] == "author_site").sum()) == expected


def test_repeated_cli_capture_arguments_build_and_validate(tmp_path: Path) -> None:
    output = tmp_path / "cli.parquet"
    manifest = tmp_path / "cli.manifest.json"
    args = [
        "--people",
        str(PEOPLE),
        "--memberships",
        str(MEMBERSHIPS),
        "--departments",
        str(DEPARTMENTS),
        "--cache-dir",
        str(tmp_path / "empty-cache"),
        "--author-site-capture",
        str(IYER),
        "--author-site-capture",
        str(ANDERSON),
        "--snapshot",
        str(output),
        "--manifest",
        str(manifest),
    ]
    assert snapshot.build_main(args) == 0
    assert (
        snapshot.validate_main(
            [
                "--people",
                str(PEOPLE),
                "--memberships",
                str(MEMBERSHIPS),
                "--departments",
                str(DEPARTMENTS),
                "--author-site-capture",
                str(ANDERSON),
                "--author-site-capture",
                str(IYER),
                "--snapshot",
                str(output),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )


def test_missing_or_substituted_capture_cannot_validate(tmp_path: Path) -> None:
    output, manifest_path, manifest = _build(tmp_path, [IYER, ANDERSON], "both")
    with pytest.raises(ValueError, match="Author-site capture does not match"):
        _validate(output, manifest_path, [IYER])
    manifest["author_site_captures"][1]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="capture list does not match"):
        _validate(output, manifest_path, [IYER, ANDERSON])


def test_duplicate_person_and_native_id_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="Duplicate author-site person"):
        _build(tmp_path, [IYER, IYER], "duplicate-person")
    original = snapshot.load_author_site_rows
    iyer_native = original(
        IYER, snapshot.load_registry(PEOPLE, MEMBERSHIPS, DEPARTMENTS)
    )[0]["source_native_id"]

    def duplicate_native(path: Path, registry):
        rows = original(path, registry)
        if path == ANDERSON:
            rows[0]["source_native_id"] = iyer_native
        return rows

    monkeypatch.setattr(snapshot, "load_author_site_rows", duplicate_native)
    with pytest.raises(ValueError, match="Duplicate author-site native ID"):
        _build(tmp_path, [IYER, ANDERSON], "duplicate-native")


def test_changed_snapshot_row_fails_even_when_parquet_checksum_is_updated(
    tmp_path: Path,
) -> None:
    output, manifest_path, manifest = _build(tmp_path, [IYER, ANDERSON], "both")
    frame = pandas.read_parquet(output)
    index = frame.index[frame["source_type"] == "author_site"][0]
    frame.at[index, "title"] = "A substituted publication title"
    frame.to_parquet(output, index=False, engine="pyarrow")
    manifest["snapshot_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="observation changed"):
        _validate(output, manifest_path, [IYER, ANDERSON])


def test_missing_anderson_row_fails_even_when_counts_and_hash_are_updated(
    tmp_path: Path,
) -> None:
    output, manifest_path, manifest = _build(tmp_path, [IYER, ANDERSON], "both")
    frame = pandas.read_parquet(output)
    index = frame.index[
        (frame["source_type"] == "author_site")
        & (frame["person_id"] == "person-theresaanderson-5ed760a9")
    ][0]
    frame = frame.drop(index=index)
    frame.to_parquet(output, index=False, engine="pyarrow")
    manifest["snapshot_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest["publication_rows"] -= 1
    manifest["author_site_rows"] -= 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete Anderson author-site profile"):
        _validate(output, manifest_path, [IYER, ANDERSON])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ([], "must be an object"),
        ({"person_id": "", "expected_entry_count": 0}, "identity or row count"),
    ],
)
def test_malformed_capture_manifest_is_rejected_before_build(
    tmp_path: Path, content: object, message: str
) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _build(tmp_path, invalid, "invalid")


def test_duplicate_manifest_filename_is_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / IYER.name
    duplicate.write_bytes(ANDERSON.read_bytes())
    with pytest.raises(ValueError, match="Duplicate author-site manifest filename"):
        _build(tmp_path, [IYER, duplicate], "duplicate-filename")


def test_partial_loaded_capture_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = snapshot.load_author_site_rows

    def partial(path: Path, registry):
        rows = original(path, registry)
        return rows[:-1] if path == ANDERSON else rows

    monkeypatch.setattr(snapshot, "load_author_site_rows", partial)
    with pytest.raises(ValueError, match="Incomplete author-site capture"):
        _build(tmp_path, [IYER, ANDERSON], "partial")


def test_legacy_manifest_cannot_hide_multiple_captures(tmp_path: Path) -> None:
    output, manifest_path, manifest = _build(tmp_path, [IYER, ANDERSON], "both")
    manifest.pop("author_site_captures")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Legacy snapshot manifest cannot bind"):
        _validate(output, manifest_path, [IYER, ANDERSON])
    with pytest.raises(
        ValueError, match="Snapshot author-site captures are incomplete"
    ):
        _validate(output, manifest_path, None)


@pytest.mark.parametrize(
    ("field", "expected_message"),
    [
        ("source_native_id", "duplicate author-site native IDs"),
        ("author_pub_id", "observation identities changed"),
    ],
)
def test_tampered_snapshot_id_is_rejected_with_new_checksum(
    tmp_path: Path, field: str, expected_message: str
) -> None:
    output, manifest_path, manifest = _build(tmp_path, [IYER, ANDERSON], "both")
    frame = pandas.read_parquet(output)
    author_indices = frame.index[frame["source_type"] == "author_site"]
    if field == "source_native_id":
        frame.at[author_indices[1], field] = frame.at[author_indices[0], field]
    else:
        frame.at[author_indices[0], field] = "author-site:substituted"
    frame.to_parquet(output, index=False, engine="pyarrow")
    manifest["snapshot_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=expected_message):
        _validate(output, manifest_path, [IYER, ANDERSON])


def test_validated_combined_snapshot_remains_blocked_from_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from map_of_research import publisher

    output, manifest_path, _ = _build(tmp_path, [IYER, ANDERSON], "both")
    validated = _validate(output, manifest_path, [IYER, ANDERSON])
    monkeypatch.setattr(
        publisher, "_require_current_dataset_revision", lambda **kwargs: None
    )
    monkeypatch.setattr(
        publisher, "implicit_primary_baseline_for_people", lambda *args: {}
    )
    monkeypatch.setattr(
        publisher,
        "validate_included_scholar_primary_homes",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        publisher, "validate_snapshot", lambda *args, **kwargs: validated
    )

    def unexpected_write(_intent):
        pytest.fail("Publisher attempted a Hub write")

    with pytest.raises(ValueError, match="exact reviewed release gate"):
        publisher.publish_snapshot(
            output,
            manifest_path,
            people_path=PEOPLE,
            memberships_path=MEMBERSHIPS,
            departments_path=DEPARTMENTS,
            hf_token="unused",
            prior_revision="bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd",
            atomic_intent_sink=unexpected_write,
        )
