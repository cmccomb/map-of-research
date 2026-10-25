"""The exact Chemistry review replaces only its 19 historical region flips."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas
import pytest

from map_of_research import chem_region_review as chem_review
from map_of_research.chem_region_review import (
    CHEM_REVIEW_SHA256,
    DECISION_COUNTS,
    DEFAULT_CHEM_REGION_REVIEW_PATH,
    SOURCE_SHA256,
    load_chem_region_review,
)
from map_of_research.registry import load_registry


def _save_manifest(path: Path, source: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    path.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(
        chem_review, "CHEM_REVIEW_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Keep all 19 reviewed identities and eight full profile lengths."""

    source = json.loads(DEFAULT_CHEM_REGION_REVIEW_PATH.read_text())
    source_rows = []
    for item in source["decisions"]:
        raw = json.dumps({"author_pub_id": item["author_pub_id"]})
        item["source_record_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        source_rows.append(
            {
                "person_id": item["person_id"],
                "scholar_id": item["scholar_id"],
                "author_pub_id": item["author_pub_id"],
                "source_type": "scholar",
                "title": item["captured_title"],
                "source_url": item["saved_scholar_url"],
                "source_record_json": raw,
            }
        )

    cache_dir = tmp_path / "authors"
    cache_dir.mkdir()
    for scholar_id, profile in source["profiles"].items():
        rows = [
            {
                "author_pub_id": f"{scholar_id}:fixture-{index}",
                "capture_row_index": index,
                "pub_url": "https://scholar.google.com/fixture",
            }
            for index in range(1, profile["expected_source_rows"] + 1)
        ]
        for item in source["decisions"]:
            if item["scholar_id"] == scholar_id:
                rows[item["capture_row_index"] - 1] = {
                    "author_pub_id": item["author_pub_id"],
                    "capture_row_index": item["capture_row_index"],
                    "pub_url": item["saved_scholar_url"],
                }
        assert len({row["author_pub_id"] for row in rows}) == len(rows)
        cache = {
            "scholar_id": scholar_id,
            "publication_count": len(rows),
            "browser_capture": {
                "coverage": "complete_visible_profile",
                "more_remain": False,
            },
            "publications": rows,
        }
        cache_path = cache_dir / f"{scholar_id}.json"
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
        profile["cache_sha256"] = hashlib.sha256(cache_path.read_bytes()).hexdigest()

    manifest_path = tmp_path / "chem-region-review.json"
    _save_manifest(manifest_path, source, monkeypatch)
    registry = load_registry(
        Path("registry/people.csv"),
        Path("registry/memberships.csv"),
        Path("registry/departments.csv"),
    )
    return manifest_path, pandas.DataFrame(source_rows), cache_dir, registry


def _load(args: tuple):
    path, frame, cache_dir, registry = args
    return load_chem_region_review(
        path,
        snapshot_frame=frame,
        snapshot_sha256=SOURCE_SHA256,
        cache_dir=cache_dir,
        registry=registry,
    )


def test_checked_in_chem_review_is_exact_and_preserves_source() -> None:
    path = DEFAULT_CHEM_REGION_REVIEW_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CHEM_REVIEW_SHA256
    source = json.loads(path.read_text())
    decisions = source["decisions"]
    repair = json.loads(
        Path("data/curated/region-remediation-bernhard.json").read_text()
    )
    math = json.loads(Path("data/curated/scholar-region-review.json").read_text())
    assert len(decisions) == len({item["work_id"] for item in decisions}) == 19
    assert len({item["author_pub_id"] for item in decisions}) == 19
    assert Counter((item["transition"], item["decision"]) for item in decisions) == (
        DECISION_COUNTS
    )
    assert {item["work_id"] for item in decisions}.isdisjoint(
        {item["work_id"] for item in math["decisions"]}
    )
    for item in decisions:
        key = (
            "restore_eligible_work_ids"
            if item["transition"] == "restore"
            else "restore_outlier_work_ids"
        )
        assert item["work_id"] in repair[key]
        assert item["saved_scholar_url"].startswith("https://scholar.google.com/")
        assert len(item["source_record_sha256"]) == 64
        assert len(item["work_fingerprint_sha256"]) == 64
    assert {
        item["reason_code"] for item in decisions if item["decision"] == "exclude"
    } == {
        "false_profile_attribution",
        "corrupted_duplicate",
    }
    assert {item["work_id"] for item in decisions if item.get("output_type")} == {
        "work-4e63e66d343de69522c77ffde7a55853382c39d01a9379ca47f4fa4eb939e802",
        "work-f07078afd572186934d04127dcd8526c089e5089ad69b1c4504702ac49c4329b",
    }
    assert all(
        item["output_type"] == "conference_presentation"
        for item in decisions
        if item.get("output_type")
    )


def test_modified_chem_review_fails_before_accessing_source(tmp_path: Path) -> None:
    changed = tmp_path / "chem-region-review.json"
    changed.write_bytes(DEFAULT_CHEM_REGION_REVIEW_PATH.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest bytes changed"):
        load_chem_region_review(
            changed,
            snapshot_frame=None,
            snapshot_sha256="f8d7ffec683eb1830dadc77e612cbad7f3fa18ff199696041a71a9b8edbcefa2",
            cache_dir=tmp_path,
            registry=None,
        )


def test_chem_review_loader_binds_all_19_rows_and_eight_complete_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    source = json.loads(args[0].read_text())
    reviewed = _load(args)
    assert len(reviewed) == 19
    assert Counter((item.transition, item.decision) for item in reviewed) == (
        DECISION_COUNTS
    )
    assert {item.author_pub_id for item in reviewed} == {
        item["author_pub_id"] for item in source["decisions"]
    }
    assert {item.scholar_id for item in reviewed} == set(source["profiles"])
    assert sum(item.visible for item in reviewed) == 10
    assert sum(item.output_type == "conference_presentation" for item in reviewed) == 2
    assert all(
        item.evidence_scope
        == (
            "exact_work_primary_evidence"
            if item.decision == "include"
            else "reviewed_unmapped_work"
        )
        for item in reviewed
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("snapshot_sha", "source or parent changed"),
        ("missing_source_row", "source observation set changed"),
        ("raw_source_changed", "exact source decision drift"),
        ("cache_bytes_changed", "complete Scholar cache changed"),
        ("cache_incomplete", "Scholar profile is incomplete"),
        ("cache_duplicate_id", "cache has duplicate rows"),
        ("cache_row_index_changed", "exact source decision drift"),
        ("chem_membership_removed", "exact source decision drift"),
    ],
)
def test_chem_review_loader_rejects_source_cache_or_home_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, message: str
) -> None:
    args = list(_fixture(tmp_path, monkeypatch))
    source = json.loads(args[0].read_text())
    first = source["decisions"][0]
    if case == "snapshot_sha":
        with pytest.raises(ValueError, match=message):
            load_chem_region_review(
                args[0],
                snapshot_frame=args[1],
                snapshot_sha256="0" * 64,
                cache_dir=args[2],
                registry=args[3],
            )
        return
    if case == "missing_source_row":
        args[1] = args[1].iloc[1:].copy()
    elif case == "raw_source_changed":
        args[1].at[0, "source_record_json"] = "changed raw Scholar row"
    elif case == "chem_membership_removed":
        args[3] = replace(
            args[3],
            memberships=tuple(
                membership
                for membership in args[3].memberships
                if not (
                    membership.person_id == first["person_id"]
                    and membership.department_id == "chem"
                    and membership.included
                )
            ),
        )
    else:
        scholar_id = first["scholar_id"]
        cache_path = args[2] / f"{scholar_id}.json"
        if case == "cache_bytes_changed":
            cache_path.write_bytes(cache_path.read_bytes() + b" ")
        else:
            cache = json.loads(cache_path.read_text())
            if case == "cache_incomplete":
                cache["browser_capture"]["more_remain"] = True
            elif case == "cache_duplicate_id":
                cache["publications"][1] = cache["publications"][0]
            else:
                cache["publications"][first["capture_row_index"] - 1][
                    "capture_row_index"
                ] += 1
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            source["profiles"][scholar_id]["cache_sha256"] = hashlib.sha256(
                cache_path.read_bytes()
            ).hexdigest()
            _save_manifest(args[0], source, monkeypatch)
    with pytest.raises(ValueError, match=message):
        _load(tuple(args))
