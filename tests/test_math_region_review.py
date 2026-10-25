"""Frozen Math review is source-bound and affects only its six live rows."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas
import pytest

from map_of_research.publisher import require_math_region_mode
from map_of_research.region_continuity import (
    WORK_FINGERPRINT_FIELDS,
    _fingerprint_fields,
    reconcile_region_decisions,
)
from map_of_research.region_review import (
    CURRENT_REVISION,
    DEFAULT_REGION_REVIEW_PATH,
    SNAPSHOT_SHA256,
    SOURCE_COMMIT,
    ReviewedRegionDecision,
    load_region_review,
)
from map_of_research.registry import load_registry
from tests.test_region_continuity import observation, work


def _fixture(tmp_path):
    source = copy.deepcopy(json.loads(DEFAULT_REGION_REVIEW_PATH.read_text()))
    cache_dir = tmp_path / "authors"
    cache_dir.mkdir()
    frame = []
    excluded = set()
    held = set()
    for item in source["decisions"]:
        author_pub_id = item["author_pub_id"]
        source_record = json.dumps({"id": author_pub_id})
        item["source_record_sha256"] = hashlib.sha256(
            source_record.encode()
        ).hexdigest()
        frame.append(
            {
                "person_id": item["person_id"],
                "scholar_id": item["scholar_id"],
                "author_pub_id": author_pub_id,
                "source_type": "scholar",
                "title": item["captured_title"],
                "source_url": item["saved_scholar_url"],
                "source_record_json": source_record,
            }
        )
        key = item["scholar_id"], author_pub_id
        if item["decision"] == "exclude":
            excluded.add(key)
        elif item["decision"] == "hold":
            held.add(key)
    for scholar_id, profile in source["profiles"].items():
        rows = [
            {
                "author_pub_id": f"{scholar_id}:synthetic-{index}",
                "capture_row_index": index,
                "pub_url": "https://scholar.google.com/synthetic",
            }
            for index in range(1, profile["expected_source_rows"] + 1)
        ]
        for decision in source["decisions"]:
            if decision["scholar_id"] == scholar_id:
                rows[decision["capture_row_index"] - 1] = {
                    "author_pub_id": decision["author_pub_id"],
                    "capture_row_index": decision["capture_row_index"],
                    "pub_url": decision["saved_scholar_url"],
                }
        cache = {
            "scholar_id": scholar_id,
            "publication_count": len(rows),
            "browser_capture": {
                "coverage": "complete_visible_profile",
                "more_remain": False,
            },
            "publications": rows,
        }
        path = cache_dir / f"{scholar_id}.json"
        path.write_text(json.dumps(cache))
        profile["cache_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "region-review.json"
    manifest_path.write_text(json.dumps(source))
    registry = load_registry(
        Path("registry/people.csv"),
        Path("registry/memberships.csv"),
        Path("registry/departments.csv"),
    )
    return manifest_path, pandas.DataFrame(frame), cache_dir, registry, excluded, held


def _load(args):
    path, frame, cache_dir, registry, excluded, held = args
    return load_region_review(
        path,
        snapshot_frame=frame,
        snapshot_sha256=SNAPSHOT_SHA256,
        cache_dir=cache_dir,
        registry=registry,
        exclusions=excluded,
        holds=held,
    )


def test_checked_in_math_manifest_has_exact_decision_counts_and_no_aliases():
    source = json.loads(DEFAULT_REGION_REVIEW_PATH.read_text())
    rows = source["decisions"]
    assert len(rows) == len({row["work_id"] for row in rows}) == 31
    assert len({row["author_pub_id"] for row in rows}) == 31
    assert Counter((row["transition"], row["decision"]) for row in rows) == {
        ("restore", "include"): 4,
        ("restore", "exclude"): 1,
        ("restore", "hold"): 12,
        ("reexclude", "include"): 12,
        ("reexclude", "exclude"): 2,
    }
    assert all("candidate" not in row and "alias" not in row for row in rows)


def test_math_manifest_uses_runtime_fingerprints_of_frozen_bb88_works():
    fixture_path = Path("tests/fixtures/bernhard-math-bb88-works.json")
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == (
        "bf7a75361ea5a26ff866969e486440bde7389cfb942279a4755a17d417fe3a6e"
    )
    baseline = json.loads(fixture_path.read_text())
    assert baseline["hub_revision"] == CURRENT_REVISION
    fields = (*WORK_FINGERPRINT_FIELDS, "quality_assessment_version")
    assert baseline["fingerprint_fields"] == list(fields)
    works = {row["work_id"]: row for row in baseline["works"]}
    assert len(works) == len(baseline["works"]) == 31
    decisions = json.loads(DEFAULT_REGION_REVIEW_PATH.read_text())["decisions"]
    assert set(works) == {item["work_id"] for item in decisions}
    assert all(
        item["work_fingerprint_sha256"]
        == _fingerprint_fields(works[item["work_id"]], fields)
        for item in decisions
    )


def test_math_region_loader_binds_all_31_source_rows_and_primary_homes(tmp_path):
    decisions = _load(_fixture(tmp_path))
    assert len(decisions) == 31
    assert Counter(item.decision for item in decisions) == {
        "include": 16,
        "exclude": 3,
        "hold": 12,
    }
    assert sum(item.visible for item in decisions) == 16


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("snapshot_sha", "frozen repair"),
        ("missing_row", "source rows differ"),
        ("duplicate_row", "source rows differ"),
        ("raw_row_changed", "differs from raw row"),
        ("cache_changed", "cache changed"),
        ("cache_incomplete", "cache changed"),
        ("home_changed", "primary home or membership changed"),
        ("include_without_primary_evidence", "conflicts with exact review"),
        ("exclude_without_manifest", "conflicts with exact review"),
    ],
)
def test_math_region_loader_rejects_drift(tmp_path, case, message):
    args = list(_fixture(tmp_path))
    if case == "snapshot_sha":
        with pytest.raises(ValueError, match=message):
            load_region_review(
                args[0],
                snapshot_frame=args[1],
                snapshot_sha256="0" * 64,
                cache_dir=args[2],
                registry=args[3],
                exclusions=args[4],
                holds=args[5],
            )
        return
    if case == "missing_row":
        args[1] = args[1].iloc[1:].copy()
    elif case == "duplicate_row":
        args[1] = pandas.concat([args[1], args[1].iloc[[0]]])
    elif case == "raw_row_changed":
        args[1].at[0, "source_record_json"] = "changed raw capture"
    elif case in {"cache_changed", "cache_incomplete"}:
        scholar_id = next(iter(json.loads(args[0].read_text())["profiles"]))
        path = args[2] / f"{scholar_id}.json"
        path.write_text(path.read_text() + " ")
    elif case == "home_changed":
        source = json.loads(args[0].read_text())
        source["decisions"][0]["primary_home_id"] = "not-math"
        args[0].write_text(json.dumps(source))
    elif case == "include_without_primary_evidence":
        source = json.loads(args[0].read_text())
        item = next(x for x in source["decisions"] if x["decision"] == "include")
        item["evidence_url"] = item["saved_scholar_url"]
        args[0].write_text(json.dumps(source))
    else:
        excluded = next(
            x
            for x in json.loads(args[0].read_text())["decisions"]
            if x["decision"] == "exclude"
        )
        args[4].remove((excluded["scholar_id"], excluded["author_pub_id"]))
    with pytest.raises(ValueError, match=message):
        _load(tuple(args))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("bad_header", "not bound to the frozen"),
        ("missing_profile", "profile set differs"),
        ("bad_profile", "profile source is invalid"),
        ("incomplete_capture", "capture is incomplete"),
        ("duplicate_cache_id", "cache IDs repeat"),
        ("nonobject_entry", "requires 31 exact decisions"),
        ("invalid_work_id", "invalid exact decision fields"),
        ("invalid_date", "date is invalid"),
        ("wrong_counts", "decision totals differ"),
    ],
)
def test_math_region_loader_rejects_each_frozen_manifest_boundary(
    tmp_path, case, message
):
    args = _fixture(tmp_path)
    path, _, cache_dir, *_ = args
    source = json.loads(path.read_text())
    scholar_id = next(iter(source["profiles"]))
    if case == "bad_header":
        source["source_commit"] = "0" * 40
    elif case == "missing_profile":
        source["profiles"].pop(scholar_id)
    elif case == "bad_profile":
        source["profiles"][scholar_id]["expected_source_rows"] = "not a count"
    elif case in {"incomplete_capture", "duplicate_cache_id"}:
        cache_path = cache_dir / f"{scholar_id}.json"
        cache = json.loads(cache_path.read_text())
        if case == "incomplete_capture":
            cache["browser_capture"]["more_remain"] = True
        else:
            assert len(cache["publications"]) > 1
            cache["publications"][1]["author_pub_id"] = cache["publications"][0][
                "author_pub_id"
            ]
        cache_path.write_text(json.dumps(cache))
        source["profiles"][scholar_id]["cache_sha256"] = hashlib.sha256(
            cache_path.read_bytes()
        ).hexdigest()
    elif case == "nonobject_entry":
        source["decisions"][0] = None
    elif case == "invalid_work_id":
        source["decisions"][0]["work_id"] = "not-a-work-id"
    elif case == "invalid_date":
        source["decisions"][0]["reviewed_at"] = "2026-99-99"
    else:
        entry = next(
            item
            for item in source["decisions"]
            if item["transition"] == "restore" and item["decision"] == "exclude"
        )
        entry["transition"] = "reexclude"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match=message):
        _load(args)


def test_math_mode_fails_closed_before_any_upload():
    with pytest.raises(ValueError, match="requires its exact reviewed publish flag"):
        require_math_region_mode(
            requested=False,
            prior_revision=CURRENT_REVISION,
            region_comparison_revision=CURRENT_REVISION,
            source_commit=SOURCE_COMMIT,
        )
    with pytest.raises(ValueError, match="requires its exact reviewed publish flag"):
        require_math_region_mode(
            requested=False,
            prior_revision=CURRENT_REVISION,
            region_comparison_revision="15498d5f78e2e269f4b11c630bc9b4ef56efb628",
            source_commit=SOURCE_COMMIT,
        )
    require_math_region_mode(
        requested=True,
        prior_revision=CURRENT_REVISION,
        region_comparison_revision=CURRENT_REVISION,
        source_commit=SOURCE_COMMIT,
    )
    with pytest.raises(ValueError, match="requires frozen bb88/f7 ordinary publish"):
        require_math_region_mode(
            requested=True,
            prior_revision=CURRENT_REVISION,
            region_comparison_revision="15498d5f78e2e269f4b11c630bc9b4ef56efb628",
            source_commit=SOURCE_COMMIT,
        )
    with pytest.raises(ValueError, match="requires frozen bb88/f7 ordinary publish"):
        require_math_region_mode(
            requested=True,
            prior_revision=CURRENT_REVISION,
            region_comparison_revision=CURRENT_REVISION,
            source_commit="0" * 40,
        )
    require_math_region_mode(
        requested=False,
        prior_revision="1" * 40,
        region_comparison_revision=None,
        source_commit=None,
    )


def test_selective_reconciliation_changes_only_four_gains_and_two_losses():
    decision_types = [
        *(("restore", "include") for _ in range(4)),
        ("restore", "exclude"),
        *(("restore", "hold") for _ in range(12)),
        *(("reexclude", "include") for _ in range(12)),
        *(("reexclude", "exclude") for _ in range(2)),
    ]
    prior_works = []
    current_works = []
    prior_rows = []
    current_rows = []
    decisions = []
    for index, (transition, action) in enumerate(decision_types):
        work_id = f"work-{index}"
        person_id = f"person-{index}"
        author_pub_id = f"profile:row-{index}"
        raw = json.dumps({"row": index})
        previous = observation(person_id, work_id, source=raw)
        previous["author_pub_id"] = author_pub_id
        current = copy.deepcopy(previous)
        current["map_eligible"] = action == "include"
        if action == "exclude":
            current["review_exclusion_code"] = "nonpublication"
            current["map_exclusion_reasons"] = ["reviewed_nonpublication"]
        elif action == "hold":
            current["map_exclusion_reasons"] = ["attribution_provisional_hold"]
        prior = work(
            work_id,
            eligible=transition == "reexclude",
            outlier=transition == "restore",
        )
        now = work(
            work_id,
            eligible=action == "include" and transition == "reexclude",
            outlier=action == "include" and transition == "restore",
            department=action == "include",
        )
        prior_rows.append(previous)
        current_rows.append(current)
        prior_works.append(prior)
        current_works.append(now)
        decisions.append(
            ReviewedRegionDecision(
                work_id,
                person_id,
                "profile",
                author_pub_id,
                index + 1,
                hashlib.sha256(raw.encode()).hexdigest(),
                _fingerprint_fields(
                    prior,
                    (*WORK_FINGERPRINT_FIELDS, "quality_assessment_version"),
                ),
                "A paper",
                transition,
                action,
                "https://primary.test/work",
                "exact_work_primary_evidence",
                "https://scholar.google.com/saved",
                "Checked review",
                "2026-10-25",
            )
        )
    output, receipt = reconcile_region_decisions(
        pandas.DataFrame(current_works),
        pandas.DataFrame(current_rows),
        prior_works,
        prior_rows,
        prior_revision=CURRENT_REVISION,
        reviewed_math_decisions=tuple(decisions),
    )
    assert output.map_eligible.sum() == 16
    assert len(receipt["reviewed_math_gains"]) == 4
    assert len(receipt["reviewed_losses"]) == 2
    assert receipt["unexplained_gains"] == receipt["unexplained_losses"] == []
    assert (
        receipt["current_visible_observations"] - receipt["prior_visible_observations"]
        == 2
    )

    def reconcile(
        *,
        works_override=None,
        decisions_override=None,
        revision=CURRENT_REVISION,
    ):
        return reconcile_region_decisions(
            pandas.DataFrame(works_override or current_works),
            pandas.DataFrame(current_rows),
            prior_works,
            prior_rows,
            prior_revision=revision,
            reviewed_math_decisions=tuple(decisions_override or decisions),
        )

    with pytest.raises(ValueError, match="frozen bb88 baseline"):
        reconcile(revision="0" * 40)
    with pytest.raises(ValueError, match="duplicate work IDs"):
        reconcile(works_override=[*current_works, copy.deepcopy(current_works[0])])
    bad_decisions = decisions.copy()
    bad_decisions[0] = replace(bad_decisions[0], source_record_sha256="0" * 64)
    with pytest.raises(ValueError, match="no longer matches exact"):
        reconcile(decisions_override=bad_decisions)
    no_outlier_works = copy.deepcopy(current_works)
    no_outlier_works[0]["map_region_outlier"] = False
    no_outlier_works[0]["map_eligible"] = False
    with pytest.raises(ValueError, match="no longer a region outlier"):
        reconcile(works_override=no_outlier_works)
    wrong_visibility_works = copy.deepcopy(current_works)
    included_reexclude = next(
        index
        for index, decision in enumerate(decisions)
        if decision.transition == "reexclude" and decision.visible
    )
    wrong_visibility_works[included_reexclude]["map_eligible"] = False
    with pytest.raises(ValueError, match="final visibility differs"):
        reconcile(works_override=wrong_visibility_works)
    with pytest.raises(ValueError, match="outside its exact \\+4/-2 scope"):
        reconcile(decisions_override=decisions[:-1])
