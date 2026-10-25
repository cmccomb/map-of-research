"""The ordinary publisher retains the 19 source-bound Chemistry decisions."""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas
import pytest

import map_of_research.chem_continuity as continuity
from map_of_research.chem_continuity import (
    CHEMISTRY_HUB_REVISION,
    PRE_CHEMISTRY_REVISIONS,
    merge_chemistry_review_inputs,
    needs_chemistry_continuity,
    reconcile_chemistry_person_counts,
    validate_chemistry_candidate,
    validate_chemistry_parent,
)
from map_of_research.chem_region_review import (
    CHEM_REVIEW_SHA256,
    SOURCE_COMMIT,
    SOURCE_SHA256,
)
from map_of_research.region_review import ReviewedRegionDecision


def _decisions() -> tuple[ReviewedRegionDecision, ...]:
    entries = json.loads(
        Path("data/curated/scholar-chem-region-review.json").read_text()
    )["decisions"]
    fields = (
        "work_id",
        "person_id",
        "scholar_id",
        "author_pub_id",
        "capture_row_index",
        "source_record_sha256",
        "work_fingerprint_sha256",
        "captured_title",
        "transition",
        "decision",
        "evidence_url",
        "saved_scholar_url",
        "reason",
        "reviewed_at",
    )
    return tuple(
        ReviewedRegionDecision(
            **{field: entry[field] for field in fields},
            evidence_scope="exact_work_primary_evidence",
            reason_code=entry.get("reason_code", ""),
            output_type=entry.get("output_type", ""),
        )
        for entry in entries
    )


def _files(root: Path, decisions: tuple[ReviewedRegionDecision, ...]) -> dict:
    rows = {
        name: [] for name in ("works", "profile_publications", "people", "authorships")
    }
    for item in decisions:
        observation_id = f"observation-{item.author_pub_id}"
        rows["profile_publications"].append(
            {
                "observation_id": observation_id,
                "author_pub_id": item.author_pub_id,
                "person_id": item.person_id,
                "work_id": item.work_id,
                "map_eligible": item.visible,
                "collaboration_eligible": item.visible,
                "review_evidence_url": item.evidence_url
                if item.decision == "exclude"
                else "",
            }
        )
        rows["works"].append(
            {
                "work_id": item.work_id,
                "observation_ids": [observation_id],
                "map_eligible": item.visible,
                "collaboration_eligible": item.visible,
                "map_region_outlier": False,
                "keyword_id": "broad" if item.visible else "",
                "detail_keyword_id": "detail" if item.visible else "",
                "map_region_audit_version": "low-information-semantic-region-v2",
                "map_region_documented_override": False,
                "x": 0.25,
                "title": item.captured_title,
            }
        )
        if item.visible:
            rows["authorships"].append(
                {
                    "observation_id": observation_id,
                    "work_id": item.work_id,
                    "person_id": item.person_id,
                }
            )
    rows["people"] = [
        {"person_id": person_id, "display_name": person_id, "mapped_work_count": 10}
        for person_id in sorted({item.person_id for item in decisions})
    ]
    paths = {}
    for name, values in rows.items():
        path = root / name / "train-00000-of-00001.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pandas.DataFrame(values).to_parquet(path, index=False)
        paths[f"{name}/{path.name}"] = path
    artifact = {
        "point_count": 10,
        "points": [
            {
                "work_id": item.work_id,
                "keyword_ids": ["broad", "detail"],
                "faculty_ids": [item.person_id],
                "x": 0.25,
            }
            for item in decisions
            if item.visible
        ],
    }
    maps = root / "maps"
    maps.mkdir()
    point_path = maps / "publications.json"
    point_path.write_text(json.dumps(artifact))
    paths["maps/publications.json"] = point_path
    marker = {
        "version": "exact-chemistry-post-337-delta-v1",
        "prior_revision": "38f119d71692cdb9112210217b532a7dd7c1633a",
        "source_commit": SOURCE_COMMIT,
        "source_snapshot_sha256": SOURCE_SHA256,
        "chem_region_review_sha256": CHEM_REVIEW_SHA256,
        "restored_work_ids": sorted(
            item.work_id
            for item in decisions
            if item.transition == "restore" and item.visible
        ),
        "removed_work_ids": sorted(
            item.work_id
            for item in decisions
            if item.transition == "reexclude" and not item.visible
        ),
    }
    manifest_path = maps / "manifest.json"
    manifest_path.write_text(json.dumps({"region_continuity": marker}))
    paths["maps/manifest.json"] = manifest_path
    return paths


def _rewrite_rows(path: Path, change) -> None:
    rows = pandas.read_parquet(path).to_dict("records")
    change(rows)
    pandas.DataFrame(rows).to_parquet(path, index=False)


def _post_bio_parent(
    files: dict, decisions: tuple[ReviewedRegionDecision, ...]
) -> None:
    """Model only the reviewed Bio stage derivations, including parent counts."""

    _rewrite_rows(
        files["works/train-00000-of-00001.parquet"],
        lambda rows: [
            row.update(
                map_region_prior_eligible_override=(
                    row["work_id"] in continuity.POST_BIO_PRIOR_ELIGIBLE_WORK_IDS
                ),
                map_region_prior_excluded_override=False,
                map_region_audit_version=continuity.POST_BIO_REGION_AUDIT_VERSION,
                map_region_documented_override=(
                    row["work_id"] == continuity.POST_BIO_DOCUMENTED_OVERRIDE_WORK_ID
                ),
            )
            for row in rows
        ],
    )
    restored = Counter(
        item.person_id for item in decisions if item.transition == "restore"
    )
    _rewrite_rows(
        files["people/train-00000-of-00001.parquet"],
        lambda rows: [
            row.update(
                publication_hold_reason="",
                mapped_work_count=row["mapped_work_count"] - restored[row["person_id"]],
            )
            for row in rows
        ],
    )


def test_chemistry_review_inputs_keep_primary_evidence_and_exact_scope() -> None:
    decisions = _decisions()
    exclusions, holds, contexts = merge_chemistry_review_inputs(decisions, {}, {}, {})
    assert (len(exclusions), len(holds), len(contexts)) == (4, 5, 2)
    for item in decisions:
        key = (item.scholar_id, item.author_pub_id)
        if item.decision == "hold":
            assert holds[key].evidence_url == item.evidence_url
            assert holds[key].evidence_url != item.saved_scholar_url
        if item.output_type:
            assert contexts[key].publication_type == "conference_presentation"
    assert merge_chemistry_review_inputs(decisions, exclusions, holds, contexts) == (
        exclusions,
        holds,
        contexts,
    )
    saved_holds = {
        key: replace(
            hold,
            evidence_url=next(
                item.saved_scholar_url
                for item in decisions
                if (item.scholar_id, item.author_pub_id) == key
            ),
            primary_evidence_url=hold.evidence_url,
        )
        for key, hold in holds.items()
    }
    merged = merge_chemistry_review_inputs(decisions, exclusions, saved_holds, contexts)
    assert merged == (
        exclusions,
        holds,
        contexts,
    )
    with pytest.raises(ValueError, match="conflicts with existing review"):
        merge_chemistry_review_inputs(
            decisions,
            {
                next(iter(exclusions)): replace(
                    next(iter(exclusions.values())), reason="different review"
                )
            },
            {},
            {},
        )
    with pytest.raises(ValueError, match="conflicts with existing review"):
        merge_chemistry_review_inputs(
            decisions,
            exclusions,
            {
                next(iter(holds)): replace(
                    next(iter(holds.values())), reason="different review"
                )
            },
            contexts,
        )
    with pytest.raises(ValueError, match="exact 19 decisions"):
        merge_chemistry_review_inputs(decisions[:-1], {}, {}, {})
    without_context = tuple(replace(item, output_type="") for item in decisions)
    with pytest.raises(ValueError, match="lacks two presentations"):
        merge_chemistry_review_inputs(without_context, {}, {}, {})


def test_chemistry_continuity_uses_post_delta_parent_and_source_presence() -> None:
    assert needs_chemistry_continuity(set(), CHEMISTRY_HUB_REVISION, composite=False)
    assert needs_chemistry_continuity({"ILambBMAAAAJ"}, "a" * 40, composite=False)
    assert not needs_chemistry_continuity(set(), "a" * 40, composite=False)
    assert needs_chemistry_continuity(
        set(), "a" * 40, composite=False, snapshot_sha256="b" * 64
    )
    assert not needs_chemistry_continuity(
        {"ILambBMAAAAJ"}, next(iter(PRE_CHEMISTRY_REVISIONS)), composite=False
    )
    assert not needs_chemistry_continuity(
        {"ILambBMAAAAJ"}, CHEMISTRY_HUB_REVISION, composite=True
    )


def test_exact_seven_restores_correct_only_reviewed_people_counts() -> None:
    decisions = _decisions()
    people = pandas.DataFrame(
        [
            {"person_id": person_id, "mapped_work_count": 100}
            for person_id in sorted({item.person_id for item in decisions})
        ]
        + [{"person_id": "unrelated", "mapped_work_count": 8}]
    )
    adjusted = reconcile_chemistry_person_counts(people, decisions)
    expected = {
        person_id: int(row.mapped_work_count)
        + sum(
            item.transition == "restore" and item.person_id == person_id
            for item in decisions
        )
        for row in people.itertuples(index=False)
        for person_id in (row.person_id,)
    }
    assert (
        dict(zip(adjusted["person_id"], adjusted["mapped_work_count"], strict=True))
        == expected
    )
    assert people.iloc[-1]["mapped_work_count"] == 8
    restore = next(item for item in decisions if item.transition == "restore")
    with pytest.raises(ValueError, match="restore set changed"):
        reconcile_chemistry_person_counts(
            people, tuple(item for item in decisions if item is not restore)
        )
    missing = people[people["person_id"] != restore.person_id]
    with pytest.raises(ValueError, match="missing or repeated"):
        reconcile_chemistry_person_counts(missing, decisions)
    invalid = people.copy()
    invalid.loc[invalid["person_id"] == restore.person_id, "mapped_work_count"] = -1
    with pytest.raises(ValueError, match="person count is invalid"):
        reconcile_chemistry_person_counts(invalid, decisions)


def test_chemistry_parent_loader_pins_every_file_to_immutable_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decisions = _decisions()
    baseline = _files(tmp_path / "baseline", decisions)
    later = _files(tmp_path / "later", decisions)
    calls = []

    def download(_repo, name, *, repo_type, revision, token):
        calls.append((name, repo_type, revision, token))
        return str((baseline if revision == CHEMISTRY_HUB_REVISION else later)[name])

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    assert (
        continuity.load_chemistry_parent(decisions, CHEMISTRY_HUB_REVISION, "test")[
            "visible_works"
        ]
        == 10
    )
    assert len(calls) == len(continuity.PARENT_FILES)
    assert {revision for _, _, revision, _ in calls} == {CHEMISTRY_HUB_REVISION}
    calls.clear()
    assert (
        continuity.load_chemistry_parent(decisions, "a" * 40, "test")["lineage"]
        == "exact_dcb_reviewed_rows"
    )
    assert len(calls) == 2 * len(continuity.PARENT_FILES)
    assert {revision for _, _, revision, _ in calls} == {
        CHEMISTRY_HUB_REVISION,
        "a" * 40,
    }


def test_later_parent_may_add_unrelated_biology_rows(tmp_path: Path) -> None:
    decisions = _decisions()
    baseline = _files(tmp_path / "baseline", decisions)
    parent = _files(tmp_path / "parent", decisions)
    _rewrite_rows(
        parent["profile_publications/train-00000-of-00001.parquet"],
        lambda rows: rows.append(
            {**rows[0], "author_pub_id": "other:publication", "observation_id": "other"}
        ),
    )
    _rewrite_rows(
        parent["authorships/train-00000-of-00001.parquet"],
        lambda rows: rows.append({**rows[0], "observation_id": "other"}),
    )
    path = parent["maps/publications.json"]
    artifact = json.loads(path.read_text())
    artifact["points"].append(
        {"work_id": "work-unrelated", "keyword_ids": ["other", "detail"]}
    )
    artifact["point_count"] += 1
    path.write_text(json.dumps(artifact))
    assert (
        validate_chemistry_parent(
            decisions,
            baseline_files=baseline,
            parent_files=parent,
            prior_revision="a" * 40,
        )["visible_works"]
        == 10
    )


def test_post_bio_parent_retains_reviewed_rows_and_uses_immediate_parent_counts(
    tmp_path: Path,
) -> None:
    decisions = _decisions()
    baseline = _files(tmp_path / "baseline", decisions)
    parent = _files(tmp_path / "parent", decisions)
    _post_bio_parent(parent, decisions)
    state = validate_chemistry_parent(
        decisions,
        baseline_files=baseline,
        parent_files=parent,
        prior_revision="a" * 40,
    )
    assert state["reviewed_works"] == 19
    assert state["lineage"] == "exact_dcb_reviewed_rows"
    baseline_counts = {
        key: row["mapped_work_count"]
        for key, row in continuity._selected_rows(
            baseline["people/train-00000-of-00001.parquet"],
            "person_id",
            {item.person_id for item in decisions},
        ).items()
    }
    parent_counts = {
        key: row["mapped_work_count"]
        for key, row in state["parent_rows"]["people"].items()
    }
    decreases = {
        person_id: baseline_counts[person_id] - parent_counts[person_id]
        for person_id in baseline_counts
        if baseline_counts[person_id] != parent_counts[person_id]
    }
    assert decreases == Counter(
        item.person_id for item in decisions if item.transition == "restore"
    )
    assert len(decreases) == 5
    assert sum(decreases.values()) == 7
    tables = {
        name: pandas.read_parquet(parent[f"{name}/train-00000-of-00001.parquet"])
        for name in ("works", "profile_publications", "authorships", "people")
    }
    validate_chemistry_candidate(tables, state, require_count_parity=True)
    tables["people"].loc[0, "mapped_work_count"] += 1
    with pytest.raises(ValueError, match="candidate people changed"):
        validate_chemistry_candidate(tables, state, require_count_parity=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_work", "unreviewed schema fields"),
        ("extra_person", "unreviewed schema fields"),
        ("missing_added_work", "unreviewed schema fields"),
        ("missing_added_person", "unreviewed schema fields"),
        ("missing_dcb_work", "work schema changed"),
        ("missing_dcb_person", "person schema changed"),
        ("wrong_eligible", "work derivation changed"),
        ("wrong_excluded", "work derivation changed"),
        ("wrong_audit", "work derivation changed"),
        ("wrong_override", "work derivation changed"),
        ("other_override", "work derivation changed"),
        ("person_hold", "person hold reason changed"),
    ],
)
def test_post_bio_parent_rejects_schema_and_derivation_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    decisions = _decisions()
    baseline = _files(tmp_path / "baseline", decisions)
    parent = _files(tmp_path / "parent", decisions)
    _post_bio_parent(parent, decisions)
    works = parent["works/train-00000-of-00001.parquet"]
    people = parent["people/train-00000-of-00001.parquet"]
    if mutation == "extra_work":
        _rewrite_rows(works, lambda rows: [row.update(extra="") for row in rows])
    elif mutation == "extra_person":
        _rewrite_rows(people, lambda rows: [row.update(extra="") for row in rows])
    elif mutation == "missing_added_work":
        _rewrite_rows(
            works,
            lambda rows: [
                row.pop("map_region_prior_excluded_override") for row in rows
            ],
        )
    elif mutation == "missing_added_person":
        _rewrite_rows(
            people, lambda rows: [row.pop("publication_hold_reason") for row in rows]
        )
    elif mutation == "missing_dcb_work":
        _rewrite_rows(
            works, lambda rows: [row.pop("map_region_audit_version") for row in rows]
        )
    elif mutation == "missing_dcb_person":
        _rewrite_rows(people, lambda rows: [row.pop("display_name") for row in rows])
    elif mutation == "wrong_eligible":
        _rewrite_rows(
            works,
            lambda rows: rows[0].update(
                map_region_prior_eligible_override=(
                    not rows[0]["map_region_prior_eligible_override"]
                )
            ),
        )
    elif mutation == "wrong_excluded":
        _rewrite_rows(
            works, lambda rows: rows[0].update(map_region_prior_excluded_override=True)
        )
    elif mutation == "wrong_audit":
        _rewrite_rows(
            works,
            lambda rows: rows[0].update(map_region_audit_version="wrong"),
        )
    elif mutation == "wrong_override":
        _rewrite_rows(
            works,
            lambda rows: next(
                row
                for row in rows
                if row["work_id"] == continuity.POST_BIO_DOCUMENTED_OVERRIDE_WORK_ID
            ).update(map_region_documented_override=False),
        )
    elif mutation == "other_override":
        _rewrite_rows(
            works,
            lambda rows: next(
                row
                for row in rows
                if row["work_id"] != continuity.POST_BIO_DOCUMENTED_OVERRIDE_WORK_ID
            ).update(map_region_documented_override=True),
        )
    elif mutation == "person_hold":
        _rewrite_rows(
            people, lambda rows: rows[0].update(publication_hold_reason="hold")
        )
    with pytest.raises(ValueError, match=message):
        validate_chemistry_parent(
            decisions,
            baseline_files=baseline,
            parent_files=parent,
            prior_revision="a" * 40,
        )


def test_parent_row_comparison_normalizes_nulls_and_rejects_missing_identity() -> None:
    assert continuity._equal_value(float("nan")) is None
    assert continuity._equal_value({"values": [float("nan")]}) == {"values": [None]}
    with pytest.raises(ValueError, match="identities changed"):
        continuity._same_rows(
            {"reviewed": {"value": 1}},
            {"other": {"value": 1}},
            label="reviewed",
        )


def test_exact_dcb_parent_and_later_content_lineage(tmp_path: Path) -> None:
    decisions = _decisions()
    baseline = _files(tmp_path / "baseline", decisions)
    state = validate_chemistry_parent(
        decisions,
        baseline_files=baseline,
        parent_files=baseline,
        prior_revision=CHEMISTRY_HUB_REVISION,
    )
    assert state["reviewed_works"] == 19
    assert state["visible_works"] == 10
    assert state["lineage"] == "exact_dcb_marker"
    parent = _files(tmp_path / "later", decisions)
    path = parent["works/train-00000-of-00001.parquet"]
    _rewrite_rows(path, lambda rows: rows[0].update(x=0.75, keyword_id="new-broad"))
    _rewrite_rows(
        parent["people/train-00000-of-00001.parquet"],
        lambda rows: rows[0].update(mapped_work_count=7),
    )
    artifact_path = parent["maps/publications.json"]
    artifact = json.loads(artifact_path.read_text())
    if decisions[0].visible:
        point = next(
            point
            for point in artifact["points"]
            if point["work_id"] == decisions[0].work_id
        )
        point["keyword_ids"][0] = "new-broad"
        point["x"] = 0.75
    artifact_path.write_text(json.dumps(artifact))
    later = validate_chemistry_parent(
        decisions,
        baseline_files=baseline,
        parent_files=parent,
        prior_revision="a" * 40,
    )
    assert later["lineage"] == "exact_dcb_reviewed_rows"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("marker", "release marker changed"),
        ("dcb_parent_marker", "parent marker changed"),
        ("profile", "profile payload changed"),
        ("profile_duplicate", "repeats author_pub_id"),
        ("profile_missing", "lacks reviewed author_pub_id"),
        ("work", "parent decision changed"),
        ("work_schema", "unreviewed schema fields"),
        ("authorship", "authored visibility changed"),
        ("authorship_duplicate", "repeats authorship"),
        ("point", "map visibility changed"),
        ("point_count", "map points are invalid"),
        ("point_duplicate", "repeats map point"),
        ("point_topic", "topic point changed"),
        ("person", "person payload changed"),
        ("map_point_payload", "map point payload changed"),
        ("later_marker", "lineage marker changed"),
    ],
)
def test_parent_rejects_reviewed_state_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    decisions = _decisions()
    baseline = _files(tmp_path / "baseline", decisions)
    parent = _files(tmp_path / "parent", decisions)
    if mutation == "marker":
        path = baseline["maps/manifest.json"]
        source = json.loads(path.read_text())
        source["region_continuity"]["chem_region_review_sha256"] = "0" * 64
        path.write_text(json.dumps(source))
    elif mutation == "dcb_parent_marker":
        path = parent["maps/manifest.json"]
        source = json.loads(path.read_text())
        source["region_continuity"]["chem_region_review_sha256"] = "0" * 64
        path.write_text(json.dumps(source))
    elif mutation == "profile":
        _rewrite_rows(
            parent["profile_publications/train-00000-of-00001.parquet"],
            lambda rows: rows[0].update(review_evidence_url="changed"),
        )
    elif mutation == "profile_duplicate":
        _rewrite_rows(
            parent["profile_publications/train-00000-of-00001.parquet"],
            lambda rows: rows.append(copy.deepcopy(rows[0])),
        )
    elif mutation == "profile_missing":
        _rewrite_rows(
            parent["profile_publications/train-00000-of-00001.parquet"],
            lambda rows: rows.pop(),
        )
    elif mutation == "work":
        _rewrite_rows(
            parent["works/train-00000-of-00001.parquet"],
            lambda rows: rows[0].update(observation_ids=["other"]),
        )
    elif mutation == "work_schema":
        _rewrite_rows(
            parent["works/train-00000-of-00001.parquet"],
            lambda rows: [row.update(extra="unreviewed") for row in rows],
        )
    elif mutation == "authorship":
        _rewrite_rows(
            parent["authorships/train-00000-of-00001.parquet"],
            lambda rows: rows.pop(),
        )
    elif mutation == "authorship_duplicate":
        _rewrite_rows(
            parent["authorships/train-00000-of-00001.parquet"],
            lambda rows: rows.append(copy.deepcopy(rows[0])),
        )
    elif mutation == "point":
        path = parent["maps/publications.json"]
        source = json.loads(path.read_text())
        source["points"].pop()
        source["point_count"] -= 1
        path.write_text(json.dumps(source))
    elif mutation in {"point_count", "point_duplicate", "point_topic"}:
        path = parent["maps/publications.json"]
        source = json.loads(path.read_text())
        if mutation == "point_count":
            source["point_count"] += 1
        elif mutation == "point_duplicate":
            source["points"].append(copy.deepcopy(source["points"][0]))
            source["point_count"] += 1
        else:
            source["points"][0]["keyword_ids"] = ["wrong", "detail"]
        path.write_text(json.dumps(source))
    elif mutation == "person":
        _rewrite_rows(
            parent["people/train-00000-of-00001.parquet"],
            lambda rows: rows[0].update(display_name="changed"),
        )
    elif mutation == "map_point_payload":
        path = parent["maps/publications.json"]
        source = json.loads(path.read_text())
        source["points"][0]["faculty_ids"] = ["changed"]
        path.write_text(json.dumps(source))
    elif mutation == "later_marker":
        path = parent["maps/manifest.json"]
        source = json.loads(path.read_text())
        source["region_continuity"]["chem_region_review_sha256"] = "0" * 64
        path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match=message):
        validate_chemistry_parent(
            decisions,
            baseline_files=baseline,
            parent_files=parent,
            prior_revision=(
                CHEMISTRY_HUB_REVISION if mutation == "dcb_parent_marker" else "a" * 40
            ),
        )


def test_bio_region_refit_tolerance_requires_exact_parent_source_and_refit() -> None:
    kwargs = {
        "prior_revision": continuity.BIO_REGION_REFIT_PARENT,
        "source_sha256": continuity.BIO_REGION_REFIT_SOURCE_SHA256,
        "layout_reused": False,
    }
    assert continuity.allows_bio_region_refit(**kwargs)
    for changed in (
        {"prior_revision": CHEMISTRY_HUB_REVISION},
        {"source_sha256": "0" * 64},
        {"layout_reused": True},
    ):
        assert not continuity.allows_bio_region_refit(**(kwargs | changed))


def test_candidate_retains_all_reviewed_rows_and_hidden_authorships(
    tmp_path: Path,
) -> None:
    decisions = _decisions()
    files = _files(tmp_path / "baseline", decisions)
    state = validate_chemistry_parent(
        decisions,
        baseline_files=files,
        parent_files=files,
        prior_revision=CHEMISTRY_HUB_REVISION,
    )
    tables = {
        name: pandas.read_parquet(files[f"{name}/train-00000-of-00001.parquet"])
        for name in ("works", "profile_publications", "authorships", "people")
    }
    validate_chemistry_candidate(tables, state)
    visible = set(state["parent_rows"]["points"])
    assert len(visible) == 10
    armitage_id = (
        "work-a0cfe68d0794a4afe20e8a55b401afed795f14a8445cb9dc54275a84acb6b3d4"
    )
    donahue_id = "work-4e63e66d343de69522c77ffde7a55853382c39d01a9379ca47f4fa4eb939e802"
    assert {armitage_id, donahue_id} <= visible
    for row in state["parent_rows"]["works"].values():
        row["map_region_prior_eligible_override"] = False
        row["map_region_prior_excluded_override"] = False
        row["map_exclusion_reasons"] = []
    tables["works"]["map_region_prior_eligible_override"] = False
    tables["works"]["map_region_prior_excluded_override"] = False
    tables["works"]["map_exclusion_reasons"] = [[] for _ in range(len(tables["works"]))]
    changed = copy.deepcopy(tables)
    changed["works"].loc[
        changed["works"]["work_id"].eq(donahue_id), "map_region_documented_override"
    ] = True
    changed["works"].loc[
        changed["works"]["work_id"].eq(armitage_id),
        "map_region_prior_eligible_override",
    ] = True
    with pytest.raises(ValueError, match="candidate works changed"):
        validate_chemistry_candidate(changed, state)
    validate_chemistry_candidate(changed, state, allow_region_refit=True)
    hidden_id = next(
        work_id for work_id in state["parent_rows"]["works"] if work_id not in visible
    )
    for work_id, field, value in (
        (hidden_id, "map_region_documented_override", True),
        (hidden_id, "map_region_prior_eligible_override", True),
        (armitage_id, "map_region_prior_excluded_override", True),
        (armitage_id, "map_region_audit_version", "changed"),
        (armitage_id, "map_eligible", False),
        (armitage_id, "map_region_outlier", True),
        (armitage_id, "map_exclusion_reasons", ["changed"]),
        (armitage_id, "collaboration_eligible", False),
        (armitage_id, "title", "changed"),
    ):
        invalid = copy.deepcopy(changed)
        invalid["works"].loc[invalid["works"]["work_id"].eq(work_id), field] = [value]
        with pytest.raises(ValueError, match="candidate works changed"):
            validate_chemistry_candidate(invalid, state, allow_region_refit=True)
    changed = copy.deepcopy(tables)
    changed["profile_publications"].loc[0, "review_evidence_url"] = "changed"
    with pytest.raises(ValueError, match="candidate profile_publications changed"):
        validate_chemistry_candidate(changed, state)
    changed = copy.deepcopy(tables)
    changed["profile_publications"] = changed["profile_publications"].drop(
        columns=["review_evidence_url"]
    )
    with pytest.raises(ValueError, match="candidate profile_publications schema"):
        validate_chemistry_candidate(changed, state)
    changed = copy.deepcopy(tables)
    changed["people"].loc[0, "display_name"] = "changed"
    with pytest.raises(ValueError, match="candidate people changed"):
        validate_chemistry_candidate(changed, state)
    changed = copy.deepcopy(tables)
    changed["people"].loc[0, "mapped_work_count"] += 1
    validate_chemistry_candidate(changed, state)
    with pytest.raises(ValueError, match="candidate people changed"):
        validate_chemistry_candidate(changed, state, require_count_parity=True)
    hidden = next(item for item in decisions if not item.visible)
    changed = copy.deepcopy(tables)
    changed["authorships"] = pandas.concat(
        [
            changed["authorships"],
            pandas.DataFrame(
                [
                    {
                        "observation_id": f"observation-{hidden.author_pub_id}",
                        "work_id": hidden.work_id,
                        "person_id": hidden.person_id,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="authored visibility changed"):
        validate_chemistry_candidate(changed, state)


@pytest.mark.parametrize("table", ["profile_publications", "works", "authorships"])
def test_candidate_rejects_duplicate_reviewed_identity(
    tmp_path: Path, table: str
) -> None:
    decisions = _decisions()
    files = _files(tmp_path / "baseline", decisions)
    state = validate_chemistry_parent(
        decisions,
        baseline_files=files,
        parent_files=files,
        prior_revision=CHEMISTRY_HUB_REVISION,
    )
    tables = {
        name: pandas.read_parquet(files[f"{name}/train-00000-of-00001.parquet"])
        for name in ("works", "profile_publications", "authorships", "people")
    }
    tables[table] = pandas.concat(
        [tables[table], tables[table].iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError, match=f"candidate {table} lost reviewed rows"):
        validate_chemistry_candidate(tables, state)
