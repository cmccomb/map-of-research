"""Release control for retained research-map presentation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas
import pytest

from map_of_research.presentation_drift import (
    PresentationDriftError,
    compare_presentation,
    preserve_additive_report_presentation,
    preserve_deletion_only_presentation,
    preserve_reviewed_additive_presentation,
    reported_coauthor_evidence_changed,
    require_held_additive_staged_works,
    require_presentation_review,
)


def artifact(count: int = 120, *, broad: str = "systems") -> dict:
    return {
        "keywords": [
            {
                "keyword_id": "broad",
                "label": broad,
                "extracted_label": broad,
                "label_reviewed": False,
                "level": 0,
                "parent_keyword_id": None,
            },
            {
                "keyword_id": "detail",
                "label": "methods",
                "extracted_label": "methods",
                "label_reviewed": False,
                "level": 1,
                "parent_keyword_id": "broad",
            },
        ],
        "points": [
            {
                "work_id": f"work-{index:04d}",
                "x": round(index / count, 7),
                "y": 0.0,
                "tsne_x": round(index / count, 7),
                "tsne_y": 0.0,
                "keyword_ids": ["broad", "detail"],
            }
            for index in range(count)
        ],
    }


def report(parent: dict, candidate: dict, *, source: str = "s" * 64) -> dict:
    return compare_presentation(
        parent,
        candidate,
        prior_revision="a" * 40,
        source_snapshot_sha256=source,
        parent_map_sha256="p" * 64,
        candidate_map_sha256="c" * 64,
    )


def approval(path: Path, item: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "decision": "accept_refit",
                "prior_revision": item["prior_revision"],
                "source_snapshot_sha256": item["source_snapshot_sha256"],
                "parent_map_sha256": item["parent_map_sha256"],
                "candidate_map_sha256": item["candidate_map_sha256"],
                "report_sha256": item["report_sha256"],
                "reviewed_at": "2026-10-30",
                "rationale": (
                    "Reviewed the exact refit and accepted its presentation changes."
                ),
            }
        )
    )


def parent_work_rows(parent: dict) -> list[dict]:
    return [
        {
            "work_id": point["work_id"],
            "x": point["x"],
            "y": point["y"],
            "tsne_x": point["tsne_x"],
            "tsne_y": point["tsne_y"],
            "keyword_id": "broad",
            "keyword": "systems",
            "keyword_extracted": "systems",
            "keyword_label_reviewed": False,
            "detail_keyword_id": "detail",
            "detail_keyword": "methods",
            "detail_keyword_extracted": "methods",
            "detail_keyword_label_reviewed": False,
        }
        for point in parent["points"]
    ]


def test_deletion_only_preserves_retained_coordinates_and_topics() -> None:
    parent = artifact(3)
    current = artifact(3, broad="robotics")
    current["points"] = current["points"][:2]
    for point in current["points"]:
        point["x"] = point["tsne_x"] = 0.9
    works = pandas.DataFrame(parent_work_rows(parent))
    works["keyword"] = "robotics"
    works["x"] = works["tsne_x"] = 0.9
    restored, receipt = preserve_deletion_only_presentation(
        works, parent_work_rows(parent), parent, current
    )
    assert receipt == {
        "status": "parent_presentation_preserved_for_deletions",
        "parent_points": 3,
        "retained_points": 2,
        "removed_points": 1,
        "new_point_policy": "no_new_public_points_allowed",
    }
    assert restored["x"].tolist() == [0.0, round(1 / 3, 7), 0.9]
    assert restored["keyword"].tolist() == ["systems", "systems", "robotics"]
    assert works["keyword"].tolist() == ["robotics", "robotics", "robotics"]
    for point, row in zip(
        current["points"], restored.to_dict("records")[:2], strict=True
    ):
        for field in ("x", "y", "tsne_x", "tsne_y"):
            point[field] = row[field]
    current["keywords"][0]["label"] = "systems"
    assert report(parent, current)["material_reasons"] == []


def test_deletion_preservation_never_places_new_public_points() -> None:
    parent = artifact(3)
    current = artifact(3)
    current["points"] = current["points"][:2]
    current["points"][1]["work_id"] = "new-work"
    works = pandas.DataFrame(parent_work_rows(parent)[:2])
    restored, receipt = preserve_deletion_only_presentation(
        works, parent_work_rows(parent), parent, current
    )
    assert restored is works
    assert receipt is None


@pytest.mark.parametrize("defect", ["duplicate", "missing", "wrong_topic"])
def test_deletion_preservation_rejects_bad_parent_work_rows(defect: str) -> None:
    parent = artifact(3)
    current = artifact(3)
    current["points"] = current["points"][:2]
    rows = parent_work_rows(parent)
    if defect == "duplicate":
        rows.append(rows[0])
    elif defect == "missing":
        rows.pop()
    else:
        rows[0]["keyword"] = "wrong"
    with pytest.raises(ValueError, match="Deletion-only parent"):
        preserve_deletion_only_presentation(
            pandas.DataFrame(parent_work_rows(parent)[:2]), rows, parent, current
        )


def test_deletion_preservation_rejects_missing_presentation_fields() -> None:
    parent = artifact(3)
    current = artifact(3)
    current["points"] = current["points"][:2]
    works = pandas.DataFrame(parent_work_rows(parent)[:2]).drop(columns=["keyword"])
    with pytest.raises(ValueError, match="lack presentation fields"):
        preserve_deletion_only_presentation(
            works, parent_work_rows(parent), parent, current
        )
    prior = parent_work_rows(parent)
    prior[0].pop("keyword_extracted")
    with pytest.raises(ValueError, match="lacks topics"):
        preserve_deletion_only_presentation(
            pandas.DataFrame(parent_work_rows(parent)[:2]), prior, parent, current
        )


def test_deletion_preservation_rejects_duplicate_candidate_work() -> None:
    parent = artifact(3)
    current = artifact(3)
    current["points"] = current["points"][:2]
    works = pandas.DataFrame(parent_work_rows(parent)[:2])
    works = pandas.concat([works, works.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique mapped IDs"):
        preserve_deletion_only_presentation(
            works, parent_work_rows(parent), parent, current
        )


def _additive_case():
    parent = artifact(2)
    candidate = artifact(2, broad="newly fitted topic")
    for old_point, new_point in zip(parent["points"], candidate["points"], strict=True):
        old_point["title"] = new_point["title"] = old_point["work_id"]
        new_point["x"] = new_point["tsne_x"] = 0.9
    candidate["points"][0]["reported_faculty_coauthors"] = [
        {"person_id": "coauthor", "source_person_id": "reporter"}
    ]
    prior = parent_work_rows(parent)
    works = pandas.DataFrame(prior)
    works["x"] = works["tsne_x"] = 0.9
    works["keyword"] = "newly fitted topic"
    return parent, candidate, prior, works


def _preserve_additive(parent, candidate, prior, works, *, parent_source="s" * 64):
    return preserve_additive_report_presentation(
        works,
        prior,
        parent,
        candidate,
        source_snapshot_sha256="s" * 64,
        parent_source_snapshot_sha256=parent_source,
    )


def test_additive_coauthor_evidence_preserves_same_source_presentation() -> None:
    parent, candidate, prior, works = _additive_case()
    assert reported_coauthor_evidence_changed(parent, candidate)
    unmapped = works.iloc[[0]].copy()
    unmapped["work_id"] = "unmapped-work"
    works = pandas.concat([works, unmapped], ignore_index=True)
    restored, receipt = _preserve_additive(parent, candidate, prior, works)
    assert receipt == {
        "status": "parent_presentation_preserved_for_additive_coauthor_evidence",
        "parent_points": 2,
        "retained_points": 2,
        "added_reported_coauthors": 1,
        "source_snapshot_sha256": "s" * 64,
        "new_point_policy": "no_public_point_changes_allowed",
    }
    assert restored["x"].tolist() == [0.0, 0.5, 0.9]
    assert restored["keyword"].tolist() == [
        "systems",
        "systems",
        "newly fitted topic",
    ]
    assert works["keyword"].tolist() == ["newly fitted topic"] * 3


def test_persistent_reports_do_not_trigger_additive_preservation() -> None:
    parent, candidate, _, _ = _additive_case()
    parent["points"][0]["reported_faculty_coauthors"] = candidate["points"][0][
        "reported_faculty_coauthors"
    ]
    assert not reported_coauthor_evidence_changed(parent, candidate)
    candidate["points"].append({**candidate["points"][1], "work_id": "new-work"})
    assert not reported_coauthor_evidence_changed(parent, candidate)


@pytest.mark.parametrize("defect", ["new_point", "removed_point", "changed_title"])
def test_additive_preservation_rejects_other_public_changes(defect: str) -> None:
    parent, candidate, prior, works = _additive_case()
    if defect == "new_point":
        candidate["points"].append({**candidate["points"][0], "work_id": "new"})
    elif defect == "removed_point":
        candidate["points"].pop()
    else:
        candidate["points"][1]["title"] = "different"
    with pytest.raises(
        ValueError, match="identical public work IDs|public work content"
    ):
        _preserve_additive(parent, candidate, prior, works)


def test_additive_preservation_requires_pinned_same_raw_source() -> None:
    parent, candidate, prior, works = _additive_case()
    with pytest.raises(ValueError, match="pinned parent source"):
        _preserve_additive(parent, candidate, prior, works, parent_source="t" * 64)


@pytest.mark.parametrize("defect", ["no_new", "removed_old", "bad_shape", "duplicate"])
def test_additive_preservation_rejects_bad_report_delta(defect: str) -> None:
    parent, candidate, prior, works = _additive_case()
    report = candidate["points"][0]["reported_faculty_coauthors"][0]
    if defect == "no_new":
        candidate["points"][0].pop("reported_faculty_coauthors")
    elif defect == "removed_old":
        parent["points"][0]["reported_faculty_coauthors"] = [
            {"person_id": "old", "source_person_id": "reporter"}
        ]
    elif defect == "bad_shape":
        candidate["points"][0]["reported_faculty_coauthors"] = ["bad"]
    else:
        candidate["points"][0]["reported_faculty_coauthors"] = [report, report]
    with pytest.raises(
        ValueError, match="coauthor evidence|existing evidence|invalid report"
    ):
        _preserve_additive(parent, candidate, prior, works)


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_prior",
        "missing_prior",
        "missing_field",
        "wrong_topic",
        "duplicate_work",
        "missing_work_field",
    ],
)
def test_additive_preservation_rejects_bad_work_rows(defect: str) -> None:
    parent, candidate, prior, works = _additive_case()
    if defect == "duplicate_prior":
        prior.append(prior[0])
    elif defect == "missing_prior":
        prior.pop()
    elif defect == "missing_field":
        prior[0].pop("keyword_extracted")
    elif defect == "wrong_topic":
        prior[0]["keyword"] = "different"
    elif defect == "duplicate_work":
        works = pandas.concat([works, works.iloc[[0]]], ignore_index=True)
    else:
        works = works.drop(columns=["keyword"])
    with pytest.raises(
        ValueError, match="parent works|parent work|candidate works|parent topic"
    ):
        _preserve_additive(parent, candidate, prior, works)


def test_identical_retained_presentation_passes_without_approval() -> None:
    old = artifact()
    current = artifact()
    current["points"].append(
        {
            "work_id": "new-work",
            "x": 0.1,
            "y": 0.2,
            "tsne_x": 0.1,
            "tsne_y": 0.2,
            "keyword_ids": ["broad", "detail"],
        }
    )
    item = report(old, current)
    assert item["point_counts"] == {
        "parent": 120,
        "candidate": 121,
        "retained": 120,
        "added": 1,
        "removed": 0,
    }
    assert item["material_reasons"] == []
    assert require_presentation_review(item)["status"] == "pass_no_material_drift"
    assert report(old, current)["report_sha256"] == item["report_sha256"]


def test_material_semantic_drift_fails_before_review() -> None:
    item = report(artifact(), artifact(broad="robotics"))
    assert item["retained"]["broad_semantic_label_changes"] == 120
    assert "broad_semantic_topic_churn" in item["material_reasons"]
    assert "broad_region_name_change" in item["material_reasons"]
    with pytest.raises(
        PresentationDriftError, match="lacks a reviewed approval"
    ) as error:
        require_presentation_review(item)
    assert error.value.receipt["report_sha256"] == item["report_sha256"]
    assert error.value.receipt["status"] == "review_required"


def test_material_movement_requires_exact_review(tmp_path: Path) -> None:
    old = artifact()
    current = artifact()
    for point in current["points"]:
        point["tsne_y"] = 0.2
    item = report(old, current)
    assert item["retained"]["tsne_distance_over_0_1"] == 120
    path = tmp_path / "approval.json"
    approval(path, item)
    receipt = require_presentation_review(item, path)
    assert receipt["status"] == "pass_reviewed_refit"
    assert receipt["reviewed_approval_sha256"]


def test_pca_movement_also_requires_review() -> None:
    old, current = artifact(), artifact()
    for point in current["points"]:
        point["y"] = 0.2
    item = report(old, current)
    assert item["retained"]["pca_distance_over_0_1"] == 120
    assert "retained_pca_point_movement" in item["material_reasons"]


@pytest.mark.parametrize(
    "field", ["prior_revision", "source_snapshot_sha256", "report_sha256"]
)
def test_stale_or_mismatched_approval_fails(tmp_path: Path, field: str) -> None:
    item = report(artifact(), artifact(broad="robotics"))
    path = tmp_path / "approval.json"
    approval(path, item)
    document = json.loads(path.read_text())
    document[field] = "0" * len(document[field])
    path.write_text(json.dumps(document))
    with pytest.raises(
        PresentationDriftError, match="differs from exact parent"
    ) as error:
        require_presentation_review(item, path)
    assert error.value.receipt["status"] == "stale_approval"


def test_unknown_topic_or_duplicate_work_fails_closed() -> None:
    old = artifact()
    current = artifact()
    current["points"][0]["keyword_ids"] = ["missing", "detail"]
    with pytest.raises(ValueError, match="unknown keyword ID"):
        report(old, current)
    current = artifact()
    current["points"].append(current["points"][0])
    with pytest.raises(ValueError, match="duplicate work ID"):
        report(old, current)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda item: item.pop("points"), "lacks a point list"),
        (lambda item: item["points"].append(None), "non-object point"),
        (lambda item: item.pop("keywords"), "lacks a keyword catalog"),
        (lambda item: item["keywords"].append(None), "non-object row"),
        (
            lambda item: item["keywords"].append(item["keywords"][0]),
            "duplicate keyword label",
        ),
        (lambda item: item["points"][0].pop("keyword_ids"), "exactly two keyword IDs"),
        (lambda item: item["points"][0].pop("x"), "invalid map coordinates"),
        (
            lambda item: item["points"][0].update(x=float("inf")),
            "non-finite map coordinates",
        ),
    ],
)
def test_malformed_parent_or_candidate_map_fails_closed(change, message) -> None:
    old, current = artifact(), artifact()
    change(current)
    with pytest.raises(ValueError, match=message):
        report(old, current)


def test_no_retained_points_cannot_be_approved() -> None:
    old, current = artifact(1), artifact(1)
    current["points"][0]["work_id"] = "different"
    with pytest.raises(ValueError, match="no retained works"):
        report(old, current)


def test_changed_topic_ids_with_same_labels_and_changed_detail_label() -> None:
    old, current = artifact(), artifact()
    current["keywords"] = [
        {"keyword_id": "broad-new", "label": "systems"},
        {"keyword_id": "detail-new", "label": "new methods"},
    ]
    for point in current["points"]:
        point["keyword_ids"] = ["broad-new", "detail-new"]
    item = report(old, current)
    assert item["retained"]["topic_id_changes"] == 120
    assert item["retained"]["broad_semantic_label_changes"] == 0
    assert item["retained"]["detail_semantic_label_changes"] == 120
    assert "retained_keyword_id_change" in item["material_reasons"]
    assert "detail_semantic_topic_churn" in item["material_reasons"]


def test_keyword_id_renumbering_alone_requires_exact_review() -> None:
    parent, candidate = artifact(), artifact()
    candidate["keywords"] = [
        {"keyword_id": "broad-new", "label": "systems"},
        {"keyword_id": "detail-new", "label": "methods"},
    ]
    for point in candidate["points"]:
        point["keyword_ids"] = ["broad-new", "detail-new"]
    item = report(parent, candidate)
    assert item["retained"]["topic_id_changes"] == 120
    assert item["retained"]["broad_semantic_label_changes"] == 0
    assert item["retained"]["detail_semantic_label_changes"] == 0
    assert item["material_reasons"] == ["retained_keyword_id_change"]
    with pytest.raises(PresentationDriftError, match="reviewed approval"):
        require_presentation_review(item)


def test_single_retained_broad_reassignment_requires_exact_review() -> None:
    parent, candidate = artifact(), artifact()
    for item in (parent, candidate):
        item["keywords"].append({"keyword_id": "robotics", "label": "robotics"})
        item["points"][-1]["keyword_ids"][0] = "robotics"
    candidate["points"][0]["keyword_ids"][0] = "robotics"
    item = report(parent, candidate)
    assert item["retained"]["broad_semantic_label_changes"] == 1
    assert item["retained"]["topic_id_changes"] == 1
    assert item["retained"]["new_broad_region_labels"] == []
    assert "broad_semantic_topic_churn" in item["material_reasons"]
    with pytest.raises(PresentationDriftError, match="reviewed approval"):
        require_presentation_review(item)


def test_single_retained_detail_reassignment_requires_exact_review() -> None:
    parent, candidate = artifact(), artifact()
    for item in (parent, candidate):
        item["keywords"].append({"keyword_id": "other-detail", "label": "other"})
        item["points"][-1]["keyword_ids"][1] = "other-detail"
    candidate["points"][0]["keyword_ids"][1] = "other-detail"
    item = report(parent, candidate)
    assert item["retained"]["detail_semantic_label_changes"] == 1
    assert "detail_semantic_topic_churn" in item["material_reasons"]
    with pytest.raises(PresentationDriftError, match="reviewed approval"):
        require_presentation_review(item)


@pytest.mark.parametrize(
    ("level", "reason"),
    [
        (0, "broad_semantic_topic_churn"),
        (1, "detail_semantic_topic_churn"),
    ],
)
def test_single_catalog_label_change_with_same_keyword_id_requires_review(
    level: int, reason: str
) -> None:
    parent, candidate = artifact(), artifact()
    for item in (parent, candidate):
        item["keywords"].append({"keyword_id": "unique", "label": "old label"})
        item["points"][-1]["keyword_ids"][level] = "unique"
    candidate["keywords"][-1]["label"] = "changed label"
    item = report(parent, candidate)
    assert item["retained"]["topic_id_changes"] == 0
    field = (
        "broad_semantic_label_changes"
        if level == 0
        else "detail_semantic_label_changes"
    )
    assert item["retained"][field] == 1
    assert reason in item["material_reasons"]
    with pytest.raises(PresentationDriftError, match="reviewed approval"):
        require_presentation_review(item)


def test_unneeded_or_malformed_approval_fails_closed(tmp_path: Path) -> None:
    unchanged = report(artifact(), artifact())
    path = tmp_path / "approval.json"
    approval(path, unchanged)
    with pytest.raises(PresentationDriftError, match="unused approval"):
        require_presentation_review(unchanged, path)
    with pytest.raises(ValueError, match="lacks material reasons"):
        require_presentation_review({})
    changed = report(artifact(), artifact(broad="robotics"))
    path.unlink()
    with pytest.raises(PresentationDriftError, match="approval cannot be read"):
        require_presentation_review(changed, path)
    path.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(PresentationDriftError, match="schema is invalid") as error:
        require_presentation_review(changed, path)
    assert error.value.receipt["status"] == "invalid_approval"
    approval(path, changed)
    document = json.loads(path.read_text())
    document["rationale"] = "too short"
    path.write_text(json.dumps(document))
    with pytest.raises(PresentationDriftError, match="lacks review date or rationale"):
        require_presentation_review(changed, path)
    document["rationale"] = (
        "Reviewed exact semantic and placement drift for this release."
    )
    document["reviewed_at"] = "yesterday"
    path.write_text(json.dumps(document))
    with pytest.raises(PresentationDriftError, match="approval date is invalid"):
        require_presentation_review(changed, path)


def additive_case() -> tuple[dict, dict, pandas.DataFrame]:
    parent = artifact(3)
    candidate = deepcopy(parent)
    for point in candidate["points"]:
        point["x"] = point["tsne_x"] = 0.9
    for work_id in ("work-new", "work-hold"):
        candidate["points"].append(
            {
                "work_id": work_id,
                "x": 0.7,
                "y": 0.1,
                "tsne_x": 0.6,
                "tsne_y": -0.2,
                "keyword_ids": ["broad", "detail"],
            }
        )
    rows = parent_work_rows(parent)
    for row in rows:
        row["x"] = row["tsne_x"] = 0.9
        row["keyword"] = "refitted label"
        row["map_eligible"] = True
        row["map_exclusion_reasons"] = []
    for work_id in ("work-new", "work-hold"):
        rows.append(
            {
                **rows[0],
                "work_id": work_id,
                "map_exclusion_reasons": [],
            }
        )
    return parent, candidate, pandas.DataFrame(rows)


def additive_review(path: Path, receipt: dict, *, hold: bool = True) -> dict:
    new_points = [
        {
            "work_id": "work-new",
            "decision": "place_unclassified",
            "coordinates": {
                "x": 0.21,
                "y": 0.3,
                "tsne_x": 0.24,
                "tsne_y": -0.1,
            },
            "rationale": "Reviewed nearby work titles and approved this placement.",
        },
        {
            "work_id": "work-hold",
            "decision": "hold_pending_placement" if hold else "place_unclassified",
            "coordinates": None
            if hold
            else {"x": 0.31, "y": 0.1, "tsne_x": 0.2, "tsne_y": 0.4},
            "rationale": "Placement is uncertain and remains off the public map.",
        },
    ]
    review = {
        "schema_version": 1,
        "decision": "freeze_retained_review_new",
        "prior_revision": receipt["prior_revision"],
        "source_snapshot_sha256": receipt["source_snapshot_sha256"],
        "selected_ids_sha256": receipt["selected_ids_sha256"],
        "parent_map_sha256": receipt["parent_map_sha256"],
        "candidate_map_content_sha256": receipt["candidate_map_content_sha256"],
        "new_work_ids_sha256": receipt["new_work_ids_sha256"],
        "reviewed_at": "2026-10-30",
        "new_points": new_points,
    }
    path.write_text(json.dumps(review))
    return review


def freeze_additive(
    parent: dict,
    candidate: dict,
    works: pandas.DataFrame,
    path: Path | None,
) -> tuple[pandas.DataFrame, dict]:
    return preserve_reviewed_additive_presentation(
        works,
        parent_work_rows(parent),
        parent,
        candidate,
        prior_revision="a" * 40,
        source_snapshot_sha256="s" * 64,
        selected_ids_sha256="i" * 64,
        parent_map_sha256="p" * 64,
        review_path=path,
    )


def test_additive_freezes_parent_and_reviews_each_new_point(tmp_path: Path) -> None:
    parent, candidate, works = additive_case()
    with pytest.raises(
        PresentationDriftError, match="lack reviewed placement"
    ) as error:
        freeze_additive(parent, candidate, works, None)
    receipt = error.value.receipt
    path = tmp_path / "placement-review.json"
    additive_review(path, receipt)
    output, item = freeze_additive(parent, candidate, works, path)
    assert item["status"] == "parent_frozen_new_points_reviewed"
    assert item["placed_new_points"] == 1
    assert item["held_new_points"] == 1
    assert output["x"].tolist()[:3] == [0.0, round(1 / 3, 7), round(2 / 3, 7)]
    assert output["keyword"].tolist()[:3] == ["systems"] * 3
    placed = output.loc[output.work_id.eq("work-new")].iloc[0]
    assert placed.keyword_id == "keyword-unclassified"
    assert placed.detail_keyword == "Pending topic review"
    assert (placed.x, placed.y, placed.tsne_x, placed.tsne_y) == (
        0.21,
        0.3,
        0.24,
        -0.1,
    )
    held = output.loc[output.work_id.eq("work-hold")].iloc[0]
    assert not held.map_eligible
    assert held.map_exclusion_reasons == ["pending_placement_review"]
    assert held.keyword_id == ""
    assert all(pandas.isna(held[field]) for field in ("x", "y", "tsne_x", "tsne_y"))
    assert works["x"].tolist() == [0.9] * 5


def test_additive_staged_works_rejects_unreviewed_held_placement(
    tmp_path: Path,
) -> None:
    parent, candidate, works = additive_case()
    with pytest.raises(PresentationDriftError) as error:
        freeze_additive(parent, candidate, works, None)
    review_path = tmp_path / "placement-review.json"
    additive_review(review_path, error.value.receipt)
    output, receipt = freeze_additive(parent, candidate, works, review_path)
    staged_path = tmp_path / "works.parquet"
    output.to_parquet(staged_path, index=False)
    require_held_additive_staged_works(staged_path, receipt)
    require_held_additive_staged_works(
        staged_path, {**receipt, "held_new_work_ids": []}
    )

    held_index = output.index[output.work_id.eq("work-hold")][0]
    for field in ("x", "y", "tsne_x", "tsne_y"):
        broken = output.copy(deep=True)
        broken.at[held_index, field] = 0.2
        broken.to_parquet(staged_path, index=False)
        with pytest.raises(ValueError, match="leaks coordinates"):
            require_held_additive_staged_works(staged_path, receipt)

    broken = output.copy(deep=True)
    broken.at[held_index, "map_eligible"] = True
    broken.to_parquet(staged_path, index=False)
    with pytest.raises(ValueError, match="lacks placement hold"):
        require_held_additive_staged_works(staged_path, receipt)

    broken = output.copy(deep=True)
    broken.at[held_index, "keyword"] = "Unreviewed topic"
    broken.to_parquet(staged_path, index=False)
    with pytest.raises(ValueError, match="leaks topic labels"):
        require_held_additive_staged_works(staged_path, receipt)

    broken = output.loc[output.work_id.ne("work-hold")].copy()
    broken.to_parquet(staged_path, index=False)
    with pytest.raises(ValueError, match="lack unique held work IDs"):
        require_held_additive_staged_works(staged_path, receipt)


@pytest.mark.parametrize(
    "defect,match",
    [
        ("source", "stale or mismatched"),
        ("selected", "stale or mismatched"),
        ("parent", "stale or mismatched"),
        ("candidate", "stale or mismatched"),
        ("unreadable", "cannot be read"),
        ("schema", "schema is invalid"),
        ("date", "date is invalid"),
        ("missing", "cover every new point"),
        ("duplicate", "repeats or is unexpected"),
        ("row_schema", "row schema is invalid"),
        ("decision", "decision is invalid"),
        ("held_coordinates", "held placement has coordinates"),
        ("coordinate_shape", "lacks four coordinates"),
        ("coordinate_type", "coordinate is invalid"),
        ("nan", "not finite normalized"),
        ("precision", "exceeds map precision"),
        ("unreviewed", "rationale is too short"),
    ],
)
def test_additive_review_fails_closed(tmp_path: Path, defect: str, match: str) -> None:
    parent, candidate, works = additive_case()
    with pytest.raises(PresentationDriftError) as error:
        freeze_additive(parent, candidate, works, None)
    path = tmp_path / "placement-review.json"
    item = additive_review(path, error.value.receipt)
    if defect == "source":
        item["source_snapshot_sha256"] = "x" * 64
    elif defect == "selected":
        item["selected_ids_sha256"] = "x" * 64
    elif defect == "parent":
        item["prior_revision"] = "b" * 40
    elif defect == "candidate":
        item["candidate_map_content_sha256"] = "x" * 64
    elif defect == "unreadable":
        path.unlink()
    elif defect == "schema":
        item.pop("reviewed_at")
    elif defect == "date":
        item["reviewed_at"] = "yesterday"
    elif defect == "missing":
        item["new_points"].pop()
    elif defect == "duplicate":
        item["new_points"][1]["work_id"] = "work-new"
    elif defect == "row_schema":
        item["new_points"][0].pop("rationale")
    elif defect == "decision":
        item["new_points"][0]["decision"] = "auto_assign_topic"
    elif defect == "held_coordinates":
        item["new_points"][1]["coordinates"] = {
            "x": 0.1,
            "y": 0.2,
            "tsne_x": 0.3,
            "tsne_y": 0.4,
        }
    elif defect == "coordinate_shape":
        item["new_points"][0]["coordinates"].pop("x")
    elif defect == "coordinate_type":
        item["new_points"][0]["coordinates"]["x"] = "bad"
    elif defect == "nan":
        item["new_points"][0]["coordinates"]["x"] = float("nan")
    elif defect == "precision":
        item["new_points"][0]["coordinates"]["x"] = 0.123456789
    else:
        item["new_points"][0]["rationale"] = "not reviewed"
    if defect != "unreadable":
        path.write_text(json.dumps(item))
    with pytest.raises(PresentationDriftError, match=match):
        freeze_additive(parent, candidate, works, path)


def test_additive_rejects_prior_excluded_promotion_and_topic_id_collision(
    tmp_path: Path,
) -> None:
    parent, candidate, works = additive_case()
    prior = parent_work_rows(parent)
    prior.append({**prior[0], "work_id": "work-new"})
    with pytest.raises(ValueError, match="separate promotion review"):
        preserve_reviewed_additive_presentation(
            works,
            prior,
            parent,
            candidate,
            prior_revision="a" * 40,
            source_snapshot_sha256="s" * 64,
            selected_ids_sha256="i" * 64,
            parent_map_sha256="p" * 64,
            review_path=tmp_path / "missing.json",
        )
    parent["keywords"].append(
        {
            "keyword_id": "keyword-unclassified",
            "label": "A conflicting old label",
        }
    )
    with pytest.raises(ValueError, match="collides with parent"):
        freeze_additive(parent, candidate, works, None)


def test_additive_requires_new_points_and_stable_parent_versions() -> None:
    parent, candidate, works = additive_case()
    candidate["points"] = candidate["points"][:3]
    with pytest.raises(ValueError, match="no new map points"):
        freeze_additive(parent, candidate, works, None)
    candidate["points"] = candidate["points"][1:]
    with pytest.raises(ValueError, match="cannot remove parent map points"):
        freeze_additive(parent, candidate, works, None)
    candidate = deepcopy(parent)
    candidate["points"].append(
        {
            "work_id": "work-new",
            "x": 0.1,
            "y": 0.1,
            "tsne_x": 0.1,
            "tsne_y": 0.1,
            "keyword_ids": ["broad", "detail"],
        }
    )
    parent["layout_version"] = "older"
    candidate["layout_version"] = "newer"
    with pytest.raises(ValueError, match="layout_version differs"):
        freeze_additive(parent, candidate, works, None)


def test_additive_requires_complete_parent_topic_metadata(tmp_path: Path) -> None:
    parent, candidate, works = additive_case()
    prior = parent_work_rows(parent)
    prior[0]["detail_keyword_extracted"] = "different extracted label"
    with pytest.raises(PresentationDriftError) as error:
        freeze_additive(parent, candidate, works, None)
    # A valid review cannot override a parent work/catalog mismatch.
    path = tmp_path / "review.json"
    additive_review(path, error.value.receipt)
    with pytest.raises(ValueError, match="parent topic differs from map"):
        preserve_reviewed_additive_presentation(
            works,
            prior,
            parent,
            candidate,
            prior_revision="a" * 40,
            source_snapshot_sha256="s" * 64,
            selected_ids_sha256="i" * 64,
            parent_map_sha256="p" * 64,
            review_path=path,
        )


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("duplicate_parent", "duplicate IDs"),
        ("missing_parent", "matching works rows"),
        ("missing_field", "lack placement or topic fields"),
        ("duplicate_current", "unique mapped IDs"),
        ("missing_current", "unique mapped IDs"),
    ],
)
def test_additive_rejects_malformed_work_inputs(defect: str, message: str) -> None:
    parent, candidate, works = additive_case()
    prior = parent_work_rows(parent)
    if defect == "duplicate_parent":
        prior.append(prior[0])
    elif defect == "missing_parent":
        prior.pop()
    elif defect == "missing_field":
        works = works.drop(columns=["detail_keyword"])
    elif defect == "duplicate_current":
        works = pandas.concat([works, works.iloc[[0]]], ignore_index=True)
    else:
        works = works.loc[works.work_id.ne("work-new")]
    with pytest.raises(ValueError, match=message):
        preserve_reviewed_additive_presentation(
            works,
            prior,
            parent,
            candidate,
            prior_revision="a" * 40,
            source_snapshot_sha256="s" * 64,
            selected_ids_sha256="i" * 64,
            parent_map_sha256="p" * 64,
            review_path=None,
        )


def test_additive_rejects_parent_with_missing_topic_field(tmp_path: Path) -> None:
    parent, candidate, works = additive_case()
    with pytest.raises(PresentationDriftError) as error:
        freeze_additive(parent, candidate, works, None)
    path = tmp_path / "review.json"
    additive_review(path, error.value.receipt)
    prior = parent_work_rows(parent)
    prior[0].pop("detail_keyword_extracted")
    with pytest.raises(ValueError, match="parent work lacks topic fields"):
        preserve_reviewed_additive_presentation(
            works,
            prior,
            parent,
            candidate,
            prior_revision="a" * 40,
            source_snapshot_sha256="s" * 64,
            selected_ids_sha256="i" * 64,
            parent_map_sha256="p" * 64,
            review_path=path,
        )


def test_additive_review_is_stale_if_candidate_changes_after_review(
    tmp_path: Path,
) -> None:
    parent, candidate, works = additive_case()
    with pytest.raises(PresentationDriftError) as error:
        freeze_additive(parent, candidate, works, None)
    path = tmp_path / "placement-review.json"
    additive_review(path, error.value.receipt)
    next(point for point in candidate["points"] if point["work_id"] == "work-new")[
        "tsne_x"
    ] = 0.9
    with pytest.raises(PresentationDriftError, match="stale or mismatched"):
        freeze_additive(parent, candidate, works, path)
