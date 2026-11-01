"""Release control for retained research-map presentation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pandas
import pytest

from map_of_research.composite_region import BLANK_TOPICS
from map_of_research.presentation_drift import (
    RETAINED_POINT_GROWTH_FIELDS,
    RETAINED_POINT_PRESENTATION_FIELDS,
    RETAINED_WORK_GROWTH_FIELDS,
    RETAINED_WORK_PRESENTATION_FIELDS,
    PresentationDriftError,
    compare_presentation,
    preserve_additive_report_presentation,
    preserve_deletion_only_presentation,
    preserve_reviewed_additive_presentation,
    reported_coauthor_evidence_changed,
    require_excluded_additive_staged_works,
    require_held_additive_staged_works,
    require_presentation_review,
    require_preserved_additive_artifact,
    require_preserved_additive_staged_works,
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
                "coordinates": {
                    "pca": {"x": 0.0, "y": 0.0},
                    "tsne": {"x": 0.0, "y": 0.0},
                },
            },
            {
                "keyword_id": "detail",
                "label": "methods",
                "extracted_label": "methods",
                "label_reviewed": False,
                "level": 1,
                "parent_keyword_id": "broad",
                "coordinates": {
                    "pca": {"x": 0.1, "y": 0.1},
                    "tsne": {"x": 0.1, "y": 0.1},
                },
            },
        ],
        "points": [
            {
                "work_id": f"work-{index:04d}",
                "title": f"Title {index}",
                "authors": "A. Author",
                "year": 2025,
                "venue": "A Journal",
                "citation_count": 7,
                "doi": "10.1000/example",
                "publication_type": "publication",
                "source_document_id": "",
                "source_verified_at": "2026-10-30",
                "collaboration_eligible": True,
                "display_contributors": "",
                "source_url": "",
                "faculty_ids": ["person-one"],
                "department_ids": ["ece"],
                "school_ids": ["engineering"],
                "external_coauthor_ids": [],
                "source_types": ["scholar"],
                "source_native_ids": ["original-record"],
                "source_verification_urls": [],
                "context_evidence_urls": [],
                "observation_count": 1,
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
            "title": point["title"],
            "authors": point["authors"],
            "year": point["year"],
            "venue": point["venue"],
            "citation": "Full citation",
            "citation_count": point["citation_count"],
            "doi": point["doi"],
            "publication_type": point["publication_type"],
            "source_document_id": point["source_document_id"],
            "source_verified_at": point["source_verified_at"],
            "collaboration_eligible": point["collaboration_eligible"],
            "display_contributors": point["display_contributors"],
            "source_urls": [],
            "title_variants": [point["title"]],
            "author_variants": [point["authors"]],
            "year_variants": [point["year"]],
            "venue_variants": [point["venue"]],
            "citation_variants": ["Full citation"],
            "external_coauthors": [],
            "faculty": ["A. Author"],
            "department_titles": ["Electrical & Computer Engineering"],
            "school_titles": ["College of Engineering"],
            "person_ids": ["person-one"],
            "memberships": [{"person_id": "person-one", "department_id": "ece"}],
            "department_ids": ["ece"],
            "school_ids": ["engineering"],
            "observation_ids": ["first-observation"],
            "author_pub_ids": ["first-author-pub"],
            "scholar_ids": ["scholar-one"],
            "external_coauthor_ids": [],
            "source_types": ["scholar"],
            "source_native_ids": ["original-record"],
            "source_verification_urls": [],
            "context_evidence_urls": [],
            "observation_count": 1,
            "map_eligible": True,
            "map_exclusion_reasons": [],
            "layout_version": "test-layout-v1",
            "keyword_model_version": "test-keywords-v1",
            "topic_label_review_version": "test-topic-review-v1",
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
            "map_region_audit_version": "reviewed-region-v1",
            "map_region_outlier": False,
            "map_region_documented_override": False,
            "map_region_prior_eligible_override": False,
            "map_region_prior_excluded_override": False,
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
        "reported_coauthor_source_sha256": receipt["reported_coauthor_source_sha256"],
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
        reported_coauthor_source_sha256="a" * 64,
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


def test_additive_binding_only_freezes_parent_with_empty_exact_review(
    tmp_path: Path,
) -> None:
    parent, candidate, works = additive_case()
    candidate["points"] = candidate["points"][:3]
    works = works.iloc[:3].copy()
    works.at[0, "person_ids"] = ["person-one", "person-two"]
    works.at[0, "observation_ids"] = ["first-observation", "new-observation"]
    works.at[0, "observation_count"] = 2
    with pytest.raises(PresentationDriftError) as error:
        freeze_additive(parent, candidate, works, None)
    assert error.value.receipt["new_point_count"] == 0
    path = tmp_path / "binding-only-review.json"
    review = additive_review(path, error.value.receipt)
    review["new_points"] = []
    path.write_text(json.dumps(review))

    frozen, receipt = freeze_additive(parent, candidate, works, path)
    assert receipt["placed_new_work_ids"] == []
    assert receipt["held_new_work_ids"] == []
    assert receipt["excluded_new_work_reasons"] == {}
    assert frozen["x"].tolist() == [0.0, round(1 / 3, 7), round(2 / 3, 7)]
    assert frozen["keyword"].tolist() == ["systems"] * 3
    assert frozen.at[0, "person_ids"] == ["person-one", "person-two"]
    assert frozen.at[0, "observation_ids"] == ["first-observation", "new-observation"]
    assert frozen.at[0, "observation_count"] == 2
    assert works["x"].tolist() == [0.9] * 3

    for field, value in (
        ("source_snapshot_sha256", "x" * 64),
        ("candidate_map_content_sha256", "y" * 64),
    ):
        changed = {**review, field: value}
        path.write_text(json.dumps(changed))
        with pytest.raises(PresentationDriftError, match="stale or mismatched"):
            freeze_additive(parent, candidate, works, path)

    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="no new map points"):
        freeze_additive(
            parent,
            candidate,
            pandas.concat(
                [works, works.iloc[[0]].assign(work_id="work-novel")],
                ignore_index=True,
            ),
            path,
        )


def test_additive_keeps_existing_spatial_work_type_when_new_source_is_typed(
    tmp_path: Path,
) -> None:
    parent, candidate, works = additive_case()
    works.at[0, "publication_type"] = "conference_paper"
    with pytest.raises(PresentationDriftError) as error:
        freeze_additive(parent, candidate, works, None)
    review_path = tmp_path / "placement-review.json"
    additive_review(review_path, error.value.receipt)
    frozen, _ = freeze_additive(parent, candidate, works, review_path)
    assert frozen.at[0, "publication_type"] == "publication"
    assert works.at[0, "publication_type"] == "conference_paper"
    prior = parent_work_rows(parent)
    del prior[0]["publication_type"]
    with pytest.raises(ValueError, match="lacks presentation fields"):
        preserve_reviewed_additive_presentation(
            works,
            prior,
            parent,
            candidate,
            prior_revision="a" * 40,
            source_snapshot_sha256="s" * 64,
            selected_ids_sha256="i" * 64,
            parent_map_sha256="p" * 64,
            reported_coauthor_source_sha256="a" * 64,
            review_path=review_path,
        )


def test_additive_freezes_prior_bibliography_on_spatial_and_off_map_works(
    tmp_path: Path,
) -> None:
    parent, candidate, works = additive_case()
    prior = parent_work_rows(parent)
    off_map = {
        **prior[0],
        "work_id": "prior-off-map",
        "map_eligible": False,
        "map_exclusion_reasons": ["pending_placement_review"],
    }
    prior.append(off_map)
    works = pandas.concat([works, pandas.DataFrame([off_map])], ignore_index=True)
    for index in (0, len(works) - 1):
        for field, value in (
            ("title", "Refitted title"),
            ("authors", "B. Author"),
            ("year", 2026),
            ("venue", "Changed venue"),
            ("citation", "Changed citation"),
            ("citation_count", 8),
            ("doi", "10.1000/changed"),
            ("publication_type", "conference_paper"),
            ("source_document_id", "changed-document"),
            ("source_verified_at", "2026-10-31"),
            ("collaboration_eligible", False),
            ("display_contributors", "Changed contributors"),
            ("map_exclusion_reasons", ["different_reason"]),
            ("layout_version", "changed-layout"),
            ("keyword_model_version", "changed-model"),
            ("topic_label_review_version", "changed-topic-review"),
        ):
            works.at[index, field] = value
    with pytest.raises(PresentationDriftError) as error:
        preserve_reviewed_additive_presentation(
            works,
            prior,
            parent,
            candidate,
            prior_revision="a" * 40,
            source_snapshot_sha256="s" * 64,
            selected_ids_sha256="i" * 64,
            parent_map_sha256="p" * 64,
            reported_coauthor_source_sha256="a" * 64,
            review_path=None,
        )
    path = tmp_path / "placement-review.json"
    additive_review(path, error.value.receipt)
    frozen, _ = preserve_reviewed_additive_presentation(
        works,
        prior,
        parent,
        candidate,
        prior_revision="a" * 40,
        source_snapshot_sha256="s" * 64,
        selected_ids_sha256="i" * 64,
        parent_map_sha256="p" * 64,
        reported_coauthor_source_sha256="a" * 64,
        review_path=path,
    )
    prior_by_id = {row["work_id"]: row for row in prior}
    for row in frozen.to_dict("records"):
        if row["work_id"] not in prior_by_id:
            continue
        for field in RETAINED_WORK_PRESENTATION_FIELDS:
            assert row[field] == prior_by_id[row["work_id"]][field]


@pytest.mark.parametrize("field", RETAINED_WORK_PRESENTATION_FIELDS)
def test_additive_candidate_rejects_missing_protected_column(field: str) -> None:
    parent, candidate, works = additive_case()
    with pytest.raises(ValueError, match="candidate works lack"):
        freeze_additive(parent, candidate, works.drop(columns=field), None)


def test_additive_preserves_prior_off_map_work_presentation(
    tmp_path: Path,
) -> None:
    parent, candidate, works = additive_case()
    with pytest.raises(PresentationDriftError) as error:
        freeze_additive(parent, candidate, works, None)
    review_path = tmp_path / "placement-review.json"
    additive_review(review_path, error.value.receipt)

    off_map = {
        **parent_work_rows(parent)[0],
        "work_id": "work-prior-off-map",
        "map_eligible": False,
        "x": 0.12,
        "y": -0.23,
        "tsne_x": 0.34,
        "tsne_y": -0.45,
    }
    prior = [*parent_work_rows(parent), off_map]
    refitted = {
        **works.iloc[0].to_dict(),
        "work_id": off_map["work_id"],
        "publication_type": "conference_paper",
        "map_eligible": False,
        "x": 0.9,
        "y": 0.9,
        "tsne_x": 0.9,
        "tsne_y": 0.9,
        "keyword": "unreviewed refit",
    }
    works = pandas.concat([works, pandas.DataFrame([refitted])], ignore_index=True)
    output, _ = preserve_reviewed_additive_presentation(
        works,
        prior,
        parent,
        candidate,
        prior_revision="a" * 40,
        source_snapshot_sha256="s" * 64,
        selected_ids_sha256="i" * 64,
        parent_map_sha256="p" * 64,
        reported_coauthor_source_sha256="a" * 64,
        review_path=review_path,
    )
    restored = output.loc[output.work_id.eq(off_map["work_id"])].iloc[0]
    assert [restored[field] for field in ("x", "y", "tsne_x", "tsne_y")] == [
        0.12,
        -0.23,
        0.34,
        -0.45,
    ]
    assert restored.keyword == off_map["keyword"]
    assert restored.publication_type == "publication"
    assert works.iloc[-1]["x"] == 0.9

    changed_region = works.copy()
    changed_region.at[changed_region.index[0], "map_region_prior_eligible_override"] = (
        True
    )
    changed_region.at[changed_region.index[-1], "map_region_documented_override"] = True
    restored_region, _ = preserve_reviewed_additive_presentation(
        changed_region,
        prior,
        parent,
        candidate,
        prior_revision="a" * 40,
        source_snapshot_sha256="s" * 64,
        selected_ids_sha256="i" * 64,
        parent_map_sha256="p" * 64,
        reported_coauthor_source_sha256="a" * 64,
        review_path=review_path,
    )
    assert not restored_region["map_region_prior_eligible_override"].any()
    assert not restored_region["map_region_documented_override"].any()

    changed_eligibility = works.copy()
    changed_eligibility.at[changed_eligibility.index[-1], "map_eligible"] = True
    with pytest.raises(ValueError, match="off-map work changes map eligibility"):
        preserve_reviewed_additive_presentation(
            changed_eligibility,
            prior,
            parent,
            candidate,
            prior_revision="a" * 40,
            source_snapshot_sha256="s" * 64,
            selected_ids_sha256="i" * 64,
            parent_map_sha256="p" * 64,
            reported_coauthor_source_sha256="a" * 64,
            review_path=review_path,
        )

    without_prior_off_map = works.loc[works.work_id.ne(off_map["work_id"])]
    with pytest.raises(ValueError, match="drops prior work IDs"):
        preserve_reviewed_additive_presentation(
            without_prior_off_map,
            prior,
            parent,
            candidate,
            prior_revision="a" * 40,
            source_snapshot_sha256="s" * 64,
            selected_ids_sha256="i" * 64,
            parent_map_sha256="p" * 64,
            reported_coauthor_source_sha256="a" * 64,
            review_path=review_path,
        )

    prior_missing_field = deepcopy(prior)
    prior_missing_field[-1].pop("tsne_y")
    with pytest.raises(ValueError, match="parent work lacks presentation fields"):
        preserve_reviewed_additive_presentation(
            works,
            prior_missing_field,
            parent,
            candidate,
            prior_revision="a" * 40,
            source_snapshot_sha256="s" * 64,
            selected_ids_sha256="i" * 64,
            parent_map_sha256="p" * 64,
            reported_coauthor_source_sha256="a" * 64,
            review_path=review_path,
        )

    prior_missing_region = deepcopy(prior)
    prior_missing_region[-1].pop("map_region_prior_eligible_override")
    with pytest.raises(ValueError, match="lacks presentation fields"):
        preserve_reviewed_additive_presentation(
            works,
            prior_missing_region,
            parent,
            candidate,
            prior_revision="a" * 40,
            source_snapshot_sha256="s" * 64,
            selected_ids_sha256="i" * 64,
            parent_map_sha256="p" * 64,
            reported_coauthor_source_sha256="a" * 64,
            review_path=review_path,
        )


def test_additive_staged_works_preserve_prior_region_flags(tmp_path: Path) -> None:
    parent = [{**row, "map_eligible": True} for row in parent_work_rows(artifact(2))]
    staged = pandas.DataFrame(parent)
    path = tmp_path / "works.parquet"
    staged.to_parquet(path, index=False)
    require_preserved_additive_staged_works(path, parent)

    for field in (
        "map_region_outlier",
        "map_region_documented_override",
        "map_region_prior_eligible_override",
        "map_region_prior_excluded_override",
    ):
        broken = staged.copy()
        broken.at[0, field] = True
        broken.to_parquet(path, index=False)
        with pytest.raises(ValueError, match="change prior presentation"):
            require_preserved_additive_staged_works(path, parent)

    broken = staged.copy()
    broken.at[0, "map_region_audit_version"] = "unreviewed-refit"
    broken.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="change prior presentation"):
        require_preserved_additive_staged_works(path, parent)

    broken = staged.copy()
    broken.at[0, "publication_type"] = "conference_paper"
    broken.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="change prior presentation"):
        require_preserved_additive_staged_works(path, parent)

    staged.iloc[1:].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="lose or duplicate prior work IDs"):
        require_preserved_additive_staged_works(path, parent)

    with pytest.raises(ValueError, match="parent works lack presentation fields"):
        require_preserved_additive_staged_works(path, [{"work_id": "work-0000"}])
    require_preserved_additive_staged_works(path, [])


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("title", "Changed title"),
        ("authors", "B. Author"),
        ("year", 2026),
        ("venue", "Another journal"),
        ("citation", "Altered citation"),
        ("citation_count", 8),
        ("doi", "10.1000/changed"),
        ("publication_type", "conference_paper"),
        ("source_document_id", "new-source"),
        ("source_verified_at", "2026-10-31"),
        ("collaboration_eligible", False),
        ("display_contributors", "New contributor"),
        ("map_eligible", False),
        ("map_exclusion_reasons", ["unreviewed_exclusion"]),
        ("layout_version", "new-layout"),
        ("keyword_model_version", "new-keywords"),
        ("topic_label_review_version", "new-topic-review"),
    ],
)
@pytest.mark.parametrize("off_map", [False, True])
def test_additive_staged_works_rejects_bibliographic_drift_on_every_prior_work(
    tmp_path: Path, field: str, changed: object, off_map: bool
) -> None:
    parent = parent_work_rows(artifact(2))
    parent.append(
        {
            **parent[0],
            "work_id": "prior-off-map",
            "map_eligible": False,
            "map_exclusion_reasons": ["pending_placement_review"],
        }
    )
    staged = pandas.DataFrame(parent)
    index = 2 if off_map else 0
    if field == "map_eligible" and off_map:
        changed = True
    if field == "map_exclusion_reasons" and off_map:
        changed = ["different_exclusion"]
    staged.at[index, field] = changed
    path = tmp_path / "works.parquet"
    staged.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="change prior presentation"):
        require_preserved_additive_staged_works(path, parent)


@pytest.mark.parametrize("field", RETAINED_WORK_PRESENTATION_FIELDS)
def test_additive_staged_works_rejects_missing_protected_field(
    tmp_path: Path, field: str
) -> None:
    parent = parent_work_rows(artifact(1))
    path = tmp_path / "works.parquet"
    pandas.DataFrame(parent).drop(columns=field).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="staged works lack presentation fields"):
        require_preserved_additive_staged_works(path, parent)
    missing_parent = deepcopy(parent)
    del missing_parent[0][field]
    pandas.DataFrame(parent).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="parent works lack presentation fields"):
        require_preserved_additive_staged_works(path, missing_parent)


def test_additive_staged_works_rejects_scalar_exclusion_reasons(
    tmp_path: Path,
) -> None:
    parent = parent_work_rows(artifact(1))
    staged = pandas.DataFrame(parent)
    staged["map_exclusion_reasons"] = "not-a-list"
    path = tmp_path / "works.parquet"
    staged.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="lack exclusion reasons"):
        require_preserved_additive_staged_works(path, parent)


@pytest.mark.parametrize("field", RETAINED_WORK_GROWTH_FIELDS)
def test_additive_staged_works_rejects_lost_binding_or_provenance(
    tmp_path: Path, field: str
) -> None:
    parent = parent_work_rows(artifact(1))
    parent[0][field] = (
        [{"person_id": "person-one"}] if field == "memberships" else ["old"]
    )
    staged = pandas.DataFrame(parent)
    staged.at[0, field] = []
    path = tmp_path / "works.parquet"
    staged.to_parquet(path, index=False)
    with pytest.raises(ValueError, match=f"loses prior {field}"):
        require_preserved_additive_staged_works(path, parent)


def test_additive_staged_works_allows_binding_and_provenance_growth(
    tmp_path: Path,
) -> None:
    parent = parent_work_rows(artifact(1))
    staged = pandas.DataFrame(parent)
    for field in RETAINED_WORK_GROWTH_FIELDS:
        if field == "year_variants":
            new = 2026
        elif field in ("memberships", "external_coauthors"):
            new = {"person_id": "person-two"}
        else:
            new = "new"
        staged.at[0, field] = [*parent[0][field], new]
    staged.at[0, "observation_count"] = 2
    path = tmp_path / "works.parquet"
    staged.to_parquet(path, index=False)
    require_preserved_additive_staged_works(path, parent)


@pytest.mark.parametrize("side", ["parent", "staged"])
def test_additive_staged_works_rejects_scalar_binding_list(
    tmp_path: Path, side: str
) -> None:
    parent = parent_work_rows(artifact(1))
    staged = pandas.DataFrame(parent)
    if side == "parent":
        parent[0]["person_ids"] = "not-a-list"
    else:
        staged["person_ids"] = "not-a-list"
    path = tmp_path / "works.parquet"
    staged.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="lacks person_ids list"):
        require_preserved_additive_staged_works(path, parent)


@pytest.mark.parametrize("count", [0, "bad-count", True])
def test_additive_staged_works_rejects_lost_or_invalid_observation_count(
    tmp_path: Path, count: object
) -> None:
    parent = parent_work_rows(artifact(1))
    staged = pandas.DataFrame(parent)
    staged["observation_count"] = count
    path = tmp_path / "works.parquet"
    staged.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="observation count"):
        require_preserved_additive_staged_works(path, parent)


@pytest.mark.parametrize("field", (*RETAINED_WORK_GROWTH_FIELDS, "observation_count"))
def test_additive_staged_works_rejects_missing_growth_column(
    tmp_path: Path, field: str
) -> None:
    parent = parent_work_rows(artifact(1))
    path = tmp_path / "works.parquet"
    pandas.DataFrame(parent).drop(columns=field).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="staged works lack presentation fields"):
        require_preserved_additive_staged_works(path, parent)


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


def test_additive_new_excluded_work_has_no_public_placement(
    tmp_path: Path,
) -> None:
    parent, candidate, works = additive_case()
    excluded = {
        **works.iloc[0].to_dict(),
        "work_id": "work-region-excluded",
        "map_eligible": False,
        "map_exclusion_reasons": ["low_information_region"],
        "x": 0.93,
        "y": -0.28,
        "tsne_x": 0.83,
        "tsne_y": -0.19,
        "keyword": "Unreviewed region",
    }
    works = pandas.concat([works, pandas.DataFrame([excluded])], ignore_index=True)
    with pytest.raises(PresentationDriftError) as error:
        freeze_additive(parent, candidate, works, None)
    review_path = tmp_path / "placement-review.json"
    additive_review(review_path, error.value.receipt)
    output, receipt = freeze_additive(parent, candidate, works, review_path)
    assert receipt["excluded_new_work_reasons"] == {
        "work-region-excluded": ["low_information_region"]
    }
    row = output.loc[output.work_id.eq("work-region-excluded")].iloc[0]
    assert not bool(row.map_eligible)
    assert row.map_exclusion_reasons == ["low_information_region"]
    assert all(pandas.isna(row[field]) for field in ("x", "y", "tsne_x", "tsne_y"))
    assert all(row[field] == value for field, value in BLANK_TOPICS.items())

    staged_path = tmp_path / "works.parquet"
    output.to_parquet(staged_path, index=False)
    require_excluded_additive_staged_works(staged_path, receipt)
    require_excluded_additive_staged_works(
        staged_path, {**receipt, "excluded_new_work_reasons": {}}
    )
    with pytest.raises(ValueError, match="receipt is missing"):
        require_excluded_additive_staged_works(staged_path, {})
    for field, value, message in (
        ("x", 0.2, "leaks coordinates"),
        ("keyword", "Unreviewed topic", "leaks topic labels"),
        ("map_eligible", True, "changes exclusion"),
        ("map_exclusion_reasons", ["pending_placement_review"], "changes exclusion"),
    ):
        broken = output.copy(deep=True)
        excluded_index = broken.index[broken.work_id.eq("work-region-excluded")][0]
        broken.at[excluded_index, field] = value
        broken.to_parquet(staged_path, index=False)
        with pytest.raises(ValueError, match=message):
            require_excluded_additive_staged_works(staged_path, receipt)

    missing = output.loc[output.work_id.ne("work-region-excluded")].copy()
    missing.to_parquet(staged_path, index=False)
    with pytest.raises(ValueError, match="lack unique excluded new work IDs"):
        require_excluded_additive_staged_works(staged_path, receipt)


def test_additive_new_unmapped_eligible_work_fails_closed(tmp_path: Path) -> None:
    parent, candidate, works = additive_case()
    extra = {**works.iloc[0].to_dict(), "work_id": "work-unmapped"}
    works = pandas.concat([works, pandas.DataFrame([extra])], ignore_index=True)
    with pytest.raises(PresentationDriftError) as error:
        freeze_additive(parent, candidate, works, None)
    review_path = tmp_path / "placement-review.json"
    additive_review(review_path, error.value.receipt)
    with pytest.raises(ValueError, match="lacks a separate exclusion"):
        freeze_additive(parent, candidate, works, review_path)


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
            reported_coauthor_source_sha256="a" * 64,
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
            reported_coauthor_source_sha256="a" * 64,
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
            reported_coauthor_source_sha256="a" * 64,
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
    with pytest.raises(ValueError, match="parent work lacks presentation fields"):
        preserve_reviewed_additive_presentation(
            works,
            prior,
            parent,
            candidate,
            prior_revision="a" * 40,
            source_snapshot_sha256="s" * 64,
            selected_ids_sha256="i" * 64,
            parent_map_sha256="p" * 64,
            reported_coauthor_source_sha256="a" * 64,
            review_path=path,
        )


def test_additive_review_binds_new_nonspatial_metadata_not_refit_coordinates(
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
    next(point for point in candidate["points"] if point["work_id"] == "work-new")[
        "keyword_ids"
    ] = ["different", "refit"]
    freeze_additive(parent, candidate, works, path)
    next(point for point in candidate["points"] if point["work_id"] == "work-new")[
        "title"
    ] = "Changed source title"
    with pytest.raises(PresentationDriftError, match="stale or mismatched"):
        freeze_additive(parent, candidate, works, path)


def test_additive_rejects_changed_reports_or_unexpected_new_metadata() -> None:
    parent, candidate, works = additive_case()
    candidate["points"][0]["reported_faculty_coauthors"] = [
        {"person_id": "person-other", "source_person_id": "person-author"}
    ]
    with pytest.raises(ValueError, match="changes retained reported coauthors"):
        freeze_additive(parent, candidate, works, None)
    candidate["points"][0].pop("reported_faculty_coauthors")
    candidate["points"][-1]["unexpected_public_field"] = "new"
    with pytest.raises(ValueError, match="unexpected point metadata"):
        freeze_additive(parent, candidate, works, None)


def test_additive_requires_exact_report_source_sha() -> None:
    parent, candidate, works = additive_case()
    with pytest.raises(ValueError, match="lacks exact reported coauthor source SHA"):
        preserve_reviewed_additive_presentation(
            works,
            parent_work_rows(parent),
            parent,
            candidate,
            prior_revision="a" * 40,
            source_snapshot_sha256="s" * 64,
            selected_ids_sha256="i" * 64,
            parent_map_sha256="p" * 64,
            reported_coauthor_source_sha256="invalid",
            review_path=None,
        )


def test_additive_staged_map_preserves_retained_reported_coauthors() -> None:
    parent = artifact(1)
    parent["points"][0]["publication_type"] = "publication"
    parent["points"][0]["reported_faculty_coauthors"] = [
        {"person_id": "person-other", "source_person_id": "person-author"}
    ]
    staged = deepcopy(parent)
    receipt = {
        "placed_new_work_ids": [],
        "held_new_work_ids": [],
        "placed_new_coordinates_sha256": hashlib.sha256(b"[]").hexdigest(),
    }
    require_preserved_additive_artifact(parent, staged, receipt)
    staged["points"][0]["publication_type"] = "conference_paper"
    with pytest.raises(ValueError, match="changes retained publication type"):
        require_preserved_additive_artifact(parent, staged, receipt)
    staged["points"][0]["publication_type"] = "publication"
    missing_parent_type = deepcopy(parent)
    del missing_parent_type["points"][0]["publication_type"]
    with pytest.raises(ValueError, match="lacks retained publication type"):
        require_preserved_additive_artifact(missing_parent_type, staged, receipt)
    missing_staged_type = deepcopy(staged)
    del missing_staged_type["points"][0]["publication_type"]
    with pytest.raises(ValueError, match="lacks retained publication type"):
        require_preserved_additive_artifact(parent, missing_staged_type, receipt)
    staged["points"][0]["reported_faculty_coauthors"] = []
    with pytest.raises(ValueError, match="changes retained reported coauthors"):
        require_preserved_additive_artifact(parent, staged, receipt)


def _retained_only_receipt() -> dict:
    return {
        "placed_new_work_ids": [],
        "held_new_work_ids": [],
        "placed_new_coordinates_sha256": hashlib.sha256(b"[]").hexdigest(),
    }


@pytest.mark.parametrize("field", RETAINED_POINT_PRESENTATION_FIELDS)
def test_additive_staged_map_rejects_each_changed_bibliographic_field(
    field: str,
) -> None:
    parent = artifact(1)
    staged = deepcopy(parent)
    original = staged["points"][0][field]
    staged["points"][0][field] = (
        not original if isinstance(original, bool) else f"changed-{original}"
    )
    with pytest.raises(ValueError, match="changes retained"):
        require_preserved_additive_artifact(parent, staged, _retained_only_receipt())


@pytest.mark.parametrize("field", RETAINED_POINT_PRESENTATION_FIELDS)
@pytest.mark.parametrize("missing_side", ["parent", "staged"])
def test_additive_staged_map_rejects_each_missing_bibliographic_field(
    field: str, missing_side: str
) -> None:
    parent = artifact(1)
    staged = deepcopy(parent)
    target = parent if missing_side == "parent" else staged
    del target["points"][0][field]
    with pytest.raises(ValueError, match="lacks retained"):
        require_preserved_additive_artifact(parent, staged, _retained_only_receipt())


def test_additive_staged_map_accepts_reviewed_binding_and_provenance_growth() -> None:
    parent = artifact(1)
    staged = deepcopy(parent)
    staged_point = staged["points"][0]
    for field in RETAINED_POINT_GROWTH_FIELDS:
        staged_point[field].append("new-reviewed-entry")
    staged_point["observation_count"] = 2
    require_preserved_additive_artifact(parent, staged, _retained_only_receipt())


@pytest.mark.parametrize("field", RETAINED_POINT_GROWTH_FIELDS)
def test_additive_staged_map_rejects_lost_binding_or_provenance(field: str) -> None:
    parent = artifact(1)
    parent["points"][0][field] = ["prior-entry"]
    staged = deepcopy(parent)
    staged["points"][0][field] = []
    with pytest.raises(ValueError, match=f"loses prior {field}"):
        require_preserved_additive_artifact(parent, staged, _retained_only_receipt())


@pytest.mark.parametrize("field", RETAINED_POINT_GROWTH_FIELDS)
@pytest.mark.parametrize("missing_side", ["parent", "staged"])
def test_additive_staged_map_rejects_missing_growth_field(
    field: str, missing_side: str
) -> None:
    parent = artifact(1)
    staged = deepcopy(parent)
    target = parent if missing_side == "parent" else staged
    del target["points"][0][field]
    with pytest.raises(ValueError, match=f"lacks retained {field}"):
        require_preserved_additive_artifact(parent, staged, _retained_only_receipt())


@pytest.mark.parametrize("count", [0, "bad-count", True])
def test_additive_staged_map_rejects_lost_or_invalid_observation_count(
    count: object,
) -> None:
    parent = artifact(1)
    staged = deepcopy(parent)
    staged["points"][0]["observation_count"] = count
    with pytest.raises(ValueError, match="observation count"):
        require_preserved_additive_artifact(parent, staged, _retained_only_receipt())


def test_additive_staged_map_rejects_missing_observation_count() -> None:
    parent = artifact(1)
    staged = deepcopy(parent)
    del staged["points"][0]["observation_count"]
    with pytest.raises(ValueError, match="lacks retained observation count"):
        require_preserved_additive_artifact(parent, staged, _retained_only_receipt())


def test_additive_staged_map_rejects_keyword_coordinate_drift_and_missing() -> None:
    parent = artifact(1)
    staged = deepcopy(parent)
    staged["keywords"][0]["coordinates"]["pca"]["x"] = 0.4
    with pytest.raises(ValueError, match="changes parent topic metadata"):
        require_preserved_additive_artifact(parent, staged, _retained_only_receipt())
    staged = deepcopy(parent)
    del staged["keywords"][0]["coordinates"]
    with pytest.raises(ValueError, match="lacks parent topic metadata"):
        require_preserved_additive_artifact(parent, staged, _retained_only_receipt())


def test_additive_staged_map_rejects_missing_keyword_ids() -> None:
    parent = artifact(1)
    staged = deepcopy(parent)
    del staged["points"][0]["keyword_ids"]
    with pytest.raises(ValueError, match="lacks retained topic IDs"):
        require_preserved_additive_artifact(parent, staged, _retained_only_receipt())
