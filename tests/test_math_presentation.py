"""Frozen Math releases preserve every existing map point payload."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy
import pandas
import pytest

from map_of_research import publisher
from map_of_research.region_review import ReviewedRegionDecision
from tests.test_publisher import registry, work_frame

GENERATED_AT = "2026-10-25T10:00:00+00:00"


def _decision(work_id: str, *, restored: bool) -> ReviewedRegionDecision:
    return ReviewedRegionDecision(
        work_id=work_id,
        person_id="person-one",
        scholar_id="ScholarOne",
        author_pub_id=f"ScholarOne:{work_id}",
        capture_row_index=1,
        source_record_sha256="a" * 64,
        work_fingerprint_sha256="b" * 64,
        captured_title=work_id,
        transition="restore" if restored else "reexclude",
        decision="include" if restored else "exclude",
        evidence_url="https://example.test/work",
        evidence_scope="exact_work_primary_evidence",
        saved_scholar_url="https://scholar.google.com/citations",
        reason="Reviewed exact source",
        reviewed_at="2026-10-25",
    )


def _fixture(monkeypatch):
    monkeypatch.setattr(publisher, "MATH_BASELINE_MAP_POINTS", 7)
    original = work_frame().iloc[0].to_dict()
    rows = []
    for work_id in [
        *(f"common-{index}" for index in range(5)),
        *(f"gain-{index}" for index in range(4)),
        *(f"loss-{index}" for index in range(2)),
    ]:
        row = copy.deepcopy(original)
        row["work_id"] = work_id
        row["title"] = f"Publication {work_id}"
        row["title_variants"] = [row["title"]]
        row["venue_variants"] = [row["venue"]]
        row["citation_variants"] = ["A Journal 2025"]
        row["author_variants"] = [row["authors"]]
        row["x"] += len(rows) / 1000
        row["tsne_x"] += len(rows) / 1000
        row["map_region_outlier"] = work_id.startswith("gain-")
        row["map_eligible"] = not work_id.startswith("gain-")
        if work_id.startswith("gain-"):
            for field in publisher.MATH_TOPIC_FIELDS:
                row[field] = False if field.endswith("_label_reviewed") else ""
        rows.append(row)
    old_frame = pandas.DataFrame(rows)
    artifact = publisher.build_map_artifact(
        old_frame, registry(), generated_at_utc=GENERATED_AT
    )
    current = old_frame.copy(deep=True)
    for field in publisher.MATH_TRANSIENT_REGION_FIELDS:
        current[field] = False
    for index, row in current.iterrows():
        work_id = row.work_id
        current.at[index, "map_eligible"] = not work_id.startswith("loss-")
        current.at[index, "map_region_outlier"] = work_id.startswith("loss-")
        if not work_id.startswith("loss-"):
            current.at[index, "x"] = 0.8 + index / 1000
            current.at[index, "tsne_x"] = 0.9 + index / 1000
            for field in publisher.MATH_TOPIC_FIELDS:
                current.at[index, field] = (
                    False if field.endswith("_label_reviewed") else "new cluster"
                )
    decisions = tuple(
        [*(_decision(f"gain-{index}", restored=True) for index in range(4))]
        + [*(_decision(f"loss-{index}", restored=False) for index in range(2))]
    )
    return current, old_frame.to_dict("records"), artifact, decisions


def _preserve(inputs):
    current, previous, artifact, decisions = inputs
    return publisher.preserve_reviewed_math_presentation(
        current,
        previous,
        artifact,
        decisions,
        registry(),
        generated_at_utc=GENERATED_AT,
    )


def test_math_presentation_reuses_frozen_points_and_coordinates(monkeypatch):
    current, previous, artifact, decisions = _fixture(monkeypatch)
    output, receipt = _preserve((current, previous, artifact, decisions))
    assert not set(publisher.MATH_TRANSIENT_REGION_FIELDS).intersection(output.columns)
    rebuilt = publisher.build_map_artifact(
        output, registry(), generated_at_utc=GENERATED_AT
    )
    before = {row["work_id"]: row for row in artifact["points"]}
    after = {row["work_id"]: row for row in rebuilt["points"]}
    assert set(after) - set(before) == {f"gain-{index}" for index in range(4)}
    assert set(before) - set(after) == {f"loss-{index}" for index in range(2)}
    assert all(after[work_id] == before[work_id] for work_id in before.keys() & after)
    for row in previous:
        if row["work_id"].startswith("gain-"):
            point = after[row["work_id"]]
            assert point["keyword_ids"] == [
                "keyword-unclassified",
                "keyword-unclassified-pending-review",
            ]
            assert all(
                point[field] == round(float(row[field]), 7)
                for field in publisher.MATH_COORDINATE_FIELDS
            )
    assert receipt == {
        "unchanged_shared_points": 5,
        "restored_points": 4,
        "removed_points": 2,
        "frozen_coordinate_rows": 9,
        "map_artifact_sha256": publisher._math_artifact_digest(rebuilt),
    }


def test_math_presentation_restores_only_source_variant_order(monkeypatch):
    current, previous, artifact, decisions = _fixture(monkeypatch)
    previous[0]["title_variants"] = ["Design", "design"]
    current.at[0, "title_variants"] = ["design", "Design"]
    output, _ = _preserve((current, previous, artifact, decisions))
    assert output.at[0, "title_variants"] == ["Design", "design"]
    current.at[0, "title_variants"] = ["design", "Unreviewed title"]
    with pytest.raises(ValueError, match="changed source variants"):
        _preserve((current, previous, artifact, decisions))
    previous[0]["title_variants"] = ["Design", "Design", "design"]
    current.at[0, "title_variants"] = ["Design", "design", "design"]
    with pytest.raises(ValueError, match="changed source variants"):
        _preserve((current, previous, artifact, decisions))


def test_math_presentation_keeps_unaffected_topic_exact(monkeypatch):
    current, previous, _, decisions = _fixture(monkeypatch)
    previous[0].update(
        keyword_id="keyword-other",
        keyword="Other topic",
        keyword_extracted="Other topic",
        detail_keyword_id="keyword-other-detail",
        detail_keyword="Other detail",
        detail_keyword_extracted="Other detail",
    )
    artifact = publisher.build_map_artifact(
        pandas.DataFrame(previous), registry(), generated_at_utc=GENERATED_AT
    )
    output, _ = _preserve((current, previous, artifact, decisions))
    assert output.at[0, "keyword_id"] == "keyword-other"

    original = publisher.build_map_artifact

    def drifted(*args, **kwargs):
        result = original(*args, **kwargs)
        next(
            item
            for item in result["keywords"]
            if item["keyword_id"] == "keyword-other"
        )["publication_count"] += 1
        return result

    monkeypatch.setattr(publisher, "build_map_artifact", drifted)
    with pytest.raises(ValueError, match="shared topic metadata changed"):
        _preserve((current, previous, artifact, decisions))


def test_math_presentation_keeps_external_coauthors_of_common_points(monkeypatch):
    current, previous, artifact, decisions = _fixture(monkeypatch)
    person_id = "external-shared"
    for work_id in ("common-0", "loss-0"):
        point = next(
            item for item in artifact["points"] if item["work_id"] == work_id
        )
        point["external_coauthor_ids"] = [*point["external_coauthor_ids"], person_id]
    artifact["catalogs"]["external_coauthors"].append(
        {"person_id": person_id, "display_name": "Shared coauthor"}
    )
    current["external_coauthor_ids"] = [[] for _ in range(len(current))]
    current.at[0, "external_coauthor_ids"] = [person_id]
    with pytest.raises(ValueError, match="changed external coauthor catalog"):
        _preserve((current, previous, artifact, decisions))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("wrong_decisions", "four gains and two losses"),
        ("wrong_schema", "baseline artifact differs"),
        ("wrong_quality_version", "baseline artifact differs"),
        ("wrong_point_count", "baseline artifact differs"),
        ("duplicate_point", "baseline identities are invalid"),
        ("duplicate_work", "baseline identities are invalid"),
        ("topic_id_conflict", "baseline identities are invalid"),
        ("missing_gain", "map IDs exceed exact"),
        ("missing_region_diagnostics", "lacks reconciled region diagnostics"),
        ("missing_prior_work", "baseline identities are invalid"),
        ("missing_prior_coordinate", "no frozen coordinates"),
        ("nonfinite_prior_coordinate", "no frozen coordinates"),
        ("wrong_prior_gain_state", "not a prior region outlier"),
        ("prior_gain_topic", "already has a frozen topic"),
        ("prior_shared_coordinate", "differs from frozen works"),
        ("shared_payload_drift", "shared point payload changed"),
        ("shared_topic_drift", "shared topic metadata changed"),
        ("baseline_topic_count_drift", "shared topic membership changed"),
        ("shared_catalog_drift", "changed faculty catalog"),
        ("shared_level_drift", "changed keyword-level metadata"),
        ("candidate_metadata_drift", "changed artifact metadata"),
        ("candidate_point_ids_drift", "artifact point IDs differ"),
        ("candidate_header_count_drift", "changed artifact count"),
        ("candidate_gain_topics_drift", "restored point placement or topics"),
        ("candidate_topic_ids_drift", "topic IDs beyond exact review"),
        ("candidate_new_topic_drift", "new topic metadata changed"),
        ("candidate_topic_count_drift", "shared topic metadata changed"),
        ("candidate_topic_centroid_drift", "shared topic metadata changed"),
        ("candidate_catalog_count_drift", "changed faculty catalog"),
        ("candidate_level_count_drift", "changed keyword-level metadata"),
        ("candidate_external_catalog_drift", "changed external coauthor catalog"),
        ("candidate_external_catalog_loss", "changed external coauthor catalog"),
    ],
)
def test_math_presentation_fails_closed(monkeypatch, case, message):
    current, previous, artifact, decisions = _fixture(monkeypatch)
    if case == "wrong_decisions":
        decisions = decisions[:-1]
    elif case == "wrong_schema":
        artifact["layout_version"] = "other"
    elif case == "wrong_quality_version":
        artifact["quality_assessment_version"] = "other"
    elif case == "wrong_point_count":
        artifact["point_count"] -= 1
    elif case == "duplicate_point":
        artifact["points"][-1]["work_id"] = artifact["points"][0]["work_id"]
    elif case == "duplicate_work":
        previous[-1]["work_id"] = previous[0]["work_id"]
    elif case == "topic_id_conflict":
        artifact["keywords"][0]["keyword_id"] = "keyword-unclassified"
    elif case == "missing_gain":
        current.loc[current.work_id.eq("gain-0"), "map_eligible"] = False
    elif case == "missing_region_diagnostics":
        current = current.drop(columns=[publisher.MATH_TRANSIENT_REGION_FIELDS[0]])
    elif case == "missing_prior_work":
        previous.pop(0)
    elif case == "missing_prior_coordinate":
        next(row for row in previous if row["work_id"] == "gain-0")["x"] = None
    elif case == "nonfinite_prior_coordinate":
        next(row for row in previous if row["work_id"] == "gain-0")["x"] = float("nan")
    elif case == "wrong_prior_gain_state":
        next(row for row in previous if row["work_id"] == "gain-0")[
            "map_region_outlier"
        ] = False
    elif case == "prior_gain_topic":
        next(row for row in previous if row["work_id"] == "gain-0")["keyword_id"] = (
            "keyword-01"
        )
    elif case == "prior_shared_coordinate":
        next(row for row in previous if row["work_id"] == "common-0")["x"] = 1.0
    elif case == "shared_payload_drift":
        current.loc[current.work_id.eq("common-0"), "title"] = "Wrong title"
    elif case == "shared_topic_drift":
        artifact["keywords"][0]["label"] = "Wrong topic label"
    elif case == "baseline_topic_count_drift":
        artifact["keywords"][0]["publication_count"] += 1
    elif case == "shared_catalog_drift":
        artifact["catalogs"]["faculty"][0]["display_name"] = "Wrong person"
    elif case == "shared_level_drift":
        artifact["keyword_levels"][0]["label"] = "Wrong level"
    elif case.startswith("candidate_"):
        if case == "candidate_external_catalog_loss":
            artifact["catalogs"]["external_coauthors"].append(
                {"person_id": "prior-external", "display_name": "Prior coauthor"}
            )
        original = publisher.build_map_artifact

        def drifted(*args, **kwargs):
            result = original(*args, **kwargs)
            if case == "candidate_metadata_drift":
                result["title"] = "Wrong title"
            elif case == "candidate_point_ids_drift":
                result["points"].pop()
            elif case == "candidate_header_count_drift":
                result["keyword_count"] += 1
            elif case == "candidate_gain_topics_drift":
                next(item for item in result["points"] if item["work_id"] == "gain-0")[
                    "keyword_ids"
                ] = ["wrong", "topics"]
            elif case == "candidate_topic_ids_drift":
                result["keywords"].append(
                    {**result["keywords"][0], "keyword_id": "new"}
                )
            elif case == "candidate_new_topic_drift":
                next(
                    item
                    for item in result["keywords"]
                    if item["keyword_id"] == "keyword-unclassified"
                )["label"] = "Wrong new topic"
            elif case == "candidate_topic_count_drift":
                result["keywords"][0]["publication_count"] += 1
            elif case == "candidate_topic_centroid_drift":
                result["keywords"][0]["coordinates"]["pca"]["x"] += 0.1
            elif case == "candidate_catalog_count_drift":
                result["catalogs"]["faculty"][0]["publication_count"] += 1
            elif case == "candidate_level_count_drift":
                result["keyword_levels"][0]["keyword_count"] += 1
            elif case == "candidate_external_catalog_drift":
                result["catalogs"]["external_coauthors"].append(
                    {"person_id": "unreviewed", "display_name": "Unknown"}
                )
            return result

        monkeypatch.setattr(publisher, "build_map_artifact", drifted)
    with pytest.raises((ValueError, TypeError), match=message):
        _preserve((current, previous, artifact, decisions))


def test_published_math_artifact_reads_exact_immutable_revision(monkeypatch, tmp_path):
    artifact = {"point_count": 7}
    path = tmp_path / "publications.json"
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(
        publisher,
        "MATH_BASELINE_MAP_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(path)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    assert (
        publisher._published_map_artifact(
            hf_token="", revision=publisher.MATH_REGION_BASELINE
        )
        == artifact
    )
    assert calls == [
        (
            (publisher.REPO_ID,),
            {
                "filename": "maps/publications.json",
                "repo_type": "dataset",
                "revision": publisher.MATH_REGION_BASELINE,
                "token": None,
            },
        )
    ]
    with pytest.raises(ValueError, match="frozen bb88 artifact"):
        publisher._published_map_artifact(hf_token="", revision="a" * 40)
    path.write_text("{}")
    with pytest.raises(ValueError, match="artifact bytes changed"):
        publisher._published_map_artifact(
            hf_token="", revision=publisher.MATH_REGION_BASELINE
        )


def _work_table_fixture(monkeypatch):
    works = pandas.DataFrame(
        {
            "work_id": ["one", "two"],
            "title": ["First", "Second"],
            "variants": [["Alpha", "alpha"], ["Beta"]],
            "embedding": [[0.25, 0.5], [0.75, 1.0]],
        }
    )
    fields = sorted(str(field) for field in works.columns)
    monkeypatch.setattr(
        publisher,
        "MATH_BASELINE_WORK_SCHEMA_SHA256",
        hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest(),
    )
    monkeypatch.setattr(publisher, "MATH_BASELINE_MAP_POINTS", 4)
    return works, works.to_dict("records")


def test_math_work_table_checks_every_shared_field(monkeypatch):
    works, prior = _work_table_fixture(monkeypatch)
    assert (
        publisher.require_reviewed_math_work_table_parity(
            works, iter(prior), {"one", "two"}
        )
        == 2
    )
    changed = works.copy(deep=True)
    changed.at[0, "title"] = "Changed"
    with pytest.raises(ValueError, match="field changed: one: title"):
        publisher.require_reviewed_math_work_table_parity(
            changed, iter(prior), {"one", "two"}
        )
    changed = works.copy(deep=True)
    changed.at[0, "embedding"] = [0.25, 0.6]
    with pytest.raises(ValueError, match="field changed: one: embedding"):
        publisher.require_reviewed_math_work_table_parity(
            changed, iter(prior), {"one", "two"}
        )
    hidden = works.iloc[:1].copy(deep=True)
    hidden.at[0, "work_id"] = "hidden"
    with_hidden = pandas.concat([works, hidden], ignore_index=True)
    assert (
        publisher.require_reviewed_math_work_table_parity(
            with_hidden, iter([*prior, hidden.iloc[0].to_dict()]), {"one", "two"}
        )
        == 2
    )


def test_math_work_table_normalizes_nested_arrow_values():
    assert publisher._published_field_value(
        {"values": (numpy.float32(1.25), numpy.array([2.5, float("nan")]))}
    ) == {"values": [1.25, [2.5, None]]}


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("schema", "schema or reviewed point count changed"),
        ("common_count", "schema or reviewed point count changed"),
        ("candidate_missing", "common works are missing or duplicated"),
        ("candidate_duplicate", "common works are missing or duplicated"),
        ("prior_missing", "baseline is missing common work rows"),
        ("prior_duplicate", "baseline has duplicate common work IDs"),
    ],
)
def test_math_work_table_fails_closed(monkeypatch, case, message):
    works, prior = _work_table_fixture(monkeypatch)
    ids = {"one", "two"}
    if case == "schema":
        works["unexpected"] = 1
    elif case == "common_count":
        ids.add("three")
    elif case == "candidate_missing":
        works = works.iloc[:1]
    elif case == "candidate_duplicate":
        works = pandas.concat([works, works.iloc[:1]], ignore_index=True)
    elif case == "prior_missing":
        prior.pop()
    elif case == "prior_duplicate":
        prior.append(prior[0])
    with pytest.raises(ValueError, match=message):
        publisher.require_reviewed_math_work_table_parity(works, iter(prior), ids)
