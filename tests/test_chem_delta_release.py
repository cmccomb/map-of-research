"""Fail-closed checks for the separate post-#337 Chemistry delta path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from map_of_research import chem_delta_release as delta
from map_of_research.atomic_release import RELEASE_PATHS
from map_of_research.region_continuity import (
    WORK_FINGERPRINT_FIELDS,
    _fingerprint_fields,
)
from map_of_research.region_review import ReviewedRegionDecision


def _decisions() -> tuple[ReviewedRegionDecision, ...]:
    source = json.loads(
        Path("data/curated/scholar-chem-region-review.json").read_text()
    )
    return tuple(
        ReviewedRegionDecision(
            **entry,
            evidence_scope="exact_work_primary_evidence"
            if entry["decision"] == "include"
            else "reviewed_unmapped_work",
        )
        for entry in source["decisions"]
    )


def _parquet(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_live_parent_rejects_wrong_revision_before_download() -> None:
    calls = []

    class Api:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="2" * 40)

    with pytest.raises(ValueError, match="Live Hub head"):
        delta.require_live_337_parent(
            "1" * 40,
            hf_token=None,
            api=Api(),
            download=lambda *_args, **_kwargs: calls.append(True),
        )
    assert calls == []
    with pytest.raises(ValueError, match="not yet"):
        delta.require_live_337_parent(delta.OLD_PARENT, hf_token=None, api=Api())


def test_chemistry_release_requires_new_readback_intent_path(tmp_path: Path) -> None:
    path = tmp_path / "intent.json"
    sink = delta.durable_intent_sink(path)
    sink({"intent_status": "before_write", "parent_revision": "1" * 40})
    assert json.loads(path.read_text())["intent_status"] == "before_write"
    sink({"intent_status": "commit_returned_unverified", "artifact_revision": "2" * 40})
    assert json.loads(path.read_text())["artifact_revision"] == "2" * 40
    with pytest.raises(ValueError, match="must be new"):
        delta.durable_intent_sink(path)


def test_live_parent_rejects_wrong_seven_file_hash(tmp_path: Path) -> None:
    class Api:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="1" * 40)

    paths = {name: tmp_path / name for name in sorted(RELEASE_PATHS)}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wrong parent bytes")
    assert set(delta.PARENT_SHA256_BY_PATH) == RELEASE_PATHS
    with pytest.raises(ValueError, match="differs from #337 receipt"):
        delta.require_live_337_parent(
            "1" * 40,
            hf_token=None,
            api=Api(),
            download=lambda _repo, name, **_kwargs: str(paths[name]),
        )


@pytest.mark.parametrize("tampered_path", sorted(RELEASE_PATHS))
def test_each_parent_file_hash_is_required(tmp_path: Path, tampered_path: str) -> None:
    paths = {name: tmp_path / name for name in sorted(RELEASE_PATHS)}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"receipt-matched")
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    paths[tampered_path].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs from #337 receipt"):
        delta.require_exact_parent_files(paths, expected_hashes=hashes)


def test_missing_or_extra_chemistry_observation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decisions = _decisions()
    works = [{"work_id": item.work_id} for item in decisions]
    profiles = [
        {
            "work_id": item.work_id,
            "author_pub_id": item.author_pub_id,
            "observation_id": f"observation-{index}",
        }
        for index, item in enumerate(decisions)
    ]
    authored = [
        {"work_id": row["work_id"], "observation_id": row["observation_id"]}
        for row in profiles
    ]
    people = [
        {"person_id": person_id, "unique_work_count": 20, "mapped_work_count": 20}
        for person_id in {
            item.person_id
            for item in decisions
            if not item.visible or item.visible and item.transition == "restore"
        }
    ]
    monkeypatch.setattr(delta, "_target_work", lambda row, _decision: row)
    monkeypatch.setattr(delta, "_target_profile", lambda row, _decision: row)

    def parent(
        profile_rows: list[dict[str, object]],
        author_rows=authored,
        people_rows=None,
    ):
        files = {
            f"{name}/train-00000-of-00001.parquet": _parquet(
                tmp_path / name / "train-00000-of-00001.parquet", rows
            )
            for name, rows in (
                ("works", works),
                ("profile_publications", profile_rows),
                ("authorships", author_rows),
                ("people", people if people_rows is None else people_rows),
            )
        }
        return delta.ExactParent("1" * 40, files)

    with pytest.raises(ValueError, match="missing or repeats"):
        delta._parent_targets(parent(profiles[:-1]), decisions)
    with pytest.raises(ValueError, match="missing or repeats"):
        delta._parent_targets(
            parent(
                [
                    *profiles,
                    {
                        "work_id": profiles[0]["work_id"],
                        "author_pub_id": "extra:row",
                        "observation_id": "observation-extra",
                    },
                ]
            ),
            decisions,
        )
    with pytest.raises(ValueError, match="19 authored"):
        delta._parent_targets(
            parent(
                profiles,
                [
                    *authored,
                    {
                        "work_id": profiles[0]["work_id"],
                        "observation_id": "observation-extra",
                    },
                ],
            ),
            decisions,
        )
    targets, _ = delta._parent_targets(parent(profiles), decisions)
    by_person = targets["people"]
    assert len(by_person) == 6
    assert {
        person_id: (
            row["unique_work_count"] - 20,
            row["mapped_work_count"] - 20,
        )
        for person_id, row in by_person.items()
    } == {
        "person-donahue-f4c5ce86": (0, 2),
        "person-krzysztofmatyjaszewski-29a2013e": (-3, -2),
        "person-mariakurnikova-215a60c6": (-1, 0),
        "person-robertorgil-90b0cbe9": (-1, -1),
        "person-terrencejcollins-61cf6ed9": (-3, -1),
        "person-washburn-3a49d96a": (-1, 0),
    }
    with pytest.raises(ValueError, match="lacks six affected people"):
        delta._parent_targets(parent(profiles, people_rows=people[:-1]), decisions)


@pytest.mark.parametrize("changed_table", ["works", "authorships"])
def test_unreviewed_table_payload_drift_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_table: str
) -> None:
    sample = {
        "people": [{"person_id": "person-other", "payload": "original"}],
        "works": [{"work_id": "work-other", "payload": "original"}],
        "authorships": [{"observation_id": "observation-other", "payload": "original"}],
        "profile_publications": [
            {"observation_id": "observation-other", "payload": "original"}
        ],
    }
    parent_files = {}
    stage_files = {}
    for name, rows in sample.items():
        relpath = f"{name}/train-00000-of-00001.parquet"
        parent_files[relpath] = _parquet(tmp_path / "parent" / relpath, rows)
        staged_rows = [dict(row) for row in rows]
        if name == changed_table:
            staged_rows[0]["payload"] = "unreviewed change"
        stage_files[relpath] = _parquet(tmp_path / "staged" / relpath, staged_rows)
    monkeypatch.setattr(delta, "PARENT_TABLE_ROWS", dict.fromkeys(sample, 1))
    monkeypatch.setattr(delta, "TARGET_TABLE_ROWS", dict.fromkeys(sample, 1))
    with pytest.raises(ValueError, match="payload drift"):
        delta.validate_exact_tables(
            delta.ExactParent("1" * 40, parent_files),
            stage_files,
            {name: {} for name in sample},
            set(),
        )


def test_non19_map_point_or_shared_payload_drift_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = _decisions()
    gain = {
        item.work_id
        for item in decisions
        if item.visible and item.transition == "restore"
    }
    loss = {
        item.work_id
        for item in decisions
        if not item.visible and item.transition == "reexclude"
    }
    before = [
        {"work_id": key, "x": 1}
        for key in sorted(loss | {"work-other-1", "work-other-2"})
    ]
    after = [
        {"work_id": key, "x": 1}
        for key in sorted(gain | {"work-other-1", "work-other-2"})
    ]
    presentation_people = {
        item.work_id: item.person_id for item in decisions if item.output_type
    }
    for point in after:
        if person_id := presentation_people.get(point["work_id"]):
            point.update(
                faculty_ids=[person_id],
                external_coauthor_ids=[],
                collaboration_eligible=True,
                publication_type="conference_presentation",
            )
    monkeypatch.setattr(delta, "PARENT_POINT_COUNT", len(before))
    monkeypatch.setattr(delta, "TARGET_POINT_COUNT", len(after))
    prior = {
        "point_count": len(before),
        "excluded_work_count": 0,
        "keyword_count": 0,
        "keywords": [],
        "points": before,
        "catalogs": {},
    }
    candidate = {
        "point_count": len(after),
        "excluded_work_count": len(before) - len(after),
        "keyword_count": 0,
        "keywords": [],
        "points": after,
        "catalogs": {},
    }
    assert delta.validate_exact_map(prior, candidate, decisions)["status"] == "pass"
    changed = json.loads(json.dumps(candidate))
    next(row for row in changed["points"] if row["work_id"] == "work-other-1")["x"] = 2
    with pytest.raises(ValueError, match="existing point payload"):
        delta.validate_exact_map(prior, changed, decisions)
    extra = json.loads(json.dumps(candidate))
    next(row for row in extra["points"] if row["work_id"] == "work-other-1")[
        "work_id"
    ] = "work-unreviewed-extra"
    with pytest.raises(ValueError, match="more than exact"):
        delta.validate_exact_map(prior, extra, decisions)
    added_null_metadata = {**candidate, "unexpected_metadata": None}
    with pytest.raises(ValueError, match="metadata keys"):
        delta.validate_exact_map(prior, added_null_metadata, decisions)
    wrong_excluded_count = {**candidate, "excluded_work_count": 999}
    with pytest.raises(ValueError, match="excluded-work count"):
        delta.validate_exact_map(prior, wrong_excluded_count, decisions)

    prior_with_keyword = json.loads(json.dumps(prior))
    candidate_with_keyword = json.loads(json.dumps(candidate))
    for artifact in (prior_with_keyword, candidate_with_keyword):
        for point in artifact["points"]:
            point.update(y=1, tsne_x=1, tsne_y=1, keyword_ids=["keyword-test"])
        artifact["keyword_count"] = 1
        artifact["keywords"] = [
            {
                "keyword_id": "keyword-test",
                "label": "Original topic",
                "extracted_label": "Original topic",
                "label_reviewed": False,
                "level": 0,
                "parent_keyword_id": None,
                "publication_count": len(artifact["points"]),
                "coordinates": {
                    "pca": {"x": 1.0, "y": 1.0},
                    "tsne": {"x": 1.0, "y": 1.0},
                },
            }
        ]
    assert (
        delta.validate_exact_map(prior_with_keyword, candidate_with_keyword, decisions)[
            "status"
        ]
        == "pass"
    )
    candidate_with_keyword["keywords"][0]["label"] = "UNRELATED TAMPER"
    with pytest.raises(ValueError, match="keyword metadata"):
        delta.validate_exact_map(prior_with_keyword, candidate_with_keyword, decisions)


def test_map_catalogs_allow_only_point_derived_count_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = _decisions()
    gains = {
        item.work_id
        for item in decisions
        if item.visible and item.transition == "restore"
    }
    losses = {
        item.work_id
        for item in decisions
        if not item.visible and item.transition == "reexclude"
    }
    presentation_people = {
        item.work_id: item.person_id for item in decisions if item.output_type
    }

    def point(work_id: str) -> dict[str, object]:
        person_id = presentation_people.get(work_id, "person-chem")
        return {
            "work_id": work_id,
            "school_ids": ["mcs"],
            "department_ids": ["chem"],
            "faculty_ids": [person_id],
            "external_coauthor_ids": [],
            "collaboration_eligible": True,
            "publication_type": "conference_presentation"
            if work_id in presentation_people
            else "publication",
        }

    before = [point(key) for key in sorted(losses | {"shared-a", "shared-b"})]
    after = [point(key) for key in sorted(gains | {"shared-a", "shared-b"})]
    monkeypatch.setattr(delta, "PARENT_POINT_COUNT", len(before))
    monkeypatch.setattr(delta, "TARGET_POINT_COUNT", len(after))

    def catalogs(
        count: int, presentation_count: int
    ) -> dict[str, list[dict[str, object]]]:
        return {
            "schools": [
                {"school_id": "mcs", "title": "MCS", "publication_count": count}
            ],
            "departments": [
                {
                    "department_id": "chem",
                    "title": "Chemistry",
                    "publication_count": count,
                }
            ],
            "faculty": [
                {
                    "person_id": "person-chem",
                    "memberships": ["chem"],
                    "publication_count": count - presentation_count,
                },
                {
                    "person_id": "person-donahue-f4c5ce86",
                    "memberships": ["chem"],
                    "publication_count": presentation_count,
                },
            ],
            "external_coauthors": [],
        }

    prior = {
        "point_count": len(before),
        "excluded_work_count": 0,
        "keyword_count": 0,
        "keywords": [],
        "points": before,
        "catalogs": catalogs(len(before), 0),
        "source_data_newest_at_utc": "source-date",
    }
    candidate = {
        "point_count": len(after),
        "excluded_work_count": len(before) - len(after),
        "keyword_count": 0,
        "keywords": [],
        "points": after,
        "catalogs": catalogs(len(after), 2),
        "source_data_newest_at_utc": "source-date",
    }
    assert delta.validate_exact_map(prior, candidate, decisions)["status"] == "pass"

    wrong_count = json.loads(json.dumps(candidate))
    wrong_count["catalogs"]["faculty"][0]["publication_count"] += 1
    with pytest.raises(ValueError, match="metadata or count"):
        delta.validate_exact_map(prior, wrong_count, decisions)

    wrong_home = json.loads(json.dumps(candidate))
    wrong_home["catalogs"]["faculty"][0]["memberships"] = ["other"]
    with pytest.raises(ValueError, match="metadata or count"):
        delta.validate_exact_map(prior, wrong_home, decisions)

    wrong_source = json.loads(json.dumps(candidate))
    wrong_source["source_data_newest_at_utc"] = "changed-date"
    with pytest.raises(ValueError, match="non-derived metadata"):
        delta.validate_exact_map(prior, wrong_source, decisions)


def test_reviewed_context_and_hidden_rows_keep_raw_scholar_fields() -> None:
    decisions = _decisions()
    raw = "saved Scholar observation"
    for decision in decisions:
        profile = {
            "person_id": decision.person_id,
            "scholar_id": decision.scholar_id,
            "author_pub_id": decision.author_pub_id,
            "source_type": "scholar",
            "work_id": decision.work_id,
            "source_record_json": raw,
            "map_eligible": True,
            "collaboration_eligible": True,
            "map_exclusion_reasons": [],
            "publication_type": "publication",
            "review_exclusion_code": "",
            "review_attribution_status": "",
            "title": "untouched raw title",
        }
        profile_decision = replace(
            decision,
            source_record_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        )
        profile_target = delta._target_profile(profile, profile_decision)
        assert profile_target["source_record_json"] == raw
        assert profile_target["title"] == "untouched raw title"
        if decision.output_type:
            assert profile_target["publication_type"] == "conference_presentation"
            assert profile_target["context_evidence_url"] == decision.evidence_url
            assert profile_target["collaboration_eligible"] is True
        if not decision.visible:
            assert profile_target["map_eligible"] is False
            assert profile_target["collaboration_eligible"] is False
            assert profile_target["map_exclusion_reasons"] == [
                "attribution_provisional_hold"
                if decision.decision == "hold"
                else f"reviewed_{decision.reason_code}"
            ]

        visible = decision.transition == "reexclude"
        work = {
            "work_id": decision.work_id,
            "map_eligible": visible,
            "map_region_outlier": not visible,
            "map_exclusion_reasons": [] if visible else ["low_information_region"],
            "publication_type": "publication",
            "collaboration_eligible": True,
            "context_evidence_urls": [],
            "quality_assessment_version": "test",
            "faculty": [{"person_id": decision.person_id}],
            "person_ids": [decision.person_id],
            "memberships": [{"person_id": decision.person_id}],
            "department_ids": ["chem"],
            "department_titles": ["Chemistry"],
            "school_ids": ["mcs"],
            "school_titles": ["Mellon College of Science"],
        }
        work_decision = replace(
            decision,
            work_fingerprint_sha256=_fingerprint_fields(
                work, (*WORK_FINGERPRINT_FIELDS, "quality_assessment_version")
            ),
        )
        work_target = delta._target_work(work, work_decision)
        if decision.output_type:
            assert work_target["publication_type"] == "conference_presentation"
            assert work_target["context_evidence_urls"] == [decision.evidence_url]
            assert work_target["collaboration_eligible"] is True
        if not decision.visible:
            assert work_target["map_eligible"] is False
            assert work_target["collaboration_eligible"] is False
            assert work_target["faculty"] == []
            assert work_target["person_ids"] == []
            assert work_target["memberships"] == []
