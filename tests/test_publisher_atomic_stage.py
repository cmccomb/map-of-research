"""The publisher stages complete release files before its only Hub call."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas
import pytest

import map_of_research.publisher as publisher
from map_of_research.atomic_release import RELEASE_PATHS


def _stage(monkeypatch, tmp_path, *, bio=False, gate_status="pass", pending=False):
    readme = tmp_path / "parent-README.md"
    readme.write_text("---\nlicense: mit\n---\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda *args, **kwargs: str(readme)),
    )
    calls = []

    def map_files(works, registry, *, maps_dir, **kwargs):
        maps_dir.mkdir(parents=True)
        artifact = {"point_count": 2, "points": []}
        artifact_path = maps_dir / "publications.json"
        manifest_path = maps_dir / "manifest.json"
        artifact_path.write_text(json.dumps(artifact))
        manifest_path.write_text(json.dumps(kwargs))
        calls.append(("map", kwargs))
        return artifact_path, manifest_path, artifact

    class Dataset:
        def __init__(self, frame):
            self.frame = frame

        def to_parquet(self, path):
            self.frame.to_parquet(path, index=False)

    def commit(files, **kwargs):
        assert set(files) == RELEASE_PATHS
        assert all(path.is_file() and path.stat().st_size for path in files.values())
        assert (
            json.loads(files["maps/manifest.json"].read_text())["dataset_commits"] == {}
        )
        calls.append(("commit", kwargs))
        return "b" * 40, {"mode": "single_parent_commit"}

    monkeypatch.setattr(publisher, "_write_map_artifacts", map_files)
    monkeypatch.setattr(publisher, "_to_hub_dataset", Dataset)
    monkeypatch.setattr(publisher, "build_updated_dataset_card", lambda *args: "card")
    monkeypatch.setattr(publisher, "commit_atomic_release", commit)
    manifest = tmp_path / "repair.json"
    if pending:
        manifest.write_text(json.dumps({"restore_eligible_work_ids": ["work"]}))
    if bio:
        monkeypatch.setattr(
            publisher,
            "load_bio_baselines",
            lambda **kwargs: {
                "repair_works": [
                    {
                        "work_id": "work",
                        "keyword_id": "keyword-unclassified" if pending else "other",
                        "detail_keyword_id": (
                            "keyword-unclassified-pending-review"
                            if pending
                            else "other"
                        ),
                        "map_eligible": True,
                    }
                ]
            },
        )
        monkeypatch.setattr(
            publisher,
            "preserve_pending_review_topics",
            lambda works, baseline, review: calls.append(("preserve", review)) or works,
        )
        monkeypatch.setattr(
            publisher,
            "validate_bio_release_prewrite",
            lambda **kwargs: calls.append(("gate", kwargs)) or {"status": gate_status},
        )
    tables = {
        name: pandas.DataFrame({"value": [name]}) for name in publisher.DATASET_CONFIGS
    }
    return calls, tables, manifest


@pytest.mark.parametrize("bio", [False, True])
def test_atomic_publisher_stages_all_files_and_runs_gate_before_commit(
    monkeypatch, tmp_path, bio
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path, bio=bio)
    revision, receipt = publisher._publish_atomic_release(
        tables,
        SimpleNamespace(),
        source_frame=pandas.DataFrame({"source": [1]}),
        source_manifest={
            "snapshot_sha256": publisher.BIO_SOURCE_SHA256 if bio else "other"
        },
        hf_token="token",
        prior_revision="a" * 40,
        generated_at_utc="2026-10-25T00:00:00+00:00",
        continuity_receipt={"prior_revision": "a" * 40},
        repair_manifest_path=manifest,
        provisional_holds={},
    )
    assert revision == "b" * 40
    assert receipt["mode"] == "single_parent_commit"
    assert calls[-1][0] == "commit"
    assert calls[-1][1]["prewrite_gate"]["status"] == "pass"
    if bio:
        assert [call[0] for call in calls] == ["preserve", "map", "gate", "commit"]
    else:
        assert [call[0] for call in calls] == ["map", "commit"]


def test_atomic_bio_staging_requires_manifest_only_for_pending_parent(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path, bio=True, pending=True)
    manifest.unlink()
    with pytest.raises(FileNotFoundError, match="reviewed repair manifest"):
        publisher._publish_atomic_release(
            tables,
            SimpleNamespace(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={"snapshot_sha256": publisher.BIO_SOURCE_SHA256},
            hf_token="token",
            prior_revision="a" * 40,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={"prior_revision": "a" * 40},
            repair_manifest_path=manifest,
            provisional_holds={},
        )
    assert calls == []


def test_atomic_bio_staging_reads_pending_manifest_before_gate(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path, bio=True, pending=True)
    revision, _ = publisher._publish_atomic_release(
        tables,
        SimpleNamespace(),
        source_frame=pandas.DataFrame({"source": [1]}),
        source_manifest={"snapshot_sha256": publisher.BIO_SOURCE_SHA256},
        hf_token="token",
        prior_revision="a" * 40,
        generated_at_utc="2026-10-25T00:00:00+00:00",
        continuity_receipt={"prior_revision": "a" * 40},
        repair_manifest_path=manifest,
        provisional_holds={},
    )
    assert revision == "b" * 40
    assert calls[0] == ("preserve", {"restore_eligible_work_ids": ["work"]})


def test_atomic_generic_staging_rejects_unclean_region_gate_before_commit(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="clean region-continuity"):
        publisher._publish_atomic_release(
            tables,
            SimpleNamespace(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={"snapshot_sha256": "other"},
            hf_token="token",
            prior_revision="a" * 40,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={
                "prior_revision": "a" * 40,
                "unexplained_gains": ["bad"],
            },
            repair_manifest_path=manifest,
            provisional_holds={},
        )
    assert all(call[0] != "commit" for call in calls)


def test_explicit_bio_intent_rejects_drift_before_hub_stage(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path, bio=False)
    with pytest.raises(ValueError, match="Bio release source differs"):
        publisher._publish_atomic_release(
            tables,
            SimpleNamespace(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={"snapshot_sha256": "drifted-source"},
            hf_token="token",
            prior_revision="a" * 40,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={"prior_revision": "a" * 40},
            repair_manifest_path=manifest,
            provisional_holds={},
            require_bio_release=True,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("source_hash", "prior_revision", "atomic_release", "message"),
    [
        ("changed", "a" * 40, None, "Bio release source differs"),
        (publisher.BIO_SOURCE_SHA256, None, None, "atomic parent-pinned"),
        (publisher.BIO_SOURCE_SHA256, "a" * 40, False, "atomic parent-pinned"),
    ],
)
def test_publish_bio_intent_fails_before_enrichment_or_hub_write(
    monkeypatch, tmp_path, source_hash, prior_revision, atomic_release, message
) -> None:
    calls = []
    monkeypatch.setattr(
        publisher, "_require_current_dataset_revision", lambda **kwargs: None
    )
    monkeypatch.setattr(publisher, "load_registry", lambda *args: SimpleNamespace())
    monkeypatch.setattr(
        publisher, "implicit_primary_baseline_for_people", lambda *args: {}
    )
    monkeypatch.setattr(
        publisher,
        "validate_included_scholar_primary_homes",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (
            pandas.DataFrame({"source": [1]}),
            {"snapshot_sha256": source_hash},
        ),
    )
    monkeypatch.setattr(
        publisher, "add_embeddings", lambda *args, **kwargs: calls.append("enrich")
    )
    monkeypatch.setattr(
        publisher, "commit_atomic_release", lambda *args, **kwargs: calls.append("hub")
    )
    with pytest.raises(ValueError, match=message):
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
            prior_revision=prior_revision,
            atomic_release=atomic_release,
            require_bio_release=True,
        )
    assert calls == []
