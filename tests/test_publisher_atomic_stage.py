"""The publisher stages complete release files before its only Hub call."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas
import pytest

import map_of_research.publisher as publisher
from map_of_research.atomic_release import RELEASE_PATHS


def _bio_registry():
    return SimpleNamespace(
        memberships=[
            SimpleNamespace(person_id="bio-person", included=True, department_id="bio")
        ]
    )


def _prior_bio_people():
    return [{"person_id": "bio-person", "included_department_ids": ["bio"]}]


def test_manual_bio_dispatch_passes_release_intent() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/upload.yml"
    ).read_text(encoding="utf-8")
    assert (
        "REQUIRE_BIO_RELEASE: ${{ github.event_name == 'workflow_dispatch' "
        "&& inputs.require_bio_release || false }}"
    ) in workflow
    assert "BIO_RELEASE_ARGS+=(--require-bio-release)" in workflow
    assert '"${BIO_RELEASE_ARGS[@]}"' in workflow
    assert "MAP_DURABLE_INTENT_REPO: snapshot" in workflow
    assert "MAP_DURABLE_INTENT_BRANCH: automation/map-snapshot" in workflow


def test_atomic_intent_is_pushed_before_hub_write_and_keeps_success_baseline(
    monkeypatch, tmp_path
) -> None:
    remote = tmp_path / "remote.git"
    checkout = tmp_path / "snapshot"

    def git(*args):
        return subprocess.run(
            ["git", *map(str, args)], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--bare", remote)
    git("clone", remote, checkout)
    git("-C", checkout, "checkout", "-b", "automation/map-snapshot")
    status_path = checkout / "status" / "last-upload.json"
    status_path.parent.mkdir()
    status_path.write_text('{"status":"success"}')
    successful_path = checkout / "status" / "last-successful-upload.json"
    successful_path.write_text('{"artifact_commit":"old"}')
    git("-C", checkout, "add", "status")
    git(
        "-C",
        checkout,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.test",
        "commit",
        "-m",
        "Initial receipt",
    )
    git("-C", checkout, "push", "origin", "HEAD:automation/map-snapshot")
    monkeypatch.setenv("MAP_DURABLE_INTENT_REPO", str(checkout))
    monkeypatch.setenv("MAP_DURABLE_INTENT_BRANCH", "automation/map-snapshot")
    status = {"workflow_url": "https://github.test/actions/runs/1"}
    intent = {
        "intent_status": "before_write",
        "commit_nonce": "unique-nonce",
        "parent_revision": "a" * 40,
        "release_context": {
            "producer_commit": "c" * 40,
            "snapshot_commit": "d" * 40,
        },
        "size_by_path": {"README.md": 10},
        "sha256_by_path": {"README.md": "b" * 64},
    }
    publisher._persist_atomic_intent(intent, status=status, status_path=status_path)
    saved = json.loads(
        git(
            "--git-dir",
            remote,
            "show",
            "automation/map-snapshot:status/last-upload.json",
        )
    )
    assert saved["status"] == "failure"
    assert saved["atomic_release_intent"] == intent
    assert json.loads(
        git(
            "--git-dir",
            remote,
            "show",
            "automation/map-snapshot:status/last-successful-upload.json",
        )
    ) == {"artifact_commit": "old"}

    returned = {**intent, "intent_status": "commit_returned_unverified"}
    returned["artifact_revision"] = "c" * 40
    publisher._persist_atomic_intent(returned, status=status, status_path=status_path)
    saved = json.loads(
        git(
            "--git-dir",
            remote,
            "show",
            "automation/map-snapshot:status/last-upload.json",
        )
    )
    assert saved["atomic_release_intent"]["artifact_revision"] == "c" * 40

    # An intervening source push must stop the next checkpoint before any Hub call.
    other = tmp_path / "other"
    git("clone", "--branch", "automation/map-snapshot", remote, other)
    (other / "status" / "unrelated.txt").write_text("advanced")
    git("-C", other, "add", "status/unrelated.txt")
    git(
        "-C",
        other,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.test",
        "commit",
        "-m",
        "Advance snapshot branch",
    )
    git("-C", other, "push", "origin", "HEAD:automation/map-snapshot")
    with pytest.raises(RuntimeError, match="Snapshot branch moved"):
        publisher._persist_atomic_intent(
            {**intent, "intent_status": "before_write", "commit_nonce": "new"},
            status=status,
            status_path=status_path,
        )


@pytest.mark.parametrize(
    ("branch", "status_location", "message"),
    [
        ("wrong-branch", "correct", "requires the snapshot branch"),
        ("automation/map-snapshot", "outside", "outside snapshot checkout"),
        ("automation/map-snapshot", "wrong-path", "must checkpoint"),
    ],
)
def test_atomic_intent_rejects_unreviewed_checkpoint_location(
    monkeypatch, tmp_path, branch, status_location, message
) -> None:
    repo = tmp_path / "snapshot"
    repo.mkdir()
    status_path = {
        "correct": repo / "status" / "last-upload.json",
        "outside": tmp_path / "last-upload.json",
        "wrong-path": repo / "status" / "other.json",
    }[status_location]
    monkeypatch.setenv("MAP_DURABLE_INTENT_REPO", str(repo))
    monkeypatch.setenv("MAP_DURABLE_INTENT_BRANCH", branch)
    with pytest.raises(ValueError, match=message):
        publisher._persist_atomic_intent(
            {
                "parent_revision": "a" * 40,
                "release_context": {
                    "producer_commit": "c" * 40,
                    "snapshot_commit": "d" * 40,
                },
            },
            status={},
            status_path=status_path,
        )


@pytest.mark.parametrize("context", ["invalid", {"producer_commit": "c" * 40}])
def test_atomic_intent_requires_producer_and_source_ids(
    monkeypatch, tmp_path, context
) -> None:
    repo = tmp_path / "snapshot"
    repo.mkdir()
    monkeypatch.setenv("MAP_DURABLE_INTENT_REPO", str(repo))
    monkeypatch.setenv("MAP_DURABLE_INTENT_BRANCH", "automation/map-snapshot")
    with pytest.raises(ValueError):
        publisher._persist_atomic_intent(
            {"release_context": context},
            status={},
            status_path=repo / "status" / "last-upload.json",
        )


@pytest.mark.parametrize("prior_revision", [None, "main", "a" * 39])
def test_unpinned_publish_cannot_reach_any_hub_write(
    monkeypatch, tmp_path, prior_revision
) -> None:
    monkeypatch.setattr(
        publisher,
        "_to_hub_dataset",
        lambda *args: pytest.fail("unversioned dataset write"),
    )
    monkeypatch.setattr(
        publisher,
        "_publish_atomic_release",
        lambda *args, **kwargs: pytest.fail("unversioned atomic write"),
    )
    with pytest.raises(ValueError, match="revision"):
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
            prior_revision=prior_revision,
        )


def test_pinned_publish_without_intent_sink_cannot_reach_hub(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        publisher,
        "_require_current_dataset_revision",
        lambda **kwargs: pytest.fail("Hub revision probe reached"),
    )
    monkeypatch.setattr(
        publisher,
        "_publish_atomic_release",
        lambda *args, **kwargs: pytest.fail("Hub write reached"),
    )
    with pytest.raises(ValueError, match="durable prewrite intent sink"):
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
            prior_revision="a" * 40,
        )


def _stage(
    monkeypatch,
    tmp_path,
    *,
    bio=False,
    gate_status="pass",
    pending=False,
    point_count=2,
    staged_map_drift=False,
    staged_works_drift=False,
    staged_works_schema_drift=False,
    invalid_staged_map=False,
):
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
        artifact = {"point_count": point_count, "points": []}
        artifact_path = maps_dir / "publications.json"
        manifest_path = maps_dir / "manifest.json"
        artifact_path.write_text(
            "null"
            if invalid_staged_map
            else json.dumps(
                {**artifact, "point_count": point_count + 1}
                if staged_map_drift
                else artifact
            )
        )
        manifest_path.write_text(json.dumps(kwargs))
        calls.append(("map", kwargs))
        return artifact_path, manifest_path, artifact

    class Dataset:
        def __init__(self, frame):
            self.frame = frame

        def to_parquet(self, path):
            frame = self.frame
            if staged_works_drift and "work_id" in frame.columns:
                frame = frame.assign(title="Changed after parity")
            if staged_works_schema_drift and "work_id" in frame.columns:
                frame = frame.assign(unreviewed_field="new")
            frame.to_parquet(path, index=False)

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


def test_direct_atomic_stage_without_intent_sink_stops_before_hub(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="durable prewrite intent sink"):
        publisher._publish_atomic_release(
            tables,
            _bio_registry(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={"snapshot_sha256": "other"},
            hf_token="token",
            prior_revision="a" * 40,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={"prior_revision": "a" * 40},
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
        )
    assert calls == []


@pytest.mark.parametrize("bio", [False, True])
def test_atomic_publisher_stages_all_files_and_runs_gate_before_commit(
    monkeypatch, tmp_path, bio
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path, bio=bio)
    revision, receipt = publisher._publish_atomic_release(
        tables,
        _bio_registry(),
        source_frame=pandas.DataFrame({"source": [1]}),
        source_manifest={
            "snapshot_sha256": publisher.BIO_SOURCE_SHA256 if bio else "other"
        },
        hf_token="token",
        prior_revision="a" * 40,
        atomic_intent_sink=lambda intent: None,
        generated_at_utc="2026-10-25T00:00:00+00:00",
        continuity_receipt={"prior_revision": "a" * 40},
        repair_manifest_path=manifest,
        provisional_holds={},
        prior_profiles=[],
        prior_people=_prior_bio_people(),
    )
    assert revision == "b" * 40
    assert receipt["mode"] == "single_parent_commit"
    assert calls[-1][0] == "commit"
    assert calls[-1][1]["prewrite_gate"]["status"] == "pass"
    if bio:
        assert [call[0] for call in calls] == ["preserve", "map", "gate", "commit"]
    else:
        assert [call[0] for call in calls] == ["map", "commit"]


def test_atomic_math_presentation_receipt_is_in_prewrite_intent(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(publisher, "MATH_BASELINE_MAP_POINTS", 3)
    calls, tables, manifest = _stage(monkeypatch, tmp_path, point_count=5)
    tables["works"] = pandas.DataFrame(
        {"work_id": ["retained"], "title": ["Original"], "embedding": [[0.1, 0.2]]}
    )
    monkeypatch.setattr(
        publisher,
        "MATH_BASELINE_WORK_SCHEMA_SHA256",
        hashlib.sha256(
            json.dumps(sorted(tables["works"].columns), separators=(",", ":")).encode()
        ).hexdigest(),
    )
    parity = {
        "unchanged_shared_points": 1,
        "restored_points": 4,
        "removed_points": 2,
        "frozen_coordinate_rows": 5,
        "shared_work_rows": 1,
        "map_artifact_sha256": publisher._math_artifact_digest(
            {"point_count": 5, "points": []}
        ),
    }
    revision, _ = publisher._publish_atomic_release(
        tables,
        _bio_registry(),
        source_frame=pandas.DataFrame({"source": [1]}),
        source_manifest={"snapshot_sha256": publisher.MATH_REGION_SNAPSHOT_SHA256},
        hf_token="token",
        prior_revision=publisher.MATH_REGION_BASELINE,
        atomic_intent_sink=lambda intent: None,
        generated_at_utc="2026-10-25T00:00:00+00:00",
        continuity_receipt={
            "prior_revision": publisher.MATH_REGION_BASELINE,
            "math_presentation_parity": parity,
        },
        repair_manifest_path=manifest,
        provisional_holds={},
        prior_profiles=[],
        prior_people=_prior_bio_people(),
        math_region_review=True,
        math_common_work_ids={"retained"},
    )
    assert revision == "b" * 40
    assert calls[-1][1]["prewrite_gate"]["math_presentation_parity"] == parity
    assert calls[-1][1]["prewrite_gate"]["gate"] == (
        "frozen-math-presentation-parity-v1"
    )


@pytest.mark.parametrize(
    "case",
    ("bio_source", "wrong_prior", "wrong_source", "wrong_points", "wrong_parity"),
)
def test_atomic_math_presentation_gate_fails_before_commit(
    monkeypatch, tmp_path, case
) -> None:
    monkeypatch.setattr(publisher, "MATH_BASELINE_MAP_POINTS", 3)
    calls, tables, manifest = _stage(
        monkeypatch,
        tmp_path,
        bio=case == "bio_source",
        point_count=2 if case == "wrong_points" else 5,
    )
    tables["works"] = pandas.DataFrame(
        {"work_id": ["retained"], "title": ["Original"], "embedding": [[0.1, 0.2]]}
    )
    monkeypatch.setattr(
        publisher,
        "MATH_BASELINE_WORK_SCHEMA_SHA256",
        hashlib.sha256(
            json.dumps(sorted(tables["works"].columns), separators=(",", ":")).encode()
        ).hexdigest(),
    )
    prior_revision = (
        "a" * 40 if case == "wrong_prior" else publisher.MATH_REGION_BASELINE
    )
    snapshot_sha = (
        publisher.BIO_SOURCE_SHA256
        if case == "bio_source"
        else "other"
        if case == "wrong_source"
        else publisher.MATH_REGION_SNAPSHOT_SHA256
    )
    parity = {
        "unchanged_shared_points": 1,
        "restored_points": 4,
        "removed_points": 2,
        "frozen_coordinate_rows": 5,
        "shared_work_rows": 1,
        "map_artifact_sha256": publisher._math_artifact_digest(
            {"point_count": 2 if case == "wrong_points" else 5, "points": []}
        ),
    }
    if case == "wrong_parity":
        parity["shared_work_rows"] -= 1
    with pytest.raises(ValueError, match="exact presentation parity"):
        publisher._publish_atomic_release(
            tables,
            _bio_registry(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={"snapshot_sha256": snapshot_sha},
            hf_token="token",
            prior_revision=prior_revision,
            atomic_intent_sink=lambda intent: None,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={
                "prior_revision": prior_revision,
                "math_presentation_parity": parity,
            },
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
            math_region_review=True,
            math_common_work_ids={"retained"},
        )
    assert all(call[0] != "commit" for call in calls)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("map_bytes", "exact presentation parity"),
        ("work_values", "shared work-table field changed"),
        ("work_schema", "staged works Parquet schema or row count changed"),
        ("missing_ids", "lacks retained work identities"),
        ("invalid_map", "staged map artifact is invalid"),
    ),
)
def test_atomic_math_rejects_staged_payload_drift_before_commit(
    monkeypatch, tmp_path, case, message
) -> None:
    monkeypatch.setattr(publisher, "MATH_BASELINE_MAP_POINTS", 3)
    calls, tables, manifest = _stage(
        monkeypatch,
        tmp_path,
        point_count=5,
        staged_map_drift=case == "map_bytes",
        staged_works_drift=case == "work_values",
        staged_works_schema_drift=case == "work_schema",
        invalid_staged_map=case == "invalid_map",
    )
    tables["works"] = pandas.DataFrame(
        {"work_id": ["retained"], "title": ["Original"], "embedding": [[0.1, 0.2]]}
    )
    monkeypatch.setattr(
        publisher,
        "MATH_BASELINE_WORK_SCHEMA_SHA256",
        hashlib.sha256(
            json.dumps(sorted(tables["works"].columns), separators=(",", ":")).encode()
        ).hexdigest(),
    )
    parity = {
        "unchanged_shared_points": 1,
        "restored_points": 4,
        "removed_points": 2,
        "frozen_coordinate_rows": 5,
        "shared_work_rows": 1,
        "map_artifact_sha256": publisher._math_artifact_digest(
            {"point_count": 5, "points": []}
        ),
    }
    with pytest.raises(ValueError, match=message):
        publisher._publish_atomic_release(
            tables,
            _bio_registry(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={
                "snapshot_sha256": publisher.MATH_REGION_SNAPSHOT_SHA256
            },
            hf_token="token",
            prior_revision=publisher.MATH_REGION_BASELINE,
            atomic_intent_sink=lambda intent: None,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={
                "prior_revision": publisher.MATH_REGION_BASELINE,
                "math_presentation_parity": parity,
            },
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
            math_region_review=True,
            math_common_work_ids=None if case == "missing_ids" else {"retained"},
        )
    assert all(call[0] != "commit" for call in calls)


def test_atomic_bio_staging_requires_manifest_only_for_pending_parent(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path, bio=True, pending=True)
    manifest.unlink()
    with pytest.raises(FileNotFoundError, match="reviewed repair manifest"):
        publisher._publish_atomic_release(
            tables,
            _bio_registry(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={"snapshot_sha256": publisher.BIO_SOURCE_SHA256},
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={"prior_revision": "a" * 40},
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
        )
    assert calls == []


def test_atomic_bio_staging_reads_pending_manifest_before_gate(
    monkeypatch, tmp_path
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path, bio=True, pending=True)
    revision, _ = publisher._publish_atomic_release(
        tables,
        _bio_registry(),
        source_frame=pandas.DataFrame({"source": [1]}),
        source_manifest={"snapshot_sha256": publisher.BIO_SOURCE_SHA256},
        hf_token="token",
        prior_revision="a" * 40,
        atomic_intent_sink=lambda intent: None,
        generated_at_utc="2026-10-25T00:00:00+00:00",
        continuity_receipt={"prior_revision": "a" * 40},
        repair_manifest_path=manifest,
        provisional_holds={},
        prior_profiles=[],
        prior_people=_prior_bio_people(),
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
            _bio_registry(),
            source_frame=pandas.DataFrame({"source": [1]}),
            source_manifest={"snapshot_sha256": "other"},
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={
                "prior_revision": "a" * 40,
                "unexplained_gains": ["bad"],
            },
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
        )
    assert all(call[0] != "commit" for call in calls)


@pytest.mark.parametrize(
    ("require_bio_release", "bio_source_changed"),
    [(True, False), (False, True)],
)
def test_bio_intent_or_changed_cohort_rejects_drift_before_hub_stage(
    monkeypatch, tmp_path, require_bio_release, bio_source_changed
) -> None:
    calls, tables, manifest = _stage(monkeypatch, tmp_path, bio=False)
    memberships = (
        [SimpleNamespace(person_id="bio-person", included=True, department_id="bio")]
        if bio_source_changed
        else []
    )
    source = pandas.DataFrame(
        [
            {
                "person_id": "bio-person",
                "scholar_id": "bio-scholar",
                "author_pub_id": "bio-scholar:one",
                "source_type": "scholar",
                "source_record_json": '{"title":"one"}',
            }
        ]
    )
    with pytest.raises(ValueError, match="Bio release source differs"):
        publisher._publish_atomic_release(
            tables,
            SimpleNamespace(memberships=memberships),
            source_frame=source,
            source_manifest={"snapshot_sha256": "drifted-source"},
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
            generated_at_utc="2026-10-25T00:00:00+00:00",
            continuity_receipt={"prior_revision": "a" * 40},
            repair_manifest_path=manifest,
            provisional_holds={},
            prior_profiles=[],
            prior_people=_prior_bio_people(),
            require_bio_release=require_bio_release,
        )
    assert calls == []


@pytest.mark.parametrize(
    (
        "source_hash",
        "prior_revision",
        "atomic_release",
        "require_bio_release",
        "message",
    ),
    [
        ("changed", "a" * 40, None, True, "Bio release source differs"),
        (
            publisher.BIO_SOURCE_SHA256,
            None,
            None,
            True,
            "immutable prior Hub revision",
        ),
        (publisher.BIO_SOURCE_SHA256, "a" * 40, False, True, "one atomic Hub commit"),
        (publisher.BIO_SOURCE_SHA256, "a" * 40, False, False, "one atomic Hub commit"),
    ],
)
def test_publish_bio_intent_fails_before_enrichment_or_hub_write(
    monkeypatch,
    tmp_path,
    source_hash,
    prior_revision,
    atomic_release,
    require_bio_release,
    message,
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
            atomic_intent_sink=lambda intent: None,
            require_bio_release=require_bio_release,
        )
    assert calls == []
