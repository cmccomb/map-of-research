"""The hosted rehearsal must stop at the real seven-file write boundary."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import map_of_research
from map_of_research.region_continuity import RegionReleaseDiffError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preview_atomic_release.py"
PARENT = "b" * 40
PRODUCER = "c" * 40
SNAPSHOT = "d" * 40
STATUS = "e" * 40
RELEASE_PATHS = frozenset(
    {
        "README.md",
        "maps/publications.json",
        "maps/manifest.json",
        *(
            f"{name}/train-00000-of-00001.parquet"
            for name in (
                "people",
                "works",
                "authorships",
                "profile_publications",
            )
        ),
    }
)


def _load_script():
    spec = importlib.util.spec_from_file_location("preview_atomic_release", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_preview(
    monkeypatch,
    tmp_path,
    *,
    fail_before_write: bool,
    approval: bool = False,
    binding_approval: bool = False,
    region_diff: bool = False,
):
    preview = _load_script()
    producer = tmp_path / "producer"
    snapshot = tmp_path / "snapshot"
    receipt_checkout = tmp_path / "receipt"
    output = tmp_path / "output"
    producer.mkdir()
    if approval:
        curated = producer / "data/curated"
        curated.mkdir(parents=True)
        (curated / "bio-mcmanus-poster-loss-approval.json").write_text("{}")
        if binding_approval:
            (curated / "bio-exact-binding-approval-38f.json").write_text("{}")
    (snapshot / "snapshots").mkdir(parents=True)
    (receipt_checkout / "status").mkdir(parents=True)
    raw = snapshot / "snapshots/cmu-engineering-publications.parquet"
    raw.write_bytes(b"raw-snapshot")
    raw_sha = preview._sha256(raw)
    (
        snapshot / "snapshots/cmu-engineering-publications.parquet.manifest.json"
    ).write_text(json.dumps({"snapshot_sha256": raw_sha}), encoding="utf-8")
    (receipt_checkout / "status/last-upload.json").write_text(
        json.dumps({"status": "success", "artifact_commit": PARENT}), encoding="utf-8"
    )
    args = SimpleNamespace(
        producer_dir=producer,
        snapshot_dir=snapshot,
        receipt_dir=receipt_checkout,
        output_dir=output,
        producer_commit=PRODUCER,
        snapshot_commit=SNAPSHOT,
        status_commit=STATUS,
        snapshot_sha256=raw_sha,
        prior_revision=PARENT,
        region_comparison_revision=PARENT,
        require_bio_release=False,
    )
    monkeypatch.setattr(preview, "_parse_args", lambda: args)
    monkeypatch.setattr(
        preview,
        "_git_head",
        lambda path: (
            PRODUCER if path == producer else SNAPSHOT if path == snapshot else STATUS
        ),
    )
    monkeypatch.setattr(preview, "_require_clean_tracked_files", lambda path: None)

    class FakeHfApi:
        def __init__(self, token=None):
            assert token is None

        def repo_info(self, repo_id, repo_type):
            assert repo_id == "ccm/cmu-engineering-publications"
            assert repo_type == "dataset"
            return SimpleNamespace(sha=PARENT)

        def create_commit(self, **kwargs):
            raise AssertionError("Unpatched Hub write reached")

    class FakeDataset:
        def push_to_hub(self, *args, **kwargs):
            raise AssertionError("Unpatched dataset write reached")

    class FakeReconciliation(RuntimeError):
        pass

    def fake_publish(*_args, atomic_intent_sink, **kwargs):
        approval_path = producer / "data/curated/bio-mcmanus-poster-loss-approval.json"
        assert kwargs["historical_loss_approval_path"] == (
            approval_path if approval else None
        )
        assert kwargs["bio_binding_approval_path"] == (
            producer / "data/curated/bio-exact-binding-approval-38f.json"
            if binding_approval
            else None
        )
        if region_diff:
            raise RegionReleaseDiffError(
                {
                    "unexplained_losses": [],
                    "unexplained_gains": [],
                    "unapproved_mapped_binding_changes": [
                        {"observation": "person-a/scholar-a/citation-a/scholar"}
                    ],
                }
            )
        if fail_before_write:
            raise ValueError("review gate failed")
        staging = tmp_path / "publisher-stage"
        files = {}
        for name in sorted(RELEASE_PATHS):
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            files[name] = path
        atomic_intent_sink(
            {
                "parent_revision": PARENT,
                "prewrite_gate": {"status": "pass"},
                "sha256_by_path": {
                    name: preview._sha256(path) for name, path in files.items()
                },
                "size_by_path": {
                    name: path.stat().st_size for name, path in files.items()
                },
            }
        )
        try:
            FakeHfApi().create_commit(
                repo_id="ccm/cmu-engineering-publications",
                repo_type="dataset",
                parent_commit=PARENT,
                operations=[
                    SimpleNamespace(path_in_repo=name, path_or_fileobj=path)
                    for name, path in files.items()
                ],
            )
        except Exception as error:
            raise FakeReconciliation("write stopped") from error

    fake_publisher = ModuleType("map_of_research.publisher")
    fake_publisher.REPO_ID = "ccm/cmu-engineering-publications"
    fake_publisher.publish_snapshot = fake_publish
    fake_atomic = ModuleType("map_of_research.atomic_release")
    fake_atomic.RELEASE_PATHS = RELEASE_PATHS
    fake_atomic.AtomicReleaseReconciliationRequired = FakeReconciliation
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=FakeDataset))
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeHfApi)
    )
    monkeypatch.setitem(sys.modules, "map_of_research.publisher", fake_publisher)
    monkeypatch.setitem(sys.modules, "map_of_research.atomic_release", fake_atomic)
    monkeypatch.setattr(map_of_research, "publisher", fake_publisher, raising=False)
    code = preview.main()
    return code, json.loads((output / "receipt.json").read_text(encoding="utf-8"))


def test_no_write_preview_passes_only_at_pinned_seven_file_boundary(
    monkeypatch, tmp_path
):
    code, receipt = _run_preview(monkeypatch, tmp_path, fail_before_write=False)
    assert code == 0
    assert receipt["status"] == "passed_no_upload"
    assert receipt["blocked_hub_write_calls"] == 1
    assert receipt["hub_writes_completed"] == 0
    assert receipt["hub_head_before"] == receipt["hub_head_after"] == PARENT
    assert set(receipt["stage_files"]) == RELEASE_PATHS
    assert receipt["status_commit"] == STATUS
    assert receipt["local_atomic_intent"]["prewrite_gate"]["status"] == "pass"


def test_no_write_preview_fails_job_when_release_gate_fails(monkeypatch, tmp_path):
    code, receipt = _run_preview(monkeypatch, tmp_path, fail_before_write=True)
    assert code == 1
    assert receipt["status"] == "failed_before_hub_write"
    assert receipt["error"] == "review gate failed"
    assert receipt["blocked_hub_write_calls"] == 0
    assert receipt["hub_writes_completed"] == 0


def test_no_write_preview_records_exact_region_diff_before_write(monkeypatch, tmp_path):
    code, receipt = _run_preview(
        monkeypatch, tmp_path, fail_before_write=False, region_diff=True
    )
    assert code == 1
    assert receipt["status"] == "failed_before_hub_write"
    assert receipt["blocked_hub_write_calls"] == 0
    assert receipt["region_release_diff"]["unapproved_mapped_binding_changes"] == [
        {"observation": "person-a/scholar-a/citation-a/scholar"}
    ]


def test_no_write_preview_passes_one_time_bio_loss_approval(monkeypatch, tmp_path):
    code, receipt = _run_preview(
        monkeypatch,
        tmp_path,
        fail_before_write=False,
        approval=True,
        binding_approval=True,
    )
    assert code == 0
    assert receipt["status"] == "passed_no_upload"


def test_hosted_preview_has_read_only_permissions_and_upload_lane_lock():
    workflow = (ROOT / ".github/workflows/release-preview.yml").read_text(
        encoding="utf-8"
    )
    upload = (ROOT / ".github/workflows/upload.yml").read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in workflow
    assert "id-token: write" not in workflow
    assert "persist-credentials: false" in workflow
    assert "--bio-binding-approval" in upload
    assert "--status-dir receipt/status" in workflow
    assert "--current-receipt receipt/status/last-upload.json" in workflow
    assert "--receipt-dir receipt" in workflow
    assert '--status-commit "$STATUS_COMMIT"' in workflow
    assert 'HF_HUB_DISABLE_IMPLICIT_TOKEN: "1"' in workflow
    assert "central-publication-upload-v1" in workflow
    assert "central-publication-upload-v1" in upload
    assert "preview_atomic_release.py" in workflow
    assert "map-validate-math-coverage" in workflow
    assert "map-validate-scholar-coverage" in workflow
    assert "validate_primary_homes.py" in workflow
    assert "upload-artifact@" in workflow
    assert "git push" not in workflow
    assert "hf auth token" not in workflow
    assert "MAP_DURABLE_INTENT_REPO" not in workflow
