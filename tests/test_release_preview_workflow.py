"""The hosted rehearsal must stop at the real seven-file write boundary."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import map_of_research
from map_of_research.presentation_drift import PresentationDriftError
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


def test_hosted_preview_coverage_uses_upload_prior_id_crosswalks():
    """A preview must accept the same reviewed prior-ID continuity as upload."""

    upload = (ROOT / ".github/workflows/upload.yml").read_text()
    preview = (ROOT / ".github/workflows/release-preview.yml").read_text()

    def crosswalks(workflow: str) -> set[str]:
        coverage_command = workflow.split("map-validate-scholar-coverage \\", 1)[1]
        coverage_command = coverage_command.split("--curated-works", 1)[0]
        return set(
            re.findall(
                r"--(?:additional-)?prior-id-crosswalk\s+([^\s\\]+)",
                coverage_command,
            )
        )

    assert crosswalks(upload)
    assert crosswalks(preview) == crosswalks(upload)


def test_hosted_preview_uses_the_uploads_source_review_manifests():
    """The Linux rehearsal must exercise the same additive review as upload."""

    upload = (ROOT / ".github/workflows/upload.yml").read_text()
    preview = (ROOT / ".github/workflows/release-preview.yml").read_text()
    checkout = preview.split("- name: Check out collected snapshot", 1)[1].split(
        "- name: Check out current upload receipts", 1
    )[0]
    stage = preview.split(
        "- name: Stage and validate the seven-file release without uploading", 1
    )[1].split("- name: Save inspectable preview receipt", 1)[0]

    for name, option in (
        ("scholar-additive-placement-review.json", "--additive-placement-review"),
        (
            "scholar-prior-person-continuity-review.json",
            "--prior-person-continuity-review",
        ),
    ):
        assert name in upload
        assert f"data/curated/{name}" in checkout
        assert f"snapshot/data/curated/{name}" in stage
        assert option in stage
    assert "use_additive_presentation_review:" in preview
    assert '"$USE_ADDITIVE_PRESENTATION_REVIEW" = "true"' in stage
    assert 'if [ -f "$PERSON_REVIEW" ]; then' in stage
    assert '"${REVIEW_ARGS[@]}"' in stage


def _load_script():
    spec = importlib.util.spec_from_file_location("preview_atomic_release", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _imported_package(root: Path) -> dict[str, ModuleType]:
    package = ModuleType("map_of_research")
    package.__file__ = str(root / "__init__.py")
    package.__path__ = [str(root)]
    publisher = ModuleType("map_of_research.publisher")
    publisher.__file__ = str(root / "publisher.py")
    return {"map_of_research": package, "map_of_research.publisher": publisher}


def _pinned_test_producer(tmp_path: Path) -> tuple[Path, Path, str]:
    producer = tmp_path / "producer"
    package = producer / "map_of_research"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "publisher.py").write_text("trusted = True\n", encoding="utf-8")
    _git(producer, "init", "-q")
    _git(producer, "config", "user.name", "Preview Test")
    _git(producer, "config", "user.email", "preview@example.test")
    _git(producer, "add", "map_of_research")
    _git(producer, "commit", "-qm", "pinned package")
    return producer, package, _git(producer, "rev-parse", "HEAD")


def test_preview_accepts_imports_from_pinned_producer_checkout(tmp_path):
    preview = _load_script()
    producer, package, commit = _pinned_test_producer(tmp_path)
    assert preview._require_producer_imports(
        producer, commit, _imported_package(package)
    ) == {
        "mode": "producer_checkout",
        "loaded_modules_checked": 2,
    }


def test_preview_accepts_only_byte_identical_installed_package(tmp_path):
    preview = _load_script()
    producer, package, commit = _pinned_test_producer(tmp_path)
    installed = tmp_path / "venv/lib/python3.12/site-packages/map_of_research"
    shutil.copytree(package, installed)
    assert preview._require_producer_imports(
        producer, commit, _imported_package(installed)
    ) == {"mode": "byte_identical_installed_package", "loaded_modules_checked": 2}
    (installed / "publisher.py").write_text("trusted = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from pinned commit"):
        preview._require_producer_imports(
            producer, commit, _imported_package(installed)
        )


def test_preview_rejects_other_editable_checkout_even_with_identical_bytes(tmp_path):
    preview = _load_script()
    producer, package, commit = _pinned_test_producer(tmp_path)
    other = tmp_path / "other/map_of_research"
    shutil.copytree(package, other)
    with pytest.raises(ValueError, match="another editable checkout"):
        preview._require_producer_imports(producer, commit, _imported_package(other))


def test_preview_rejects_untracked_or_changed_package_python(tmp_path):
    preview = _load_script()
    producer, package, commit = _pinned_test_producer(tmp_path)
    modules = _imported_package(package)
    extra = package / "untracked.py"
    extra.write_text("surprise = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its pinned Git commit"):
        preview._require_producer_imports(producer, commit, modules)
    extra.unlink()
    (package / "publisher.py").write_text("trusted = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its pinned Git commit"):
        preview._require_producer_imports(producer, commit, modules)


def test_preview_rejects_wheel_matching_dirty_checkout_instead_of_commit(tmp_path):
    preview = _load_script()
    producer, package, commit = _pinned_test_producer(tmp_path)
    (package / "publisher.py").write_text("trusted = False\n", encoding="utf-8")
    installed = tmp_path / "venv/lib/python3.12/site-packages/map_of_research"
    shutil.copytree(package, installed)
    with pytest.raises(ValueError, match="differs from its pinned Git commit"):
        preview._require_producer_imports(
            producer, commit, _imported_package(installed)
        )


def test_preview_rejects_missing_or_mixed_import_origins(tmp_path):
    preview = _load_script()
    producer, package, commit = _pinned_test_producer(tmp_path)
    with pytest.raises(ValueError, match="Cannot verify the imported producer"):
        preview._require_producer_imports(producer, commit, {})
    modules = _imported_package(package)
    modules["map_of_research"].__path__ = [str(tmp_path / "other")]
    with pytest.raises(ValueError, match="unexpected search path"):
        preview._require_producer_imports(producer, commit, modules)
    modules = _imported_package(package)
    del modules["map_of_research.publisher"].__file__
    with pytest.raises(ValueError, match="Cannot verify imported producer module"):
        preview._require_producer_imports(producer, commit, modules)
    modules = _imported_package(package)
    modules["map_of_research.publisher"].__file__ = str(tmp_path / "other.py")
    with pytest.raises(ValueError, match="module comes from another checkout"):
        preview._require_producer_imports(producer, commit, modules)


def _git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True
    ).strip()


def _lineage_checkouts(tmp_path: Path) -> tuple[Path, Path, Path, str, str, bytes]:
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-q")
    _git(seed, "config", "user.name", "Preview Test")
    _git(seed, "config", "user.email", "preview@example.test")
    status = seed / "status/last-successful-upload.json"
    status.parent.mkdir()
    receipt_bytes = b'{"status":"success","artifact_commit":"parent"}\n'
    status.write_bytes(receipt_bytes)
    _git(seed, "add", "status/last-successful-upload.json")
    _git(seed, "commit", "-qm", "successful receipt")
    status_commit = _git(seed, "rev-parse", "HEAD")

    receipt_checkout = tmp_path / "receipt"
    subprocess.run(["git", "clone", "-q", str(seed), str(receipt_checkout)], check=True)
    (seed / "snapshot-marker").write_text("candidate", encoding="utf-8")
    _git(seed, "add", "snapshot-marker")
    _git(seed, "commit", "-qm", "candidate snapshot")
    snapshot_commit = _git(seed, "rev-parse", "HEAD")
    snapshot = tmp_path / "snapshot"
    subprocess.run(["git", "clone", "-q", str(seed), str(snapshot)], check=True)
    return (
        seed,
        snapshot,
        receipt_checkout,
        snapshot_commit,
        status_commit,
        receipt_bytes,
    )


def test_receipt_lineage_accepts_descendant_with_identical_committed_bytes(tmp_path):
    preview = _load_script()
    _, snapshot, receipt, snapshot_commit, status_commit, expected = _lineage_checkouts(
        tmp_path
    )
    assert (
        preview._require_receipt_lineage(
            snapshot,
            receipt,
            snapshot_commit=snapshot_commit,
            status_commit=status_commit,
        )
        == expected
    )


def test_receipt_lineage_rejects_diverged_snapshot_with_same_receipt_bytes(tmp_path):
    preview = _load_script()
    seed, _, receipt, _, status_commit, _ = _lineage_checkouts(tmp_path)
    # The successful receipt is the first commit, so make an independent root
    # with the same file bytes while retaining the original branch in the clone.
    _git(seed, "checkout", "--orphan", "diverged")
    _git(seed, "add", "status/last-successful-upload.json")
    _git(seed, "commit", "-qm", "independent snapshot")
    diverged_commit = _git(seed, "rev-parse", "HEAD")
    diverged = tmp_path / "diverged"
    subprocess.run(["git", "clone", "-q", str(seed), str(diverged)], check=True)
    with pytest.raises(ValueError, match="not an ancestor"):
        preview._require_receipt_lineage(
            diverged,
            receipt,
            snapshot_commit=diverged_commit,
            status_commit=status_commit,
        )


def test_receipt_lineage_rejects_candidate_receipt_byte_change(tmp_path):
    preview = _load_script()
    seed, _, receipt, _, status_commit, _ = _lineage_checkouts(tmp_path)
    (seed / "status/last-successful-upload.json").write_bytes(
        b'{"status":"success","artifact_commit":"other"}\n'
    )
    _git(seed, "add", "status/last-successful-upload.json")
    _git(seed, "commit", "-qm", "changed candidate receipt")
    altered_commit = _git(seed, "rev-parse", "HEAD")
    altered = tmp_path / "altered"
    subprocess.run(["git", "clone", "-q", str(seed), str(altered)], check=True)
    with pytest.raises(ValueError, match="Snapshot successful receipt differs"):
        preview._require_receipt_lineage(
            altered,
            receipt,
            snapshot_commit=altered_commit,
            status_commit=status_commit,
        )


def test_receipt_lineage_rejects_uncommitted_candidate_receipt(tmp_path):
    preview = _load_script()
    _, snapshot, receipt, snapshot_commit, status_commit, _ = _lineage_checkouts(
        tmp_path
    )
    (snapshot / "status/last-successful-upload.json").write_bytes(b"altered\n")
    with pytest.raises(ValueError, match="Snapshot checkout differs"):
        preview._require_receipt_lineage(
            snapshot,
            receipt,
            snapshot_commit=snapshot_commit,
            status_commit=status_commit,
        )


def test_receipt_lineage_requires_committed_success_file(tmp_path):
    preview = _load_script()
    seed, _, receipt, _, status_commit, _ = _lineage_checkouts(tmp_path)
    _git(seed, "rm", "status/last-successful-upload.json")
    _git(seed, "commit", "-qm", "removed successful receipt")
    missing_commit = _git(seed, "rev-parse", "HEAD")
    missing = tmp_path / "missing"
    subprocess.run(["git", "clone", "-q", str(seed), str(missing)], check=True)
    with pytest.raises(ValueError, match="missing from commit"):
        preview._require_receipt_lineage(
            missing,
            receipt,
            snapshot_commit=missing_commit,
            status_commit=status_commit,
        )


def _run_preview(
    monkeypatch,
    tmp_path,
    *,
    fail_before_write: bool,
    approval: bool = False,
    binding_approval: bool = False,
    region_diff: bool = False,
    presentation_diff: bool = False,
    require_bio_release: bool | None = None,
    chem_manifest_present: bool = True,
    additive_placement_review: Path | None = None,
    prior_person_continuity_review: Path | None = None,
    presentation_gate_failure: bool = False,
    import_origin_mismatch: bool = False,
):
    preview = _load_script()
    producer = tmp_path / "producer"
    snapshot = tmp_path / "snapshot"
    receipt_checkout = tmp_path / "receipt"
    output = tmp_path / "output"
    producer.mkdir()
    curated = producer / "data/curated"
    curated.mkdir(parents=True)
    if approval:
        (curated / "bio-mcmanus-poster-loss-approval.json").write_text("{}")
        if binding_approval:
            (curated / "bio-exact-binding-approval-dcb.json").write_text("{}")
    if chem_manifest_present:
        (curated / "scholar-chem-region-review.json").write_text("reviewed")
    (snapshot / "snapshots").mkdir(parents=True)
    (receipt_checkout / "status").mkdir(parents=True)
    raw = snapshot / "snapshots/cmu-engineering-publications.parquet"
    raw.write_bytes(b"raw-snapshot")
    raw_sha = preview._sha256(raw)
    (
        snapshot / "snapshots/cmu-engineering-publications.parquet.manifest.json"
    ).write_text(json.dumps({"snapshot_sha256": raw_sha}), encoding="utf-8")
    success_bytes = json.dumps(
        {"status": "success", "artifact_commit": PARENT}
    ).encode()
    (receipt_checkout / "status/last-successful-upload.json").write_bytes(success_bytes)
    (receipt_checkout / "status/last-upload.json").write_bytes(success_bytes)
    require_bio_release = (
        approval if require_bio_release is None else require_bio_release
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
        require_bio_release=require_bio_release,
        additive_placement_review=additive_placement_review,
        prior_person_continuity_review=prior_person_continuity_review,
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
    if import_origin_mismatch:

        def reject_imports(_producer, _commit):
            raise ValueError(
                "Imported producer package comes from another editable checkout"
            )

        monkeypatch.setattr(preview, "_require_producer_imports", reject_imports)
    else:
        monkeypatch.setattr(
            preview, "_require_producer_imports", lambda _producer, _commit: {}
        )
    monkeypatch.setattr(
        preview, "_require_receipt_lineage", lambda *args, **kwargs: success_bytes
    )

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
            approval_path if approval and require_bio_release else None
        )
        assert kwargs["bio_binding_approval_path"] == (
            producer / "data/curated/bio-exact-binding-approval-dcb.json"
            if binding_approval and require_bio_release
            else None
        )
        assert kwargs["person_publication_holds_path"] == (
            producer / "data/curated/person-publication-holds.json"
        )
        assert kwargs["additive_placement_review_path"] == additive_placement_review
        assert (
            kwargs["prior_person_continuity_review_path"]
            == prior_person_continuity_review
        )
        if presentation_gate_failure:
            raise PresentationDriftError(
                {"candidate_map_content_sha256": "f" * 64},
                "new points lack reviewed placement",
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
        if presentation_diff:
            raise PresentationDriftError(
                {
                    "material_reasons": ["retained_point_movement"],
                    "status": "review_required",
                },
                "material drift lacks a reviewed approval",
            )
        chem_manifest = kwargs["chem_region_review_path"]
        assert chem_manifest == curated / "scholar-chem-region-review.json"
        if not chem_manifest.is_file():
            raise FileNotFoundError(f"Missing Chemistry review: {chem_manifest}")
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


def test_preview_rejects_wrong_import_before_any_hub_access(monkeypatch, tmp_path):
    code, receipt = _run_preview(
        monkeypatch, tmp_path, fail_before_write=False, import_origin_mismatch=True
    )
    assert code == 1
    assert receipt["status"] == "failed_before_hub_write"
    assert "another editable checkout" in receipt["error"]
    assert "hub_head_before" not in receipt
    assert receipt["blocked_hub_write_calls"] == receipt["hub_writes_completed"] == 0


def test_no_write_preview_forwards_additive_placement_review(monkeypatch, tmp_path):
    review = tmp_path / "review.json"
    code, receipt = _run_preview(
        monkeypatch,
        tmp_path,
        fail_before_write=False,
        additive_placement_review=review,
    )
    assert code == 0
    assert receipt["status"] == "passed_no_upload"


def test_no_write_preview_forwards_prior_person_continuity_review(
    monkeypatch, tmp_path
):
    review = tmp_path / "person-review.json"
    code, receipt = _run_preview(
        monkeypatch,
        tmp_path,
        fail_before_write=False,
        prior_person_continuity_review=review,
    )
    assert code == 0
    assert receipt["status"] == "passed_no_upload"


def test_no_write_preview_records_unapproved_additive_candidate(monkeypatch, tmp_path):
    code, receipt = _run_preview(
        monkeypatch,
        tmp_path,
        fail_before_write=False,
        additive_placement_review=tmp_path / "absent-review.json",
        presentation_gate_failure=True,
    )
    assert code == 1
    assert receipt["status"] == "failed_before_hub_write"
    assert receipt["presentation_drift"]["candidate_map_content_sha256"] == "f" * 64
    assert receipt["blocked_hub_write_calls"] == receipt["hub_writes_completed"] == 0


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


def test_no_write_preview_records_exact_presentation_diff_before_write(
    monkeypatch, tmp_path
):
    code, receipt = _run_preview(
        monkeypatch, tmp_path, fail_before_write=False, presentation_diff=True
    )
    assert code == 1
    assert receipt["status"] == "failed_before_hub_write"
    assert receipt["blocked_hub_write_calls"] == 0
    assert receipt["presentation_drift"] == {
        "material_reasons": ["retained_point_movement"],
        "status": "review_required",
    }


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


def test_later_preview_ignores_historical_bio_approvals(monkeypatch, tmp_path):
    code, receipt = _run_preview(
        monkeypatch,
        tmp_path,
        fail_before_write=False,
        approval=True,
        binding_approval=True,
        require_bio_release=False,
    )
    assert code == 0
    assert receipt["status"] == "passed_no_upload"


def test_no_write_preview_requires_producer_chemistry_review(monkeypatch, tmp_path):
    code, receipt = _run_preview(
        monkeypatch, tmp_path, fail_before_write=False, chem_manifest_present=False
    )
    assert code == 1
    assert receipt["status"] == "failed_before_hub_write"
    assert "Missing Chemistry review:" in receipt["error"]
    assert receipt["blocked_hub_write_calls"] == 0
    assert receipt["hub_writes_completed"] == 0


def test_hosted_preview_has_read_only_permissions_and_upload_lane_lock():
    workflow = (ROOT / ".github/workflows/release-preview.yml").read_text(
        encoding="utf-8"
    )
    upload = (ROOT / ".github/workflows/upload.yml").read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in workflow
    assert "id-token: write" not in workflow
    assert "persist-credentials: false" in workflow
    assert "--bio-binding-approval" in upload
    assert "fetch-depth: 0" in workflow
    assert "--status-dir receipt/status" in workflow
    assert "--current-receipt receipt/status/last-upload.json" in workflow
    assert "--receipt-dir receipt" in workflow
    assert '--status-commit "$STATUS_COMMIT"' in workflow
    assert 'HF_HUB_DISABLE_IMPLICIT_TOKEN: "1"' in workflow
    assert "central-publication-upload-v1" in workflow
    assert "central-publication-upload-v1" in upload
    assert "cache: pip" not in workflow
    assert "cache: pip" not in upload
    assert "preview_atomic_release.py" in workflow
    assert "map-validate-math-coverage" in workflow
    assert "map-validate-scholar-coverage" in workflow
    assert (
        "--person-publication-holds source/data/curated/person-publication-holds.json"
    ) in upload
    assert "validate_primary_homes.py" in workflow
    assert "upload-artifact@" in workflow
    assert "git push" not in workflow
    assert "hf auth token" not in workflow
    assert "MAP_DURABLE_INTENT_REPO" not in workflow
