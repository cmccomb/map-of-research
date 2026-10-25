"""The trusted upload workflow must select exactly one publication path."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/upload.yml"
PRODUCER = "a" * 40
PARENT = "b" * 40


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps():
    return {step["name"]: step for step in _workflow()["jobs"]["upload"]["steps"]}


def _gate(
    *,
    mode,
    event="workflow_dispatch",
    ref="refs/heads/main",
    sha=PRODUCER,
    reviewed=PRODUCER,
    parent=PARENT,
    restore="false",
) -> int:
    script = _steps()["Require an exact release dispatch"]["run"]
    env = {
        **os.environ,
        "RELEASE_MODE": mode,
        "GITHUB_EVENT_NAME": event,
        "GITHUB_REF": ref,
        "GITHUB_SHA": sha,
        "REVIEWED_PRODUCER_COMMIT": reviewed,
        "EXPECTED_PARENT_REVISION": parent,
        "RESTORE_PRIOR_REGIONS": restore,
    }
    return subprocess.run(["bash", "-e", "-c", script], env=env, check=False).returncode


@pytest.mark.parametrize(
    ("mode", "event", "expected"),
    [
        ("snapshot", "push", 0),
        ("snapshot", "workflow_dispatch", 0),
        ("chemistry", "workflow_dispatch", 0),
        ("chemistry", "push", 1),
        ("unknown", "workflow_dispatch", 1),
    ],
)
def test_release_mode_gate(mode: str, event: str, expected: int) -> None:
    assert (_gate(mode=mode, event=event) == 0) == (expected == 0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ref": "refs/heads/feature"},
        {"sha": "c" * 40},
        {"reviewed": ""},
        {"reviewed": "not-a-sha"},
        {"parent": ""},
        {"parent": "not-a-sha"},
        {"restore": "true"},
    ],
)
def test_chemistry_dispatch_rejects_unreviewed_context(
    overrides: dict[str, str],
) -> None:
    assert _gate(mode="chemistry", **overrides) != 0


def test_trusted_job_keeps_normal_and_chemistry_paths_mutually_exclusive() -> None:
    workflow = _workflow()
    job = workflow["jobs"]["upload"]
    steps = _steps()
    assert workflow["name"] == "Publish central publication dataset"
    assert set(workflow["jobs"]) == {"upload"}
    assert workflow["permissions"] == {"contents": "write", "id-token": "write"}
    assert workflow["concurrency"]["group"] == "central-publication-upload-v1"
    assert job["environment"] == "huggingface"
    assert "inputs.release_mode || 'snapshot'" in job["env"]["RELEASE_MODE"]
    assert (
        "source/pyproject.toml"
        in steps["Set up Python"]["with"]["cache-dependency-path"]
    )

    snapshot_only = (
        "Check out collected snapshot",
        "Check out current upload receipts",
        "Install trusted publisher",
        "Pin previous hosted release",
        "Select region comparison",
        "Require evidenced primary homes for included Scholar profiles",
        "Require complete Mathematical Sciences Scholar captures",
        "Verify complete Scholar sources before publication",
        "Validate, enrich, and publish snapshot",
    )
    chemistry_only = (
        "Check out Chemistry status for remote intent",
        "Check out frozen Chemistry source data",
        "Install exact Chemistry producer",
        "Validate, checkpoint remotely, and publish Chemistry delta",
    )
    for name in snapshot_only:
        assert steps[name]["if"] == "${{ env.RELEASE_MODE == 'snapshot' }}"
    for name in chemistry_only:
        assert steps[name]["if"] == "${{ env.RELEASE_MODE == 'chemistry' }}"
    assert steps["Record upload receipt"]["if"] == (
        "${{ always() && env.RELEASE_MODE == 'snapshot' }}"
    )
    assert (
        "env.RELEASE_MODE == 'snapshot'"
        in (steps["Check out reviewed prior upload receipt"]["if"])
    )
    assert steps["Check out Chemistry status for remote intent"]["with"] == {
        "ref": "automation/map-snapshot",
        "path": "snapshot",
        "persist-credentials": True,
        "sparse-checkout": "status\n",
        "sparse-checkout-cone-mode": False,
    }
    frozen = steps["Check out frozen Chemistry source data"]["with"]
    assert frozen["ref"] == "8be6a2400cf72dd959c874bfcada8624540820ef"
    assert frozen["path"] == "source-data"
    assert frozen["persist-credentials"] is False
    assert "registry/people.csv" in frozen["sparse-checkout"]
    assert "data/curated/works.json" in frozen["sparse-checkout"]
    release = steps["Validate, checkpoint remotely, and publish Chemistry delta"]
    assert release["env"]["HF_OIDC_RESOURCE"] == (
        "datasets/ccm/cmu-engineering-publications"
    )
    assert "--snapshot-checkout snapshot" in release["run"]
    assert "--source-data-checkout source-data" in release["run"]
    assert "--producer-checkout source" in release["run"]
    assert '--expected-parent-revision "$EXPECTED_PARENT_REVISION"' in release["run"]
    assert '--expected-producer-commit "$REVIEWED_PRODUCER_COMMIT"' in release["run"]
    assert '--intent-path "$attempt_dir/chem-intent.json"' in release["run"]
    assert steps["Check out trusted publisher"]["with"]["path"] == "source"
    assert steps["Install exact Chemistry producer"]["run"] == (
        "python -m pip install -e './source[publisher]'"
    )
    auth_steps = {
        name for name, step in steps.items() if "hf auth token" in step.get("run", "")
    }
    assert auth_steps == {
        "Validate, enrich, and publish snapshot",
        "Validate, checkpoint remotely, and publish Chemistry delta",
    }
    assert not (ROOT / ".github/workflows/chem-region-release.yml").exists()
