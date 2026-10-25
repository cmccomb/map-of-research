"""The one-time region correction cannot become an arbitrary baseline switch."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from types import SimpleNamespace

import numpy
import pandas
import pytest

import map_of_research.region_repair as repair


def repair_kwargs(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.parquet"
    snapshot.write_bytes(b"frozen Bernhard test snapshot")
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    monkeypatch.setattr(repair, "SNAPSHOT_SHA256", digest)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"snapshot_sha256": digest}))
    current_receipt = tmp_path / "current-receipt.json"
    current_receipt.write_text(
        json.dumps(
            {
                "status": "success",
                "artifact_commit": repair.CURRENT_HUB_REVISION,
                "raw_snapshot_sha256": digest,
                "workflow_url": repair.CURRENT_UPLOAD_RUN,
                "raw_snapshot_observations": 51823,
                "raw_snapshot_profiles": 310,
            }
        )
    )
    historical_receipt = tmp_path / "historical-receipt.json"
    historical_receipt.write_text(
        json.dumps(
            {
                "status": "success",
                "artifact_commit": repair.REGION_COMPARISON_REVISION,
                "raw_snapshot_sha256": repair.HISTORICAL_SNAPSHOT_SHA256,
                "workflow_url": repair.HISTORICAL_UPLOAD_RUN,
                "raw_snapshot_observations": 51595,
                "raw_snapshot_profiles": 309,
            }
        )
    )
    return {
        "requested": True,
        "event_name": "workflow_dispatch",
        "run_ref": "refs/heads/main",
        "run_commit": "a" * 40,
        "producer_commit": "a" * 40,
        "snapshot_ref": repair.SNAPSHOT_BRANCH,
        "snapshot_commit": repair.SNAPSHOT_COMMIT,
        "historical_commit": repair.HISTORICAL_RECEIPT_COMMIT,
        "snapshot_path": snapshot,
        "manifest_path": manifest,
        "current_receipt_path": current_receipt,
        "historical_receipt_path": historical_receipt,
        "current_revision": repair.CURRENT_HUB_REVISION,
    }


def test_normal_upload_uses_current_hub_revision_without_repair_files(tmp_path):
    revision = repair.select_region_comparison_revision(
        requested=False,
        event_name="push",
        run_ref="refs/heads/automation/map-snapshot",
        run_commit="not checked in normal mode",
        producer_commit="not checked in normal mode",
        snapshot_ref="whatever",
        snapshot_commit="whatever",
        historical_commit="whatever",
        snapshot_path=tmp_path / "missing.parquet",
        manifest_path=tmp_path / "missing.json",
        current_receipt_path=tmp_path / "missing-current.json",
        historical_receipt_path=tmp_path / "missing-historical.json",
        current_revision="c" * 40,
    )
    assert revision == "c" * 40


def test_frozen_dispatch_selects_historical_region_revision(tmp_path, monkeypatch):
    kwargs = repair_kwargs(tmp_path, monkeypatch)
    assert repair.select_region_comparison_revision(**kwargs) == (
        repair.REGION_COMPARISON_REVISION
    )
    assert (
        repair.require_approved_comparison_revision(
            current_revision=repair.CURRENT_HUB_REVISION,
            comparison_revision=repair.REGION_COMPARISON_REVISION,
            snapshot_sha256=repair.SNAPSHOT_SHA256,
        )
        == repair.REGION_COMPARISON_REVISION
    )


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("event_name", "push"),
        ("run_ref", "refs/heads/feature"),
        ("run_commit", "b" * 40),
        ("producer_commit", "b" * 40),
        ("snapshot_ref", "unreviewed-source"),
        ("snapshot_commit", "b" * 40),
        ("historical_commit", "b" * 40),
        ("current_revision", "b" * 40),
    ],
)
def test_repair_rejects_wrong_run_source_or_hub_head(tmp_path, monkeypatch, field, bad):
    kwargs = repair_kwargs(tmp_path, monkeypatch)
    kwargs[field] = bad
    with pytest.raises(ValueError, match="frozen dispatch and Hub head"):
        repair.select_region_comparison_revision(**kwargs)


def test_repair_rejects_changed_parquet_or_manifest(tmp_path, monkeypatch):
    kwargs = repair_kwargs(tmp_path, monkeypatch)
    kwargs["snapshot_path"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen Parquet"):
        repair.select_region_comparison_revision(**kwargs)
    kwargs = repair_kwargs(tmp_path, monkeypatch)
    kwargs["manifest_path"].write_text(json.dumps({"snapshot_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="frozen Parquet"):
        repair.select_region_comparison_revision(**kwargs)


@pytest.mark.parametrize("receipt_name", ["current", "historical"])
def test_repair_rejects_changed_upload_receipt(tmp_path, monkeypatch, receipt_name):
    kwargs = repair_kwargs(tmp_path, monkeypatch)
    path = kwargs[f"{receipt_name}_receipt_path"]
    receipt = json.loads(path.read_text())
    receipt["status"] = "failure"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipts do not match"):
        repair.select_region_comparison_revision(**kwargs)


def test_arbitrary_comparison_and_replay_are_rejected(tmp_path, monkeypatch):
    repair_kwargs(tmp_path, monkeypatch)
    for current, comparison, snapshot_sha in (
        (repair.CURRENT_HUB_REVISION, "b" * 40, repair.SNAPSHOT_SHA256),
        ("c" * 40, repair.REGION_COMPARISON_REVISION, repair.SNAPSHOT_SHA256),
        (repair.CURRENT_HUB_REVISION, repair.REGION_COMPARISON_REVISION, "0" * 64),
    ):
        with pytest.raises(ValueError, match="Unapproved region comparison"):
            repair.require_approved_comparison_revision(
                current_revision=current,
                comparison_revision=comparison,
                snapshot_sha256=snapshot_sha,
            )
    assert (
        repair.require_approved_comparison_revision(
            current_revision="c" * 40,
            comparison_revision="c" * 40,
            snapshot_sha256="new snapshot",
        )
        == "c" * 40
    )


def test_repair_checks_live_source_branch_tip(tmp_path, monkeypatch):
    calls = []

    def ls_remote(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            stdout=f"{repair.SNAPSHOT_COMMIT}\trefs/heads/{repair.SNAPSHOT_BRANCH}\n"
        )

    monkeypatch.setattr(repair.subprocess, "run", ls_remote)
    repair.require_frozen_source_tip(tmp_path)
    assert calls == [
        (
            [
                "git",
                "-C",
                str(tmp_path),
                "ls-remote",
                "origin",
                f"refs/heads/{repair.SNAPSHOT_BRANCH}",
            ],
            {"capture_output": True, "check": True, "text": True},
        )
    ]
    monkeypatch.setattr(
        repair.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"{'b' * 40}\trefs/heads/{repair.SNAPSHOT_BRANCH}\n"
        ),
    )
    with pytest.raises(RuntimeError, match="branch moved"):
        repair.require_frozen_source_tip(tmp_path)


def test_repair_checks_live_producer_main_tip(tmp_path, monkeypatch):
    expected = "a" * 40
    monkeypatch.setattr(
        repair.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"{expected}\trefs/heads/main\n"
        ),
    )
    repair.require_frozen_producer_tip(tmp_path, expected)
    with pytest.raises(ValueError, match="immutable"):
        repair.require_frozen_producer_tip(tmp_path, "main")
    monkeypatch.setattr(
        repair.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"{'b' * 40}\trefs/heads/main\n"
        ),
    )
    with pytest.raises(RuntimeError, match="Producer main moved"):
        repair.require_frozen_producer_tip(tmp_path, expected)


def reviewed_preview():
    return {
        "work_count": 47693,
        "map_point_count": 42869,
        "new_embedding_count": 0,
        "layout_reused": True,
        "continuity_receipt": {
            "prior_revision": repair.REGION_COMPARISON_REVISION,
            "current_hub_revision": repair.CURRENT_HUB_REVISION,
            "prior_visible_observations": 48953,
            "current_visible_observations": 49136,
            "stable_works": 45437,
            "reviewed_losses": [],
            "verified_gains": [],
            "new_source_gains": 183,
            "unexplained_losses": [],
            "unexplained_gains": [],
        },
    }


def test_exact_repair_preview_is_required_before_upload():
    expected = reviewed_preview()
    repair.require_expected_repair_preview(**expected)
    required_keys = tuple(expected["continuity_receipt"])
    # The raw rebuild can already agree with 21 historical decisions before
    # continuity runs, so the published-table override counts are not a gate.
    expected["continuity_receipt"].update(
        prior_eligible_overrides=107, prior_excluded_overrides=152
    )
    repair.require_expected_repair_preview(**expected)
    for field, value in (
        ("work_count", 47694),
        ("map_point_count", 42870),
        ("new_embedding_count", 1),
        ("layout_reused", False),
    ):
        changed = {**expected, field: value}
        with pytest.raises(RuntimeError, match="map points, embeddings, or layout"):
            repair.require_expected_repair_preview(**changed)
    for key in required_keys:
        changed = reviewed_preview()
        changed["continuity_receipt"][key] = "different"
        with pytest.raises(RuntimeError, match="reviewed region replay"):
            repair.require_expected_repair_preview(**changed)


def test_repair_requires_all_historical_region_decisions(monkeypatch):
    monkeypatch.setattr(repair, "HISTORICAL_WORK_COUNT", 2)
    historical = [
        {
            "work_id": "work-visible",
            "map_eligible": True,
            "map_region_outlier": False,
            "map_exclusion_reasons": [],
        },
        {
            "work_id": "work-hidden",
            "map_eligible": False,
            "map_region_outlier": True,
            "map_exclusion_reasons": ["missing_source", "low_information_region"],
        },
    ]
    final = pandas.DataFrame([*historical, {**historical[0], "work_id": "new-work"}])
    repair.require_historical_region_decisions(historical, final)
    for field, changed_value in (
        ("map_eligible", True),
        ("map_region_outlier", False),
        ("map_exclusion_reasons", ["low_information_region", "missing_source"]),
    ):
        changed = final.copy(deep=True)
        changed.at[1, field] = changed_value
        with pytest.raises(RuntimeError, match="changed historical region decisions"):
            repair.require_historical_region_decisions(historical, changed)
    with pytest.raises(RuntimeError, match="changed historical region decisions"):
        repair.require_historical_region_decisions(historical, final.iloc[[0, 2]])


def test_repair_rejects_incomplete_or_ambiguous_historical_decisions(monkeypatch):
    monkeypatch.setattr(repair, "HISTORICAL_WORK_COUNT", 2)
    row = {
        "work_id": "work-one",
        "map_eligible": False,
        "map_region_outlier": True,
        "map_exclusion_reasons": ["low_information_region"],
    }
    final = pandas.DataFrame([row])
    with pytest.raises(RuntimeError, match="complete historical work set"):
        repair.require_historical_region_decisions([row], final)
    with pytest.raises(RuntimeError, match="duplicate historical work IDs"):
        repair.require_historical_region_decisions([row, row], final)
    with pytest.raises(RuntimeError, match="lacks historical region decisions"):
        repair.require_historical_region_decisions(
            [row, {"work_id": "work-two", "map_eligible": False}], final
        )
    with pytest.raises(RuntimeError, match="lacks final region decision fields"):
        repair.require_historical_region_decisions(
            [row, {**row, "work_id": "work-two"}], final.drop(columns="map_eligible")
        )
    with pytest.raises(RuntimeError, match="duplicate final work IDs"):
        repair.require_historical_region_decisions(
            [row, {**row, "work_id": "work-two"}], pandas.concat([final, final])
        )


def test_repair_rejects_null_flags_but_accepts_numpy_booleans(monkeypatch):
    monkeypatch.setattr(repair, "HISTORICAL_WORK_COUNT", 1)
    row = {
        "work_id": "work-one",
        "map_eligible": False,
        "map_region_outlier": True,
        "map_exclusion_reasons": ["low_information_region"],
    }
    final = pandas.DataFrame([row])
    repair.require_historical_region_decisions(
        [{**row, "map_eligible": numpy.bool_(False)}], final
    )
    for bad in (None, float("nan")):
        with pytest.raises(RuntimeError, match="lacks historical region decisions"):
            repair.require_historical_region_decisions(
                [{**row, "map_eligible": bad}], final
            )
        with pytest.raises(RuntimeError, match="lacks historical region decisions"):
            repair.require_historical_region_decisions(
                [row], pandas.DataFrame([{**row, "map_region_outlier": bad}])
            )


def test_cli_emits_selected_revision(tmp_path, monkeypatch, capsys):
    kwargs = repair_kwargs(tmp_path, monkeypatch)
    repair.main(
        [
            "--restore-prior-regions",
            "true",
            "--event-name",
            kwargs["event_name"],
            "--run-ref",
            kwargs["run_ref"],
            "--run-commit",
            kwargs["run_commit"],
            "--producer-commit",
            kwargs["producer_commit"],
            "--snapshot-ref",
            kwargs["snapshot_ref"],
            "--snapshot-commit",
            kwargs["snapshot_commit"],
            "--historical-commit",
            kwargs["historical_commit"],
            "--snapshot",
            str(kwargs["snapshot_path"]),
            "--manifest",
            str(kwargs["manifest_path"]),
            "--current-receipt",
            str(kwargs["current_receipt_path"]),
            "--historical-receipt",
            str(kwargs["historical_receipt_path"]),
            "--current-revision",
            kwargs["current_revision"],
        ]
    )
    assert capsys.readouterr().out == (
        f"revision={repair.REGION_COMPARISON_REVISION}\n"
    )


def test_module_entrypoint_defaults_to_current_hub_revision(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m map_of_research.region_repair",
            "--restore-prior-regions",
            "false",
            "--event-name",
            "push",
            "--run-ref",
            "refs/heads/automation/map-snapshot",
            "--run-commit",
            "unused",
            "--producer-commit",
            "unused",
            "--snapshot-ref",
            "unused",
            "--snapshot-commit",
            "unused",
            "--historical-commit",
            "unused",
            "--snapshot",
            str(tmp_path / "missing.parquet"),
            "--manifest",
            str(tmp_path / "missing.json"),
            "--current-receipt",
            str(tmp_path / "missing-current.json"),
            "--historical-receipt",
            str(tmp_path / "missing-historical.json"),
            "--current-revision",
            "c" * 40,
        ],
    )
    with monkeypatch.context() as patch:
        patch.delitem(sys.modules, "map_of_research.region_repair")
        runpy.run_module("map_of_research.region_repair", run_name="__main__")
    assert capsys.readouterr().out == f"revision={'c' * 40}\n"
