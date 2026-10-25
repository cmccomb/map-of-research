"""The first failed upload must not erase the prior successful release."""

from __future__ import annotations

import json
import runpy
import sys

import pytest

from map_of_research.release_baseline import pin_prior_release

REVISION = "a" * 40


def test_first_upload_pins_success_before_a_failure_and_retry(
    tmp_path, monkeypatch, capsys
):
    receipt = {"status": "success", "artifact_commit": REVISION}
    latest = tmp_path / "last-upload.json"
    latest.write_text(json.dumps(receipt) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["python -m map_of_research.release_baseline", "--status-dir", str(tmp_path)],
    )
    with monkeypatch.context() as patch:
        patch.delitem(sys.modules, "map_of_research.release_baseline")
        runpy.run_module("map_of_research.release_baseline", run_name="__main__")
    assert capsys.readouterr().out == f"revision={REVISION}\n"
    saved = tmp_path / "last-successful-upload.json"
    assert saved.read_bytes() == latest.read_bytes()

    latest.write_text(json.dumps({"status": "failure"}) + "\n")
    assert pin_prior_release(tmp_path) == REVISION
    assert json.loads(saved.read_text()) == receipt


@pytest.mark.parametrize(
    "receipt",
    [
        {"status": "failure", "artifact_commit": REVISION},
        {"status": "success", "artifact_commit": "main"},
        {"status": "success"},
    ],
)
def test_unvalidated_receipt_cannot_be_pinned(tmp_path, receipt):
    (tmp_path / "last-upload.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        pin_prior_release(tmp_path)
    assert not (tmp_path / "last-successful-upload.json").exists()


def test_invalid_saved_success_is_not_replaced_by_later_receipt(tmp_path):
    (tmp_path / "last-successful-upload.json").write_text(
        json.dumps({"status": "failure", "artifact_commit": REVISION})
    )
    (tmp_path / "last-upload.json").write_text(
        json.dumps({"status": "success", "artifact_commit": "b" * 40})
    )
    with pytest.raises(ValueError, match="not successful"):
        pin_prior_release(tmp_path)


def test_missing_previous_receipt_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        pin_prior_release(tmp_path)
