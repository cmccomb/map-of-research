"""Exercise the collection workflow's actual Git staging helper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/stage_collector_result.py"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write(repo: Path, relative: str, content: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "snapshot"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    _write(repo, "registry/people.csv", "old registry\n")
    _write(repo, "data/authors/alpha.json", "old cache\n")
    _write(repo, "status/authors.json", "old state\n")
    _write(repo, "status/last-collection.json", '{"schema_version":1,"status":"old"}\n')
    _write(repo, "snapshots/cmu-engineering-publications.parquet", "old snapshot\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "seed")
    _write(repo, "registry/people.csv", "new registry\n")
    _write(repo, "data/authors/alpha.json", "partial cache\n")
    _write(repo, "status/authors.json", "new state\n")
    _write(
        repo,
        "status/last-collection.json",
        '{"schema_version":1,"status":"success","refreshed_profile_ids":["alpha"]}\n',
    )
    _write(repo, "snapshots/cmu-engineering-publications.parquet", "new snapshot\n")
    return repo


def _stage(repo: Path, exit_code: str, snapshot_outcome: str) -> dict:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--collector-exit-code",
            exit_code,
            "--snapshot-outcome",
            snapshot_outcome,
        ],
        cwd=repo,
        check=True,
    )
    return json.loads((repo / "status/last-collection.json").read_text())


@pytest.mark.parametrize(
    ("exit_code", "snapshot_outcome", "reason"),
    [
        ("0", "failure", "source_or_snapshot_validation_failed"),
        ("1", "skipped", "collector_failed"),
        ("", "skipped", "collector_failed"),
    ],
)
def test_failed_collection_stages_only_failure_receipt(
    tmp_path: Path, exit_code: str, snapshot_outcome: str, reason: str
) -> None:
    repo = _repo(tmp_path)
    status = _stage(repo, exit_code, snapshot_outcome)
    assert (
        _git(repo, "diff", "--cached", "--name-only") == "status/last-collection.json"
    )
    assert status["snapshot_validation"] == {"status": "failure", "reason": reason}
    assert "data/authors/alpha.json" in _git(repo, "diff", "--name-only")
    assert "snapshots/cmu-engineering-publications.parquet" in _git(
        repo, "diff", "--name-only"
    )


def test_success_stages_validated_snapshot_and_cache(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    status = _stage(repo, "0", "success")
    assert status["snapshot_validation"] == {"status": "success"}
    assert _git(repo, "diff", "--cached", "--name-only").splitlines() == [
        "data/authors/alpha.json",
        "registry/people.csv",
        "snapshots/cmu-engineering-publications.parquet",
        "status/authors.json",
        "status/last-collection.json",
    ]


def test_noop_does_not_stage_unvalidated_data(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    status = _stage(repo, "0", "skipped")
    assert status["snapshot_validation"] == {"status": "skipped"}
    assert (
        _git(repo, "diff", "--cached", "--name-only") == "status/last-collection.json"
    )


def test_missing_collector_status_still_records_failure(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "status/last-collection.json").unlink()
    status = _stage(repo, "1", "skipped")
    assert status == {
        "schema_version": 1,
        "status": "failure",
        "snapshot_validation": {"status": "failure", "reason": "collector_failed"},
    }
    assert (
        _git(repo, "diff", "--cached", "--name-only") == "status/last-collection.json"
    )
