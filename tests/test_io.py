import hashlib

import pytest

import map_of_research.io as io


def test_json_round_trip_and_streaming_digest(tmp_path) -> None:
    path = tmp_path / "nested" / "state.json"
    io.atomic_write_json(path, {"message": "café", "count": 2})
    assert io.load_json(path) == {"count": 2, "message": "café"}
    assert io.sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_atomic_write_removes_temporary_file_after_failure(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "state.json"

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(io.json, "dump", fail)
    with pytest.raises(OSError, match="disk full"):
        io.atomic_write_json(path, {})

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
