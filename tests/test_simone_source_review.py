"""Regression checks for the whole-site Simone source capture."""

import hashlib
import json
import shutil

import pytest

from scripts.validate_simone_source_review import DEFAULT_BUNDLE, validate_bundle


def test_simone_candidate_covers_all_author_entries_and_exact_arxiv_records():
    assert validate_bundle() == {
        "public_families": 11,
        "author_page_entries": 10,
        "arxiv_entries": 10,
        "journal_dois": 8,
        "theses": 1,
        "held_in_preparation": 1,
    }


def test_missing_family_fails_even_with_rehashed_candidate(tmp_path):
    bundle = tmp_path / "simone"
    shutil.copytree(DEFAULT_BUNDLE, bundle)
    path = bundle / "candidate-11.json"
    candidate = json.loads(path.read_text())
    candidate["works"].pop(0)
    path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="11 public work families"):
        validate_bundle(bundle)
