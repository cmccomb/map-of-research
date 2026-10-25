"""Keep Simone's author-site bibliography one complete, typed source batch."""

import datetime as dt
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from map_of_research.dataset import _curated_publication_type, build_dataset_tables
from map_of_research.registry import load_registry
from map_of_research.snapshot import build_snapshot, validate_snapshot

ROOT = Path(__file__).resolve().parents[1]
PERSON_ID = "person-jonathansimone-9415e63e"
SOURCE_COMMIT = "3efca3dfe67eecc89884bc82b2e90521a1657f86"


def test_simone_complete_batch_builds_eleven_distinct_mapped_works(tmp_path):
    bundle = ROOT / "data/source_reviews/jonathan-simone-2026-10-25"
    candidate_path = bundle / "candidate-11.json"
    candidate = json.loads(candidate_path.read_text())
    source_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    curated_path = ROOT / "data/curated/works.json"
    curated = json.loads(curated_path.read_text())
    batch = [
        row
        for row in curated["works"]
        if row.get("author_site_source_batch") == "jonathan-simone-2026-10-25"
    ]
    assert len(candidate["works"]) == len(batch) == 11
    assert {row["arxiv_id"] for row in batch if row["arxiv_id"]} == {
        row["arxiv_id"] for row in candidate["works"] if "arxiv_id" in row
    }
    assert all(row["person_id"] == PERSON_ID for row in batch)
    assert all(row["author_site_source_commit"] == SOURCE_COMMIT for row in batch)
    assert all(row["author_site_candidate_sha256"] == source_hash for row in batch)
    assert not any("X- and H-plumbings" in row["title"] for row in batch)
    assert (
        sum(row.get("publication_type") == "doctoral_dissertation" for row in batch)
        == 1
    )

    cache_dir = tmp_path / "empty-cache"
    cache_dir.mkdir()
    snapshot_path = tmp_path / "source.parquet"
    manifest_path = tmp_path / "source.manifest.json"
    now = dt.datetime(2026, 10, 25, 14, 0, tzinfo=dt.UTC)
    build_snapshot(
        cache_dir=cache_dir,
        curated_path=curated_path,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        now=now,
    )
    frame, manifest = validate_snapshot(
        snapshot_path, manifest_path, curated_path=curated_path, now=now
    )
    assert manifest["curated_work_rows"] == len(curated["works"])
    simone = frame.loc[frame["person_id"] == PERSON_ID].copy()
    assert len(simone) == 11
    assert simone["source_type"].eq("curated_verified").all()
    assert simone["scholar_id"].eq("").all()
    simone["embedding"] = [[0.0, 1.0] for _ in range(len(simone))]
    registry = load_registry()
    tables = build_dataset_tables(simone, registry)
    works = tables["works"]
    authorships = tables["authorships"]
    assert len(tables["profile_publications"]) == len(authorships) == 11
    observations = tables["profile_publications"]
    assert observations["source_type"].eq("curated_verified").all()
    assert observations["scholar_id"].eq("").all()
    assert observations["person_id"].eq(PERSON_ID).all()
    assert (
        observations["source_record_json"]
        .map(
            lambda value: (
                json.loads(value)["author_site_source_commit"] == SOURCE_COMMIT
            )
        )
        .all()
    )
    assert len(works) == 11
    assert works["map_eligible"].all()
    assert works["work_id"].nunique() == 11
    assert works["source_types"].map(lambda value: value == ["curated_verified"]).all()
    assert works["person_ids"].map(lambda value: value == [PERSON_ID]).all()
    assert works["department_ids"].map(lambda value: value == ["math"]).all()
    assert works["school_ids"].map(lambda value: value == ["mcs"]).all()
    assert works["external_coauthor_ids"].map(lambda value: value == []).all()
    assert set(authorships["person_id"]) == {PERSON_ID}
    assert set(authorships["work_id"]) == set(works["work_id"])
    assert authorships["department_ids"].map(lambda value: value == ["math"]).all()
    assert authorships["school_ids"].map(lambda value: value == ["mcs"]).all()
    thesis = works.loc[works["publication_type"] == "doctoral_dissertation"]
    assert len(thesis) == 1
    assert thesis.iloc[0]["doi"] == "10.18130/v39z90b6j"
    assert not thesis.iloc[0]["collaboration_eligible"]
    papers = works.loc[works["publication_type"] == "publication"]
    assert len(papers) == 10
    assert papers["collaboration_eligible"].all()


@pytest.mark.parametrize(
    ("source_record", "message"),
    [
        ("not json", "source record"),
        ("[]", "must be an object"),
        ('{"publication_type":"invented_type"}', "unknown publication type"),
    ],
)
def test_curated_publication_type_rejects_bad_source_metadata(source_record, message):
    row = SimpleNamespace(
        source_type="curated_verified", source_record_json=source_record
    )
    with pytest.raises(ValueError, match=message):
        _curated_publication_type(row)


def test_scholar_row_does_not_inherit_curated_publication_type():
    row = SimpleNamespace(source_type="scholar", source_record_json="not json")
    assert _curated_publication_type(row) == "publication"
