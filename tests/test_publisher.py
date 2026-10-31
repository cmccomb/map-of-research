import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy
import pandas
import pytest

import map_of_research.publisher as publisher
from map_of_research.composite_region import (
    BASELINE_REVISION as SULLIVAN_BASELINE_REVISION,
)
from map_of_research.composite_region import (
    SOURCE_COMMIT as SULLIVAN_SOURCE_COMMIT,
)
from map_of_research.composite_region import (
    SOURCE_SHA256 as SULLIVAN_SOURCE_SHA256,
)
from map_of_research.composite_region import load_reviewed_sullivan_embeddings
from map_of_research.presentation_drift import (
    PresentationDriftError,
    compare_presentation,
    preserve_reviewed_additive_presentation,
    require_preserved_additive_artifact,
)
from map_of_research.public_subsets import ids_sha256
from map_of_research.region_continuity import RegionReleaseDiffError
from map_of_research.registry import Department, Membership, Person, Registry, School


@pytest.fixture(autouse=True)
def isolate_generic_publisher_tests_from_reviewed_person_holds(monkeypatch) -> None:
    """These synthetic release fixtures do not include the held Math person."""

    monkeypatch.setattr(publisher, "load_person_publication_holds", lambda path: ())


class Rows(list):
    pass


def test_pending_generic_region_protection_is_manifest_source_bound(tmp_path) -> None:
    profiles = pandas.DataFrame(
        [
            {
                "author_pub_id": "owner:new",
                "scholar_id": "owner",
                "work_id": "work-new",
                "source_type": "scholar",
                "publication_type": "publication",
                "work_match_method": "profile_publication",
                "doi": "",
                "collaboration_eligible": False,
                "map_eligible": True,
            },
            {
                "author_pub_id": "owner:prior",
                "scholar_id": "owner",
                "work_id": "work-prior",
                "source_type": "scholar",
                "publication_type": "publication",
                "work_match_method": "profile_publication",
                "doi": "",
                "collaboration_eligible": False,
                "map_eligible": True,
            },
            {
                "author_pub_id": "owner:authored",
                "scholar_id": "owner",
                "work_id": "work-authored",
                "source_type": "scholar",
                "publication_type": "conference_paper",
                "work_match_method": "reviewed_context_doi",
                "doi": "10.1000/authored",
                "collaboration_eligible": True,
                "map_eligible": True,
            },
            {
                "author_pub_id": "owner:unreviewed",
                "scholar_id": "owner",
                "work_id": "work-unreviewed",
                "source_type": "scholar",
                "publication_type": "publication",
                "work_match_method": "profile_publication",
                "doi": "",
                "collaboration_eligible": False,
                "map_eligible": True,
            },
        ]
    )
    works = pandas.DataFrame(
        [
            {
                "work_id": work_id,
                "publication_type": publication_type,
                "doi": doi,
                "collaboration_eligible": collaboration,
                "map_eligible": True,
            }
            for work_id, publication_type, doi, collaboration in (
                ("work-new", "publication", "", False),
                ("work-prior", "publication", "", False),
                ("work-authored", "conference_paper", "10.1000/authored", True),
                ("work-unreviewed", "publication", "", False),
            )
        ]
    )
    contexts = {
        ("owner", "owner:new"): SimpleNamespace(publication_type="publication"),
        ("owner", "owner:prior"): SimpleNamespace(publication_type="publication"),
        ("owner", "owner:authored"): SimpleNamespace(
            publication_type="conference_paper"
        ),
    }
    review = {
        "prior_revision": "a" * 40,
        "source_snapshot_sha256": "b" * 64,
        "selected_ids_sha256": ids_sha256(set(profiles["author_pub_id"])),
        "new_points": [
            {
                "work_id": "work-new",
                "decision": "hold_pending_placement",
                "coordinates": None,
            },
            {
                "work_id": "work-authored",
                "decision": "hold_pending_placement",
                "coordinates": None,
            },
            {
                "work_id": "work-unreviewed",
                "decision": "hold_pending_placement",
                "coordinates": None,
            },
        ],
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))

    protected = publisher._reviewed_new_generic_placement_work_ids(
        profiles,
        works,
        contexts,
        [{"work_id": "work-prior"}],
        path,
        prior_revision="a" * 40,
        source_snapshot_sha256="b" * 64,
    )
    assert protected == frozenset({"work-new"})

    missing_generic_review = dict(review)
    missing_generic_review["new_points"] = [
        item for item in review["new_points"] if item["work_id"] != "work-new"
    ]
    path.write_text(json.dumps(missing_generic_review))
    with pytest.raises(ValueError, match="lacks all-held placement review"):
        publisher._reviewed_new_generic_placement_work_ids(
            profiles,
            works,
            contexts,
            [{"work_id": "work-prior"}],
            path,
            prior_revision="a" * 40,
            source_snapshot_sha256="b" * 64,
        )

    work_type_drift = works.copy()
    work_type_drift.loc[
        work_type_drift["work_id"] == "work-new", "publication_type"
    ] = "conference_paper"
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="ineligible canonical work"):
        publisher._reviewed_new_generic_placement_work_ids(
            profiles,
            work_type_drift,
            contexts,
            [{"work_id": "work-prior"}],
            path,
            prior_revision="a" * 40,
            source_snapshot_sha256="b" * 64,
        )

    review["prior_revision"] = "c" * 40
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="source or parent changed"):
        publisher._reviewed_new_generic_placement_work_ids(
            profiles,
            works,
            contexts,
            [{"work_id": "work-prior"}],
            path,
            prior_revision="a" * 40,
            source_snapshot_sha256="b" * 64,
        )

    review["prior_revision"] = "a" * 40
    path.write_text(json.dumps(review))
    ineligible = profiles.copy()
    ineligible.loc[ineligible["work_id"] == "work-new", "map_eligible"] = False
    with pytest.raises(ValueError, match="ineligible source work"):
        publisher._reviewed_new_generic_placement_work_ids(
            ineligible,
            works,
            contexts,
            [{"work_id": "work-prior"}],
            path,
            prior_revision="a" * 40,
            source_snapshot_sha256="b" * 64,
        )

    wrong_source_type = profiles.copy()
    wrong_source_type.loc[wrong_source_type["work_id"] == "work-new", "doi"] = (
        "10.1000/unreviewed"
    )
    with pytest.raises(ValueError, match="ineligible source work"):
        publisher._reviewed_new_generic_placement_work_ids(
            wrong_source_type,
            works,
            contexts,
            [{"work_id": "work-prior"}],
            path,
            prior_revision="a" * 40,
            source_snapshot_sha256="b" * 64,
        )

    path.write_text(json.dumps(missing_generic_review))
    assert (
        publisher._reviewed_new_generic_placement_work_ids(
            ineligible,
            works,
            contexts,
            [{"work_id": "work-prior"}],
            path,
            prior_revision="a" * 40,
            source_snapshot_sha256="b" * 64,
        )
        == frozenset()
    )
    path.write_text(json.dumps(review))

    with pytest.raises(ValueError, match="lacks work eligibility fields"):
        publisher._reviewed_new_generic_placement_work_ids(
            profiles,
            works.drop(columns=["doi"]),
            contexts,
            [{"work_id": "work-prior"}],
            path,
            prior_revision="a" * 40,
            source_snapshot_sha256="b" * 64,
        )
    duplicate_review = dict(review)
    duplicate_review["new_points"] = [*review["new_points"], review["new_points"][0]]
    path.write_text(json.dumps(duplicate_review))
    with pytest.raises(ValueError, match="stale or duplicate work IDs"):
        publisher._reviewed_new_generic_placement_work_ids(
            profiles,
            works,
            contexts,
            [{"work_id": "work-prior"}],
            path,
            prior_revision="a" * 40,
            source_snapshot_sha256="b" * 64,
        )
    path.write_text("{}")
    with pytest.raises(ValueError, match="lacks exact new work decisions"):
        publisher._reviewed_new_generic_placement_work_ids(
            profiles,
            works,
            contexts,
            [{"work_id": "work-prior"}],
            path,
            prior_revision="a" * 40,
            source_snapshot_sha256="b" * 64,
        )


def install_published_rows(monkeypatch, loader) -> None:
    monkeypatch.setattr(publisher, "_published_rows", loader)


def stub_before_physics_review_gate(monkeypatch, frame) -> None:
    """Keep synthetic one-column frames focused on the post-hold review gate."""

    monkeypatch.setattr(publisher, "load_corrections", lambda path: {})
    monkeypatch.setattr(publisher, "apply_corrections", lambda frame, _: (frame, 0))
    monkeypatch.setattr(publisher, "load_contexts", lambda path: {})
    monkeypatch.setattr(
        publisher,
        "load_attribution_reviews",
        lambda path: SimpleNamespace(confirmed={}),
    )
    monkeypatch.setattr(publisher, "load_profile_releases", lambda path: ())
    monkeypatch.setattr(publisher, "load_provisional_holds", lambda path: ())
    monkeypatch.setattr(
        publisher,
        "select_released_observations",
        lambda *args, **kwargs: (frame, [], set()),
    )


def test_published_rows_ignore_extra_parquet_columns_and_read_all_shards(
    monkeypatch,
    tmp_path,
) -> None:
    directory = tmp_path / "profile_publications"
    directory.mkdir()
    for index in range(2):
        pandas.DataFrame(
            [
                {
                    "author_pub_id": f"author:{index}",
                    "title": f"Title {index}",
                    "embedding": [float(index)] * publisher.EMBEDDING_DIMENSION,
                    "newer_column": "not in older Hub feature metadata",
                }
            ]
        ).to_parquet(directory / f"train-{index:05d}-of-00002.parquet", index=False)
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    rows = list(
        publisher._published_rows(
            "profile_publications",
            ("author_pub_id", "title", "embedding"),
            hf_token="token",
        )
    )
    assert [row["author_pub_id"] for row in rows] == ["author:0", "author:1"]
    assert all("newer_column" not in row for row in rows)
    assert calls == [
        (
            publisher.REPO_ID,
            {
                "repo_type": "dataset",
                "allow_patterns": "profile_publications/train-*.parquet",
                "token": "token",
            },
        )
    ]
    list(
        publisher._published_rows(
            "profile_publications",
            ("author_pub_id", "title", "embedding"),
            hf_token="token",
            revision="a" * 40,
        )
    )
    assert calls[-1][1]["revision"] == "a" * 40
    assert len(publisher._existing_embeddings(hf_token="token")) == 2
    assert len(publisher._existing_embeddings(hf_token="token", revision="a" * 40)) == 2
    assert calls[-1][1]["revision"] == "a" * 40

    works_directory = tmp_path / "works"
    works_directory.mkdir()
    pandas.DataFrame(
        [
            {
                "work_id": "work-one",
                "layout_version": publisher.LAYOUT_VERSION,
                "x": 0.1,
                "y": 0.2,
                "tsne_x": 0.3,
                "tsne_y": 0.4,
                "newer_column": "not in older Hub feature metadata",
            }
        ]
    ).to_parquet(works_directory / "train-00000-of-00001.parquet", index=False)
    assert publisher._existing_layouts(hf_token="token") == {
        "work-one": {"x": 0.1, "y": 0.2, "tsne_x": 0.3, "tsne_y": 0.4}
    }
    assert publisher._existing_layouts(hf_token="token", revision="a" * 40)
    assert calls[-1][1]["revision"] == "a" * 40


def test_published_rows_skip_unusable_or_missing_parquet_config(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda *args, **kwargs: str(tmp_path)),
    )
    with pytest.raises(FileNotFoundError, match="default"):
        list(publisher._published_rows("default", ("embedding",), hf_token=""))

    directory = tmp_path / "works"
    directory.mkdir()
    pandas.DataFrame([{"work_id": "one"}]).to_parquet(
        directory / "train-00000-of-00001.parquet", index=False
    )
    assert (
        list(
            publisher._published_rows(
                "works", ("work_id", "layout_version"), hf_token=""
            )
        )
        == []
    )


def test_pinned_dataset_revision_must_still_be_hub_head(monkeypatch) -> None:
    class HfApi:
        def __init__(self, token):
            assert token == "token"

        def repo_info(self, repo_id, repo_type):
            assert (repo_id, repo_type) == (publisher.REPO_ID, "dataset")
            return SimpleNamespace(sha="a" * 40)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=HfApi))
    publisher._require_current_dataset_revision(
        hf_token="token", prior_revision="a" * 40
    )
    with pytest.raises(RuntimeError, match="moved after preview"):
        publisher._require_current_dataset_revision(
            hf_token="token", prior_revision="b" * 40
        )


def test_embedding_refresh_reuses_existing_vectors(monkeypatch) -> None:
    frame = pandas.DataFrame(
        [
            {"author_pub_id": "a:one", "title": "Existing"},
            {"author_pub_id": "b:two", "title": "New"},
        ]
    )
    monkeypatch.setattr(
        publisher,
        "_existing_embeddings",
        lambda **_: {("a:one", "Existing"): [0.0] * 768},
    )

    class Encoder:
        def encode(self, titles, **kwargs):
            assert titles == ["New"]
            assert kwargs["normalize_embeddings"] is True
            return numpy.ones((1, 768), dtype=numpy.float32)

    enriched, new_count = publisher.add_embeddings(
        frame,
        hf_token="test",
        encoder=Encoder(),
    )

    assert new_count == 1
    assert enriched.loc[0, "embedding"] == [0.0] * 768
    assert enriched.loc[1, "embedding"] == [1.0] * 768


def test_reviewed_sullivan_vectors_are_exact_despite_encoder_and_row_order(
    monkeypatch,
) -> None:
    curated = Path(__file__).resolve().parents[1] / "data/curated"
    vectors = load_reviewed_sullivan_embeddings(
        curated,
        source_commit=SULLIVAN_SOURCE_COMMIT,
        source_sha256=SULLIVAN_SOURCE_SHA256,
        hub_parent=SULLIVAN_BASELINE_REVISION,
    )
    assert len(vectors) == 70
    monkeypatch.setattr(publisher, "_existing_embeddings", lambda **_: {})

    class PerturbedEncoder:
        def encode(self, titles, **kwargs):
            raise AssertionError("Reviewed Sullivan vectors must never be re-encoded")

    keys = list(vectors)
    results = []
    for ordered in (keys, list(reversed(keys))):
        frame = pandas.DataFrame(
            [
                {"author_pub_id": author_pub_id, "title": title}
                for author_pub_id, title in ordered
            ]
        )
        output, new_count = publisher.add_embeddings(
            frame,
            hf_token="test",
            encoder=PerturbedEncoder(),
            reviewed_new_vectors=vectors,
        )
        assert new_count == 70
        results.append(
            {
                (row.author_pub_id, row.title): row.embedding
                for row in output.itertuples(index=False)
            }
        )
    assert results[0] == results[1] == vectors


def test_reviewed_sullivan_vectors_reject_unknown_new_title(monkeypatch) -> None:
    monkeypatch.setattr(publisher, "_existing_embeddings", lambda **_: {})
    frame = pandas.DataFrame([{"author_pub_id": "new:one", "title": "Unreviewed"}])
    with pytest.raises(ValueError, match="differ from reviewed Sullivan"):
        publisher.add_embeddings(
            frame,
            hf_token="test",
            encoder=SimpleNamespace(encode=lambda *args, **kwargs: None),
            reviewed_new_vectors={},
        )


def test_reviewed_sullivan_vectors_reject_wrong_source_and_fixture_bytes(
    tmp_path,
) -> None:
    curated = Path(__file__).resolve().parents[1] / "data/curated"
    kwargs = {
        "source_commit": SULLIVAN_SOURCE_COMMIT,
        "source_sha256": SULLIVAN_SOURCE_SHA256,
        "hub_parent": SULLIVAN_BASELINE_REVISION,
    }
    with pytest.raises(ValueError, match="provenance changed"):
        load_reviewed_sullivan_embeddings(
            curated, **{**kwargs, "source_commit": "wrong source"}
        )
    fixture = curated / "sullivan-reviewed-new-embeddings.json"
    altered = tmp_path / fixture.name
    altered.write_bytes(fixture.read_bytes() + b" ")
    with pytest.raises(ValueError, match="fixture bytes changed"):
        load_reviewed_sullivan_embeddings(tmp_path, **kwargs)


def test_list_normalization_handles_scalar_tuple_and_array() -> None:
    assert publisher._as_list(None) == []
    assert publisher._as_list(["a"]) == ["a"]
    assert publisher._as_list(("a", "b")) == ["a", "b"]
    assert publisher._as_list(numpy.asarray(["a", "b"])) == ["a", "b"]
    assert publisher._as_list(numpy.asarray(3)) == [3]
    assert publisher._as_list("a") == ["a"]


def test_existing_embeddings_searches_compatible_configs(monkeypatch) -> None:
    calls = []

    def published_rows(name, columns, *, hf_token):
        calls.append((name, columns, hf_token))
        if name == "profile_publications":
            raise FileNotFoundError
        if name == "publications":
            return Rows([{"title": "No vectors"}])
        return Rows(
            [
                {
                    "author_pub_id": "a:one",
                    "title": "Reusable",
                    "embedding": [1] * publisher.EMBEDDING_DIMENSION,
                },
                {"author_pub_id": "", "title": "Ignored", "embedding": []},
                {"author_pub_id": "a:short", "title": "Short", "embedding": [1]},
            ]
        )

    install_published_rows(monkeypatch, published_rows)

    assert publisher._existing_embeddings(hf_token="token") == {
        ("a:one", "Reusable"): [1.0] * publisher.EMBEDDING_DIMENSION
    }
    assert [call[0] for call in calls] == list(publisher.REUSABLE_EMBEDDING_CONFIGS)
    assert calls[0][1] == ("author_pub_id", "title", "embedding")
    assert calls[0][2] == "token"


def test_existing_embeddings_returns_empty_when_no_vectors_exist(monkeypatch) -> None:
    invalid = Rows([{"author_pub_id": "", "title": "", "embedding": []}])
    install_published_rows(monkeypatch, lambda *args, **kwargs: invalid)
    assert publisher._existing_embeddings(hf_token="") == {}


def test_existing_embeddings_stop_on_unexpected_read_failure(monkeypatch) -> None:
    install_published_rows(
        monkeypatch,
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("corrupt shard")),
    )
    with pytest.raises(RuntimeError, match="Could not read reusable"):
        publisher._existing_embeddings(hf_token="")


@pytest.mark.parametrize(
    ("vector_shape", "message"),
    [((2, publisher.EMBEDDING_DIMENSION), "row count"), ((1, 3), "dimension")],
)
def test_embedding_refresh_rejects_invalid_encoder_output(
    monkeypatch,
    vector_shape,
    message,
) -> None:
    monkeypatch.setattr(publisher, "_existing_embeddings", lambda **_: {})
    frame = pandas.DataFrame([{"author_pub_id": "a:one", "title": "One"}])
    encoder = SimpleNamespace(encode=lambda *args, **kwargs: numpy.ones(vector_shape))
    with pytest.raises(RuntimeError, match=message):
        publisher.add_embeddings(frame, hf_token="token", encoder=encoder)


def test_embedding_refresh_constructs_the_pinned_encoder(monkeypatch) -> None:
    captured = {}

    class Encoder:
        def __init__(self, model_name, **kwargs):
            captured.update(model_name=model_name, **kwargs)

        def encode(self, titles, **kwargs):
            return numpy.zeros((len(titles), publisher.EMBEDDING_DIMENSION))

    monkeypatch.setattr(publisher, "_existing_embeddings", lambda **_: {})
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=Encoder),
    )
    frame = pandas.DataFrame([{"author_pub_id": "a:one", "title": "One"}])

    _, new_count = publisher.add_embeddings(frame, hf_token="token")

    assert new_count == 1
    assert captured == {
        "model_name": publisher.MODEL_NAME,
        "revision": publisher.MODEL_REVISION,
        "trust_remote_code": False,
    }


def test_embedding_refresh_does_nothing_when_every_vector_is_reusable(
    monkeypatch,
) -> None:
    vector = [0.0] * publisher.EMBEDDING_DIMENSION
    monkeypatch.setattr(
        publisher,
        "_existing_embeddings",
        lambda **_: {("a:one", "One"): vector},
    )
    frame = pandas.DataFrame([{"author_pub_id": "a:one", "title": "One"}])
    output, new_count = publisher.add_embeddings(frame, hf_token="token")
    assert new_count == 0
    assert output.loc[0, "embedding"] == vector


def test_embedding_refresh_reuses_exact_title_for_new_curated_observation(
    monkeypatch,
) -> None:
    vector = [0.25] * publisher.EMBEDDING_DIMENSION
    monkeypatch.setattr(
        publisher,
        "_existing_embeddings",
        lambda **_: {("scholar:old", "Same title"): vector},
    )
    frame = pandas.DataFrame([{"author_pub_id": "curated:new", "title": "Same title"}])
    output, new_count = publisher.add_embeddings(frame, hf_token="token")
    assert new_count == 0
    assert output.loc[0, "embedding"] == vector


def test_embedding_refresh_encodes_when_published_title_vectors_disagree(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        publisher,
        "_existing_embeddings",
        lambda **_: {
            ("one", "Same title"): [0.0] * publisher.EMBEDDING_DIMENSION,
            ("two", "Same title"): [1.0] * publisher.EMBEDDING_DIMENSION,
        },
    )
    frame = pandas.DataFrame([{"author_pub_id": "curated:new", "title": "Same title"}])
    encoder = SimpleNamespace(
        encode=lambda titles, **kwargs: numpy.full(
            (len(titles), publisher.EMBEDDING_DIMENSION), 0.5
        )
    )
    output, new_count = publisher.add_embeddings(
        frame, hf_token="token", encoder=encoder
    )
    assert new_count == 1
    assert output.loc[0, "embedding"] == [0.5] * publisher.EMBEDDING_DIMENSION


def work_frame() -> pandas.DataFrame:
    return pandas.DataFrame(
        [
            {
                "work_id": "work-one",
                "publication_type": "publication",
                "map_eligible": True,
                "map_exclusion_reasons": [],
                "map_region_audit_version": "reviewed-region-v1",
                "map_region_outlier": False,
                "map_region_documented_override": False,
                "map_region_prior_eligible_override": False,
                "map_region_prior_excluded_override": False,
                "keyword_model_version": publisher.KEYWORD_MODEL_VERSION,
                "topic_label_review_version": publisher.TOPIC_LABEL_REVIEW_VERSION,
                "layout_version": publisher.LAYOUT_VERSION,
                "keyword_id": "keyword-01",
                "keyword": "reliable systems",
                "keyword_extracted": "reliable systems",
                "keyword_label_reviewed": True,
                "detail_keyword_id": "keyword-01-01",
                "detail_keyword": "fault tolerant systems",
                "detail_keyword_extracted": "fault tolerant systems",
                "detail_keyword_label_reviewed": True,
                "memberships": [
                    {
                        "person_id": "person-one",
                        "display_name": "One Person",
                        "department_id": "ece",
                        "department_title": "Electrical & Computer Engineering",
                        "role": "faculty",
                    },
                    {
                        "person_id": "person-one",
                        "display_name": "One Person",
                        "department_id": "cmu-silicon-valley",
                        "department_title": "CMU Silicon Valley",
                        "role": "teaching",
                    },
                ],
                "x": 0.25,
                "y": -0.5,
                "tsne_x": -0.75,
                "tsne_y": 0.125,
                "title": "A Paper",
                "authors": "A. Author and B. Author",
                "year": 2025,
                "venue": "A Journal",
                "citation": "A Paper. A Journal (2025).",
                "citation_count": 7,
                "doi": "10.1000/test",
                "source_document_id": "",
                "source_verified_at": "2026-10-30",
                "collaboration_eligible": True,
                "display_contributors": "",
                "source_urls": ["https://example.test/paper"],
                "observation_count": 2,
                "last_fetched_at_utc": "2026-01-01T00:00:00+00:00",
            }
        ]
    )


def test_pending_map_schema_has_nonspatial_rows_without_changing_points(
    tmp_path: Path,
) -> None:
    mapped = work_frame()
    pending = mapped.copy(deep=True)
    pending.at[0, "work_id"] = "work-pending"
    pending.at[0, "map_eligible"] = False
    pending.at[0, "map_exclusion_reasons"] = ["pending_placement_review"]
    for field in ("x", "y", "tsne_x", "tsne_y"):
        pending.at[0, field] = None
    works = pandas.concat([mapped, pending], ignore_index=True)
    ordinary = publisher.build_map_artifact(
        works, registry(), generated_at_utc="2026-10-30T01:00:00+00:00"
    )
    selected_pending = [
        {
            "work_id": "work-pending",
            "title": "A reviewed paper",
            "year": 2026,
            "faculty_ids": ["person-one"],
            "source_url": "https://example.test/pending",
            "release_decision": "selected",
            "map_hold_reason": "pending_topic_review",
            "collaboration_eligible": True,
        }
    ]
    _, manifest_path, artifact = publisher._write_map_artifacts(
        works,
        registry(),
        maps_dir=tmp_path / "maps",
        generated_at_utc="2026-10-30T01:00:00+00:00",
        dataset_commits={},
        pending_publications=selected_pending,
    )
    manifest = json.loads(manifest_path.read_text())
    assert artifact["schema_version"] == manifest["schema_version"] == 8
    assert artifact["pending_publications"] == selected_pending
    assert artifact["pending_publication_count"] == 1
    assert manifest["artifact"]["pending_publication_count"] == 1
    assert artifact["points"] == ordinary["points"]
    assert artifact["keywords"] == ordinary["keywords"]


def test_additive_map_rebuild_preserves_parent_and_has_coherent_catalog(
    tmp_path: Path,
) -> None:
    old = work_frame()
    baseline = publisher.build_map_artifact(
        old, registry(), generated_at_utc="2026-10-30T01:00:00+00:00"
    )
    current = old.copy(deep=True)
    current.at[0, "source_urls"] = [
        "https://example.test/new-faculty-paper",
        "https://example.test/paper",
    ]
    current.at[0, "memberships"] = [
        *current.at[0, "memberships"],
        {
            "person_id": "person-two",
            "display_name": "Two Person",
            "department_id": "bme",
            "department_title": "Biomedical Engineering",
            "role": "emeritus",
        },
    ]
    added = old.copy(deep=True)
    added.at[0, "work_id"] = "work-added"
    added.at[0, "title"] = "A New Work"
    added.at[0, "x"] = added.at[0, "tsne_x"] = 0.9
    low_info = added.copy(deep=True)
    low_info.at[0, "work_id"] = "work-low-info"
    low_info.at[0, "map_eligible"] = False
    low_info.at[0, "map_exclusion_reasons"] = ["low_information_region"]
    current = pandas.concat([current, added, low_info], ignore_index=True)
    current["keyword"] = "refit label"
    current["keyword_extracted"] = "refit label"
    current["x"] = 0.9
    current["tsne_x"] = 0.9
    candidate = publisher.build_map_artifact(
        current, registry(), generated_at_utc="2026-10-30T02:00:00+00:00"
    )
    assert {point["work_id"] for point in candidate["points"]} == {
        "work-one",
        "work-added",
    }
    kwargs = {
        "prior_revision": "a" * 40,
        "source_snapshot_sha256": "s" * 64,
        "selected_ids_sha256": "i" * 64,
        "parent_map_sha256": "p" * 64,
        "reported_coauthor_source_sha256": "a" * 64,
    }
    with pytest.raises(PresentationDriftError) as error:
        preserve_reviewed_additive_presentation(
            current,
            old.to_dict("records"),
            baseline,
            candidate,
            review_path=None,
            **kwargs,
        )
    pending = error.value.receipt
    review = {
        "schema_version": 1,
        "decision": "freeze_retained_review_new",
        **{
            key: pending[key]
            for key in (
                "prior_revision",
                "source_snapshot_sha256",
                "selected_ids_sha256",
                "parent_map_sha256",
                "candidate_map_content_sha256",
                "new_work_ids_sha256",
                "reported_coauthor_source_sha256",
            )
        },
        "reviewed_at": "2026-10-30",
        "new_points": [
            {
                "work_id": "work-added",
                "decision": "place_unclassified",
                "coordinates": {"x": 0.1, "y": 0.2, "tsne_x": 0.3, "tsne_y": 0.4},
                "rationale": "Reviewed exact new work against parent placement.",
            }
        ],
    }
    review_path = tmp_path / "additive-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    candidate["generated_at_utc"] = "2026-10-30T03:00:00+00:00"
    missing_parent_link = current.copy(deep=True)
    missing_parent_link.at[0, "source_urls"] = [
        "https://example.test/new-faculty-paper"
    ]
    with pytest.raises(ValueError, match="source URL lacks provenance"):
        preserve_reviewed_additive_presentation(
            missing_parent_link,
            old.to_dict("records"),
            baseline,
            candidate,
            review_path=review_path,
            **kwargs,
        )
    with pytest.raises(ValueError, match="source URL lacks provenance"):
        preserve_reviewed_additive_presentation(
            current.drop(columns="source_urls"),
            old.to_dict("records"),
            baseline,
            candidate,
            review_path=review_path,
            **kwargs,
        )
    malformed_sources = current.copy(deep=True)
    malformed_sources.at[0, "source_urls"] = "https://example.test/paper"
    with pytest.raises(ValueError, match="lacks source URL list"):
        preserve_reviewed_additive_presentation(
            malformed_sources,
            old.to_dict("records"),
            baseline,
            candidate,
            review_path=review_path,
            **kwargs,
        )
    blank_parent_url = json.loads(json.dumps(baseline))
    blank_parent_url["points"][0]["source_url"] = ""
    with pytest.raises(ValueError, match="blank map source URL needs review"):
        preserve_reviewed_additive_presentation(
            current,
            old.to_dict("records"),
            blank_parent_url,
            candidate,
            review_path=review_path,
            **kwargs,
        )
    blank_source_list = current.copy(deep=True)
    blank_source_list.at[0, "source_urls"] = []
    blank_frozen, _ = preserve_reviewed_additive_presentation(
        blank_source_list,
        old.to_dict("records"),
        blank_parent_url,
        candidate,
        review_path=review_path,
        **kwargs,
    )
    assert (
        blank_frozen.loc[blank_frozen.work_id.eq("work-one"), "source_urls"].iloc[0]
        == []
    )
    frozen, receipt = preserve_reviewed_additive_presentation(
        current,
        old.to_dict("records"),
        baseline,
        candidate,
        review_path=review_path,
        **kwargs,
    )
    rebuilt = publisher.build_map_artifact(
        frozen, registry(), generated_at_utc="2026-10-30T04:00:00+00:00"
    )
    by_id = {point["work_id"]: point for point in rebuilt["points"]}
    assert set(by_id) == {"work-one", "work-added"}
    assert by_id["work-one"]["faculty_ids"] == ["person-one", "person-two"]
    assert by_id["work-one"]["source_url"] == baseline["points"][0]["source_url"]
    assert frozen.loc[frozen.work_id.eq("work-one"), "source_urls"].iloc[0] == [
        "https://example.test/paper",
        "https://example.test/new-faculty-paper",
    ]
    for field in ("x", "y", "tsne_x", "tsne_y", "keyword_ids"):
        assert by_id["work-one"][field] == baseline["points"][0][field]
    assert by_id["work-added"]["keyword_ids"] == [
        "keyword-unclassified",
        "keyword-unclassified-pending-review",
    ]
    assert by_id["work-added"]["x"] == 0.1
    assert receipt["placed_new_points"] == 1
    require_preserved_additive_artifact(baseline, rebuilt, receipt)
    assert (
        frozen.loc[frozen.work_id.eq("work-low-info"), "map_eligible"].eq(False).all()
    )
    assert (
        compare_presentation(
            baseline,
            rebuilt,
            prior_revision="a" * 40,
            source_snapshot_sha256="s" * 64,
            parent_map_sha256="p" * 64,
            candidate_map_sha256="c" * 64,
        )["material_reasons"]
        == []
    )
    keywords = {item["keyword_id"]: item for item in rebuilt["keywords"]}
    assert keywords["keyword-01"]["publication_count"] == 1
    assert keywords["keyword-01"]["label"] == "reliable systems"
    assert keywords["keyword-unclassified"]["publication_count"] == 1
    assert keywords["keyword-unclassified"]["label"] == "Unclassified publications"
    assert keywords["keyword-unclassified-pending-review"]["parent_keyword_id"] == (
        "keyword-unclassified"
    )
    missing = {
        **rebuilt,
        "points": [
            point for point in rebuilt["points"] if point["work_id"] != "work-one"
        ],
    }
    with pytest.raises(ValueError, match="loses parent points"):
        require_preserved_additive_artifact(baseline, missing, receipt)
    changed = json.loads(json.dumps(rebuilt))
    next(point for point in changed["points"] if point["work_id"] == "work-one")[
        "x"
    ] = 0.2
    with pytest.raises(ValueError, match="moves a retained point"):
        require_preserved_additive_artifact(baseline, changed, receipt)
    changed = json.loads(json.dumps(rebuilt))
    changed["points"].append({**changed["points"][0], "work_id": "unreviewed"})
    with pytest.raises(ValueError, match="differs from reviewed new work IDs"):
        require_preserved_additive_artifact(baseline, changed, receipt)
    changed = json.loads(json.dumps(rebuilt))
    changed["keywords"] = [
        item for item in changed["keywords"] if item["keyword_id"] != "keyword-01"
    ]
    with pytest.raises(ValueError, match="loses parent keyword IDs"):
        require_preserved_additive_artifact(baseline, changed, receipt)
    changed = json.loads(json.dumps(rebuilt))
    next(item for item in changed["keywords"] if item["keyword_id"] == "keyword-01")[
        "label"
    ] = "new label"
    with pytest.raises(ValueError, match="changes parent topic metadata"):
        require_preserved_additive_artifact(baseline, changed, receipt)
    changed = json.loads(json.dumps(rebuilt))
    next(point for point in changed["points"] if point["work_id"] == "work-one")[
        "keyword_ids"
    ] = ["keyword-unclassified", "keyword-unclassified-pending-review"]
    with pytest.raises(ValueError, match="changes retained topic IDs"):
        require_preserved_additive_artifact(baseline, changed, receipt)
    changed = json.loads(json.dumps(rebuilt))
    next(point for point in changed["points"] if point["work_id"] == "work-one")[
        "source_url"
    ] = "https://example.test/new-faculty-paper"
    with pytest.raises(ValueError, match="changes retained source URL"):
        require_preserved_additive_artifact(baseline, changed, receipt)
    changed = json.loads(json.dumps(rebuilt))
    next(point for point in changed["points"] if point["work_id"] == "work-added")[
        "keyword_ids"
    ] = ["keyword-01", "keyword-01-01"]
    with pytest.raises(ValueError, match="labels new work as reviewed"):
        require_preserved_additive_artifact(baseline, changed, receipt)
    changed = json.loads(json.dumps(rebuilt))
    next(point for point in changed["points"] if point["work_id"] == "work-added")[
        "x"
    ] = 1.2
    with pytest.raises(ValueError, match="unnormalized placement"):
        require_preserved_additive_artifact(baseline, changed, receipt)
    changed = json.loads(json.dumps(rebuilt))
    next(point for point in changed["points"] if point["work_id"] == "work-added")[
        "tsne_x"
    ] = 0.9
    with pytest.raises(ValueError, match="differs from reviewed work coordinates"):
        require_preserved_additive_artifact(baseline, changed, receipt)
    missing_digest = {**receipt, "placed_new_coordinates_sha256": ""}
    with pytest.raises(ValueError, match="differs from reviewed work coordinates"):
        require_preserved_additive_artifact(baseline, rebuilt, missing_digest)


def test_map_artifact_retains_verified_external_coauthor_and_source_type() -> None:
    works = work_frame()
    external = {
        "person_id": "external-paul-erdos",
        "display_name": "Paul Erdős",
        "source_url": "https://example.test/publisher",
    }
    works["external_coauthors"] = [[external]]
    works["external_coauthor_ids"] = [[external["person_id"]]]
    works["source_types"] = [["curated_verified"]]
    works["source_native_ids"] = [["egu2012:abstract-1"]]
    works["source_verified_at"] = "2026-10-24"
    works["source_verification_urls"] = [["https://example.test/publisher"]]
    works["last_fetched_at_utc"] = ""

    artifact = publisher.build_map_artifact(
        works, registry(), generated_at_utc="2026-10-24T12:00:00+00:00"
    )
    point = artifact["points"][0]
    assert point["source_types"] == ["curated_verified"]
    assert point["source_native_ids"] == ["egu2012:abstract-1"]
    assert point["source_verified_at"] == "2026-10-24"
    assert point["source_verification_urls"] == ["https://example.test/publisher"]
    assert point["external_coauthor_ids"] == ["external-paul-erdos"]
    assert artifact["catalogs"]["external_coauthors"] == [external]
    assert artifact["source_data_newest_at_utc"] is None

    works["publication_type"] = "patent"
    works["source_document_id"] = "US6486123B1"
    works["collaboration_eligible"] = False
    works["context_evidence_urls"] = [
        ["https://patents.google.com/patent/US6486123B1/en"]
    ]
    works["display_contributors"] = "Chien Ho; Tong-Jian Shen"
    typed_artifact = publisher.build_map_artifact(
        works, registry(), generated_at_utc="2026-10-24T12:00:00+00:00"
    )
    typed_point = typed_artifact["points"][0]
    assert typed_point["publication_type"] == "patent"
    assert typed_point["source_document_id"] == "US6486123B1"
    assert typed_point["collaboration_eligible"] is False
    assert typed_point["display_contributors"] == "Chien Ho; Tong-Jian Shen"
    assert typed_point["external_coauthor_ids"] == []
    assert typed_artifact["catalogs"]["external_coauthors"] == []
    assert typed_point["context_evidence_urls"] == [
        "https://patents.google.com/patent/US6486123B1/en"
    ]

    # Two records may name the same external person, but may not disagree.
    works["collaboration_eligible"] = True
    duplicate = works.copy()
    duplicate.at[0, "work_id"] = "work-two"
    assert (
        len(
            publisher.build_map_artifact(
                pandas.concat([works, duplicate], ignore_index=True),
                registry(),
                generated_at_utc="2026-10-24T12:00:00+00:00",
            )["catalogs"]["external_coauthors"]
        )
        == 1
    )


@pytest.mark.parametrize(
    ("external", "message"),
    [
        ("bad", "must be an object"),
        ({"person_id": "", "display_name": "Name"}, "needs identity"),
        ({"person_id": "external-x", "display_name": ""}, "needs identity"),
    ],
)
def test_map_artifact_rejects_invalid_external_identity(external, message) -> None:
    works = work_frame()
    works["external_coauthors"] = [[external]]
    with pytest.raises(ValueError, match=message):
        publisher.build_map_artifact(
            works, registry(), generated_at_utc="2026-10-24T12:00:00+00:00"
        )


def test_map_artifact_rejects_conflicting_external_identity() -> None:
    works = work_frame()
    other = works.copy()
    other.at[0, "work_id"] = "work-two"
    works["external_coauthors"] = [[{"person_id": "external-x", "display_name": "X"}]]
    other["external_coauthors"] = [[{"person_id": "external-x", "display_name": "Y"}]]
    with pytest.raises(ValueError, match="Conflicting external"):
        publisher.build_map_artifact(
            pandas.concat([works, other], ignore_index=True),
            registry(),
            generated_at_utc="2026-10-24T12:00:00+00:00",
        )


def registry() -> Registry:
    return Registry(
        people=(
            Person(
                person_id="person-one",
                display_name="One Person",
                scholar_id="ScholarOne",
                scholar_id_source_url="https://example.test/one",
                scholar_id_verified_at="2026-07-17",
                orcid="",
                homepage_url="https://example.test/one",
                notes="",
                primary_department_id="ece",
                primary_department_source_url="https://example.test/ece",
                primary_department_verified_at="2026-07-17",
                primary_department_basis="reviewed",
            ),
            Person(
                person_id="person-two",
                display_name="Two Person",
                scholar_id="",
                scholar_id_source_url="",
                scholar_id_verified_at="",
                orcid="0000-0000-0000-000X",
                homepage_url="",
                notes="No Scholar profile",
            ),
        ),
        memberships=(
            Membership(
                person_id="person-one",
                department_id="ece",
                role="faculty",
                included=True,
                legacy_label="",
                source_url="https://example.test/ece",
                verified_at="2026-07-17",
            ),
            Membership(
                person_id="person-one",
                department_id="cmu-silicon-valley",
                role="teaching",
                included=True,
                legacy_label="",
                source_url="https://example.test/sv",
                verified_at="2026-07-17",
            ),
            Membership(
                person_id="person-two",
                department_id="bme",
                role="emeritus",
                included=True,
                legacy_label="",
                source_url="https://example.test/bme",
                verified_at="2026-07-17",
            ),
        ),
        departments=(
            Department(
                department_id="bme",
                school_id="engineering",
                title="Biomedical Engineering",
                directory_url="https://example.test/bme",
                reviewed_at="2026-07-17",
                review_notes="Test",
            ),
            Department(
                department_id="cmu-silicon-valley",
                school_id="engineering",
                title="CMU Silicon Valley",
                directory_url="https://example.test/sv",
                reviewed_at="2026-07-17",
                review_notes="Test",
                unit_type="campus",
            ),
            Department(
                department_id="ece",
                school_id="engineering",
                title="Electrical & Computer Engineering",
                directory_url="https://example.test/ece",
                reviewed_at="2026-07-17",
                review_notes="Test",
            ),
        ),
        schools=(
            School(
                school_id="engineering",
                title="College of Engineering",
                directory_url="https://example.test/engineering",
            ),
        ),
    )


def test_custom_people_path_without_primary_baseline_reaches_snapshot_validation(
    monkeypatch, tmp_path
) -> None:
    class SnapshotChecked(Exception):
        pass

    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    monkeypatch.setattr(
        publisher, "_require_current_dataset_revision", lambda **kwargs: None
    )

    def check_snapshot(*args, **kwargs):
        raise SnapshotChecked

    monkeypatch.setattr(publisher, "validate_snapshot", check_snapshot)
    with pytest.raises(SnapshotChecked):
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
            people_path=tmp_path / "people.csv",
            memberships_path=tmp_path / "memberships.csv",
            departments_path=tmp_path / "departments.csv",
        )
    explicit_baseline = tmp_path / "missing-baseline.json"
    parsed = publisher._parse_args(
        [
            "--snapshot",
            str(tmp_path / "snapshot.parquet"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--prior-revision",
            "a" * 40,
            "--primary-home-baseline",
            str(explicit_baseline),
        ]
    )
    assert parsed.primary_home_baseline == explicit_baseline


def test_map_artifact_uses_shared_coordinates_and_id_relationships() -> None:
    artifact = publisher.build_map_artifact(
        work_frame(),
        registry(),
        generated_at_utc="2026-07-17T12:00:00+00:00",
    )

    assert artifact["schema_version"] == 7
    assert artifact["quality_assessment_version"] == (
        publisher.QUALITY_ASSESSMENT_VERSION
    )
    assert artifact["layout_version"] == publisher.LAYOUT_VERSION
    assert artifact["keyword_model_version"] == publisher.KEYWORD_MODEL_VERSION
    assert artifact["topic_label_review_version"] == (
        publisher.TOPIC_LABEL_REVIEW_VERSION
    )
    assert artifact["region_audit_version"] == publisher.REGION_AUDIT_VERSION
    assert artifact["default_layout_id"] == "pca"
    assert [layout["layout_id"] for layout in artifact["layouts"]] == [
        "pca",
        "tsne",
    ]
    assert artifact["point_count"] == 1
    assert artifact["points"][0]["faculty_ids"] == ["person-one"]
    assert artifact["points"][0]["department_ids"] == [
        "cmu-silicon-valley",
        "ece",
    ]
    assert artifact["points"][0]["school_ids"] == ["engineering"]
    assert artifact["school_count"] == 1
    assert artifact["catalogs"]["schools"] == [
        {
            "school_id": "engineering",
            "title": "College of Engineering",
            "directory_url": "https://example.test/engineering",
            "publication_count": 1,
        }
    ]
    assert artifact["points"][0]["x"] == 0.25
    assert artifact["points"][0]["tsne_x"] == -0.75
    assert artifact["points"][0]["keyword_ids"] == [
        "keyword-01",
        "keyword-01-01",
    ]
    assert artifact["keyword_count"] == 2
    assert artifact["keyword_levels"] == [
        {"level": 0, "label": "Topic regions", "keyword_count": 1},
        {"level": 1, "label": "Detailed topics", "keyword_count": 1},
    ]
    assert artifact["keywords"] == [
        {
            "keyword_id": "keyword-01",
            "label": "reliable systems",
            "extracted_label": "reliable systems",
            "label_reviewed": True,
            "level": 0,
            "parent_keyword_id": None,
            "publication_count": 1,
            "coordinates": {
                "pca": {"x": 0.25, "y": -0.5},
                "tsne": {"x": -0.75, "y": 0.125},
            },
        },
        {
            "keyword_id": "keyword-01-01",
            "label": "fault tolerant systems",
            "extracted_label": "fault tolerant systems",
            "label_reviewed": True,
            "level": 1,
            "parent_keyword_id": "keyword-01",
            "publication_count": 1,
            "coordinates": {
                "pca": {"x": 0.25, "y": -0.5},
                "tsne": {"x": -0.75, "y": 0.125},
            },
        },
    ]


def test_map_artifact_keeps_reported_faculty_separate_from_authorship() -> None:
    works = work_frame()
    report = {
        "person_id": "person-two",
        "source_person_id": "person-one",
        "source_observation_id": "observation-" + "a" * 64,
        "source_url": "https://example.test/byline",
        "verified_at": "2026-10-30",
    }
    artifact = publisher.build_map_artifact(
        works,
        registry(),
        generated_at_utc="2026-10-30T12:00:00+00:00",
        reported_faculty_coauthors_by_work={"work-one": [report]},
    )
    point = artifact["points"][0]
    assert point["faculty_ids"] == ["person-one"]
    assert point["reported_faculty_coauthors"] == [report]
    assert (
        next(
            item
            for item in artifact["catalogs"]["faculty"]
            if item["person_id"] == "person-one"
        )["publication_count"]
        == 1
    )
    assert (
        next(
            item
            for item in artifact["catalogs"]["faculty"]
            if item["person_id"] == "person-two"
        )["publication_count"]
        == 0
    )

    with pytest.raises(ValueError, match="no mapped point"):
        publisher.build_map_artifact(
            works,
            registry(),
            generated_at_utc="2026-10-30T12:00:00+00:00",
            reported_faculty_coauthors_by_work={"work-missing": [report]},
        )
    works["collaboration_eligible"] = False
    with pytest.raises(ValueError, match="lost collaboration eligibility"):
        publisher.build_map_artifact(
            works,
            registry(),
            generated_at_utc="2026-10-30T12:00:00+00:00",
            reported_faculty_coauthors_by_work={"work-one": [report]},
        )


def test_map_artifact_requires_consistent_keywords() -> None:
    missing = work_frame()
    missing.loc[0, "keyword_id"] = ""
    with pytest.raises(ValueError, match="both topic keyword levels"):
        publisher.build_map_artifact(
            missing,
            registry(),
            generated_at_utc="2026-07-17T12:00:00+00:00",
        )

    inconsistent = pandas.concat([work_frame(), work_frame()], ignore_index=True)
    inconsistent.loc[1, "work_id"] = "work-two"
    inconsistent.loc[1, "keyword"] = "different label"
    with pytest.raises(ValueError, match="inconsistent metadata"):
        publisher.build_map_artifact(
            inconsistent,
            registry(),
            generated_at_utc="2026-07-17T12:00:00+00:00",
        )


def test_global_layout_attaches_pca_and_tsne_coordinates(monkeypatch) -> None:
    frame = pandas.DataFrame(
        [
            {
                "work_id": "one",
                "map_eligible": True,
                "department_ids": ["ece"],
                "embedding": [3.0, 4.0, 0.0],
            },
            {
                "work_id": "two",
                "map_eligible": True,
                "department_ids": ["mse"],
                "embedding": [0.0, 0.0, 2.0],
            },
            {
                "work_id": "excluded",
                "map_eligible": False,
                "department_ids": ["ece"],
                "embedding": [1.0, 0.0, 0.0],
            },
        ]
    )
    captured = []

    def fake_pca(matrix):
        captured.append(matrix.copy())
        return numpy.asarray([[0.1, 0.2], [0.3, 0.4]])

    def fake_tsne(matrix):
        captured.append(matrix.copy())
        return numpy.asarray([[-0.5, 0.6], [0.7, -0.8]])

    monkeypatch.setattr(publisher, "_fit_pca_layout", fake_pca)
    monkeypatch.setattr(publisher, "_fit_tsne_layout", fake_tsne)

    output = publisher.add_global_layout(frame)

    assert output.loc[0, ["x", "y", "tsne_x", "tsne_y"]].tolist() == [
        0.1,
        0.2,
        -0.5,
        0.6,
    ]
    assert output.loc[1, ["x", "y", "tsne_x", "tsne_y"]].tolist() == [
        0.3,
        0.4,
        0.7,
        -0.8,
    ]
    assert output.loc[2, ["x", "y", "tsne_x", "tsne_y"]].isna().all()
    assert output["layout_version"].eq(publisher.LAYOUT_VERSION).all()
    assert len(captured) == 2
    assert numpy.allclose(captured[0][0], [0.6, 0.8, 0.0])
    assert numpy.allclose(captured[0][1], [0.0, 0.0, 1.0])


def test_global_layout_reuses_complete_unchanged_coordinates(monkeypatch) -> None:
    frame = pandas.DataFrame(
        [
            {
                "work_id": "one",
                "map_eligible": True,
                "department_ids": ["ece"],
                "embedding": [1.0, 0.0],
            },
            {
                "work_id": "two",
                "map_eligible": True,
                "department_ids": ["mse"],
                "embedding": [0.0, 1.0],
            },
        ]
    )
    existing = {
        "two": {"x": 0.2, "y": 0.3, "tsne_x": 0.4, "tsne_y": 0.5},
        "one": {"x": -0.2, "y": -0.3, "tsne_x": -0.4, "tsne_y": -0.5},
    }
    monkeypatch.setattr(
        publisher,
        "_fit_pca_layout",
        lambda _: (_ for _ in ()).throw(AssertionError("PCA should not refit")),
    )
    monkeypatch.setattr(
        publisher,
        "_fit_tsne_layout",
        lambda _: (_ for _ in ()).throw(AssertionError("t-SNE should not refit")),
    )

    output = publisher.add_global_layout(frame, existing_layouts=existing)

    assert output.loc[0, ["x", "y", "tsne_x", "tsne_y"]].tolist() == [
        -0.2,
        -0.3,
        -0.4,
        -0.5,
    ]
    assert output.loc[1, ["x", "y", "tsne_x", "tsne_y"]].tolist() == [
        0.2,
        0.3,
        0.4,
        0.5,
    ]


def test_existing_layouts_accepts_only_complete_compatible_rows(monkeypatch) -> None:
    valid = Rows(
        [
            {
                "work_id": "one",
                "layout_version": publisher.LAYOUT_VERSION,
                "x": 0,
                "y": 1,
                "tsne_x": 2,
                "tsne_y": 3,
            }
        ]
    )
    install_published_rows(monkeypatch, lambda *args, **kwargs: valid)
    assert publisher._existing_layouts(hf_token="token") == {
        "one": {"x": 0.0, "y": 1.0, "tsne_x": 2.0, "tsne_y": 3.0}
    }

    invalid_rows = [
        Rows([]),
        Rows([{**valid[0], "layout_version": "old"}]),
        Rows([{**valid[0], "x": "bad"}]),
        Rows([{**valid[0], "x": numpy.inf}]),
        Rows([{**valid[0], "work_id": ""}]),
        Rows([valid[0], valid[0]]),
    ]
    for rows in invalid_rows:
        install_published_rows(monkeypatch, lambda *args, _rows=rows, **kwargs: _rows)
        assert publisher._existing_layouts(hf_token="token") == {}

    mixed = Rows([valid[0], {**valid[0], "work_id": "excluded", "x": numpy.nan}])
    install_published_rows(monkeypatch, lambda *args, **kwargs: mixed)
    assert publisher._existing_layouts(hf_token="token") == {
        "one": {"x": 0.0, "y": 1.0, "tsne_x": 2.0, "tsne_y": 3.0}
    }


def test_existing_layouts_tolerates_missing_prior_dataset(monkeypatch) -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError

    install_published_rows(monkeypatch, missing)
    assert publisher._existing_layouts(hf_token="token") == {}

    install_published_rows(monkeypatch, lambda *args, **kwargs: Rows([]))
    assert publisher._existing_layouts(hf_token="token") == {}


def test_existing_layouts_stop_on_unexpected_read_failure(monkeypatch) -> None:
    install_published_rows(
        monkeypatch,
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("corrupt shard")),
    )
    with pytest.raises(RuntimeError, match="Could not read reusable works"):
        publisher._existing_layouts(hf_token="")


def test_existing_work_state_reads_published_order_and_missing_config(
    monkeypatch,
) -> None:
    calls = []

    def published_rows(config, fields, *, hf_token, revision=None):
        calls.append((config, hf_token, fields, revision))
        return Rows([{"work_id": "second"}, {"work_id": "first"}])

    install_published_rows(monkeypatch, published_rows)
    assert [
        row["work_id"] for row in publisher._existing_work_state(hf_token="token")
    ] == [
        "second",
        "first",
    ]
    assert calls[0][0:2] == ("works", "token")
    assert "map_region_outlier" in calls[0][2]
    publisher._existing_work_state(hf_token="token", revision="a" * 40)
    assert calls[-1][3] == "a" * 40

    def missing(*args, **kwargs):
        raise FileNotFoundError

    install_published_rows(monkeypatch, missing)
    assert publisher._existing_work_state(hf_token="token") == []

    def corrupt(*args, **kwargs):
        raise OSError("corrupt shard")

    install_published_rows(monkeypatch, corrupt)
    with pytest.raises(RuntimeError, match="order reuse"):
        publisher._existing_work_state(hf_token="token")


def test_reusable_year_key_handles_missing_and_invalid_values() -> None:
    assert publisher._year_key(None) is None
    assert publisher._year_key(numpy.nan) is None
    assert publisher._year_key("2020") == 2020
    assert publisher._year_key("unknown") is None


def reusable_work_state():
    new_id = "work-" + hashlib.sha256(b"doi:10.1000/rekey").hexdigest()

    def prior(work_id, title, *, eligible=True, outlier=False, reasons=None):
        return {
            "work_id": work_id,
            "title": title,
            "year": 2020,
            "doi": "",
            "work_match_method": "normalized_title_year",
            "map_eligible": eligible,
            "map_region_outlier": outlier,
            "map_exclusion_reasons": reasons or [],
            "department_ids": ["mse"],
            "quality_assessment_version": publisher.QUALITY_ASSESSMENT_VERSION,
            "keyword_model_version": publisher.KEYWORD_MODEL_VERSION,
            "topic_label_review_version": publisher.TOPIC_LABEL_REVIEW_VERSION,
            "map_region_audit_version": publisher.REGION_AUDIT_VERSION,
            "layout_version": publisher.LAYOUT_VERSION,
        }

    previous = [
        prior("old-title-id", "Material A"),
        prior("retained", "Material B"),
        prior(
            "audited-out",
            "Material C",
            eligible=False,
            outlier=True,
            reasons=["low_information_region"],
        ),
        prior("quality-out", "Material D", eligible=False, reasons=["non_research"]),
    ]
    current = pandas.DataFrame(
        [
            {
                "work_id": "retained",
                "title": "Material B",
                "year": 2020,
                "doi": "",
                "map_eligible": True,
                "map_exclusion_reasons": [],
                "department_ids": ["mse"],
                "embedding": [1.0, 0.0],
            },
            {
                "work_id": new_id,
                "title": "Material A",
                "year": 2020,
                "doi": "10.1000/rekey",
                "map_eligible": True,
                "map_exclusion_reasons": [],
                "department_ids": ["mse"],
                "embedding": [0.0, 1.0],
            },
            {
                "work_id": "quality-out",
                "title": "Material D",
                "year": 2020,
                "doi": "",
                "map_eligible": False,
                "map_exclusion_reasons": ["non_research"],
                "department_ids": ["mse"],
                "embedding": [0.0, 0.0],
            },
            {
                "work_id": "audited-out",
                "title": "Material C",
                "year": 2020,
                "doi": "",
                "map_eligible": True,
                "map_exclusion_reasons": [],
                "department_ids": ["mse"],
                "embedding": [1.0, 1.0],
            },
        ]
    )
    current["work_match_method"] = "normalized_title_year"
    current.loc[1, "work_match_method"] = "curated_title_year"
    current["source_types"] = [["scholar"] for _ in range(len(current))]
    current.at[1, "source_types"] = ["scholar", "curated_verified"]
    current["source_verified_at"] = ""
    current.loc[1, "source_verified_at"] = "2026-10-24"
    current["source_verification_urls"] = [[] for _ in range(len(current))]
    current.at[1, "source_verification_urls"] = ["https://example.test/doi"]
    layouts = {
        work_id: {"x": x, "y": 0.2, "tsne_x": 0.3, "tsne_y": 0.4}
        for work_id, x in [
            ("old-title-id", 0.1),
            ("retained", 0.5),
            ("audited-out", 0.9),
        ]
    }
    return current, previous, layouts, new_id


def test_reusable_work_order_preserves_doi_rekey_layout_and_topic_order(
    monkeypatch,
) -> None:
    current, previous, layouts, new_id = reusable_work_state()
    reused = publisher._reusable_work_order(current, previous, layouts)
    assert reused is not None
    ordered, remapped = reused
    assert ordered["work_id"].tolist() == [
        new_id,
        "retained",
        "audited-out",
        "quality-out",
    ]
    assert set(remapped) == {new_id, "retained", "audited-out"}
    assert remapped[new_id] == layouts["old-title-id"]

    monkeypatch.setattr(
        publisher,
        "_fit_pca_layout",
        lambda _: (_ for _ in ()).throw(AssertionError("layout refitted")),
    )
    monkeypatch.setattr(
        publisher,
        "_fit_tsne_layout",
        lambda _: (_ for _ in ()).throw(AssertionError("layout refitted")),
    )
    laid_out = publisher.add_global_layout(ordered, existing_layouts=remapped)
    assert laid_out.loc[0, "x"] == 0.1
    assert laid_out.loc[2, "x"] == 0.9
    assert numpy.isnan(laid_out.loc[3, "x"])


def reusable_alias_work_state():
    current, previous, layouts, _ = reusable_work_state()
    aliases = publisher.load_doi_aliases(publisher.DEFAULT_DOI_ALIASES_PATH)
    (scholar_id, author_pub_id), alias = next(iter(aliases.items()))
    new_id = "work-" + hashlib.sha256(f"doi:{alias.doi}".encode()).hexdigest()
    previous[0]["title"] = alias.observed_title
    previous[0]["year"] = alias.observed_year
    current.loc[1, "work_id"] = new_id
    current.loc[1, "title"] = alias.observed_title
    current.loc[1, "year"] = alias.observed_year
    current.loc[1, "doi"] = alias.doi
    current.loc[1, "work_match_method"] = "reviewed_version_alias"
    current.at[1, "source_types"] = ["scholar"]
    current.at[1, "source_verification_urls"] = list(alias.evidence_urls)
    current["scholar_ids"] = [[] for _ in range(len(current))]
    current["author_pub_ids"] = [[] for _ in range(len(current))]
    current.at[1, "scholar_ids"] = [scholar_id]
    current.at[1, "author_pub_ids"] = [author_pub_id]
    return current, previous, layouts, new_id, aliases


def test_reusable_work_order_preserves_reviewed_alias_layout(monkeypatch) -> None:
    current, previous, layouts, new_id, aliases = reusable_alias_work_state()
    reused = publisher._reusable_work_order(current, previous, layouts, aliases)
    assert reused is not None
    ordered, remapped = reused
    assert ordered["work_id"].tolist()[0] == new_id
    assert remapped[new_id] == layouts["old-title-id"]

    monkeypatch.setattr(
        publisher,
        "_fit_pca_layout",
        lambda _: (_ for _ in ()).throw(AssertionError("layout refitted")),
    )
    monkeypatch.setattr(
        publisher,
        "_fit_tsne_layout",
        lambda _: (_ for _ in ()).throw(AssertionError("layout refitted")),
    )
    laid_out = publisher.add_global_layout(ordered, existing_layouts=remapped)
    assert laid_out.loc[0, "x"] == 0.1


@pytest.mark.parametrize(
    "change",
    [
        "missing_manifest",
        "wrong_doi",
        "wrong_scholar_id",
        "wrong_observation_id",
        "missing_evidence",
        "missing_scholar_source",
        "unverified",
        "wrong_method",
    ],
)
def test_reusable_work_order_rejects_unproven_alias(change) -> None:
    current, previous, layouts, _, aliases = reusable_alias_work_state()
    if change == "missing_manifest":
        aliases = {}
    elif change == "wrong_doi":
        current.loc[1, "doi"] = "10.1000/wrong"
        current.loc[1, "work_id"] = (
            "work-" + hashlib.sha256(b"doi:10.1000/wrong").hexdigest()
        )
    elif change == "wrong_scholar_id":
        current.at[1, "scholar_ids"] = ["other"]
    elif change == "wrong_observation_id":
        current.at[1, "author_pub_ids"] = ["other:row"]
    elif change == "missing_evidence":
        current.at[1, "source_verification_urls"] = [
            current.at[1, "source_verification_urls"][0]
        ]
    elif change == "missing_scholar_source":
        current.at[1, "source_types"] = ["curated_verified"]
    elif change == "unverified":
        current.loc[1, "source_verified_at"] = ""
    elif change == "wrong_method":
        current.loc[1, "work_match_method"] = "normalized_title_year"
    assert publisher._reusable_work_order(current, previous, layouts, aliases) is None


def test_reusable_work_order_preserves_prior_order_without_new_rekeys() -> None:
    current, previous, layouts, new_id = reusable_work_state()
    previous[0]["work_id"] = new_id
    layouts[new_id] = layouts.pop("old-title-id")
    reused = publisher._reusable_work_order(current, previous, layouts)
    assert reused is not None
    assert reused[0]["work_id"].tolist()[0] == new_id


def test_reusable_work_order_caps_doi_rekeys() -> None:
    current, previous, layouts, _ = reusable_work_state()
    additions = []
    for index in range(publisher.MAX_REUSABLE_DOI_REKEYS):
        old = f"old-{index}"
        new = (
            "work-" + hashlib.sha256(f"doi:10.1000/rekey-{index}".encode()).hexdigest()
        )
        previous.append({**previous[0], "work_id": old, "title": f"Material {index}"})
        additions.append(
            {
                **current.iloc[1].to_dict(),
                "work_id": new,
                "title": f"Material {index}",
                "doi": f"10.1000/rekey-{index}",
            }
        )
        layouts[old] = layouts["old-title-id"]
    expanded = pandas.concat([current, pandas.DataFrame(additions)], ignore_index=True)
    assert publisher._reusable_work_order(expanded, previous, layouts) is None


@pytest.mark.parametrize(
    "change",
    [
        "missing_previous",
        "missing_layouts",
        "old_duplicate_id",
        "current_duplicate_id",
        "old_version",
        "more_current_works",
        "changed_title",
        "changed_year",
        "changed_eligibility",
        "changed_reasons",
        "changed_department_inclusion",
        "missing_coordinate",
        "bad_doi",
        "ambiguous_title_year",
        "old_doi_id_still_present",
        "rekey_matches_retained_title",
        "old_doi_present",
        "unverified_rekey",
    ],
)
def test_reusable_work_order_rejects_non_equivalent_corpora(change) -> None:
    current, previous, layouts, new_id = reusable_work_state()
    if change == "missing_previous":
        previous = []
    elif change == "missing_layouts":
        layouts = {}
    elif change == "old_duplicate_id":
        previous[1]["work_id"] = previous[0]["work_id"]
    elif change == "current_duplicate_id":
        current.loc[1, "work_id"] = "retained"
    elif change == "old_version":
        previous[0]["layout_version"] = "old-layout"
    elif change == "more_current_works":
        current = pandas.concat([current, current.iloc[[0]].assign(work_id="new")])
    elif change == "changed_title":
        current.loc[0, "title"] = "New title"
    elif change == "changed_year":
        current.loc[0, "year"] = 2021
    elif change == "changed_eligibility":
        current.loc[0, "map_eligible"] = False
    elif change == "changed_reasons":
        current.at[2, "map_exclusion_reasons"] = ["different_reason"]
    elif change == "changed_department_inclusion":
        current.at[0, "department_ids"] = []
    elif change == "missing_coordinate":
        layouts.pop("retained")
    elif change == "bad_doi":
        current.loc[1, "doi"] = "10.1000/wrong"
    elif change == "ambiguous_title_year":
        previous[1]["title"] = "Material A"
        current.loc[0, "title"] = "Material A"
    elif change == "old_doi_id_still_present":
        current.loc[0, "work_id"] = "old-title-id"
    elif change == "rekey_matches_retained_title":
        current.loc[0, "title"] = "Material A"
        current.loc[1, "title"] = "Different title"
    elif change == "old_doi_present":
        previous[0]["doi"] = "10.1000/old"
    elif change == "unverified_rekey":
        current.at[1, "source_verification_urls"] = []
    assert publisher._reusable_work_order(current, previous, layouts) is None


def test_layout_normalization_handles_flat_axes_and_aspect_modes() -> None:
    points = numpy.asarray([[1, 4], [1, 8]], dtype=numpy.float32)
    independent = publisher._normalize_layout(points, preserve_aspect=False)
    preserved = publisher._normalize_layout(points, preserve_aspect=True)
    assert numpy.isfinite(independent).all()
    assert numpy.max(numpy.abs(independent), axis=0).tolist() == [0.0, 1.0]
    assert numpy.max(numpy.abs(preserved)) == 1.0
    assert numpy.array_equal(
        publisher._normalize_layout(numpy.zeros((1, 2)), preserve_aspect=True),
        numpy.zeros((1, 2)),
    )


def test_real_layout_fitters_return_normalized_coordinates() -> None:
    matrix = numpy.eye(4, dtype=numpy.float32)
    assert publisher._fit_pca_layout(matrix).shape == (4, 2)
    projected = publisher._fit_tsne_layout(matrix)
    assert projected.shape == (4, 2)
    assert numpy.max(numpy.abs(projected)) <= 1


def test_global_layout_handles_single_work_and_rejects_empty_map() -> None:
    one = pandas.DataFrame(
        [
            {
                "work_id": "one",
                "map_eligible": True,
                "department_ids": ["ece"],
                "embedding": [0, 0],
            }
        ]
    )
    output = publisher.add_global_layout(one)
    assert output.loc[0, ["x", "y", "tsne_x", "tsne_y"]].tolist() == [0, 0, 0, 0]

    one.at[0, "map_eligible"] = False
    with pytest.raises(ValueError, match="No included works"):
        publisher.add_global_layout(one)


def test_map_artifact_catalog_retains_full_included_roster() -> None:
    artifact = publisher.build_map_artifact(
        work_frame(),
        registry(),
        generated_at_utc="2026-07-17T12:00:00+00:00",
    )

    assert artifact["department_count"] == 3
    assert artifact["faculty_count"] == 2
    assert [item["display_name"] for item in artifact["catalogs"]["faculty"]] == [
        "One Person",
        "Two Person",
    ]
    assert artifact["catalogs"]["faculty"][1]["publication_count"] == 0
    assert artifact["catalogs"]["faculty"][1]["memberships"] == [
        {"department_id": "bme", "role": "emeritus"}
    ]
    departments = {
        item["department_id"]: item for item in artifact["catalogs"]["departments"]
    }
    assert departments["ece"]["publication_count"] == 1
    assert departments["ece"]["school_id"] == "engineering"
    assert departments["ece"]["unit_type"] == "department"
    assert departments["bme"]["publication_count"] == 0
    assert artifact["catalogs"]["faculty"][0]["school_ids"] == ["engineering"]
    assert artifact["catalogs"]["faculty"][0]["primary_academic_home_type"] == (
        "department"
    )
    assert artifact["catalogs"]["faculty"][0]["primary_academic_home_id"] == ("ece")


def test_map_artifact_tracks_cross_school_membership_once() -> None:
    source = registry()
    cross_school = Registry(
        people=tuple(
            replace(
                person,
                primary_department_id="",
                primary_department_source_url="",
                primary_department_verified_at="",
                primary_department_basis="",
            )
            if person.person_id == "person-one"
            else person
            for person in source.people
        ),
        memberships=(
            *source.memberships,
            Membership(
                person_id="person-one",
                department_id="s3d",
                role="faculty",
                included=True,
                legacy_label="",
                source_url="https://example.test/s3d",
                verified_at="2026-10-24",
            ),
        ),
        departments=(
            *source.departments,
            Department(
                department_id="s3d",
                school_id="scs",
                title="Software and Societal Systems Department",
                directory_url="https://example.test/s3d",
                reviewed_at="2026-10-24",
                review_notes="Pilot",
            ),
        ),
        schools=(
            *source.schools,
            School(
                school_id="scs",
                title="School of Computer Science",
                directory_url="https://example.test/scs",
            ),
        ),
    )
    works = work_frame()
    works.at[0, "memberships"] = [
        *works.at[0, "memberships"],
        {
            "person_id": "person-one",
            "display_name": "One Person",
            "department_id": "s3d",
            "department_title": "Software and Societal Systems Department",
            "role": "faculty",
        },
    ]

    artifact = publisher.build_map_artifact(
        works,
        cross_school,
        generated_at_utc="2026-10-24T12:00:00+00:00",
    )

    assert artifact["point_count"] == 1
    assert artifact["points"][0]["school_ids"] == ["engineering", "scs"]
    assert artifact["catalogs"]["faculty"][0]["school_ids"] == [
        "engineering",
        "scs",
    ]
    assert artifact["catalogs"]["faculty"][0]["primary_department_id"] == ""
    assert artifact["catalogs"]["faculty"][0]["primary_academic_home_type"] == (
        "unresolved"
    )
    assert (
        artifact["catalogs"]["faculty"][0]["primary_department_basis"] == "unresolved"
    )
    school_counts = {
        school["school_id"]: school["publication_count"]
        for school in artifact["catalogs"]["schools"]
    }
    assert school_counts == {
        "engineering": 1,
        "scs": 1,
    }


def test_hub_dataset_preserves_legacy_faculty_string() -> None:
    frame = pandas.DataFrame(
        {
            "faculty": ["Legacy label"],
            "embedding": [[0.0] * publisher.EMBEDDING_DIMENSION],
        }
    )

    dataset = publisher._to_hub_dataset(frame)

    assert dataset[0]["faculty"] == "Legacy label"


def test_empty_map_artifact_is_valid_and_explicit() -> None:
    frame = work_frame()
    frame.at[0, "memberships"] = []
    artifact = publisher.build_map_artifact(
        frame,
        registry(),
        generated_at_utc="2026-07-17T12:00:00+00:00",
    )

    assert artifact["point_count"] == 0
    assert artifact["points"] == []
    assert artifact["department_count"] == 3
    assert artifact["faculty_count"] == 2
    assert artifact["source_data_newest_at_utc"] is None
    assert artifact["keyword_count"] == 0
    assert artifact["keyword_levels"] == [
        {"level": 0, "label": "Topic regions", "keyword_count": 0},
        {"level": 1, "label": "Detailed topics", "keyword_count": 0},
    ]
    assert artifact["keywords"] == []


def test_map_artifact_excludes_ineligible_works_but_retains_freshness() -> None:
    frame = work_frame()
    frame.at[0, "map_eligible"] = False
    frame.at[0, "map_exclusion_reasons"] = ["organization_or_container"]
    artifact = publisher.build_map_artifact(
        frame,
        registry(),
        generated_at_utc="2026-07-17T12:00:00+00:00",
    )

    assert artifact["point_count"] == 0
    assert artifact["excluded_work_count"] == 1
    assert artifact["source_data_newest_at_utc"] == "2026-01-01T00:00:00+00:00"


def test_map_artifact_normalizes_missing_metadata_and_rejects_bad_memberships() -> None:
    frame = work_frame()
    frame.loc[0, "year"] = numpy.nan
    frame.at[0, "source_urls"] = []
    frame.loc[0, "last_fetched_at_utc"] = ""
    artifact = publisher.build_map_artifact(
        frame,
        registry(),
        generated_at_utc="2026-07-17T12:00:00+00:00",
    )
    assert artifact["points"][0]["year"] is None
    assert artifact["points"][0]["source_url"] == ""
    assert artifact["source_data_oldest_at_utc"] is None

    frame.at[0, "memberships"] = ["not an object"]
    with pytest.raises(ValueError, match="must be an object"):
        publisher.build_map_artifact(
            frame,
            registry(),
            generated_at_utc="2026-07-17T12:00:00+00:00",
        )


def test_map_catalog_ignores_excluded_memberships_and_unrostered_people() -> None:
    source = registry()
    extended = Registry(
        people=(
            *source.people,
            Person(
                person_id="person-three",
                display_name="Three Person",
                scholar_id="",
                scholar_id_source_url="",
                scholar_id_verified_at="",
                orcid="",
                homepage_url="",
                notes="No included membership",
            ),
        ),
        memberships=(
            *source.memberships,
            Membership(
                person_id="person-one",
                department_id="ece",
                role="affiliate",
                included=False,
                legacy_label="",
                source_url="https://example.test/ece",
                verified_at="2026-07-17",
            ),
        ),
        departments=source.departments,
        schools=source.schools,
    )
    artifact = publisher.build_map_artifact(
        work_frame(),
        extended,
        generated_at_utc="2026-07-17T12:00:00+00:00",
    )
    assert artifact["faculty_count"] == 2


def test_staged_map_files_bind_dataset_to_same_revision(tmp_path) -> None:
    artifact_path, manifest_path, artifact = publisher._write_map_artifacts(
        work_frame(),
        registry(),
        maps_dir=tmp_path / "maps",
        generated_at_utc="2026-10-25T00:00:00+00:00",
        dataset_commits={},
        continuity_receipt={"prior_revision": "a" * 40},
    )
    manifest = json.loads(manifest_path.read_text())
    assert json.loads(artifact_path.read_text()) == artifact
    assert manifest["dataset_commits"] == {}
    assert manifest["dataset_revision_policy"] == "same_revision_as_map_manifest"
    assert manifest["region_continuity"]["prior_revision"] == "a" * 40

    _, prior_manifest_path, _ = publisher._write_map_artifacts(
        work_frame(),
        registry(),
        maps_dir=tmp_path / "legacy-maps",
        generated_at_utc="2026-10-25T00:00:00+00:00",
        dataset_commits={"works": "legacy-commit"},
    )
    prior_manifest = json.loads(prior_manifest_path.read_text())
    assert prior_manifest["dataset_commits"] == {"works": "legacy-commit"}
    assert "region_continuity" not in prior_manifest
    assert "dataset_revision_policy" not in prior_manifest


def test_hub_dataset_casts_list_and_embedding_features(monkeypatch) -> None:
    cast_features = []

    class Dataset:
        column_names = ["department_ids", "embedding", "title"]
        features = {name: object() for name in column_names}

        @classmethod
        def from_pandas(cls, frame, preserve_index):
            assert preserve_index is False
            return cls()

        def cast(self, features):
            cast_features.append(features)
            return self

    def value(type_name):
        return ("value", type_name)

    def list_type(item):
        return ("list", item)

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(Dataset=Dataset, Value=value, List=list_type),
    )
    publisher._to_hub_dataset(pandas.DataFrame())
    assert cast_features[0]["department_ids"] == ("list", ("value", "string"))
    assert cast_features[0]["embedding"] == ("list", ("value", "float32"))


def test_hub_dataset_avoids_unnecessary_casts() -> None:
    unchanged = publisher._to_hub_dataset(pandas.DataFrame({"faculty": ["One"]}))
    assert unchanged[0]["faculty"] == "One"
    string_list = publisher._to_hub_dataset(
        pandas.DataFrame({"department_ids": [["ece"]]})
    )
    assert string_list[0]["department_ids"] == ["ece"]


@pytest.mark.parametrize("reuse_order", [False, True])
@pytest.mark.parametrize(
    ("pinned", "missing_prior", "region_override", "atomic"),
    [
        (False, False, False, True),
        (True, False, False, True),
        (True, True, False, True),
        (True, False, True, True),
    ],
)
def test_publish_snapshot_orchestrates_all_dataset_configs(
    monkeypatch,
    tmp_path,
    reuse_order,
    pinned,
    missing_prior,
    region_override,
    atomic,
) -> None:
    monkeypatch.setattr(
        publisher,
        "apply_reported_faculty_coauthors",
        lambda *args, **kwargs: ({}, []),
    )
    monkeypatch.setattr(publisher, "require_reported_source_sync", lambda *args: False)
    bio_intent = pinned and atomic and not region_override and not missing_prior
    if bio_intent:
        # This fixture uses an abstract immutable parent for the historical Bio path.
        monkeypatch.setattr(publisher, "BIO_BINDING_PRIOR_REVISION", "a" * 40)
    snapshot_digest = {
        "value": publisher.BIO_SOURCE_SHA256 if bio_intent else "test-snapshot"
    }
    frame = pandas.DataFrame(
        {
            "scholar_id": ["one", "two", "one"],
            "author_pub_id": ["one:a", "two:a", "one:b"],
            "source_type": ["scholar"] * 3,
            "title": ["Damaged Title", "Second Title", "Third Title"],
            "year": [1712, 2020, 2021],
            "venue": ["Journal A", "Journal B", "Journal C"],
        }
    )
    corrections_path = tmp_path / "metadata-corrections.json"
    corrections_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corrections": [
                    {
                        "scholar_id": "one",
                        "author_pub_id": "one:a",
                        "expected": {
                            "title": "Damaged Title",
                            "year": 1712,
                            "venue": "Journal A",
                        },
                        "corrected": {"year": 2011},
                        "basis": "Primary journal year corrects the Scholar field.",
                        "evidence_url": "https://example.test/primary",
                        "verified_at": "2026-10-25",
                    }
                ],
            }
        )
    )
    tables = {
        name: pandas.DataFrame({"value": [1, 2]}) for name in publisher.DATASET_CONFIGS
    }
    tables["profile_publications"]["context_verified_at"] = ["", "2026-10-25"]
    tables["works"] = pandas.DataFrame(
        {
            "value": [1, 2],
            "department_ids": [["ece"], ["ece"]],
            "memberships": [[{"person_id": "person-one"}], []],
            "map_eligible": [True, False],
        }
    )
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    monkeypatch.setattr(publisher, "bio_scholar_source_changed", lambda *args: False)
    monkeypatch.setattr(publisher, "load_profile_releases", lambda path: ())
    monkeypatch.setattr(publisher, "load_provisional_holds", lambda path: ())
    chemistry_calls = []
    chemistry_candidate_options = []
    chemistry_active = bio_intent
    monkeypatch.setattr(
        publisher,
        "needs_chemistry_continuity",
        lambda *args, **kwargs: chemistry_active,
    )
    monkeypatch.setattr(
        publisher,
        "load_chem_region_review",
        lambda *args, **kwargs: chemistry_calls.append("source") or ("review",),
    )
    monkeypatch.setattr(
        publisher,
        "load_chemistry_parent",
        lambda *args: chemistry_calls.append("parent") or {"pinned": True},
    )
    monkeypatch.setattr(
        publisher,
        "merge_chemistry_review_inputs",
        lambda decisions, exclusions, holds, contexts: (
            exclusions,
            holds,
            contexts,
        ),
    )
    monkeypatch.setattr(
        publisher,
        "validate_chemistry_candidate",
        lambda tables, state, **kwargs: (
            chemistry_calls.append("candidate")
            or chemistry_candidate_options.append(kwargs)
        ),
    )
    monkeypatch.setattr(
        publisher,
        "reconcile_chemistry_person_counts",
        lambda people, decisions: chemistry_calls.append("historical_counts") or people,
    )
    monkeypatch.setattr(publisher, "validate_doi_aliases", lambda *args: None)
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (
            frame,
            {
                "created_at_utc": "2026-07-17T12:00:00+00:00",
                "snapshot_sha256": snapshot_digest["value"],
            },
        ),
    )
    monkeypatch.setattr(
        publisher,
        "add_embeddings",
        lambda *args, **kwargs: (frame, 2),
    )
    monkeypatch.setattr(publisher, "build_dataset_tables", lambda *args: tables)
    original_layouts = {"old": {"x": 0.1}}
    reused_layouts = {"new": {"x": 0.1}}
    monkeypatch.setattr(
        publisher, "_existing_layouts", lambda **kwargs: original_layouts
    )
    monkeypatch.setattr(
        publisher, "_existing_work_state", lambda **kwargs: [{"work_id": "old"}]
    )
    monkeypatch.setattr(
        publisher,
        "_reusable_work_order",
        lambda works, previous, layouts, doi_aliases: (
            (works, reused_layouts) if reuse_order else None
        ),
    )
    observed_layouts = []

    def retain_layout(works, *, existing_layouts):
        observed_layouts.append(existing_layouts)
        return works

    monkeypatch.setattr(publisher, "add_global_layout", retain_layout)
    monkeypatch.setattr(
        publisher,
        "add_publication_keywords",
        lambda works: works.assign(
            keyword_id=["keyword-01", ""],
            detail_keyword_id=["keyword-01-01", ""],
        ),
    )
    atomic_calls = []
    monkeypatch.setattr(
        publisher,
        "_publish_atomic_release",
        lambda *args, **kwargs: (
            atomic_calls.append(kwargs) or "atomic-commit",
            {"mode": "single_parent_commit"},
        ),
    )
    baseline_checks = []
    comparison_checks = []
    preview_checks = []
    historical_decision_checks = []
    source_tip_checks = []
    producer_tip_checks = []
    published_revisions = []
    published_fields = []
    approved_losses = []
    approval_loads = []
    approved_loss = object()
    monkeypatch.setattr(
        publisher,
        "load_poster_loss_approval",
        lambda path, **kwargs: approval_loads.append((path, kwargs)) or approved_loss,
    )
    if region_override:
        monkeypatch.setattr(
            publisher,
            "require_approved_comparison_revision",
            lambda **kwargs: comparison_checks.append(kwargs),
        )
        monkeypatch.setattr(
            publisher,
            "require_frozen_source_tip",
            lambda path: source_tip_checks.append(path),
        )
        monkeypatch.setenv("MAP_PRODUCER_COMMIT", "c" * 40)
        monkeypatch.setattr(
            publisher,
            "require_frozen_producer_tip",
            lambda path, commit: producer_tip_checks.append((path, commit)),
        )
        monkeypatch.setattr(
            publisher,
            "require_expected_repair_preview",
            lambda **kwargs: preview_checks.append(kwargs),
        )
        monkeypatch.setattr(
            publisher,
            "require_historical_region_decisions",
            lambda works, final: historical_decision_checks.append((works, final)),
        )
    if pinned:
        monkeypatch.setattr(
            publisher,
            "read_pinned_prior_people",
            lambda **kwargs: ([{"person_id": "fixture"}], "a" * 64),
        )
        monkeypatch.setattr(
            publisher,
            "require_prior_person_continuity",
            lambda *args, **kwargs: {"status": "pass"},
        )
        monkeypatch.setattr(
            publisher,
            "_require_current_dataset_revision",
            lambda **kwargs: baseline_checks.append(kwargs["prior_revision"]),
        )

        def prior_rows(config, fields, **kwargs):
            published_revisions.append(kwargs["revision"])
            published_fields.append((config, fields))
            return (
                []
                if missing_prior and config == "works"
                else [{"work_id": "old"}]
                if config == "works"
                else [{"person_id": "bio-person", "included_department_ids": ["bio"]}]
                if config == "people"
                else [{"author_pub_id": "one:a"}]
            )

        monkeypatch.setattr(publisher, "_published_rows", prior_rows)
        monkeypatch.setattr(
            publisher,
            "reconcile_region_decisions",
            lambda works, *args, **kwargs: (
                approved_losses.append(kwargs["approved_poster_loss"]) or works,
                {"prior_revision": kwargs["prior_revision"], "stable_works": 1},
            ),
        )

    class Dataset:
        def __init__(self, config_name):
            self.config_name = config_name

        def push_to_hub(self, *args, **kwargs):
            return SimpleNamespace(oid=f"{kwargs['config_name']}-commit")

    monkeypatch.setattr(
        publisher,
        "_to_hub_dataset",
        lambda table: Dataset(
            next(name for name, value in tables.items() if value is table)
        ),
    )
    publish_options = {
        "hf_token": "token",
        "corrections_path": corrections_path,
        "prior_revision": "a" * 40 if pinned else None,
        "atomic_intent_sink": lambda intent: None,
        "require_bio_release": bio_intent,
    }
    publish_options["atomic_release"] = True
    if region_override:
        publish_options["region_comparison_revision"] = "b" * 40
    if not pinned:
        with pytest.raises(ValueError, match="immutable prior Hub revision"):
            publisher.publish_snapshot(
                tmp_path / "snapshot.parquet",
                tmp_path / "manifest.json",
                **publish_options,
            )
        assert not atomic_calls
        return
    if missing_prior:
        with pytest.raises(RuntimeError, match="lacks region continuity tables"):
            publisher.publish_snapshot(
                tmp_path / "snapshot.parquet",
                tmp_path / "manifest.json",
                **publish_options,
            )
        return
    if atomic and region_override:
        with pytest.raises(ValueError, match="Ordinary atomic upload"):
            publisher.publish_snapshot(
                tmp_path / "snapshot.parquet",
                tmp_path / "manifest.json",
                **publish_options,
            )
        assert not atomic_calls
        return
    if bio_intent:
        snapshot_digest["value"] = "drifted-bio-source"
        publish_options["require_bio_release"] = False
        monkeypatch.setattr(publisher, "bio_scholar_source_changed", lambda *args: True)
        with pytest.raises(ValueError, match="Changed Bio Scholar source"):
            publisher.publish_snapshot(
                tmp_path / "snapshot.parquet",
                tmp_path / "manifest.json",
                **publish_options,
            )
        if reuse_order:
            with pytest.raises(ValueError, match="requires the frozen Bio source"):
                publisher.publish_snapshot(
                    tmp_path / "snapshot.parquet",
                    tmp_path / "manifest.json",
                    historical_loss_approval_path=tmp_path / "approval.json",
                    **publish_options,
                )
        assert not atomic_calls
        baseline_checks.clear()
        published_revisions.clear()
        published_fields.clear()
        observed_layouts.clear()
        snapshot_digest["value"] = publisher.BIO_SOURCE_SHA256
        publish_options["require_bio_release"] = True
        if reuse_order:
            publish_options["historical_loss_approval_path"] = (
                tmp_path / "approval.json"
            )
    result = publisher.publish_snapshot(
        tmp_path / "snapshot.parquet",
        tmp_path / "manifest.json",
        **publish_options,
    )
    if chemistry_active:
        expected_chemistry_calls = [
            "source",
            "parent",
            "source",
            "parent",
        ]
        assert [call for call in chemistry_calls if call != "historical_counts"] == [
            *expected_chemistry_calls,
            "candidate",
        ]
        assert chemistry_calls.count("historical_counts") == (
            2 if publisher.CHEMISTRY_HUB_REVISION == "a" * 40 else 0
        )
        assert chemistry_candidate_options == [
            {
                "allow_region_refit": not reuse_order,
                "require_count_parity": False,
            }
        ]
    assert baseline_checks == (["a" * 40, "a" * 40] if pinned else [])
    assert published_revisions == (
        [
            "b" * 40 if region_override else "a" * 40,
            "b" * 40 if region_override else "a" * 40,
            "a" * 40,
        ]
        if pinned
        else []
    )
    if pinned:
        assert published_fields[0][0] == "works"
        assert {
            "map_exclusion_reasons",
            "person_ids",
            "scholar_ids",
            "author_pub_ids",
            "observation_ids",
            "observation_count",
        } <= set(published_fields[0][1])
    if region_override:
        assert len(historical_decision_checks) == 1
        assert historical_decision_checks[0][0] == [{"work_id": "old"}]
        assert historical_decision_checks[0][1] is tables["works"]
    else:
        assert historical_decision_checks == []
    assert comparison_checks == (
        [
            {
                "current_revision": "a" * 40,
                "comparison_revision": "b" * 40,
                "snapshot_sha256": "test-snapshot",
            }
        ]
        if region_override
        else []
    )
    assert source_tip_checks == ([tmp_path.parent] if region_override else [])
    assert producer_tip_checks == (
        [(tmp_path.parent, "c" * 40)] if region_override else []
    )
    assert preview_checks == (
        [
            {
                "work_count": 2,
                "map_point_count": 1,
                "new_embedding_count": 2,
                "layout_reused": reuse_order,
                "continuity_receipt": {
                    "prior_revision": "b" * 40,
                    "stable_works": 1,
                    "current_hub_revision": "a" * 40,
                },
            }
        ]
        if region_override
        else []
    )
    assert result["region_continuity"] == (
        {
            "prior_revision": "b" * 40 if region_override else "a" * 40,
            "stable_works": 1,
            **({"current_hub_revision": "a" * 40} if region_override else {}),
            **(
                {
                    "chem_region_review_sha256": publisher.CHEM_REVIEW_SHA256,
                    "chemistry_review_root_revision": publisher.CHEMISTRY_HUB_REVISION,
                }
                if chemistry_active
                else {}
            ),
        }
        if pinned
        else None
    )
    assert result["region_comparison_revision"] == (
        "b" * 40 if region_override else "a" * 40 if pinned else None
    )
    assert result["dataset_commits"] == dict.fromkeys(
        publisher.DATASET_CONFIGS, "atomic-commit"
    )
    assert result["atomic_release"] == {"mode": "single_parent_commit"}
    assert len(atomic_calls) == 1
    if pinned and not bio_intent and not region_override:
        assert atomic_calls[-1]["prior_person_continuity_receipt"] == {"status": "pass"}
    if bio_intent and reuse_order:
        assert len(approval_loads) == 1
        assert approval_loads[0][0] == tmp_path / "approval.json"
        assert approval_loads[0][1]["prior_revision"] == "a" * 40
        assert approved_losses[-1] is approved_loss
    else:
        assert approval_loads == []
        if pinned:
            assert approved_losses[-1] is None
    assert observed_layouts == [reused_layouts if reuse_order else original_layouts]
    assert result["profile_count"] == 2
    assert result["new_embedding_count"] == 2
    assert result["reviewed_metadata_corrections"] == 1
    assert (
        result["reviewed_metadata_corrections_sha256"]
        == hashlib.sha256(corrections_path.read_bytes()).hexdigest()
    )
    assert result["mapped_works"] == 1
    assert result["excluded_works"] == 1
    assert result["reviewed_contexts"] == 1
    assert result["keywords"] == 1
    assert result["detail_keywords"] == 1
    assert result["topic_label_review_version"] == (
        publisher.TOPIC_LABEL_REVIEW_VERSION
    )
    if pinned and atomic and not region_override and not missing_prior and reuse_order:
        # The additive path must carry the pinned parent pending state before
        # the ordinary region continuity gate sees a rebuilt corpus.
        snapshot_digest["value"] = "test-snapshot"
        chemistry_active = False
        monkeypatch.setattr(
            publisher, "bio_scholar_source_changed", lambda *args: False
        )
        carryforward_calls = []

        def carryforward(works, profiles, prior_works, prior_profiles):
            carryforward_calls.append((works, profiles, prior_works, prior_profiles))
            return works, ["old"]

        monkeypatch.setattr(
            publisher, "carry_forward_parent_pending_works", carryforward
        )
        additive_options = {
            **publish_options,
            "require_bio_release": False,
            "additive_placement_review_path": tmp_path / "placement.json",
        }
        additive_options.pop("historical_loss_approval_path", None)
        # This synthetic orchestration fixture has no actual placement
        # manifest; exact generic-work binding is tested separately.
        monkeypatch.setattr(
            publisher,
            "_reviewed_new_generic_placement_work_ids",
            lambda *args, **kwargs: frozenset(),
        )
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            **additive_options,
        )
        assert len(carryforward_calls) == 1
        assert atomic_calls[-1]["parent_pending_work_ids"] == ["old"]


def test_historical_chemistry_count_offset_applies_only_at_its_parent(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(publisher, "CHEMISTRY_HUB_REVISION", "a" * 40)
    test_publish_snapshot_orchestrates_all_dataset_configs(
        monkeypatch,
        tmp_path,
        reuse_order=True,
        pinned=True,
        missing_prior=False,
        region_override=False,
        atomic=True,
    )


def test_region_override_requires_current_hub_revision(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (
            pandas.DataFrame(),
            {"snapshot_sha256": "test-snapshot"},
        ),
    )
    with pytest.raises(ValueError, match="immutable prior Hub revision"):
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
            region_comparison_revision="a" * 40,
        )


def test_sullivan_transition_requires_exact_source_commit(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    monkeypatch.setattr(
        publisher, "_require_current_dataset_revision", lambda **kwargs: None
    )
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (
            pandas.DataFrame(),
            {"snapshot_sha256": publisher.SULLIVAN_SOURCE_SHA256},
        ),
    )
    with pytest.raises(
        ValueError, match="Sullivan release needs its exact source commit"
    ):
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
            prior_revision=publisher.SULLIVAN_PRIOR_REVISION,
            atomic_intent_sink=lambda intent: None,
        )


def test_publish_snapshot_rejects_explicit_serial_upload(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        publisher,
        "_to_hub_dataset",
        lambda *args: pytest.fail("serial Hub write reached"),
    )
    monkeypatch.setattr(
        publisher,
        "_publish_atomic_release",
        lambda *args, **kwargs: pytest.fail("atomic Hub write reached"),
    )
    with pytest.raises(ValueError, match="one atomic Hub commit"):
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
            prior_revision="a" * 40,
            atomic_release=False,
        )


@pytest.mark.parametrize("released_gain", [False, True])
def test_publish_snapshot_holds_complete_profile_before_public_tables(
    monkeypatch, tmp_path, released_gain
) -> None:
    monkeypatch.setattr(
        publisher,
        "apply_reported_faculty_coauthors",
        lambda *args, **kwargs: ({}, []),
    )
    monkeypatch.setattr(publisher, "require_reported_source_sync", lambda *args: False)
    monkeypatch.setattr(publisher, "bio_scholar_source_changed", lambda *args: False)
    monkeypatch.setattr(
        publisher, "_require_current_dataset_revision", lambda **kwargs: None
    )
    monkeypatch.setattr(
        publisher,
        "_published_rows",
        lambda config, fields, **kwargs: (
            [{"work_id": "old"}]
            if config == "works"
            else [{"person_id": "prior-bio", "included_department_ids": ["bio"]}]
            if config == "people"
            else [{"author_pub_id": "curated:independent"}]
        ),
    )
    monkeypatch.setattr(
        publisher,
        "read_pinned_prior_people",
        lambda **kwargs: ([{"person_id": "prior-bio"}], "a" * 64),
    )
    monkeypatch.setattr(
        publisher,
        "require_prior_person_continuity",
        lambda *args, **kwargs: {"status": "pass"},
    )
    receipt = {
        "prior_revision": "a" * 40,
        "prior_visible_observations": 0,
        "current_visible_observations": int(released_gain),
        "new_source_gains": int(released_gain),
        **dict.fromkeys(
            (
                "reviewed_losses",
                "verified_gains",
                "reviewed_composite_observation_gains",
                "reviewed_composite_observation_losses",
                "approved_sullivan_held_losses",
                "approved_sullivan_rekey_losses",
                "approved_sullivan_additional_losses",
                "approved_sullivan_binding_changes",
                "unapproved_mapped_binding_changes",
                "unexplained_losses",
                "unexplained_gains",
            ),
            [],
        ),
    }
    monkeypatch.setattr(
        publisher,
        "reconcile_region_decisions",
        lambda works, *args, **kwargs: (works, receipt),
    )
    parent_map = tmp_path / "parent-publications.json"
    parent_map.write_text('{"point_count":1,"points":[],"keywords":[]}')
    freeze_calls = []

    def pinned_parent(*args, **kwargs):
        if released_gain:
            pytest.fail("Mixed release must not load a held-only parent map")
        assert kwargs["revision"] == "a" * 40
        return str(parent_map)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=pinned_parent),
    )
    monkeypatch.setattr(
        publisher,
        "build_map_artifact",
        lambda *args, **kwargs: {"point_count": 1, "points": []},
    )

    def freeze(works, **kwargs):
        freeze_calls.append(kwargs)
        return works

    monkeypatch.setattr(publisher, "preserve_held_only_topics", freeze)
    monkeypatch.setattr(publisher, "load_provisional_holds", lambda path: ())
    snapshot = tmp_path / "snapshots" / "source.parquet"
    cache_dir = tmp_path / "data" / "authors"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "ScholarOne.json"
    cache_path.write_text(
        json.dumps(
            {
                "scholar_id": "ScholarOne",
                "publication_count": 2,
                "publications": [
                    {"author_pub_id": "ScholarOne:a"},
                    {"author_pub_id": "ScholarOne:b"},
                ],
                "browser_capture": {
                    "coverage": "complete_visible_profile",
                    "more_remain": False,
                },
            }
        )
    )
    releases_path = tmp_path / "releases.json"
    releases_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profiles": [
                    {
                        "person_id": "person-one",
                        "scholar_id": "ScholarOne",
                        "state": "held",
                        "expected_source_rows": 2,
                        "cache_sha256": hashlib.sha256(
                            cache_path.read_bytes()
                        ).hexdigest(),
                        "reason": "Profile attribution under review.",
                        "reviewed_at": "2026-10-25",
                    }
                ],
            }
        )
    )
    original_release_sha = hashlib.sha256(releases_path.read_bytes()).hexdigest()
    corrections_path = tmp_path / "corrections.json"
    corrections_path.write_text(json.dumps({"schema_version": 1, "corrections": []}))
    raw = pandas.DataFrame(
        [
            {
                "person_id": "person-one",
                "scholar_id": "ScholarOne",
                "author_pub_id": f"ScholarOne:{letter}",
                "source_type": "scholar",
                "title": f"Held work {letter}",
                "year": 2020,
                "venue": "Journal",
                "authors": "One Person",
            }
            for letter in ("a", "b")
        ]
        + [
            {
                "person_id": "person-two",
                "scholar_id": "",
                "author_pub_id": "curated:independent",
                "source_type": "curated_verified",
                "title": "Independent work",
                "year": 2021,
                "venue": "Journal",
                "authors": "Two Person and One Person",
            }
        ]
    )
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (
            raw,
            {
                "created_at_utc": "2026-10-25T00:00:00+00:00",
                "snapshot_sha256": "source-sha",
            },
        ),
    )
    monkeypatch.setattr(publisher, "validate_doi_aliases", lambda *args: None)

    def embedding_input(frame, **kwargs):
        assert frame["person_id"].tolist() == ["person-two"]
        assert frame["authors"].tolist() == ["Two Person and One Person"]
        return frame, 0

    monkeypatch.setattr(publisher, "add_embeddings", embedding_input)

    def public_tables(frame, *args):
        assert frame["author_pub_id"].tolist() == ["curated:independent"]
        return {
            "profile_publications": frame.assign(
                review_exclusion_code="", context_verified_at=""
            ),
            "authorships": frame[["person_id", "scholar_id"]].copy(),
            "works": pandas.DataFrame(
                [
                    {
                        "observation_ids": ["observation-independent"],
                        "has_included_source_profile": True,
                        "map_eligible": True,
                        "department_ids": ["bme"],
                    }
                ]
            ),
            "people": pandas.DataFrame(
                [
                    {
                        "person_id": "person-one",
                        "publication_observation_count": 0,
                        "scholar_observation_count": 0,
                        "curated_observation_count": 0,
                        "unique_work_count": 0,
                        "mapped_work_count": 0,
                    }
                ]
            ),
        }

    monkeypatch.setattr(publisher, "build_dataset_tables", public_tables)
    monkeypatch.setattr(publisher, "_existing_layouts", lambda **kwargs: {})
    monkeypatch.setattr(publisher, "add_global_layout", lambda works, **kwargs: works)
    monkeypatch.setattr(
        publisher,
        "add_publication_keywords",
        lambda works: works.assign(keyword_id="k", detail_keyword_id="d"),
    )
    monkeypatch.setattr(
        publisher, "exclude_low_information_regions", lambda works, **kwargs: works
    )

    atomic_calls = []

    def upload_dataset(*args, **kwargs):
        atomic_calls.append(kwargs)
        releases_path.write_text("changed after decision load")
        return "commit", {"mode": "single_parent_commit"}

    monkeypatch.setattr(
        publisher,
        "_publish_atomic_release",
        upload_dataset,
    )
    result = publisher.publish_snapshot(
        snapshot,
        tmp_path / "manifest.json",
        hf_token="token",
        prior_revision="a" * 40,
        atomic_intent_sink=lambda intent: None,
        corrections_path=corrections_path,
        profile_releases_path=releases_path,
        source_cache_dir=cache_dir,
    )
    assert result["raw_snapshot_observations"] == 3
    assert result["raw_snapshot_profiles"] == 1
    assert result["raw_snapshot_sha256"] == "source-sha"
    assert result["held_profile_ids"] == ["ScholarOne"]
    assert result["held_source_observation_rows"] == 2
    assert result["held_observation_rows"] == 2
    assert result["profile_publications"] == 1
    assert result["profile_count"] == 0
    assert result["reviewed_metadata_corrections"] == 0
    assert result["profile_release_manifest_sha256"] == original_release_sha
    assert bool(freeze_calls) is not released_gain
    assert bool(atomic_calls[0]["held_topic_parent_artifact"]) is not released_gain
    if not released_gain:
        assert atomic_calls[0]["held_topic_parent_map_sha256"] == (
            hashlib.sha256(parent_map.read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("parent_digest", [None, "a" * 64])
def test_atomic_stage_records_held_only_parent_map_parity(
    monkeypatch, tmp_path, parent_digest
) -> None:
    class FakeDataset:
        def to_parquet(self, path):
            Path(path).write_bytes(b"staged parquet")

    parent_map = {
        "generated_at_utc": "2026-10-24T00:00:00+00:00",
        "point_count": 0,
        "keyword_count": 0,
        "points": [],
        "keywords": [],
    }
    candidate_map = {
        **parent_map,
        "generated_at_utc": "2026-10-25T00:00:00+00:00",
    }

    def write_map(*args, maps_dir, **kwargs):
        maps_dir.mkdir(parents=True)
        artifact_path = maps_dir / "publications.json"
        manifest_path = maps_dir / "manifest.json"
        artifact_path.write_text(json.dumps(candidate_map))
        manifest_path.write_text("{}")
        return artifact_path, manifest_path, candidate_map

    readme = tmp_path / "README.md"
    readme.write_text("Parent card")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda *args, **kwargs: str(readme)),
    )
    monkeypatch.setattr(publisher, "bio_scholar_source_changed", lambda *args: False)
    monkeypatch.setattr(publisher, "_to_hub_dataset", lambda frame: FakeDataset())
    monkeypatch.setattr(publisher, "_write_map_artifacts", write_map)
    monkeypatch.setattr(
        publisher,
        "retain_held_only_parent_works",
        lambda staged, **kwargs: (
            tables["works"],
            {name: 1 for name in publisher.DATASET_CONFIGS},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            Dataset=SimpleNamespace(from_parquet=lambda path: FakeDataset())
        ),
    )
    monkeypatch.setattr(
        publisher, "build_updated_dataset_card", lambda *args: "Staged card"
    )
    gates = []

    def commit(*args, prewrite_gate, **kwargs):
        gates.append(prewrite_gate)
        return "commit", {"mode": "single_parent_commit"}

    monkeypatch.setattr(publisher, "commit_atomic_release", commit)
    kwargs = dict(
        source_frame=pandas.DataFrame([{"source_type": "scholar"}]),
        source_manifest={"snapshot_sha256": "not-the-Bio-snapshot"},
        hf_token="token",
        prior_revision="a" * 40,
        generated_at_utc="2026-10-25T00:00:00+00:00",
        continuity_receipt={
            "prior_revision": "a" * 40,
            "unexplained_losses": [],
            "unexplained_gains": [],
        },
        repair_manifest_path=tmp_path / "repair.json",
        provisional_holds={},
        prior_profiles=[],
        prior_people=[],
        atomic_intent_sink=lambda intent: None,
        held_topic_parent_artifact=parent_map,
        held_topic_parent_map_sha256=parent_digest,
    )
    tables = {
        name: pandas.DataFrame([{"value": 1}]) for name in publisher.DATASET_CONFIGS
    }
    if parent_digest is None:
        with pytest.raises(ValueError, match="lacks pinned parent map digest"):
            publisher._publish_atomic_release(tables, registry(), **kwargs)
        assert not gates
    else:
        publisher._publish_atomic_release(tables, registry(), **kwargs)
        assert gates[0]["held_only_topic_parity"] == {
            "status": "pass",
            "point_count": 0,
            "keyword_count": 0,
            "parent_map_sha256": parent_digest,
            "exact_table_rows": {name: 1 for name in publisher.DATASET_CONFIGS},
        }


def test_physics_review_gate_blocks_held_raw_snapshot_before_enrichment(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        publisher, "_require_current_dataset_revision", lambda **kwargs: None
    )
    snapshot_path = tmp_path / "snapshots" / "snapshot.parquet"
    status_path = tmp_path / "status" / publisher.PHYSICS_REVIEW_STATUS_NAME
    status_path.parent.mkdir()
    status_path.write_text(
        json.dumps(
            {
                "batch": publisher.PHYSICS_REVIEW_BATCH,
                "map_quality_hold": True,
                "producer_review_ready": False,
                "unresolved_exact_ids": ["wJ_gQLoAAAAJ:held"],
                "paired_producer_review_count_by_profile": {"wJ_gQLoAAAAJ": 0},
            }
        )
    )
    frame = pandas.DataFrame({"scholar_id": ["wJ_gQLoAAAAJ"]})
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    stub_before_physics_review_gate(monkeypatch, frame)
    monkeypatch.setattr(publisher, "validate_doi_aliases", lambda *args: None)
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (
            frame,
            {"created_at_utc": "now", "snapshot_sha256": "test-snapshot"},
        ),
    )
    monkeypatch.setattr(
        publisher,
        "add_embeddings",
        lambda *args, **kwargs: pytest.fail("held rows reached enrichment"),
    )
    with pytest.raises(ValueError, match="source review is held"):
        publisher.publish_snapshot(
            snapshot_path,
            tmp_path / "manifest.json",
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
        )


@pytest.mark.parametrize(
    "scholar_id",
    ["wJ_gQLoAAAAJ", publisher.BREIVIK_REVIEW_SCHOLAR_ID],
)
def test_physics_quality_gates_receive_only_released_rows(
    monkeypatch, tmp_path, scholar_id
) -> None:
    frame = pandas.DataFrame({"scholar_id": [scholar_id]})
    monkeypatch.setattr(
        publisher, "_require_current_dataset_revision", lambda **kwargs: None
    )
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    monkeypatch.setattr(publisher, "validate_doi_aliases", lambda *args: None)
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (
            frame,
            {"created_at_utc": "now", "snapshot_sha256": "test-snapshot"},
        ),
    )
    stub_before_physics_review_gate(monkeypatch, frame)
    monkeypatch.setattr(
        publisher,
        "select_released_observations",
        lambda *args, **kwargs: (frame.iloc[:0].copy(), [], set()),
    )

    class AfterQualityGates(Exception):
        pass

    def next_gate(*args, **kwargs):
        raise AfterQualityGates

    monkeypatch.setattr(publisher, "select_released_person_observations", next_gate)
    with pytest.raises(AfterQualityGates):
        publisher.publish_snapshot(
            tmp_path / "snapshots" / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
        )


def test_physics_review_gate_requires_valid_released_status(tmp_path) -> None:
    snapshot_path = tmp_path / "snapshots" / "snapshot.parquet"
    status_path = tmp_path / "status" / publisher.PHYSICS_REVIEW_STATUS_NAME
    status_path.parent.mkdir()
    frame = pandas.DataFrame({"scholar_id": ["wJ_gQLoAAAAJ"]})
    exclusions = {("wJ_gQLoAAAAJ", "wJ_gQLoAAAAJ:reviewed"): object()}

    publisher._require_physics_review_ready(
        pandas.DataFrame({"scholar_id": ["other"]}), snapshot_path, exclusions
    )
    with pytest.raises(FileNotFoundError):
        publisher._require_physics_review_ready(frame, snapshot_path, exclusions)

    for invalid in ([], {"batch": "wrong"}):
        status_path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="invalid source status"):
            publisher._require_physics_review_ready(frame, snapshot_path, exclusions)

    ready = {
        "batch": publisher.PHYSICS_REVIEW_BATCH,
        "map_quality_hold": False,
        "producer_review_ready": True,
        "unresolved_exact_ids": [],
        "paired_producer_review_count_by_profile": {"wJ_gQLoAAAAJ": 1},
    }
    for change in (
        {"map_quality_hold": True},
        {"producer_review_ready": False},
        {"unresolved_exact_ids": ["wJ_gQLoAAAAJ:held"]},
    ):
        status_path.write_text(json.dumps({**ready, **change}))
        with pytest.raises(ValueError, match="source review is held"):
            publisher._require_physics_review_ready(frame, snapshot_path, exclusions)

    for change in (
        {"paired_producer_review_count_by_profile": None},
        {"paired_producer_review_count_by_profile": {"wJ_gQLoAAAAJ": 2}},
    ):
        status_path.write_text(json.dumps({**ready, **change}))
        with pytest.raises(ValueError, match="reviewed-exclusion counts differ"):
            publisher._require_physics_review_ready(frame, snapshot_path, exclusions)

    status_path.write_text(json.dumps(ready))
    publisher._require_physics_review_ready(frame, snapshot_path, exclusions)


def test_physics_review_gate_releases_only_source_bound_schumacher(tmp_path) -> None:
    snapshot_path = tmp_path / "snapshots" / "snapshot.parquet"
    status_path = tmp_path / "status" / publisher.PHYSICS_REVIEW_STATUS_NAME
    status_path.parent.mkdir()
    ledger_path = tmp_path / "status" / "schumacher-review.json"
    ledger_bytes = json.dumps(
        {
            "scholar_id": "LibSWRMAAAAJ",
            "reviewed_exclusion_count": 1,
            "provisional_hold_count": 2,
        }
    ).encode()
    ledger_path.write_bytes(ledger_bytes)
    decision = {
        "map_quality_hold": False,
        "producer_review_ready": True,
        "unresolved_exact_ids": [],
        "reviewed_exclusion_count": 1,
        "provisional_hold_count": 2,
        "review_ledger_path": "status/schumacher-review.json",
        "review_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
    }
    status = {
        "batch": publisher.PHYSICS_REVIEW_BATCH,
        "map_quality_hold": True,
        "producer_review_ready": False,
        "unresolved_exact_ids": ["wJ_gQLoAAAAJ:held"],
        "paired_producer_review_count_by_profile": {
            "LibSWRMAAAAJ": 1,
            "wJ_gQLoAAAAJ": 0,
        },
        "paired_profile_release_ready": {"LibSWRMAAAAJ": decision},
    }
    status_path.write_text(json.dumps(status))
    exclusions = {("LibSWRMAAAAJ", "LibSWRMAAAAJ:reviewed"): object()}
    schumacher = pandas.DataFrame({"scholar_id": ["LibSWRMAAAAJ"]})
    publisher._require_physics_review_ready(schumacher, snapshot_path, exclusions)

    both = pandas.DataFrame({"scholar_id": ["LibSWRMAAAAJ", "wJ_gQLoAAAAJ"]})
    with pytest.raises(ValueError, match="wJ_gQLoAAAAJ"):
        publisher._require_physics_review_ready(both, snapshot_path, exclusions)

    ledger_path.write_bytes(ledger_bytes + b" ")
    with pytest.raises(ValueError, match="profile review ledger changed"):
        publisher._require_physics_review_ready(schumacher, snapshot_path, exclusions)
    ledger_path.write_bytes(ledger_bytes)

    for change in (
        {"map_quality_hold": True},
        {"producer_review_ready": False},
        {"unresolved_exact_ids": ["LibSWRMAAAAJ:held"]},
    ):
        status["paired_profile_release_ready"]["LibSWRMAAAAJ"] = {
            **decision,
            **change,
        }
        status_path.write_text(json.dumps(status))
        with pytest.raises(ValueError, match="source review is held"):
            publisher._require_physics_review_ready(
                schumacher, snapshot_path, exclusions
            )

    for change in (
        {"review_ledger_path": None},
        {"review_ledger_path": "status/../schumacher-review.json"},
        {"review_ledger_sha256": None},
        {"review_ledger_sha256": "short"},
    ):
        status["paired_profile_release_ready"]["LibSWRMAAAAJ"] = {
            **decision,
            **change,
        }
        status_path.write_text(json.dumps(status))
        with pytest.raises(ValueError, match="invalid profile review ledger"):
            publisher._require_physics_review_ready(
                schumacher, snapshot_path, exclusions
            )

    for change in (
        {"reviewed_exclusion_count": 2},
        {"provisional_hold_count": 3},
    ):
        status["paired_profile_release_ready"]["LibSWRMAAAAJ"] = {
            **decision,
            **change,
        }
        status_path.write_text(json.dumps(status))
        with pytest.raises(ValueError, match="profile review ledger differs"):
            publisher._require_physics_review_ready(
                schumacher, snapshot_path, exclusions
            )


def test_publish_snapshot_corrects_metadata_before_embedding(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        publisher, "_require_current_dataset_revision", lambda **kwargs: None
    )
    import json

    from map_of_research.corrections import (
        DEFAULT_CORRECTIONS_PATH,
        load_corrections,
    )

    correction = load_corrections(DEFAULT_CORRECTIONS_PATH)[
        ("8iLLOQgAAAAJ", "8iLLOQgAAAAJ:TesyEGJKHF4C")
    ]
    selected = next(
        entry
        for entry in json.loads(DEFAULT_CORRECTIONS_PATH.read_text())["corrections"]
        if entry["author_pub_id"] == "8iLLOQgAAAAJ:TesyEGJKHF4C"
    )
    corrections_path = tmp_path / "selected-correction.json"
    corrections_path.write_text(
        json.dumps({"schema_version": 1, "corrections": [selected]})
    )
    doi_aliases_path = tmp_path / "empty-doi-aliases.json"
    doi_aliases_path.write_text(json.dumps({"schema_version": 1, "aliases": []}))
    raw = json.dumps({"pub_year": "1712", "title": correction.expected["title"]})
    frame = pandas.DataFrame(
        [
            {
                "scholar_id": "8iLLOQgAAAAJ",
                "author_pub_id": "8iLLOQgAAAAJ:TesyEGJKHF4C",
                "source_type": "scholar",
                "source_record_json": raw,
                **correction.expected,
            }
        ]
    )
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    monkeypatch.setattr(publisher, "load_profile_releases", lambda path: ())
    monkeypatch.setattr(publisher, "load_provisional_holds", lambda path: ())
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (
            frame,
            {
                "created_at_utc": "2026-10-25T00:00:00Z",
                "snapshot_sha256": "test-snapshot",
            },
        ),
    )

    class BeforeUpload(Exception):
        pass

    def check_before_embedding(effective, **kwargs):
        assert kwargs["hf_token"] == "token"
        assert effective.loc[0, "year"] == 2011
        assert effective.loc[0, "source_record_json"] == raw
        assert effective.loc[0, "metadata_correction_evidence_url"] == (
            correction.evidence_url
        )
        assert frame.loc[0, "year"] == 1712
        raise BeforeUpload

    monkeypatch.setattr(publisher, "add_embeddings", check_before_embedding)
    with pytest.raises(BeforeUpload):
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
            corrections_path=corrections_path,
            doi_aliases_path=doi_aliases_path,
        )


def test_breivik_gate_blocks_current_hold_before_enrichment(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        publisher, "_require_current_dataset_revision", lambda **kwargs: None
    )
    snapshot_path = tmp_path / "snapshots" / "snapshot.parquet"
    status_path = tmp_path / "status" / publisher.BREIVIK_REVIEW_STATUS_NAME
    status_path.parent.mkdir()
    status_path.write_text(
        json.dumps(
            {
                "batch": publisher.BREIVIK_REVIEW_BATCH,
                "ingestion_ready": False,
                "snapshot_release_hold": True,
                "preliminary_candidate_count": 8,
                "reviewed_exclusion_count_present_in_producer_main": 2,
            }
        )
    )
    frame = pandas.DataFrame({"scholar_id": [publisher.BREIVIK_REVIEW_SCHOLAR_ID]})
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    stub_before_physics_review_gate(monkeypatch, frame)
    monkeypatch.setattr(publisher, "validate_doi_aliases", lambda *args: None)
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (
            frame,
            {"created_at_utc": "now", "snapshot_sha256": "test-snapshot"},
        ),
    )
    monkeypatch.setattr(
        publisher,
        "add_embeddings",
        lambda *args, **kwargs: pytest.fail("held rows reached enrichment"),
    )
    with pytest.raises(
        ValueError, match="Breivik map-quality gate: source review is held"
    ):
        publisher.publish_snapshot(
            snapshot_path,
            tmp_path / "manifest.json",
            hf_token="token",
            prior_revision="a" * 40,
            atomic_intent_sink=lambda intent: None,
        )


def test_breivik_gate_requires_explicit_release_status(tmp_path) -> None:
    snapshot_path = tmp_path / "snapshots" / "snapshot.parquet"
    status_path = tmp_path / "status" / publisher.BREIVIK_REVIEW_STATUS_NAME
    status_path.parent.mkdir()
    frame = pandas.DataFrame({"scholar_id": [publisher.BREIVIK_REVIEW_SCHOLAR_ID]})
    exclusions = {
        (publisher.BREIVIK_REVIEW_SCHOLAR_ID, f"Breivik:reviewed-{i}"): object()
        for i in range(7)
    }
    exclusions[("other", "other:reviewed")] = object()

    publisher._require_breivik_review_ready(
        pandas.DataFrame({"scholar_id": ["other"]}), snapshot_path, exclusions
    )
    with pytest.raises(FileNotFoundError):
        publisher._require_breivik_review_ready(frame, snapshot_path, exclusions)

    status_path.write_text("{")
    with pytest.raises(json.JSONDecodeError):
        publisher._require_breivik_review_ready(frame, snapshot_path, exclusions)

    for invalid in ([], {"batch": "wrong"}):
        status_path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="invalid source status"):
            publisher._require_breivik_review_ready(frame, snapshot_path, exclusions)

    ready = {
        "batch": publisher.BREIVIK_REVIEW_BATCH,
        "ingestion_ready": True,
        "snapshot_release_hold": False,
        "preliminary_candidate_count": 0,
        "quality_review_complete": True,
        "reviewed_exclusion_count_present_in_producer_main": 7,
    }
    for change in (
        {"ingestion_ready": False},
        {"snapshot_release_hold": True},
        {"preliminary_candidate_count": 1},
        {"preliminary_candidate_count": False},
        {"quality_review_complete": False},
    ):
        status_path.write_text(json.dumps({**ready, **change}))
        with pytest.raises(ValueError, match="source review is held"):
            publisher._require_breivik_review_ready(frame, snapshot_path, exclusions)

    for missing_field in (
        "ingestion_ready",
        "snapshot_release_hold",
        "preliminary_candidate_count",
        "quality_review_complete",
    ):
        status_path.write_text(
            json.dumps(
                {key: value for key, value in ready.items() if key != missing_field}
            )
        )
        with pytest.raises(ValueError, match="source review is held"):
            publisher._require_breivik_review_ready(frame, snapshot_path, exclusions)

    for changed_count in (2, 8, True, "7", None):
        status_path.write_text(
            json.dumps(
                {
                    **ready,
                    "reviewed_exclusion_count_present_in_producer_main": changed_count,
                }
            )
        )
        with pytest.raises(ValueError, match="reviewed-exclusion count differs"):
            publisher._require_breivik_review_ready(frame, snapshot_path, exclusions)

    status_path.write_text(json.dumps(ready))
    publisher._require_breivik_review_ready(frame, snapshot_path, exclusions)


def test_workflow_url_and_argument_parser(monkeypatch, tmp_path) -> None:
    for name in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    assert publisher._workflow_url() is None
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.test")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    assert (
        publisher._workflow_url() == "https://github.test/owner/repo/actions/runs/123"
    )

    args = publisher._parse_args(
        [
            "--snapshot",
            str(tmp_path / "a.parquet"),
            "--manifest",
            str(tmp_path / "m.json"),
            "--prior-revision",
            "a" * 40,
        ]
    )
    assert args.max_age_days == publisher.DEFAULT_MAX_AGE_DAYS


@pytest.mark.parametrize("explicit_manifest", [True, False])
def test_main_uses_trusted_chemistry_manifest_from_workspace_root(
    monkeypatch, tmp_path, explicit_manifest: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = Path("source/data/curated/scholar-chem-region-review.json")
    manifest.parent.mkdir(parents=True)
    manifest.write_text("reviewed")
    monkeypatch.setenv(publisher.HF_TOKEN_ENV, "token")
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (pandas.DataFrame(), {}),
    )
    monkeypatch.setattr(publisher, "validate_math_coverage", lambda *args, **kwargs: {})
    monkeypatch.setattr(publisher, "validate_coverage", lambda **kwargs: {})
    seen = []

    def publish(*args, **kwargs):
        chem_review = kwargs["chem_region_review_path"]
        seen.append(chem_review)
        if not chem_review.is_file():
            raise FileNotFoundError(f"Chemistry review is missing: {chem_review}")
        return {"works": 4}

    monkeypatch.setattr(publisher, "publish_snapshot", publish)
    cli = [
        "--snapshot",
        "snapshot/snapshots/cmu-engineering-publications.parquet",
        "--manifest",
        "snapshot/snapshots/cmu-engineering-publications.parquet.manifest.json",
        "--prior-revision",
        "a" * 40,
        "--status-file",
        str(tmp_path / "status.json"),
    ]
    if explicit_manifest:
        cli.extend(["--chem-region-review-manifest", str(manifest)])
    result = publisher.main(cli)
    assert result == (0 if explicit_manifest else 1)
    assert seen == [
        manifest if explicit_manifest else publisher.DEFAULT_CHEM_REGION_REVIEW_PATH
    ]
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == ("success" if explicit_manifest else "failure")
    if not explicit_manifest:
        assert status["error_type"] == "FileNotFoundError"


@pytest.mark.parametrize(
    ("with_token", "region_failure", "expected"),
    [(True, False, 0), (True, True, 1), (False, False, 1)],
)
def test_main_always_writes_terminal_status(
    monkeypatch,
    tmp_path,
    with_token,
    region_failure,
    expected,
) -> None:
    status_path = tmp_path / "status.json"
    if with_token:
        monkeypatch.setenv(publisher.HF_TOKEN_ENV, "token")
    else:
        monkeypatch.delenv(publisher.HF_TOKEN_ENV, raising=False)
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (pandas.DataFrame(), {}),
    )
    monkeypatch.setattr(
        publisher,
        "validate_math_coverage",
        lambda *args, **kwargs: {"complete_scholar_profiles": 1},
    )
    monkeypatch.setattr(
        publisher,
        "validate_coverage",
        lambda **kwargs: {"legacy_profiles": 0, "complete_profiles": 1},
    )
    receipt = {
        "unexplained_losses": ["person/profile/id/scholar"],
        "unexplained_gains": [],
    }
    if region_failure:

        def reject_release(*args, **kwargs):
            raise RegionReleaseDiffError(receipt)

        monkeypatch.setattr(publisher, "publish_snapshot", reject_release)
    else:
        monkeypatch.setattr(
            publisher,
            "publish_snapshot",
            lambda *args, **kwargs: {"works": 4},
        )
    exit_code = publisher.main(
        [
            "--snapshot",
            str(tmp_path / "snapshot.parquet"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--prior-revision",
            "a" * 40,
            "--status-file",
            str(status_path),
        ]
    )
    status = json.loads(status_path.read_text())
    assert exit_code == expected
    assert status["status"] == ("success" if expected == 0 else "failure")
    if region_failure:
        assert status["region_continuity"] == receipt
    assert status["finished_at_utc"]
    if with_token:
        assert status["math_coverage"] == {"complete_scholar_profiles": 1}
        assert status["scholar_coverage"] == {
            "legacy_profiles": 0,
            "complete_profiles": 1,
        }


def test_main_persists_presentation_drift_failure_receipt(
    monkeypatch, tmp_path
) -> None:
    status_path = tmp_path / "status.json"
    monkeypatch.setenv(publisher.HF_TOKEN_ENV, "token")
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (pandas.DataFrame(), {}),
    )
    monkeypatch.setattr(
        publisher,
        "validate_math_coverage",
        lambda *args, **kwargs: {"complete_scholar_profiles": 1},
    )
    monkeypatch.setattr(
        publisher,
        "validate_coverage",
        lambda **kwargs: {"legacy_profiles": 0, "complete_profiles": 1},
    )
    receipt = {"status": "review_required", "report_sha256": "exact-report"}

    def reject_release(*args, **kwargs):
        raise publisher.PresentationDriftError(receipt, "material drift")

    monkeypatch.setattr(publisher, "publish_snapshot", reject_release)
    assert (
        publisher.main(
            [
                "--snapshot",
                str(tmp_path / "snapshot.parquet"),
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--prior-revision",
                "a" * 40,
                "--status-file",
                str(status_path),
            ]
        )
        == 1
    )
    status = json.loads(status_path.read_text())
    assert status["status"] == "failure"
    assert status["presentation_drift"] == receipt


def test_main_persists_ambiguous_hub_write_reconciliation_receipt(
    monkeypatch, tmp_path
) -> None:
    status_path = tmp_path / "status.json"
    monkeypatch.setenv(publisher.HF_TOKEN_ENV, "token")
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    monkeypatch.setattr(
        publisher, "validate_snapshot", lambda *args, **kwargs: (pandas.DataFrame(), {})
    )
    monkeypatch.setattr(publisher, "validate_math_coverage", lambda *args, **kwargs: {})
    monkeypatch.setattr(publisher, "validate_coverage", lambda **kwargs: {})
    receipt = {
        "verification_status": "reconciliation_required",
        "verification_stage": "create_commit",
        "verification_error_type": "ConnectionError",
        "verification_error_message": "response lost",
        "write_outcome": "unknown",
        "parent_revision": "a" * 40,
        "artifact_revision": None,
        "observed_head_revision": "b" * 40,
        "sha256_by_path": {"README.md": "c" * 64},
    }

    def ambiguous_write(*args, **kwargs):
        kwargs["atomic_intent_sink"](
            {
                "intent_status": "before_write",
                "parent_revision": "a" * 40,
                "sha256_by_path": {"README.md": "c" * 64},
            }
        )
        checkpoint = json.loads(status_path.read_text())
        assert checkpoint["status"] == "failure"
        assert checkpoint["atomic_release_intent"]["intent_status"] == "before_write"
        raise publisher.AtomicReleaseReconciliationRequired(receipt)

    monkeypatch.setattr(publisher, "publish_snapshot", ambiguous_write)
    exit_code = publisher.main(
        [
            "--snapshot",
            str(tmp_path / "snapshot.parquet"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--prior-revision",
            "a" * 40,
            "--status-file",
            str(status_path),
        ]
    )
    status = json.loads(status_path.read_text())
    assert exit_code == 1
    assert status["status"] == "failure"
    assert status["atomic_release_reconciliation"] == receipt
    assert status["atomic_release_intent"]["intent_status"] == "before_write"
    assert "artifact_commit" not in status
