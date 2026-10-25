import hashlib
import json
import sys
from types import SimpleNamespace

import numpy
import pandas
import pytest

import map_of_research.publisher as publisher
from map_of_research.registry import Department, Membership, Person, Registry, School


class Rows(list):
    pass


def install_published_rows(monkeypatch, loader) -> None:
    monkeypatch.setattr(publisher, "_published_rows", loader)


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
    assert len(publisher._existing_embeddings(hf_token="token")) == 2

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
                "map_eligible": True,
                "map_exclusion_reasons": [],
                "keyword_model_version": publisher.KEYWORD_MODEL_VERSION,
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
                "citation_count": 7,
                "doi": "10.1000/test",
                "source_urls": ["https://example.test/paper"],
                "observation_count": 2,
                "last_fetched_at_utc": "2026-01-01T00:00:00+00:00",
            }
        ]
    )


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
    works["source_verified_at"] = "2026-10-24"
    works["source_verification_urls"] = [["https://example.test/publisher"]]
    works["last_fetched_at_utc"] = ""

    artifact = publisher.build_map_artifact(
        works, registry(), generated_at_utc="2026-10-24T12:00:00+00:00"
    )
    point = artifact["points"][0]
    assert point["source_types"] == ["curated_verified"]
    assert point["source_verified_at"] == "2026-10-24"
    assert point["source_verification_urls"] == ["https://example.test/publisher"]
    assert point["external_coauthor_ids"] == ["external-paul-erdos"]
    assert artifact["catalogs"]["external_coauthors"] == [external]
    assert artifact["source_data_newest_at_utc"] is None

    # Two records may name the same external person, but may not disagree.
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

    def published_rows(config, fields, *, hf_token):
        calls.append((config, hf_token, fields))
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
        people=source.people,
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


def test_map_artifact_upload_writes_manifest_and_requires_commit(monkeypatch) -> None:
    uploads = []

    class HfApi:
        def __init__(self, token):
            assert token == "token"

        def upload_folder(self, **kwargs):
            folder = kwargs["folder_path"]
            uploads.append(
                (
                    kwargs,
                    json.loads((folder / publisher.MAP_ARTIFACT_NAME).read_text()),
                    json.loads((folder / "manifest.json").read_text()),
                )
            )
            return SimpleNamespace(oid="artifact-commit")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=HfApi))
    commit = publisher._upload_map_artifacts(
        work_frame(),
        registry(),
        hf_token="token",
        generated_at_utc="2026-07-17T12:00:00+00:00",
        dataset_commits={"works": "works-commit"},
    )
    assert commit == "artifact-commit"
    assert uploads[0][1]["point_count"] == 1
    assert uploads[0][2]["dataset_commits"] == {"works": "works-commit"}
    assert uploads[0][2]["topic_label_review_version"] == (
        publisher.TOPIC_LABEL_REVIEW_VERSION
    )
    assert uploads[0][2]["artifact"]["keyword_count"] == 2
    assert uploads[0][2]["artifact"]["school_count"] == 1
    assert uploads[0][2]["artifact"]["keyword_levels"][1]["keyword_count"] == 1

    HfApi.upload_folder = lambda self, **kwargs: SimpleNamespace()
    with pytest.raises(RuntimeError, match="did not return a commit"):
        publisher._upload_map_artifacts(
            work_frame(),
            registry(),
            hf_token="token",
            generated_at_utc="2026-07-17T12:00:00+00:00",
            dataset_commits={},
        )


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
def test_publish_snapshot_orchestrates_all_dataset_configs(
    monkeypatch,
    tmp_path,
    reuse_order,
) -> None:
    frame = pandas.DataFrame({"scholar_id": ["one", "two", "one"]})
    tables = {
        name: pandas.DataFrame({"value": [1, 2]}) for name in publisher.DATASET_CONFIGS
    }
    tables["works"] = pandas.DataFrame(
        {
            "value": [1, 2],
            "department_ids": [["ece"], ["ece"]],
            "map_eligible": [True, False],
        }
    )
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (
            frame,
            {"created_at_utc": "2026-07-17T12:00:00+00:00"},
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
        lambda works, previous, layouts: (
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
    monkeypatch.setattr(
        publisher,
        "_upload_map_artifacts",
        lambda *args, **kwargs: "artifact-commit",
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
    result = publisher.publish_snapshot(
        tmp_path / "snapshot.parquet",
        tmp_path / "manifest.json",
        hf_token="token",
    )
    assert result["dataset_commits"] == {
        name: f"{name}-commit" for name in publisher.DATASET_CONFIGS
    }
    assert observed_layouts == [reused_layouts if reuse_order else original_layouts]
    assert result["profile_count"] == 2
    assert result["new_embedding_count"] == 2
    assert result["mapped_works"] == 1
    assert result["excluded_works"] == 1
    assert result["keywords"] == 1
    assert result["detail_keywords"] == 1
    assert result["topic_label_review_version"] == (
        publisher.TOPIC_LABEL_REVIEW_VERSION
    )


def test_publish_snapshot_rejects_upload_without_commit(monkeypatch, tmp_path) -> None:
    frame = pandas.DataFrame({"scholar_id": ["one"]})
    tables = {
        name: pandas.DataFrame({"value": [1]}) for name in publisher.DATASET_CONFIGS
    }
    monkeypatch.setattr(publisher, "load_registry", lambda *args: registry())
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (frame, {"created_at_utc": "now"}),
    )
    monkeypatch.setattr(publisher, "add_embeddings", lambda *args, **kwargs: (frame, 0))
    monkeypatch.setattr(publisher, "build_dataset_tables", lambda *args: tables)
    monkeypatch.setattr(publisher, "_existing_layouts", lambda **kwargs: {})
    monkeypatch.setattr(publisher, "add_global_layout", lambda works, **kwargs: works)
    monkeypatch.setattr(publisher, "add_publication_keywords", lambda works: works)
    monkeypatch.setattr(
        publisher,
        "exclude_low_information_regions",
        lambda works: works,
    )
    monkeypatch.setattr(
        publisher,
        "_to_hub_dataset",
        lambda table: SimpleNamespace(
            push_to_hub=lambda *args, **kwargs: SimpleNamespace()
        ),
    )
    with pytest.raises(RuntimeError, match="upload did not return a commit"):
        publisher.publish_snapshot(
            tmp_path / "snapshot.parquet",
            tmp_path / "manifest.json",
            hf_token="token",
        )


def test_physics_review_gate_blocks_held_raw_snapshot_before_enrichment(
    monkeypatch, tmp_path
) -> None:
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
    monkeypatch.setattr(
        publisher,
        "validate_snapshot",
        lambda *args, **kwargs: (frame, {"created_at_utc": "now"}),
    )
    monkeypatch.setattr(
        publisher,
        "add_embeddings",
        lambda *args, **kwargs: pytest.fail("held rows reached enrichment"),
    )
    with pytest.raises(ValueError, match="source review is held"):
        publisher.publish_snapshot(
            snapshot_path, tmp_path / "manifest.json", hf_token="token"
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
        ]
    )
    assert args.max_age_days == publisher.DEFAULT_MAX_AGE_DAYS


@pytest.mark.parametrize(("with_token", "expected"), [(True, 0), (False, 1)])
def test_main_always_writes_terminal_status(
    monkeypatch,
    tmp_path,
    with_token,
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
            "--status-file",
            str(status_path),
        ]
    )
    status = json.loads(status_path.read_text())
    assert exit_code == expected
    assert status["status"] == ("success" if with_token else "failure")
    assert status["finished_at_utc"]
    if with_token:
        assert status["math_coverage"] == {"complete_scholar_profiles": 1}
        assert status["scholar_coverage"] == {
            "legacy_profiles": 0,
            "complete_profiles": 1,
        }
