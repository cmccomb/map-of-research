import json
import sys
from types import SimpleNamespace

import numpy
import pandas
import pytest

import map_of_research.publisher as publisher
from map_of_research.registry import Department, Membership, Person, Registry


class Rows(list):
    def __init__(self, rows, columns=None):
        super().__init__(rows)
        self.column_names = columns or list(rows[0] if rows else [])


def install_datasets(monkeypatch, loader) -> None:
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=loader),
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

    def load_dataset(repo_id, *, name, split, token):
        calls.append((repo_id, name, split, token))
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

    install_datasets(monkeypatch, load_dataset)

    assert publisher._existing_embeddings(hf_token="token") == {
        ("a:one", "Reusable"): [1.0] * publisher.EMBEDDING_DIMENSION
    }
    assert [call[1] for call in calls] == list(publisher.REUSABLE_EMBEDDING_CONFIGS)


def test_existing_embeddings_returns_empty_when_no_vectors_exist(monkeypatch) -> None:
    invalid = Rows([{"author_pub_id": "", "title": "", "embedding": []}])
    install_datasets(monkeypatch, lambda *args, **kwargs: invalid)
    assert publisher._existing_embeddings(hf_token="") == {}


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
                title="Biomedical Engineering",
                directory_url="https://example.test/bme",
                reviewed_at="2026-07-17",
                review_notes="Test",
            ),
            Department(
                department_id="cmu-silicon-valley",
                title="CMU Silicon Valley",
                directory_url="https://example.test/sv",
                reviewed_at="2026-07-17",
                review_notes="Test",
            ),
            Department(
                department_id="ece",
                title="Electrical & Computer Engineering",
                directory_url="https://example.test/ece",
                reviewed_at="2026-07-17",
                review_notes="Test",
            ),
        ),
    )


def test_map_artifact_uses_shared_coordinates_and_id_relationships() -> None:
    artifact = publisher.build_map_artifact(
        work_frame(),
        registry(),
        generated_at_utc="2026-07-17T12:00:00+00:00",
    )

    assert artifact["schema_version"] == 6
    assert artifact["quality_assessment_version"] == (
        publisher.QUALITY_ASSESSMENT_VERSION
    )
    assert artifact["layout_version"] == publisher.LAYOUT_VERSION
    assert artifact["keyword_model_version"] == publisher.KEYWORD_MODEL_VERSION
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
    assert artifact["points"][0]["x"] == 0.25
    assert artifact["points"][0]["tsne_x"] == -0.75
    assert artifact["points"][0]["keyword_id"] == "keyword-01"
    assert artifact["keyword_count"] == 1
    assert artifact["keywords"] == [
        {
            "keyword_id": "keyword-01",
            "label": "reliable systems",
            "publication_count": 1,
            "coordinates": {
                "pca": {"x": 0.25, "y": -0.5},
                "tsne": {"x": -0.75, "y": 0.125},
            },
        }
    ]


def test_map_artifact_requires_consistent_keywords() -> None:
    missing = work_frame()
    missing.loc[0, "keyword_id"] = ""
    with pytest.raises(ValueError, match="must have a topic keyword"):
        publisher.build_map_artifact(
            missing,
            registry(),
            generated_at_utc="2026-07-17T12:00:00+00:00",
        )

    inconsistent = pandas.concat([work_frame(), work_frame()], ignore_index=True)
    inconsistent.loc[1, "work_id"] = "work-two"
    inconsistent.loc[1, "keyword"] = "different label"
    with pytest.raises(ValueError, match="inconsistent labels"):
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
    install_datasets(monkeypatch, lambda *args, **kwargs: valid)
    assert publisher._existing_layouts(hf_token="token") == {
        "one": {"x": 0.0, "y": 1.0, "tsne_x": 2.0, "tsne_y": 3.0}
    }

    invalid_rows = [
        Rows([], columns=["work_id"]),
        Rows([{**valid[0], "layout_version": "old"}]),
        Rows([{**valid[0], "x": "bad"}]),
        Rows([{**valid[0], "x": numpy.inf}]),
        Rows([{**valid[0], "work_id": ""}]),
        Rows([valid[0], valid[0]]),
    ]
    for rows in invalid_rows:
        install_datasets(monkeypatch, lambda *args, _rows=rows, **kwargs: _rows)
        assert publisher._existing_layouts(hf_token="token") == {}


def test_existing_layouts_tolerates_missing_prior_dataset(monkeypatch) -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError

    install_datasets(monkeypatch, missing)
    assert publisher._existing_layouts(hf_token="token") == {}

    columns = ["work_id", "layout_version", "x", "y", "tsne_x", "tsne_y"]
    install_datasets(monkeypatch, lambda *args, **kwargs: Rows([], columns=columns))
    assert publisher._existing_layouts(hf_token="token") == {}


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
    assert departments["bme"]["publication_count"] == 0


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
    assert uploads[0][2]["artifact"]["keyword_count"] == 1

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


def test_publish_snapshot_orchestrates_all_dataset_configs(
    monkeypatch,
    tmp_path,
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
    monkeypatch.setattr(publisher, "_existing_layouts", lambda **kwargs: {})
    monkeypatch.setattr(publisher, "add_global_layout", lambda works, **kwargs: works)
    monkeypatch.setattr(
        publisher,
        "add_publication_keywords",
        lambda works: works.assign(keyword_id=["keyword-01", ""]),
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
    assert result["profile_count"] == 2
    assert result["new_embedding_count"] == 2
    assert result["mapped_works"] == 1
    assert result["excluded_works"] == 1
    assert result["keywords"] == 1


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
