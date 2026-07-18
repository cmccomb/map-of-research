import numpy
import pandas

import map_of_research.publisher as publisher
from map_of_research.registry import Department, Membership, Person, Registry


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


def work_frame() -> pandas.DataFrame:
    return pandas.DataFrame(
        [
            {
                "work_id": "work-one",
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

    assert artifact["schema_version"] == 4
    assert artifact["layout_version"] == publisher.LAYOUT_VERSION
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


def test_global_layout_attaches_pca_and_tsne_coordinates(monkeypatch) -> None:
    frame = pandas.DataFrame(
        [
            {
                "work_id": "one",
                "department_ids": ["ece"],
                "embedding": [3.0, 4.0, 0.0],
            },
            {
                "work_id": "two",
                "department_ids": ["mse"],
                "embedding": [0.0, 0.0, 2.0],
            },
            {
                "work_id": "excluded",
                "department_ids": [],
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
                "department_ids": ["ece"],
                "embedding": [1.0, 0.0],
            },
            {
                "work_id": "two",
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
