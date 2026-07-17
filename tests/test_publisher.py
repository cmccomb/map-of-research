import numpy
import pandas

import map_of_research.publisher as publisher


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
                        "map_slug": "map-of-ece",
                        "map_title": "Electrical & Computer Engineering",
                        "role": "faculty",
                    },
                    {
                        "person_id": "person-one",
                        "display_name": "One Person",
                        "map_slug": "map-of-cmu-silicon-valley",
                        "map_title": "CMU Silicon Valley",
                        "role": "teaching",
                    },
                ],
                "x": 0.25,
                "y": -0.5,
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


def test_map_artifact_uses_shared_coordinates_and_array_relationships() -> None:
    artifact = publisher.build_map_artifact(
        work_frame(),
        map_slug="map-of-cmu-silicon-valley",
        title="CMU Silicon Valley",
        generated_at_utc="2026-07-17T12:00:00+00:00",
    )

    assert artifact["schema_version"] == 2
    assert artifact["layout_version"] == publisher.LAYOUT_VERSION
    assert artifact["point_count"] == 1
    assert artifact["points"][0]["faculty"] == ["One Person"]
    assert artifact["points"][0]["groups"] == ["One Person"]
    assert artifact["points"][0]["x"] == 0.25


def test_engineering_map_groups_by_map_title() -> None:
    artifact = publisher.build_map_artifact(
        work_frame(),
        map_slug="map-of-eng",
        title="Engineering",
        generated_at_utc="2026-07-17T12:00:00+00:00",
    )

    assert artifact["points"][0]["groups"] == [
        "CMU Silicon Valley",
        "Electrical & Computer Engineering",
    ]


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
    artifact = publisher.build_map_artifact(
        work_frame(),
        map_slug="map-of-iii",
        title="Integrated Innovation Institute",
        generated_at_utc="2026-07-17T12:00:00+00:00",
    )

    assert artifact["point_count"] == 0
    assert artifact["points"] == []
    assert artifact["source_data_newest_at_utc"] is None
