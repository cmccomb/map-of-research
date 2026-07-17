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
            return numpy.ones((1, 768), dtype=numpy.float32)

    frame["map_slugs"] = [["map-of-ece"], ["map-of-ece"]]
    frame["memberships"] = [
        [{"map_slug": "map-of-ece", "department": "ECE", "faculty": "A"}],
        [{"map_slug": "map-of-ece", "department": "ECE", "faculty": "B"}],
    ]
    enriched, new_count = publisher.add_embeddings(
        frame,
        hf_token="test",
        encoder=Encoder(),
    )

    assert new_count == 1
    assert enriched.loc[0, "embedding"] == [0.0] * 768
    assert enriched.loc[1, "embedding"] == [1.0] * 768


def test_map_artifact_expands_only_requested_membership() -> None:
    frame = pandas.DataFrame(
        [
            {
                "faculty": "Canonical Name",
                "memberships": [
                    {
                        "map_slug": "map-of-ece",
                        "department": "ECE",
                        "faculty": "ECE Label",
                    },
                    {
                        "map_slug": "map-of-cmu-silicon-valley",
                        "department": "CMU Silicon Valley",
                        "faculty": "SV Label",
                    },
                ],
                "embedding": [0.0] * 768,
                "title": "A Paper",
                "year": 2025,
                "citation_count": 7,
                "source_url": "https://example.test/paper",
            }
        ]
    )

    artifact = publisher.build_map_artifact(
        frame,
        map_slug="map-of-cmu-silicon-valley",
        title="CMU Silicon Valley",
        generated_at_utc="2026-07-17T12:00:00+00:00",
    )

    assert artifact["point_count"] == 1
    assert artifact["points"][0]["faculty"] == "SV Label"
    assert artifact["points"][0]["group"] == "SV Label"


def test_map_artifact_handles_missing_year() -> None:
    frame = pandas.DataFrame(
        [
            {
                "faculty": "A",
                "memberships": [
                    {
                        "map_slug": "map-of-ece",
                        "department": "ECE",
                        "faculty": "A",
                    }
                ],
                "embedding": [0.0] * 768,
                "title": "Undated",
                "year": numpy.nan,
                "citation_count": 0,
                "source_url": "",
            }
        ]
    )

    artifact = publisher.build_map_artifact(
        frame,
        map_slug="map-of-ece",
        title="ECE",
        generated_at_utc="2026-07-17T12:00:00+00:00",
    )

    assert artifact["points"][0]["year"] is None
