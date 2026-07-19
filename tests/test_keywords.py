import numpy
import pandas
import pytest

import map_of_research.keywords as keywords


def works_frame() -> pandas.DataFrame:
    rows = []
    titles_and_coordinates = [
        ("Robot teamwork coordination", -1.0, -1.0),
        ("Robot teamwork planning", -1.1, -0.9),
        ("Robot teamwork control", -0.9, -1.1),
        ("Battery health prediction", 1.0, 1.0),
        ("Battery health monitoring", 1.1, 0.9),
        ("Battery health estimation", 0.9, 1.1),
    ]
    for index, (title, tsne_x, tsne_y) in enumerate(titles_and_coordinates):
        rows.append(
            {
                "work_id": f"work-{index}",
                "title": title,
                "map_eligible": True,
                "department_ids": ["ece"],
                "tsne_x": tsne_x,
                "tsne_y": tsne_y,
            }
        )
    rows.extend(
        [
            {
                "work_id": "excluded",
                "title": "Excluded front matter",
                "map_eligible": False,
                "department_ids": ["ece"],
                "tsne_x": numpy.nan,
                "tsne_y": numpy.nan,
            },
            {
                "work_id": "outside-roster",
                "title": "External research",
                "map_eligible": True,
                "department_ids": [],
                "tsne_x": numpy.nan,
                "tsne_y": numpy.nan,
            },
        ]
    )
    return pandas.DataFrame(rows)


def test_keywords_are_deterministic_visible_regions_and_retain_every_work() -> None:
    first = keywords.add_publication_keywords(works_frame(), keyword_count=2)
    second = keywords.add_publication_keywords(works_frame(), keyword_count=2)

    pandas.testing.assert_series_equal(first["keyword_id"], second["keyword_id"])
    pandas.testing.assert_series_equal(first["keyword"], second["keyword"])
    mapped = first[first["keyword_id"] != ""]
    assert len(mapped) == 6
    assert set(mapped["keyword"]) == {"battery health", "robot teamwork"}
    assert mapped.groupby("keyword")["keyword_id"].first().to_dict() == {
        "battery health": "keyword-01",
        "robot teamwork": "keyword-02",
    }
    assert first.loc[first["work_id"] == "excluded", "keyword_id"].item() == ""
    assert first.loc[first["work_id"] == "outside-roster", "keyword"].item() == ""
    assert set(first["keyword_model_version"]) == {keywords.KEYWORD_MODEL_VERSION}


def test_one_visible_region_uses_one_keyword() -> None:
    frame = pandas.DataFrame(
        [
            {
                "title": "Quantum sensor calibration",
                "map_eligible": True,
                "department_ids": ["ece"],
                "tsne_x": 0.0,
                "tsne_y": 0.0,
            },
            {
                "title": "Quantum sensor control",
                "map_eligible": True,
                "department_ids": ["ece"],
                "tsne_x": 0.0,
                "tsne_y": 0.0,
            },
        ]
    )

    output = keywords.add_publication_keywords(frame)

    assert output["keyword_id"].tolist() == ["keyword-01", "keyword-01"]
    assert output["keyword"].tolist() == ["quantum sensor", "quantum sensor"]


def test_keyword_generation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        keywords.add_publication_keywords(works_frame(), keyword_count=0)

    empty = works_frame()
    empty["map_eligible"] = False
    with pytest.raises(ValueError, match="No included works"):
        keywords.add_publication_keywords(empty)

    nonfinite = works_frame().iloc[:1].copy()
    nonfinite.loc[:, "tsne_x"] = numpy.nan
    with pytest.raises(ValueError, match="finite t-SNE"):
        keywords.add_publication_keywords(nonfinite)


def test_fallback_keywords_are_readable_and_unique() -> None:
    assert (
        keywords._fallback_keyword(
            ["Alpha beta"],
            used={"alpha beta"},
        )
        == "beta"
    )
    assert keywords._fallback_keyword(["the and"], used=set()) == "research topic 1"

    labels = keywords._cluster_keywords(["A", "I"], numpy.array([0, 1]))
    assert labels == {0: "research topic 1", 1: "research topic 2"}


def test_duplicate_cluster_phrase_falls_back_to_an_unused_label() -> None:
    labels = keywords._cluster_keywords(
        [
            "Robot teamwork alpha",
            "Robot teamwork beta",
            "Robot teamwork gamma",
            "Robot teamwork delta",
        ],
        numpy.array([0, 0, 1, 1]),
    )

    assert labels[0] == "robot teamwork"
    assert labels[1] != labels[0]
