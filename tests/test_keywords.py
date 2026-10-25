import numpy
import pandas
import pytest

import map_of_research.keywords as keywords
from map_of_research.doi_aliases import ReviewedDoiAlias
from map_of_research.topic_labels import apply_reviewed_labels, review_catalog


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
    options = {
        "keyword_count": 2,
        "detail_keyword_count": 4,
        "minimum_detail_cluster_size": 1,
    }
    first = keywords.add_publication_keywords(works_frame(), **options)
    second = keywords.add_publication_keywords(works_frame(), **options)

    pandas.testing.assert_series_equal(first["keyword_id"], second["keyword_id"])
    pandas.testing.assert_series_equal(first["keyword"], second["keyword"])
    mapped = first[first["keyword_id"] != ""]
    assert len(mapped) == 6
    assert set(mapped["keyword"]) == {"battery health", "robot teamwork"}
    assert mapped.groupby("keyword")["keyword_id"].first().to_dict() == {
        "battery health": "keyword-01",
        "robot teamwork": "keyword-02",
    }
    assert mapped["detail_keyword_id"].nunique() == 4
    assert all(
        detail_id.startswith(f"{keyword_id}-")
        for keyword_id, detail_id in zip(
            mapped["keyword_id"],
            mapped["detail_keyword_id"],
            strict=True,
        )
    )
    assert not set(mapped["keyword"]) & set(mapped["detail_keyword"])
    assert first.loc[first["work_id"] == "excluded", "keyword_id"].item() == ""
    assert first.loc[first["work_id"] == "outside-roster", "keyword"].item() == ""
    assert first.loc[first["work_id"] == "excluded", "detail_keyword_id"].item() == ""
    assert set(first["keyword_model_version"]) == {keywords.KEYWORD_MODEL_VERSION}
    assert first.loc[first["keyword_id"].ne(""), "keyword_extracted"].ne("").all()
    assert (
        first.loc[first["detail_keyword_id"].ne(""), "detail_keyword_extracted"]
        .ne("")
        .all()
    )
    assert not first["keyword_label_reviewed"].any()
    assert first["detail_keyword_label_reviewed"].dtype == bool


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
    assert output["detail_keyword_id"].tolist() == [
        "keyword-01-01",
        "keyword-01-01",
    ]
    assert output["detail_keyword"].nunique() == 1
    assert output["detail_keyword"].iloc[0] != "quantum sensor"


def test_keyword_generation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        keywords.add_publication_keywords(works_frame(), keyword_count=0)

    with pytest.raises(ValueError, match="at least keyword_count"):
        keywords.add_publication_keywords(
            works_frame(),
            keyword_count=3,
            detail_keyword_count=2,
        )

    with pytest.raises(ValueError, match="minimum_detail_cluster_size"):
        keywords.add_publication_keywords(
            works_frame(),
            minimum_detail_cluster_size=0,
        )

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


def test_overview_keywords_prefer_readable_phrases_over_codes() -> None:
    titles = [
        "4H SiC device",
        "4H SiC film",
        "Silicon carbide device",
        "Silicon carbide film",
    ]
    labels = numpy.zeros(4, dtype=int)

    literal = keywords._cluster_keywords(titles, labels)
    readable = keywords._cluster_keywords(titles, labels, prefer_readable=True)

    assert literal == {0: "4h sic"}
    assert readable == {0: "silicon carbide"}


def test_keyword_labels_skip_generic_phrases() -> None:
    titles = [
        "United States low carbon policy",
        "United States low carbon transition",
        "United States low carbon pathways",
    ]

    labels = keywords._cluster_keywords(titles, numpy.zeros(3, dtype=int))

    assert labels == {0: "low carbon"}


def test_keyword_labels_prefer_well_supported_specific_phrases() -> None:
    titles = [
        "Robust speech recognition in noise",
        "Robust speech recognition for calls",
        "Robust speech processing for meetings",
    ]

    labels = keywords._cluster_keywords(titles, numpy.zeros(3, dtype=int))

    assert labels == {0: "robust speech recognition"}


def test_reviewed_labels_preserve_extraction_and_reject_ambiguity() -> None:
    overview_reviews = review_catalog(0)
    detail_reviews = review_catalog(1)
    assert len(overview_reviews) == len(set(overview_reviews.values())) == 30
    assert len(detail_reviews) == len(set(detail_reviews.values())) == 120
    assert set(overview_reviews.values()).isdisjoint(detail_reviews.values())

    labels, reviewed = apply_reviewed_labels(
        {0: "access control", 1: "unreviewed phrase"},
        overview_reviews,
    )

    assert labels == {
        0: "cybersecurity & privacy",
        1: "unreviewed phrase",
    }
    assert reviewed == {0: True, 1: False}

    labels, reviewed = apply_reviewed_labels(
        {0: "access control"},
        overview_reviews,
        reserved={"cybersecurity & privacy"},
    )
    assert labels == {0: "access control"}
    assert reviewed == {0: False}
    with pytest.raises(ValueError, match="non-empty and unique"):
        apply_reviewed_labels({0: ""}, {})
    with pytest.raises(ValueError, match="non-empty and unique"):
        apply_reviewed_labels({0: "access control"}, {"access control": " "})
    with pytest.raises(ValueError, match="non-empty and unique"):
        apply_reviewed_labels({0: "one"}, {"one": "same", "two": "same"})
    labels, reviewed = apply_reviewed_labels(
        {0: "current"},
        {"unused": "parent"},
        reserved={"parent"},
    )
    assert labels == {0: "current"}
    assert reviewed == {0: False}
    with pytest.raises(ValueError, match="Unknown topic hierarchy"):
        review_catalog(2)


@pytest.mark.parametrize(
    ("extracted", "expected"),
    [
        ({0: "decision making", 1: "trauma triage"}, {0: False, 1: False}),
        ({0: "trauma triage", 1: "decision making"}, {0: False, 1: False}),
    ],
)
def test_direct_extraction_precedes_colliding_reviewed_label(
    extracted: dict[int, str], expected: dict[int, bool]
) -> None:
    labels, reviewed = apply_reviewed_labels(
        extracted,
        {"decision making": "trauma triage"},
    )

    assert labels == extracted
    assert reviewed == expected


def test_direct_extraction_precedes_another_reviewed_keys_output() -> None:
    labels, reviewed = apply_reviewed_labels(
        {0: "first", 1: "second"},
        {"first": "second", "second": "final"},
    )

    assert labels == {0: "first", 1: "final"}
    assert reviewed == {0: False, 1: True}


@pytest.mark.parametrize(
    ("extracted", "reserved"),
    [({0: "topic", 1: "topic"}, set()), ({0: "topic"}, {"topic"})],
)
def test_duplicate_or_parent_matching_extractions_remain_errors(
    extracted: dict[int, str], reserved: set[str]
) -> None:
    with pytest.raises(ValueError, match="non-empty and unique"):
        apply_reviewed_labels(extracted, {}, reserved=reserved)


def test_colliding_reviews_preserve_source_phrases_in_both_keyword_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def extracted_for_test(
        titles: list[str],
        labels: numpy.ndarray,
        *,
        reserved: set[str] | None = None,
        prefer_readable: bool = False,
    ) -> dict[int, str]:
        del titles, reserved
        clusters = sorted({int(label) for label in labels})
        if prefer_readable:
            assert len(clusters) == 2
            return dict(
                zip(
                    clusters,
                    ["access control", "cybersecurity & privacy"],
                    strict=True,
                )
            )
        assert len(clusters) == 4
        return dict(
            zip(
                clusters,
                [
                    "decision making",
                    "trauma triage",
                    "energy systems",
                    "heat exchanger",
                ],
                strict=True,
            )
        )

    monkeypatch.setattr(keywords, "_cluster_keywords", extracted_for_test)
    output = keywords.add_publication_keywords(
        works_frame(),
        keyword_count=2,
        detail_keyword_count=4,
        minimum_detail_cluster_size=1,
    )

    overview = output[["keyword", "keyword_extracted", "keyword_label_reviewed"]]
    detail = output[
        ["detail_keyword", "detail_keyword_extracted", "detail_keyword_label_reviewed"]
    ]
    assert set(overview["keyword"]) == {"access control", "cybersecurity & privacy", ""}
    assert not overview["keyword_label_reviewed"].any()
    assert set(detail["detail_keyword"]) == {
        "decision making",
        "trauma triage",
        "energy systems",
        "process integration & carbon capture",
        "",
    }
    assert (
        output.loc[
            output["detail_keyword_extracted"] == "decision making",
            "detail_keyword_label_reviewed",
        ]
        .eq(False)
        .all()
    )
    assert (
        output.loc[
            output["detail_keyword_extracted"] == "heat exchanger",
            "detail_keyword_label_reviewed",
        ]
        .eq(True)
        .all()
    )
    assert set(overview["keyword_extracted"]) == {
        "access control",
        "cybersecurity & privacy",
        "",
    }
    assert set(detail["detail_keyword_extracted"]) == {
        "decision making",
        "trauma triage",
        "energy systems",
        "heat exchanger",
        "",
    }
    assert not set(overview["keyword"]) & set(detail["detail_keyword"]) - {""}


def test_detail_review_colliding_with_parent_falls_back_to_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def extracted_for_test(
        titles: list[str],
        labels: numpy.ndarray,
        *,
        reserved: set[str] | None = None,
        prefer_readable: bool = False,
    ) -> dict[int, str]:
        del titles, labels
        if prefer_readable:
            assert reserved is None
            return {0: "block copolymers"}
        assert reserved == {"block copolymers"}
        return {0: "block copolymer"}

    monkeypatch.setattr(keywords, "_cluster_keywords", extracted_for_test)
    output = keywords.add_publication_keywords(
        works_frame(),
        keyword_count=1,
        detail_keyword_count=1,
    )
    mapped = output.loc[output["keyword_id"].ne("")]

    assert set(mapped["keyword"]) == {"block copolymers"}
    assert set(mapped["detail_keyword"]) == {"block copolymer"}
    assert set(mapped["detail_keyword_extracted"]) == {"block copolymer"}
    assert not mapped["detail_keyword_label_reviewed"].any()


def test_detail_cluster_allocation_is_proportional_and_capacity_limited() -> None:
    coordinates = numpy.asarray(
        [[-1, -1], [-1.1, -1], [-0.9, -1], [1, 1], [1.1, 1], [0.9, 1]],
        dtype=numpy.float32,
    )
    labels = numpy.asarray([0, 0, 0, 1, 1, 1])

    assert keywords._detail_cluster_allocations(
        labels,
        coordinates,
        target_count=4,
        minimum_cluster_size=1,
    ) == {0: 2, 1: 2}
    assert keywords._detail_cluster_allocations(
        labels,
        coordinates,
        target_count=20,
        minimum_cluster_size=1,
    ) == {0: 3, 1: 3}


def test_sparse_detail_clusters_merge_into_the_nearest_sibling() -> None:
    coordinates = numpy.asarray(
        [[0, 0], [0.1, 0], [5, 5], [5.1, 5], [9, 9]],
        dtype=numpy.float32,
    )
    labels = numpy.asarray([0, 0, 1, 1, 2])

    merged = keywords._merge_small_clusters(
        coordinates,
        labels,
        minimum_cluster_size=2,
    )
    single = keywords._merge_small_clusters(
        coordinates[:2],
        numpy.zeros(2, dtype=int),
        minimum_cluster_size=3,
    )

    assert merged.tolist() == [0, 0, 1, 1, 1]
    assert single.tolist() == [0, 0]


def test_low_information_region_audit_requires_all_quality_signals() -> None:
    rows = []
    dimensions = keywords.MINIMUM_AUDIT_REGION_SIZE
    for index in range(dimensions):
        embedding = numpy.zeros(dimensions, dtype=numpy.float32)
        embedding[index] = 1.0
        rows.append(
            {
                "map_eligible": True,
                "map_exclusion_reasons": [],
                "detail_keyword_id": "keyword-01-01",
                "embedding": embedding.tolist(),
                "year": None,
                "citation_count": 0,
                "title": f"Fragment {index}",
                "x": float(index),
                "y": float(index),
                "tsne_x": float(index),
                "tsne_y": float(index),
            }
        )
    for index in range(dimensions):
        rows.append(
            {
                "map_eligible": True,
                "map_exclusion_reasons": [],
                "detail_keyword_id": "keyword-02-01",
                "embedding": [1.0, *([0.0] * (dimensions - 1))],
                "year": 2025,
                "citation_count": 1,
                "title": f"A descriptive publication title number {index}",
                "x": float(index),
                "y": float(index),
                "tsne_x": float(index),
                "tsne_y": float(index),
            }
        )
    frame = pandas.DataFrame(rows)

    audited = keywords.exclude_low_information_regions(frame)

    outliers = audited[audited["detail_keyword_id"] == "keyword-01-01"]
    retained = audited[audited["detail_keyword_id"] == "keyword-02-01"]
    assert not outliers["map_eligible"].any()
    assert outliers["map_region_outlier"].all()
    assert outliers["tsne_x"].tolist() == [float(index) for index in range(dimensions)]
    assert set(outliers["map_exclusion_reasons"].map(tuple)) == {
        ("low_information_region",)
    }
    assert retained["map_eligible"].all()
    assert not retained["map_region_outlier"].any()
    assert set(audited["map_region_audit_version"]) == {keywords.REGION_AUDIT_VERSION}


def test_region_audit_skips_small_regions() -> None:
    frame = pandas.DataFrame(
        [
            {
                "map_eligible": True,
                "map_exclusion_reasons": [],
                "detail_keyword_id": "keyword-01-01",
                "embedding": [1.0, 0.0],
                "year": None,
                "citation_count": 0,
                "title": "Tiny",
                "x": 0.0,
                "y": 0.0,
                "tsne_x": 0.0,
                "tsne_y": 0.0,
            }
        ]
    )

    audited = keywords.exclude_low_information_regions(frame)

    assert audited["map_eligible"].tolist() == [True]
    assert audited["map_region_outlier"].tolist() == [False]


def test_region_audit_preserves_documented_paper_inside_weak_region() -> None:
    rows = []
    size = keywords.MINIMUM_AUDIT_REGION_SIZE
    for index in range(size):
        embedding = numpy.zeros(size, dtype=numpy.float32)
        embedding[index] = 1.0
        rows.append(
            {
                "map_eligible": True,
                "map_exclusion_reasons": [],
                "detail_keyword_id": "keyword-01-01",
                "embedding": embedding.tolist(),
                "year": None,
                "citation_count": 0,
                "title": f"Fragment {index}",
                "authors": "",
                "venue": "",
                "doi": "",
                "source_urls": [],
            }
        )
    rows[0].update(
        {
            "year": 1995,
            "title": "Independence of solution sets and minimal asymptotic bases",
            "authors": "Paul Erdős, Melvyn Nathanson, and Prasad Tetali",
            "venue": "Acta Arithmetica",
            "doi": "10.4064/aa-69-3-243-258",
            "source_urls": ["https://example.test/publisher"],
        }
    )
    rows[1].update(
        {
            "year": 1990,
            "title": "A long enough title with several descriptive words",
            "authors": "Two Authors",
            "venue": "Journal",
            "doi": "10.1000/another",
            "source_urls": [],
        }
    )

    audited = keywords.exclude_low_information_regions(pandas.DataFrame(rows))

    assert audited["map_eligible"].sum() == 1
    assert bool(audited.loc[0, "map_region_documented_override"])
    assert not bool(audited.loc[0, "map_region_outlier"])
    assert audited.loc[0, "map_exclusion_reasons"] == []
    assert not bool(audited.loc[1, "map_eligible"])
    assert bool(audited.loc[1, "map_region_outlier"])
    assert audited.loc[1, "map_exclusion_reasons"] == ["low_information_region"]
    assert (
        audited["map_region_audit_version"]
        .eq("low-information-semantic-region-v3")
        .all()
    )


def test_region_audit_preserves_only_verified_native_source_work() -> None:
    rows = []
    size = keywords.MINIMUM_AUDIT_REGION_SIZE
    for index in range(size):
        embedding = numpy.zeros(size, dtype=numpy.float32)
        embedding[index] = 1.0
        rows.append(
            {
                "map_eligible": True,
                "map_exclusion_reasons": [],
                "detail_keyword_id": "keyword-01-01",
                "embedding": embedding.tolist(),
                "year": None,
                "citation_count": 0,
                "title": f"Fragment {index}",
                "authors": "",
                "venue": "",
                "doi": "",
                "source_urls": [],
                "source_types": ["scholar"],
                "source_native_ids": [],
                "source_verified_at": "",
                "source_verification_urls": [],
            }
        )
    documented = {
        "year": 2009,
        "title": "Structured Literature Image Finder",
        "authors": "Amr Ahmed, Andrew Arnold, and Robert F. Murphy",
        "venue": "ISMB BioLINK Special Interest Group",
        "source_urls": ["https://murphylab.cbd.cmu.edu/publications/"],
        "source_types": ["scholar", "curated_verified"],
        "source_native_ids": ["biolink2009:209-212"],
        "source_verified_at": "2026-10-25",
        "source_verification_urls": ["https://murphylab.cbd.cmu.edu/publications/"],
    }
    rows[0].update(documented)
    rows[1].update(documented | {"source_types": ["scholar"]})
    rows[2].update(
        documented | {"source_verification_urls": ["http://example.test/unverified"]}
    )
    rows[3].update(documented | {"venue": ""})
    rows[4].update(documented | {"source_verified_at": "not-a-date"})
    rows[5].update(documented | {"source_types": "curated_verified"})
    rows[6].update(documented | {"source_native_ids": ["invalid-native-id"]})
    rows[7].update(
        documented | {"source_verification_urls": "https://example.test/not-a-list"}
    )

    audited = keywords.exclude_low_information_regions(pandas.DataFrame(rows))

    assert bool(audited.loc[0, "map_eligible"])
    assert bool(audited.loc[0, "map_region_documented_override"])
    for index in range(1, 8):
        assert not bool(audited.loc[index, "map_eligible"])
        assert bool(audited.loc[index, "map_region_outlier"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://example.test/paper", True),
        (" ", False),
        (["https://example.test/paper"], True),
        (("https://example.test/paper",), True),
        (numpy.asarray(["https://example.test/paper"]), True),
        ([], False),
        ([None], False),
        ([""], False),
        (None, False),
        (float("nan"), False),
    ],
)
def test_documented_work_source_url_handles_parquet_and_missing_values(
    value, expected
) -> None:
    assert keywords._has_source_url(value) is expected


def test_reviewed_short_title_exception_is_exact_to_gordon_chapter() -> None:
    scholar_id = "n6flaAsAAAAJ"
    author_pub_id = f"{scholar_id}:M3NEmzRMIkIC"
    doi = "10.1007/1-4020-3500-4_18"
    chapter_url = "https://link.springer.com/chapter/10.1007/1-4020-3500-4_18"
    alias = ReviewedDoiAlias(
        doi=doi,
        observed_title="Exchange processes",
        observed_year=2006,
        relationship="version_of_record",
        evidence_urls=(
            chapter_url,
            "https://link.springer.com/book/10.1007/1-4020-3500-4",
        ),
        verified_at="2026-10-25",
        allow_short_title_region_override=True,
    )
    other_pub_id = f"{scholar_id}:other-short-chapter"
    aliases = {
        (scholar_id, author_pub_id): alias,
        (scholar_id, other_pub_id): ReviewedDoiAlias(
            doi="10.1007/1-4020-3500-4_19",
            observed_title="Other chapter",
            observed_year=2006,
            relationship="version_of_record",
            evidence_urls=(
                "https://link.springer.com/chapter/10.1007/1-4020-3500-4_19",
            ),
            verified_at="2026-10-25",
        ),
    }
    size = keywords.MINIMUM_AUDIT_REGION_SIZE
    rows = []
    for index in range(size):
        embedding = numpy.zeros(size, dtype=numpy.float32)
        embedding[index] = 1.0
        rows.append(
            {
                "map_eligible": True,
                "map_exclusion_reasons": [],
                "detail_keyword_id": "keyword-01-01",
                "embedding": embedding.tolist(),
                "year": None,
                "citation_count": 0,
                "title": f"Fragment {index}",
                "authors": "",
                "venue": "",
                "doi": "",
                "source_urls": [],
                "source_types": ["scholar"],
                "source_verification_urls": [],
                "scholar_ids": [],
                "author_pub_ids": [],
            }
        )
    gordon = {
        "year": 2006,
        "citation_count": 3,
        "title": "Exchange processes",
        "authors": "GS Rule, TK Hitchens",
        "venue": "Fundamentals of Protein NMR Spectroscopy, 403-430, 2006",
        "doi": doi,
        "source_urls": [
            "https://scholar.google.com/citations?user=n6flaAsAAAAJ"
            "&citation_for_view=n6flaAsAAAAJ:M3NEmzRMIkIC"
        ],
        "source_verification_urls": [chapter_url],
        "source_verified_at": "2026-10-25",
        "source_types": ["scholar"],
        "scholar_ids": [scholar_id],
        "author_pub_ids": [author_pub_id],
    }
    rows[0].update(gordon)
    for index, change in enumerate(
        (
            {"doi": "10.1007/different"},
            {"title": "Other processes"},
            {"year": 2007},
            {"author_pub_ids": [f"{scholar_id}:other"]},
            {"source_verification_urls": []},
            {"scholar_ids": ["other"]},
            {"source_verified_at": "2026-10-24"},
        ),
        start=1,
    ):
        rows[index].update(gordon | change)
    rows[7].update(
        gordon
        | {
            "title": "Other chapter",
            "doi": "10.1007/1-4020-3500-4_19",
            "source_verification_urls": [
                "https://link.springer.com/chapter/10.1007/1-4020-3500-4_19"
            ],
            "author_pub_ids": [other_pub_id],
        }
    )

    without_review = keywords.exclude_low_information_regions(pandas.DataFrame(rows))
    assert not without_review["map_eligible"].any()

    audited = keywords.exclude_low_information_regions(
        pandas.DataFrame(rows), doi_aliases=aliases
    )
    assert audited["map_eligible"].sum() == 1
    assert bool(audited.loc[0, "map_region_documented_override"])
    assert not bool(audited.loc[0, "map_region_outlier"])
    assert audited.loc[1:, "map_region_outlier"].all()
    for field, value in (
        ("author_pub_ids", author_pub_id),
        ("scholar_ids", scholar_id),
        ("source_types", "scholar"),
        ("source_verification_urls", chapter_url),
        ("source_types", []),
        ("source_verified_at", "not-a-date"),
        ("year", 2006.5),
    ):
        malformed = pandas.Series(gordon | {field: value})
        assert not keywords._reviewed_short_title_alias_matches(malformed, aliases)


def test_reviewed_cell_short_title_requires_complete_pii_doi_pubmed_evidence() -> None:
    scholar_id = "28yZqbcAAAAJ"
    author_pub_id = f"{scholar_id}:UeHWp8X0CEIC"
    doi = "10.1016/j.molcel.2008.06.010"
    evidence = (
        "https://www.cell.com/molecular-cell/fulltext/S1097-2765(08)00428-0",
        "https://pubmed.ncbi.nlm.nih.gov/18614041/",
        "https://doi.org/10.1016/j.molcel.2008.06.010",
    )
    alias = ReviewedDoiAlias(
        doi=doi,
        observed_title="Getting the message out",
        observed_year=2008,
        relationship="version_of_record",
        evidence_urls=evidence,
        verified_at="2026-10-25",
        allow_short_title_region_override=True,
    )
    work = pandas.Series(
        {
            "doi": doi,
            "title": "Getting the message out",
            "year": 2008,
            "authors": "CJ McManus, BR Graveley",
            "venue": "Molecular cell 31 (1), 4-6, 2008",
            "source_types": ["scholar"],
            "scholar_ids": [scholar_id],
            "author_pub_ids": [author_pub_id],
            "source_verified_at": "2026-10-25",
            "source_verification_urls": list(evidence),
        }
    )
    aliases = {(scholar_id, author_pub_id): alias}
    assert keywords._reviewed_short_title_alias_matches(work, aliases)

    for changed in (
        {"doi": "10.1016/j.molcel.2008.06.011"},
        {"title": "Another message"},
        {"year": 2009},
        {"scholar_ids": ["another"]},
        {"author_pub_ids": [f"{scholar_id}:another"]},
        {"source_verified_at": "2026-10-24"},
        {"source_verification_urls": list(evidence[1:])},
        {"source_verification_urls": list(evidence[:1] + evidence[2:])},
        {"source_verification_urls": list(evidence[:2])},
    ):
        assert not keywords._reviewed_short_title_alias_matches(
            pandas.Series(work.to_dict() | changed), aliases
        )

    unflagged = {(scholar_id, author_pub_id): ReviewedDoiAlias(
        doi=doi,
        observed_title=alias.observed_title,
        observed_year=alias.observed_year,
        relationship=alias.relationship,
        evidence_urls=evidence,
        verified_at=alias.verified_at,
    )}
    assert not keywords._reviewed_short_title_alias_matches(work, unflagged)
