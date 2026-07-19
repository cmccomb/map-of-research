import pytest

import map_of_research.quality as quality
from map_of_research.quality import assess_publication


def assessment(
    title: str,
    *,
    year: int | None = None,
    doi: str = "",
    source_url: str = "",
) -> quality.PublicationQuality:
    return assess_publication(
        title=title,
        year=year,
        doi=doi,
        source_url=source_url,
    )


@pytest.mark.parametrize(
    ("title", "reason"),
    [
        ("Open Access Media", "navigation_or_interface"),
        ("Program co-Chairs", "conference_front_matter"),
        ("PANEL SUB-COMMITTEE", "conference_front_matter"),
        (
            "Technical Program Committee Real-Time Infrastructure Chair",
            "conference_front_matter",
        ),
        ("Guest Editorial: A Special Issue", "publication_front_matter"),
        ("Keynotes", "event_or_session_label"),
        ("EECS 2017", "event_or_session_label"),
        ("IEEE Communications Society", "organization_or_container"),
        ("Department of Biomedical Engineering", "affiliation_or_contact"),
        ("BÀI-26-RESEARCH PAPER", "placeholder_or_document_control"),
        (
            "ALTMAN, Neal CMU-ISR-20-110 BAUER, Lujo CMU-ISR-20-114",
            "person_or_citation_index",
        ),
        (
            "Behavior Forensics..... H. V. Zhao 311",
            "garbled_or_index_text",
        ),
    ],
)
def test_high_confidence_noise_families_are_excluded(title, reason) -> None:
    result = assessment(title)
    assert result.map_eligible is False
    assert reason in result.exclusion_reasons


@pytest.mark.parametrize(
    "title",
    [
        "A robust controller for uncertain nonlinear systems",
        "Privacy Tools",
        "Modeling and optimization of sustainable supply chains",
    ],
)
def test_incomplete_bibliography_is_not_an_exclusion_reason(title) -> None:
    assert assessment(title) == quality.PublicationQuality(True, ())


def test_strong_identity_evidence_protects_an_organization_shaped_title() -> None:
    result = assessment(
        "Department of Biomedical Engineering",
        doi="10.1000/example",
    )
    assert result.map_eligible is True


def test_long_identified_container_title_is_retained() -> None:
    result = assessment(
        "Proceedings of a Workshop on Reliable Distributed Systems "
        "for Safety Critical Infrastructure",
        year=2025,
        source_url="https://example.test/proceedings",
    )
    assert result.map_eligible is True


@pytest.mark.parametrize(
    "title",
    [
        "Christian Becker, becker@uni.example.edu, University of Stuttgart",
        "Rui Zhang, Carnegie Mellon University, Pittsburgh, PA 15213",
        "is an associate professor whose research interests include systems",
    ],
)
def test_contact_and_biography_records_are_excluded(title) -> None:
    result = assessment(title, year=2020)
    assert result.exclusion_reasons == ("affiliation_or_contact",)


@pytest.mark.parametrize(
    "title",
    [
        "Khoshgoftaar, Taghi 9, 83 Kim, Jihyeon 342 Kraft, Robin 350",
        "Aamer Jaleel, NVIDIA, Akanksha Jain, Google, Microsoft Research",
    ],
)
def test_person_and_citation_index_shapes_are_excluded(title) -> None:
    result = assessment(title)
    assert result.exclusion_reasons == ("person_or_citation_index",)


def test_garbled_ocr_run_is_excluded() -> None:
    result = assessment("CURRENT TRANSIENTS GGGGGGGGGGGGGGGGGGGGLLLLLLLLLLLLLLLLLLLL")
    assert result.exclusion_reasons == ("garbled_or_index_text",)


def test_overlapping_reasons_are_stable_and_not_duplicated() -> None:
    result = assessment("PAPER SELECTION COMMITTEE")
    assert result.exclusion_reasons == (
        "conference_front_matter",
        "event_or_session_label",
    )


def test_helper_predicates_cover_non_matches() -> None:
    assert quality._has_garbled_index_text("Normal title") is False
    assert (
        quality._looks_like_person_or_citation_index(
            "Normal title",
            metadata_weak=False,
        )
        is False
    )
