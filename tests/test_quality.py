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
        ("A New Point of View [President's Message]", "publication_front_matter"),
        ("2006 Associate Editors List", "publication_front_matter"),
        ("A tribute to R. Byron Bird", "publication_front_matter"),
        ("NETL-University Collaboration Title Page", "publication_front_matter"),
        ("7. Publication List", "publication_front_matter"),
        ("Preface: A new direction", "publication_front_matter"),
        (
            'Correction to" Statistical performance"',
            "publication_front_matter",
        ),
        ("Selected Proceedings from the 233rd ECS Meeting", "publication_front_matter"),
        ("Keynotes", "event_or_session_label"),
        ("EECS 2017", "event_or_session_label"),
        ("IEEE Communications Society", "organization_or_container"),
        ("Department of Biomedical Engineering", "affiliation_or_contact"),
        ("BÀI-26-RESEARCH PAPER", "placeholder_or_document_control"),
        (
            "1. REPORT DATE (DD-MM-YYYY)",
            "placeholder_or_document_control",
        ),
        ("ICES REPORT 15-15", "placeholder_or_document_control"),
        ("PC Members", "placeholder_or_document_control"),
        ("1C4-5", "placeholder_or_document_control"),
        ("Emsoft release https://doi.org/10.5281", "navigation_or_interface"),
        ("ERROR 404", "navigation_or_interface"),
        ("Back Matter", "publication_front_matter"),
        ("DETC2013-12651", "placeholder_or_document_control"),
        ("CMU-PDL-17-107 November 2017", "placeholder_or_document_control"),
        ("BWS", "underspecified_title"),
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


def test_title_case_technical_paper_is_not_mistaken_for_an_author_list() -> None:
    result = assessment(
        "A Cost Analysis Of A Fiber Upgrade For A Coaxial Cable Network "
        "To Support On-Demand Video",
        year=1990,
    )
    assert result == quality.PublicationQuality(True, ())


def test_technical_report_identifier_inside_a_real_title_is_retained() -> None:
    result = assessment(
        "A type system for transactional memory (CMU-CS-02-140)",
    )
    assert result == quality.PublicationQuality(True, ())


def test_strong_identity_evidence_protects_an_organization_shaped_title() -> None:
    result = assessment(
        "Department of Biomedical Engineering",
        doi="10.1000/example",
    )
    assert result.map_eligible is True


def test_strong_identity_evidence_protects_a_short_title() -> None:
    assert assessment("IronFleet", doi="10.1000/ironfleet").map_eligible is True


def test_long_multilingual_title_is_retained() -> None:
    assert assessment("增强电力系统的恢复力", year=2017).map_eligible is True


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
        "Pittsburgh, Pennsylvania 15213, USA",
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
        "AA Linninger and MM El-Halwagi",
        "Abdeldjebbar Kandouci; Abdelmadjid Ezzine",
        "Christopher R. Palmer Georgos Siganos Michalis Faloutsos",
        "Gupta, V., RTI International",
        "ERIK K. ANTONSSON",
        "Caroline. M Conely, Mathew. B Wasserman, O Burak Ozdoganlar",
        "Tor M. Aamodt 99, 408 Mohamad Abdel-Majeed 111 Jaume Abella 160",
        "Image and Video Representation... L. Yang, L. Jing, and MK Ng 4701",
        "JS Bow, MJ Kim, RW Carpenter and RC Glass",
        "Ezra JT Levin Anthony J. Prenni Markus Dirk Petters Sonia M. Kreidenweis",
        "and Namhun Kim",
        "Altman, Neal; Bauer, Lujo; pages 9 83 342",
    ],
)
def test_person_and_citation_index_shapes_are_excluded(title) -> None:
    result = assessment(title)
    assert result.exclusion_reasons == ("person_or_citation_index",)


def test_garbled_ocr_run_is_excluded() -> None:
    result = assessment("CURRENT TRANSIENTS GGGGGGGGGGGGGGGGGGGGLLLLLLLLLLLLLLLLLLLL")
    assert result.exclusion_reasons == ("garbled_or_index_text",)


def test_symbol_heavy_corrupted_text_is_excluded() -> None:
    result = assessment("3V* 5V5V5V $| Zh* Σ∞ ╒┬*╧ JàI⌠ yJ╧ JYVv* ZEZE+* û╩*^ B* ON*")
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
    assert (
        quality._looks_like_document_fragment(
            "Normal title",
            identity_weak=False,
        )
        is False
    )
    assert (
        quality._is_underspecified_title("Normal title", identity_weak=False) is False
    )
