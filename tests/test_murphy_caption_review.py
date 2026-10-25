"""Keep the six reviewed Murphy caption and figure works distinct."""

from map_of_research.dataset import normalize_title
from map_of_research.doi_aliases import DEFAULT_DOI_ALIASES_PATH, load_doi_aliases
from map_of_research.exclusions import DEFAULT_EXCLUSIONS_PATH, load_exclusions

SCHOLAR_ID = "qQLlBH4AAAAJ"


def _key(suffix: str) -> tuple[str, str]:
    return SCHOLAR_ID, f"{SCHOLAR_ID}:{suffix}"


def test_three_distinct_doi_works_keep_exact_published_identities() -> None:
    aliases = load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)
    expected = {
        "g5m5HwL7SMYC": "10.1145/1557019.1557031",  # KDD 2009
        "4fKUyHm3Qg0C": "10.1007/978-3-642-13131-8_4",  # Springer 2010
        "08ZZubdj9fEC": "10.1016/j.websem.2010.04.002",  # Journal 2010
    }
    assert {suffix: aliases[_key(suffix)].doi for suffix in expected} == expected
    assert all(
        aliases[_key(suffix)].relationship == "version_of_record" for suffix in expected
    )


def test_duplicate_variants_cannot_supply_works_or_coauthors() -> None:
    exclusions = load_exclusions(DEFAULT_EXCLUSIONS_PATH)
    duplicate_to_retained = {
        "unp9ATQDT5gC": "pqnbT2bcN3wC",  # PSB 2007
        "idthP5jqfYAC": "AXPGKjj_ei8C",  # BioLINK 2009
        "P7Ujq4OLJYoC": "g5m5HwL7SMYC",  # KDD 2009
    }
    for duplicate, retained in duplicate_to_retained.items():
        decision = exclusions[_key(duplicate)]
        assert decision.reason_code == "corrupted_duplicate"
        assert f"{SCHOLAR_ID}:{retained}" in decision.reason
        assert _key(retained) not in exclusions


def test_source_native_titles_match_retained_scholar_titles() -> None:
    pairs = [
        (
            "A stacked graphical model for associating sub-images with sub-captions",
            "A Stacked Graphical Model for Associating Sub-Images with Sub-Captions",
        ),
        (
            "Structured literature image finder",
            "Structured Literature Image Finder",
        ),
        (
            "Structured literature image finder: Open source software for "
            "extracting and disseminating information from text and figures in "
            "biomedical literature",
            "Structured Literature Image Finder: Open Source Software for "
            "Extracting and Disseminating Information from Text and Figures in "
            "Biomedical Literature",
        ),
    ]
    assert all(normalize_title(raw) == normalize_title(source) for raw, source in pairs)
    assert len({normalize_title(source) for _, source in pairs}) == 3
