"""Keep three exact publisher-backed Bio gains separate from a poster placeholder."""

from map_of_research.corrections import DEFAULT_CORRECTIONS_PATH, load_corrections
from map_of_research.doi_aliases import DEFAULT_DOI_ALIASES_PATH, load_doi_aliases
from map_of_research.io import load_json

REVIEWED = {
    ("FXqaMl4AAAAJ", "FXqaMl4AAAAJ:UebtZRa9Y70C"): (
        "10.1016/j.neuron.2017.06.006",
        2017,
        "https://pubmed.ncbi.nlm.nih.gov/28641104/",
    ),
    ("28yZqbcAAAAJ", "28yZqbcAAAAJ:UeHWp8X0CEIC"): (
        "10.1016/j.molcel.2008.06.010",
        2008,
        "https://pubmed.ncbi.nlm.nih.gov/18614041/",
    ),
    ("6ArMETUAAAAJ", "6ArMETUAAAAJ:nroGzMJTTpEC"): (
        "10.1371/journal.pcbi.1014780",
        2026,
        "https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014780",
    ),
}


def test_three_exact_bio_observations_have_reviewed_publication_ids() -> None:
    aliases = load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)
    for key, (doi, year, evidence_url) in REVIEWED.items():
        alias = aliases[key]
        assert alias.doi == doi
        assert alias.observed_year == year
        assert alias.relationship == "version_of_record"
        assert evidence_url in alias.evidence_urls

    mcmanus = next(
        item
        for item in load_json(DEFAULT_DOI_ALIASES_PATH)["aliases"]
        if item["author_pub_id"] == "28yZqbcAAAAJ:UeHWp8X0CEIC"
    )
    assert mcmanus["allow_short_title_region_override"] is True
    assert {
        "https://www.cell.com/molecular-cell/fulltext/S1097-2765(08)00428-0",
        "https://pubmed.ncbi.nlm.nih.gov/18614041/",
        "https://doi.org/10.1016/j.molcel.2008.06.010",
    } <= set(mcmanus["evidence_urls"])

    barth = next(
        item
        for item in load_json(DEFAULT_DOI_ALIASES_PATH)["aliases"]
        if item["author_pub_id"] == "FXqaMl4AAAAJ:UebtZRa9Y70C"
    )
    assert barth["allow_short_title_region_override"] is True
    assert {
        "https://www.sciencedirect.com/science/article/pii/S0896627317305044",
        "https://pubmed.ncbi.nlm.nih.gov/28641104/",
        "https://doi.org/10.1016/j.neuron.2017.06.006",
    } <= set(barth["evidence_urls"])


def test_bio_display_corrections_require_exact_raw_metadata() -> None:
    corrections = load_corrections(DEFAULT_CORRECTIONS_PATH)
    for key in REVIEWED:
        correction = corrections[key]
        assert set(correction.expected) == {"title", "year", "venue", "authors"}
        assert set(correction.corrected) == {"venue", "authors"}
        assert correction.expected["year"] == REVIEWED[key][1]
        assert correction.corrected["authors"]
        assert correction.corrected["venue"]


def test_willard_native_article_has_original_title_authors_and_pages() -> None:
    key = ("UZf8gAUAAAAJ", "UZf8gAUAAAAJ:u5HHmVD_uO8C")
    correction = load_corrections(DEFAULT_CORRECTIONS_PATH)[key]
    assert correction.expected == {
        "title": (
            "Controversies in neuroscience: a literature-based course for first "
            "year undergraduates that improves scientific confidence while "
            "teaching concepts"
        ),
        "year": 2014,
        "venue": "Journal of undergraduate neuroscience education 12 (2), A159, 2014",
        "authors": "AM Willard, DJ Brasier",
    }
    assert correction.corrected == {
        "title": (
            "Controversies in Neuroscience: A Literature-Based Course for First "
            "Year Undergraduates that Improves Scientific Confidence While "
            "Teaching Concepts"
        ),
        "venue": (
            "Journal of Undergraduate Neuroscience Education 12(2), A159–A166 (2014)"
        ),
        "authors": "Amanda M. Willard and D. J. Brasier",
    }
    assert (
        correction.evidence_url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC3970999/"
    )


def test_malformed_poster_is_still_source_raw_only() -> None:
    poster = ("28yZqbcAAAAJ", "28yZqbcAAAAJ:HKviVsUxM5wC")
    assert poster not in load_doi_aliases(DEFAULT_DOI_ALIASES_PATH)
    assert poster not in load_corrections(DEFAULT_CORRECTIONS_PATH)
