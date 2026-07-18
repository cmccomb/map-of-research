import datetime as dt

from map_of_research.roster_review import build_review


def test_annual_review_names_inclusion_policy_and_all_maps() -> None:
    review = build_review(today=dt.date(2026, 7, 17))

    assert "Annual faculty roster review — 2026" in review
    assert "faculty, teaching faculty, and emeriti" in review
    assert "never bulk-search Scholar" in review
    assert "Biomedical Engineering" in review
    assert "CMU-Africa" in review
