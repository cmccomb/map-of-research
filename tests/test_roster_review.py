import datetime as dt

import map_of_research.roster_review as roster_review
from map_of_research.roster_review import build_review


def test_annual_review_names_inclusion_policy_and_all_maps() -> None:
    review = build_review(today=dt.date(2026, 7, 17))

    assert "Annual faculty roster review — 2026" in review
    assert "faculty, teaching faculty, and emeriti" in review
    assert "never bulk-search Scholar" in review
    assert "Biomedical Engineering" in review
    assert "CMU-Africa" in review


def test_annual_review_defaults_to_current_utc_year(monkeypatch) -> None:
    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2031, 1, 2, tzinfo=tz)

    monkeypatch.setattr(roster_review.dt, "datetime", Clock)
    assert "Annual faculty roster review — 2031" in build_review()


def test_main_prints_or_writes_review(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(roster_review, "build_review", lambda: "Annual review\n")
    assert roster_review.main([]) == 0
    assert capsys.readouterr().out == "Annual review\n\n"

    output = tmp_path / "review.md"
    assert roster_review.main(["--output", str(output)]) == 0
    assert output.read_text() == "Annual review\n"
