# Two exact Math Scholar preprint and journal pairs

Reviewed 2026-10-25. These four captured Scholar rows describe two articles,
each appearing once as an arXiv preprint and once as the journal version. The
exact citation IDs, titles, years, venues, and raw source records remain in the
Scholar capture. Only their normalized work identity changes.

| Scholar researcher | Preprint row | Journal row | Published DOI |
| --- | --- | --- | --- |
| Prasad Tetali | `FmZaK1wAAAAJ:roLk4NBRz8UC` (2014, arXiv:1405.0608) | `FmZaK1wAAAAJ:f2IySw72cVMC` (2015, *Annales de la Faculté des sciences de Toulouse*, 24(4), 691–716) | `10.5802/afst.1460` |
| Dejan Slepčev | `tWQsy4UAAAAJ:nb7KW1ujOQ8C` (2018, arXiv:1805.09450) | `tWQsy4UAAAAJ:GnPB-g6toBAC` (2020, *Applied and Computational Harmonic Analysis*, 49(2), 655–697) | `10.1016/j.acha.2019.03.005` |

The [Tetali arXiv submission](https://arxiv.org/abs/1405.0608) names Pietro
Caputo, Georg Menz, and Prasad Tetali and states that the revised paper was to
appear in the journal. The [journal's Numdam record](https://numdam.org/articles/10.5802/afst.1460/)
has the same title and authors, the 2015 volume and pages, and the DOI.

The [Slepčev arXiv submission](https://arxiv.org/abs/1805.09450) names Matthew
M. Dunlop, Dejan Slepčev, Andrew M. Stuart, and Matthew Thorpe. The
[Elsevier journal record](https://www.sciencedirect.com/science/article/pii/S1063520318301398)
has the same title and authors, the 2020 volume and pages, and the DOI.

Each preprint row has a `preprint_of` alias and each corresponding journal row
has a `version_of_record` alias in `data/curated/scholar-doi-aliases.json`. The
producer validates each exact captured identity, title, year, and any source
DOI before use. The normalized work takes the journal year and venue, retains
both Scholar observation IDs and year variants, and carries both primary-source
URLs. The two pairs reduce the normalized and mapped work counts by one each;
they do not add a publication, change a person's Scholar coverage, or erase a
captured row. Other rows continue to follow the existing DOI and normalized
title-year identity rules; these exact aliases do not change them.
