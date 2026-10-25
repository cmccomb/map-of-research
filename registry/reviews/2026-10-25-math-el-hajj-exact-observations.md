# Layan El Hajj: exact Scholar publication review

Reviewed 2026-10-25 against the complete 19-row visible Google Scholar capture
in source PR #230 (`2kKrsXEAAAAJ`). All 19 raw observations, their citation
IDs, and their original metadata remain in the source cache and snapshot. The
producer decisions below apply only to normalized work identity or effective
title metadata after source validation.

## Version and DOI identity

| Scholar citation ID suffix | Observed row | Decision | Published DOI |
| --- | --- | --- | --- |
| `LkGwnXOMwfcC` | 2025 Qatar repository entry, *A free boundary problem for systems (The symmetric regime)* | Link repository-indexed observation to the published article | `10.3934/cpaa.2025036` |
| `ufrVoPGSRksC` | 2022 arXiv:2207.01157, *Symmetry for a fully nonlinear free boundary problem with highly singular term* | Preprint of the published article | `10.2422/2036-2145.202309_003` |
| `d1gkVwhDpl0C` | 2018 *Journal of Classical Analysis* article, *On geometrical properties of starlike logharmonic mappings* | Published version of the 2016 arXiv row | `10.7153/jca-2018-12-03` |
| `YsMSGLbcyi4C` | 2016 arXiv:1610.01087, *On geometrical properties of logharmonic mappings* | Preprint of the 2018 journal row | `10.7153/jca-2018-12-03` |

The [AIMS article](https://www.aimsciences.org/article/doi/10.3934/cpaa.2025036)
and [coauthor's KTH publication record](https://kth.diva-portal.org/smash/person.jsf?pid=authority-person%3A31384)
give the same two authors, title, 2025 journal volume and pages, and DOI for
the repository-indexed row. The [2022 arXiv manuscript](https://arxiv.org/abs/2207.01157)
and [Annali publisher record](https://journals.sns.it/index.php/annaliscienze/article/view/6423)
give the same three authors and title; the publisher dates its final issue to
2026. The [2016 arXiv manuscript](https://arxiv.org/pdf/1610.01087) itself
uses the fuller *starlike logharmonic mappings of order α* title and shares
the authors and abstract with the [2018 journal PDF](https://files.ele-math.com/articles/jca-12-03.pdf).
These two Scholar rows would otherwise remain separate because their observed
titles and years differ.

The 2018 row `u5HHmVD_uO8C`, *Stable geometrical properties of logharmonic
mappings*, is a **different article**: DOI `10.1080/17476933.2017.1351439`
in *Complex Variables and Elliptic Equations*. The
[author's ORCID/Crossref record](https://orcid.org/0000-0003-1865-2843)
lists that title and DOI separately from the *Journal of Classical Analysis*
article, so it is not an alias of the 2016/2018 arXiv and journal pair.

## Title repairs and retained observations

Scholar appended the author names after a colon in three article titles. The
published titles omit that suffix: `WF5omc3nYNoC`, [*On the Bohr’s Inequality
for Stable Mappings*](https://link.springer.com/article/10.1007/s40840-022-01401-1);
`eQOLeE2rZwMC`, [*Quadrature identities with a background PDE*](https://link.springer.com/article/10.1007/s13324-022-00674-w);
and `UeHWp8X0CEIC`, [*Logharmonic mappings on linearly connected domains*](https://api.crossref.org/works/10.1007/s13324-019-00318-6).
The reviewed corrections remove only those title suffixes; their author,
year, venue, and source fields remain unchanged.

Row `2osOgNQ5qMEC`, *On the univalence of polyharmonic mappings*, is an
ordinary [Elsevier research article](https://www.sciencedirect.com/science/article/pii/S0022247X17302755),
DOI `10.1016/j.jmaa.2017.03.033`. A secondary catalog's undifferentiated
"retracted" label is unsupported by the publisher; this review does not
exclude it. Row `qjMakFHDy7sC` is a distinct [McGill doctoral thesis](https://central.bac-lac.gc.ca/.item?app=Library&id=TC-QMM-107810&oclc_number=1032995899&op=pdf),
retained as an authored scholarly work. Row `Tyk-4Ss8FVUC`, *On convexity in
entire functions* (2007), has not been tied to a primary publication or thesis
record. Its displayed author name Layan Ahmad El Hajj agrees with the
[McGill/CRM fellowship report](https://www.crmath.ca/rapports/en/CRMannual2011-12.pdf),
but the work type remains unresolved. It stays in the raw capture without an
exclusion or metadata repair.

The publisher must validate each exact citation ID, observed title, year, and
source exclusion state against source PR #230 before applying these decisions.
