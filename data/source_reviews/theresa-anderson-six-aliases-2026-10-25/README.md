# Theresa Anderson: six held title-alias resolutions

This source-only review resolves the six website/CV citation differences flagged
in [source PR #340](https://github.com/cmccomb/map-of-research/pull/340),
[author-site staging PR #344](https://github.com/cmccomb/map-of-research/pull/344),
and [composition PR #348](https://github.com/cmccomb/map-of-research/pull/348).
The [CMU-hosted research page](https://faculty.math.cmu.edu/anderson/research/)
and [August 2026 CV](https://faculty.math.cmu.edu/anderson/wp-content/uploads/sites/2/2026/08/CV2026.pdf)
matched the pinned captures in PR #340 by SHA-256 on October 25. Their exact
six pairs of citations are preserved in `author-site-cv-raw-six.csv`.

| Site ordinal / CV number | Author-page title | CV title | Publisher or journal title | Exact work key |
| --- | --- | --- | --- | --- |
| 21 / 21 | On the Ergodic Waring-Goldbach Problem | The Ergodic Goldbach-Waring problem | On the ergodic Waring–Goldbach problem | [`10.1016/j.jfa.2021.109334`](https://doi.org/10.1016/j.jfa.2021.109334), JFA 282(5), article 109334 |
| 22 / 18 | Sharp Mei’s lemma for different bases | Sharp Mei’s lemma with different bases | Sharp Mei’s Lemma with Different Bases | [`10.1007/s00025-021-01587-z`](https://doi.org/10.1007/s00025-021-01587-z), Results Math. 77(2), article 69 |
| 23 / 20 | A structure theorem for measures with different bases | A structure theorem on doubling measures: a number theoretic approach | A structure theorem on doubling measures with different bases | [`10.1016/j.jmaa.2021.125620`](https://doi.org/10.1016/j.jmaa.2021.125620), JMAA 505(1), article 125620 |
| 24 / 22 | Bounds for discrete multilinear spherical maximal operators | Bounds for multilinear spherical maximal functions | Bounds for discrete multilinear spherical maximal functions | [`10.1007/s13348-020-00308-z`](https://doi.org/10.1007/s13348-020-00308-z), Collect. Math. 73(1), pages 75–87 |
| 25 / 15 | Bounds for discrete multilinear spherical maximal operators in higher dimensions | Bounds for multilinear spherical maximal functions in higher dimensions | Bounds for discrete multilinear spherical maximal functions in higher dimensions | [`10.1112/blms.12465`](https://doi.org/10.1112/blms.12465), BLMS 53(3), pages 855–860 |
| 33 / 9 | A Dyadic Gehring Inequality and Applications | A Dyadic Gehring Inequality on spaces of homogeneous type and applications | A dyadic Gehring inequality in spaces of homogeneous type and applications | [NYJM 24 (2018), pages 1–19](https://nyjm.albany.edu/j/2018/24-1.html); no journal DOI displayed |

For each row, the two author-source citations have matching author lists and
identical journal issue, article number, or page coordinates. The five Crossref
publisher deposits give the cited DOI, title, and authors. The NYJM journal
page gives the sixth title, authors, volume, year, and pages. Each website/CV
pair therefore refers to **one published work**, while all three raw title
forms remain visible. The NYJM article has a [separate arXiv preprint
DOI](https://arxiv.org/abs/1606.03461); that DOI is not assigned to the
journal version.

Six arXiv records are retained as version evidence: [1703.02713](https://arxiv.org/abs/1703.02713),
[2009.11906](https://arxiv.org/abs/2009.11906),
[2012.05778](https://arxiv.org/abs/2012.05778),
[1910.11409](https://arxiv.org/abs/1910.11409),
[1911.00464](https://arxiv.org/abs/1911.00464), and
[1606.03461](https://arxiv.org/abs/1606.03461). The fifth arXiv page
explicitly links the BLMS DOI; the others are supported by matching title and
byline but do not themselves assert a journal DOI link. The Collectanea paper
first appeared online in 2021 and in its cited print issue in 2022; both dates
are preserved in the Crossref deposit.

## Boundary

This review resolves six **work aliases**, not Anderson's whole-career or
Google Scholar coverage. No exact Scholar account is verified. All six rows
remain ineligible for source snapshot promotion, Hub publication, and graph
links. The loader emits neither canonical work IDs nor coauthor IDs. It leaves
PRs #340, #344, and #348 unchanged. Run
`python scripts/validate_anderson_six_aliases.py` to verify the source hashes,
publisher matching keys, original citation preservation, and hold flags.
