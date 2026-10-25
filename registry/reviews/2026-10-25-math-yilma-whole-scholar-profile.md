# Zelealem Yilma: complete Scholar profile review

Reviewed 2026-10-25. The [official CMU Mathematics faculty page](https://www.cmu.edu/math/people/faculty/yilma.html) gives Yilma's CMU Qatar appointment and `zyilma@qatar.cmu.edu`. The complete [Google Scholar profile](https://scholar.google.com/citations?user=AlNKTGsAAAAJ&hl=en) was captured in [source PR #232](https://github.com/cmccomb/map-of-research/pull/232) at commit `1df7462`, with 20 distinct citation IDs and a disabled **Show More** button. Its verified `qatar.cmu.edu` address and linked CMU Math homepage identify the account. [Identity PR #228](https://github.com/cmccomb/map-of-research/pull/228) assigns that Scholar ID and Yilma's Math primary home; this producer draft does not duplicate the registry edit. The saved HTML, structured capture, and author cache match row for row; the raw source remains unchanged by this producer review.

## Official bibliography crosswalk

Yilma's [CMU-hosted personal bibliography](https://www.math.cmu.edu/~zyilma/) lists 14 distinct papers. Every one appears in the 20-row Scholar capture. The row numbers below are the saved capture order, not cache sort order.

| CMU bibliography paper | Scholar row | Exact citation suffix |
| --- | ---: | --- |
| Number of spanning trees in 4-regular simple graphs | 12 | `3fE2CSJIrl8C` |
| Minimum number of spanning trees in regular multigraphs | 9 | `MXK_kJrjxJIC` |
| Selection on sperm size | 8 | `8k81kl-MbHgC` |
| Eccentricity of networks with structural constraints | 6 | `YsMSGLbcyi4C` |
| Erdős–Rothschild problem on edge-colourings | 2 | `W7OEmFMy1HYC` |
| Supersaturation problem for color-critical graphs | 3 | `zYLM7Y9cAGgC` |
| Closeness centralization measure for two-mode data | 11 | `Y0pCki6q_DkC` |
| A new bound for the 2/3 conjecture | 5 | `u-x6o8ySG0sC` |
| A tight bound on the set chromatic number | 13 | `qjMakFHDy7sC` |
| Multiple Petersen subdivisions in permutation graphs | 19 | `IjCSPb-OGe4C` |
| Antimagic properties of graphs with large maximum degree | 4 | `9yKSN-GCB0IC` |
| Maximum number of K3-free and K4-free edge 4-colorings | 1 | `u5HHmVD_uO8C` |
| Hypergraph with a unique perfect matching | 10 | `2osOgNQ5qMEC` |
| Set systems without a strong simplex | 7 | `d1gkVwhDpl0C` |

The remaining six Scholar observations are reviewed, not silently dropped:

| Scholar row | Citation suffix | Decision and evidence |
| ---: | --- | --- |
| 14 | `UeHWp8X0CEIC` | Retain the distinct 2011 CMU PhD thesis, [author-hosted full text](https://www.math.cmu.edu/~zyilma/docs/thesis.pdf). |
| 15–16 | `5nxA0vEk-isC`, `0EnyYjriUFMC` | Retain the 2021 even-degree and odd-degree manuscript records. The [2022 publisher article](https://www.combinatorics.org/ojs/index.php/eljc/article/view/v29i4p29) and [author-hosted final text](https://www.math.cmu.edu/~zyilma/docs/PekarekSereniYilma_SpanningTreesRegularMultigraphs.pdf) cover both cases, but the specific earlier manuscripts are not independently linked to that DOI. Do not assert an alias or exclude either source row yet. |
| 17 | `LkGwnXOMwfcC` | Retain the 2018 weak-metadata eccentricity record. Its title matches row 6; the [journal text](https://www.dmgt.uz.zgora.pl/publish/article.php?doi=2180) verifies the complete four-author 2020 article. The exact relationship of the Scholar record, which names only Yilma, still needs direct source evidence. |
| 18 | `hqOjcs7Dif8C` | Retain the 2016 weak-metadata closeness record. Its title matches row 11; [arXiv](https://arxiv.org/abs/1608.04093) and [Cambridge](https://www.cambridge.org/core/journals/network-science/article/closeness-centralization-measure-for-twomode-data-of-prescribed-sizes/0ED95B1BE14895F6BEE99EF6701A92CA) give the complete four-author work. The exact type of the Scholar record remains unverified. |
| 20 | `eQOLeE2rZwMC` | Retain the 2012 [arXiv:1204.2519](https://arxiv.org/abs/1204.2519) preprint, with a reviewed alias to the published 2013 DOI and a guarded byline repair. |

## Exact producer decisions

Four entries in `data/curated/scholar-doi-aliases.json` bind the unchanged source rows to published works:

| Row | Relationship | Published DOI | Primary evidence |
| ---: | --- | --- | --- |
| 19 | `preprint_of` | `10.37236/2239` | [arXiv:1204.1989](https://arxiv.org/abs/1204.1989), [Electronic Journal of Combinatorics](https://www.combinatorics.org/ojs/index.php/eljc/article/view/v20i1p37) |
| 20 | `preprint_of` | `10.1017/S0963548312000612` | [arXiv:1204.2519](https://arxiv.org/abs/1204.2519), [Cambridge](https://www.cambridge.org/core/journals/combinatorics-probability-and-computing/article/new-bound-for-the-23-conjecture/46A4FEA3BE07A0D1498138BD15B0DBEF) |
| 5 | `version_of_record` | `10.1017/S0963548312000612` | Same arXiv and Cambridge records |
| 9 | `version_of_record` | `10.37236/10911` | [Author-hosted final text](https://www.math.cmu.edu/~zyilma/docs/PekarekSereniYilma_SpanningTreesRegularMultigraphs.pdf), [Electronic Journal of Combinatorics](https://www.combinatorics.org/ojs/index.php/eljc/article/view/v29i4p29) |

The [Cambridge article](https://www.cambridge.org/core/journals/combinatorics-probability-and-computing/article/new-bound-for-the-23-conjecture/46A4FEA3BE07A0D1498138BD15B0DBEF), [author-hosted publisher PDF](https://www.math.cmu.edu/~zyilma/docs/KLSWY_ANewBoundforthe23Conjecture.pdf), and [author-submitted arXiv record](https://arxiv.org/abs/1204.2519) all name Daniel Král', Chun-Hung Liu, Jean-Sébastien Sereni, Peter Whalen, and Zelealem B. Yilma. Scholar row 5 omits Král'; row 20 lists only Whalen and Yilma. Two exact entries in `data/curated/scholar-metadata-corrections.json` repair the effective bylines, guarded by the observed title, year, venue, and original byline. Raw HTML, cache rows, and `source_record_json` stay as captured. The normalized two-version work uses the 2013 journal year and full five-author byline.

This is a source review and draft producer decision. It does not publish the profile or change the source snapshot. The trusted release must still combine this producer branch with identity PR #228 and source PR #232, then rerun source-bound validation after the pending Donahue/Chien release.
