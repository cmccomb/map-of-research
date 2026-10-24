# Physics four-profile source exclusions, 2026-10-24

The four complete Physics Scholar profiles were merged into the source branch through [PR #109](https://github.com/cmccomb/map-of-research/pull/109). This paired producer draft adds only six exact row exclusions for Kozii, Tarnopolsky, and Di Matteo. The source observations and caches remain intact. With the five already reviewed Kozii and Tarnopolsky exclusions on `main`, the expected 451 raw observations yield 440 map-eligible observations.

Each ID below occurs once in the saved raw capture and once in the corresponding converted author cache. The full IDs and reason codes are in [`data/curated/scholar-exclusions.json`](../../data/curated/scholar-exclusions.json).

| Profile | Capture row | ID suffix | Reviewed decision and primary evidence |
| --- | ---: | --- | --- |
| Vladyslav Kozii | 35 | `-f6ydRqryjwC` | Incomplete publisher-metadata duplicate of the [2020 *Nature Communications* article](https://www.nature.com/articles/s41467-020-14819-7), retained as complete profile row 9 (`KlAtU1dfN6UC`). |
| Grigory Tarnopolsky | 53 | `ZHo1McVdvXMC` | Misread first author and incomplete metadata duplicate the [PTEP 2016 article](https://academic.oup.com/ptep/article/2016/12/12C105/2624094), retained as complete profile row 12 (`roLk4NBRz8UC`). |
| Grigory Tarnopolsky | 54 | `dfsIfKJdRG4C` | Second record for the [arXiv:2606.19732 paper](https://arxiv.org/abs/2606.19732), retained as complete profile row 55 (`cFHS6HbyZ2cC`). |
| Tiziana Di Matteo | 238 | `z8nqeaKD1nsC` | Affiliation text extracted from the [Sijacki et al. paper](https://arxiv.org/pdf/0705.2238), retained as complete profile row 5 (`olpn-zPbct0C`); it is not a separate publication. |
| Tiziana Di Matteo | 320 | `8dzOF9BpDQoC` | Abstract fragment from [KwangHo Park's 2012 dissertation](https://www.astro.umd.edu/sites/default/files/theses/people/2012Park.pdf); Di Matteo is an external advisory committee member, not an author. |
| Tiziana Di Matteo | 328 | `mvPsJ3kp5DgC` | [Supplementary methods PDF](https://web.phys.cmu.edu/~tiziana/NatureSI/si.pdf) for the [2005 *Nature* letter](https://www.nature.com/articles/nature03335), retained as complete profile row 1 (`TFP_iSt0sucC`); it is not a separate publication. |

The other saved rows remain eligible or pending exact source review. The producer changes in this draft require the source rows from PR #109 for a combined upload; this draft does not build a snapshot or publish a dataset.
