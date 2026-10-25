# Gautam Iyer and Robert Pego: six exact DOI work identities

Reviewed on 2026-10-25. This draft links six specific Robert Pego Scholar observations to six DOI-identified entries in Gautam Iyer's 47-row author-owned bibliography. The Iyer source remains `author_site`; no Google Scholar account was inferred. The trusted publisher still rejects author-site observations.

| Iyer row | Pego Scholar citation ID | DOI work identity | Relationship | Independent evidence |
| ---: | --- | --- | --- | --- |
| 13 | `neJ3NIUAAAAJ:ML0RJ9NH7IQC` | `10.1137/22m1533207` | published version | [SIAM](https://epubs.siam.org/doi/10.1137/22M1533207), [Crossref](https://api.crossref.org/works/10.1137%2F22M1533207) |
| 24 | `neJ3NIUAAAAJ:J-pR_7NvFogC` | `10.1017/apr.2018.23` | published version | [Cambridge](https://www.cambridge.org/core/journals/advances-in-applied-probability/article/abs/coagulation-and-universal-scaling-limits-for-critical-galtonwatson-processes/AA83743F3EE751AAE422A963B37E088F), [Crossref](https://api.crossref.org/works/10.1017%2Fapr.2018.23) |
| 27 | `neJ3NIUAAAAJ:LPZeul_q3PIC` | `10.1137/15m1054730` | published version | [SIAM](https://epubs.siam.org/doi/10.1137/15M1054730), [Crossref](https://api.crossref.org/works/10.1137%2F15M1054730) |
| 32 | `neJ3NIUAAAAJ:2osOgNQ5qMEC` | `10.4310/cms.2015.v13.n1.a12` | 2013 preprint of 2015 journal article | [arXiv](https://arxiv.org/abs/1308.4735), [publisher PDF](https://archive.intlpress.com/site/pub/files/_fulltext/journals/cms/2015/0013/0001/CMS-2015-0013-0001-a012.pdf), [Crossref](https://api.crossref.org/works/10.4310%2FCMS.2015.v13.n1.a12) |
| 33 | `neJ3NIUAAAAJ:EUQCXRtRnyEC` | `10.1214/14-aap1008` | published version | [arXiv](https://arxiv.org/abs/1212.6451), [IMS issue PDF](https://imstat.org/publications/aap/aap_25_2/AAP_25_2.pdf), [Crossref](https://api.crossref.org/works/10.1214%2F14-AAP1008) |
| 34 | `neJ3NIUAAAAJ:qjMakFHDy7sC` | `10.1063/1.4738637` | published version | [arXiv](https://arxiv.org/abs/1202.5671), [author manuscript](https://www.math.cmu.edu/~gautam/sj/research/papers/201202-extdyn.pdf), [Crossref](https://api.crossref.org/works/10.1063%2F1.4738637) |

Each ledger entry pins the Scholar raw source-record SHA-256; the Iyer source type, person and publication IDs, observation ID, raw title and year, reviewed DOI, source-record SHA-256; and the Iyer capture manifest SHA-256 `421e40cf104781b1500644f9f74ad81ce6768881edcf55daf9463626d69ff6eb`. The exact six-pair fixture is `tests/fixtures/math_iyer_pego_exact_rows.json`, derived from the 51,926-row offline snapshot with SHA-256 `9bf9570577618313c333155836187e800c55e50526950d98650decc08321e364`. A changed or missing record fails validation before work grouping.

Iyer row 4 and the Slepčev arXiv retitle remain held pending a native-ID alias policy. Iyer row 22 and its two Pego Scholar preprint versions remain held pending a three-record version-family policy. Neither is in this ledger. No raw title, year, author string, citation count, or source record changes. Canonical work citation count uses Scholar's retained count rather than adding observations.

The source snapshot is unchanged. This branch does not merge the draft producer changes, publish a Hub dataset, or update a public map.
