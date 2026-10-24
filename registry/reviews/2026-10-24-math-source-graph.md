# Mathematical Sciences source graph — 2026-10-24

The [CMU Mathematical Sciences faculty directory](https://www.cmu.edu/math/people/faculty/index.html)
supplies 66 included people in the current registry: 33 faculty, 17 teaching
faculty, and 16 emeriti. Twenty have Scholar IDs linked from official CMU
pages. A profile ID alone supplies no publication observations. This pass
prioritized faculty whose work connects mathematics to engineering and
materials science.

## Direct profile batch

The [Irene Fonseca](https://www.cmu.edu/math/people/faculty/fonseca.html) and
[Giovanni Leoni](https://www.cmu.edu/math/people/faculty/leoni.html) CMU pages
link their Scholar IDs. One explicitly started local Chrome session opened
those two direct profile URLs on 2026-10-24, with 97 seconds between the first
requests. It used no search results, citation traversal, proxies, or CAPTCHA
bypass, and closed after capture. Neither profile showed a refusal.

The batch retained 200 Fonseca rows and 134 Leoni rows. Fonseca's profile still
showed more publications at the 200-row cap; her cache is marked
`capped_partial`. Leoni's visible profile showed no further rows. Every
cached publication ID, title, authors, venue, year, and citation URL was
checked against its saved source HTML. Thirteen rows have no displayed year;
37 have no displayed venue. Those fields remain blank rather than guessed.
The cache files hold per-profile capture method, coverage, source-page
checksums, and citation URLs. The dated receipt is
`status/browser-capture-2026-10-24.json` on the snapshot branch.

## Verified Mathematics–MSE links

Six curated observations identify three DOI works coauthored by
[David Kinderlehrer](https://www.cmu.edu/math/people/faculty/kinderlehrer.html),
[Shlomo Ta'asan](https://www.cmu.edu/math/people/faculty/taasan.html), and
MSE's Anthony Rollett. Neither Math faculty member has a verified Scholar ID
in the registry, so the curated records make their authorships visible without
inventing an ID. The same three titles were already present in Rollett's
Scholar cache; the DOI records establish their persistent work identities.

| Year | Work | DOI evidence |
| --- | --- | --- |
| 1998 | Extracting the relative grain boundary free energy and mobility functions from the geometry of microstructures | [Scripta Materialia](https://www.sciencedirect.com/science/article/abs/pii/S1359646297005307) |
| 2012 | Grain growth and the puzzle of its stagnation in thin films a detailed comparison of experiments and simulations | [CMU-hosted paper](https://www.math.cmu.edu/CNA/Publications/publications2010/012abs/10-CNA-012.pdf), [Materials Science Forum](https://www.scientific.net/MSF.715-716.473) |
| 2013 | Grain growth and the puzzle of its stagnation in thin films: The curious tale of a tail and an ear | [Progress in Materials Science](https://www.sciencedirect.com/science/article/abs/pii/S0079642513000339) |

`data/curated/works.json` retains the exact DOI, author list, source URLs,
and verification date for each faculty observation. The two previously
verified Tetali–Erdős DOI works remain separate curated records. Erdős is an
external historical coauthor in the collaboration map and appears only when
the viewer's **Show Erdős** control is selected. A global Erdős-distance
coloring is deferred pending a broader coauthor source graph.

The normalized pre-layout snapshot from this pass has 38,592 observations and
218 cached profiles across Engineering, SCS, and MCS. It yields 280 distinct
Math-linked works, 274 of which pass the initial map-eligibility policy; the
later spatial audit can refine visibility. Missing caches for the other Math
people remain a collection gap, not evidence of no publications.
