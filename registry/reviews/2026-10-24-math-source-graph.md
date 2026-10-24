# Mathematical Sciences source graph — 2026-10-24

The [CMU Mathematical Sciences faculty directory](https://www.cmu.edu/math/people/faculty/index.html)
supplies 66 included people in the current registry: 33 faculty, 17 teaching
faculty, and 16 emeriti. Twenty have Scholar IDs linked from official CMU
pages. A profile ID alone supplies no publication observations. This pass
prioritized faculty whose work connects mathematics to engineering and
materials science.

## First direct profile batch

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

## Second direct profile batch

The [Prasad Tetali](https://www.cmu.edu/math/people/faculty/tetali.html)
and [Dejan Slepčev](https://www.cmu.edu/math/people/faculty/slepcev.html)
CMU pages link their respective Scholar IDs. A separate, explicitly started
local Chrome batch opened those two direct profile URLs on 2026-10-24. The
profile loads were 95 seconds apart. The receipt reports no access refusal
or failed profile. The batch used no search results, citation traversal,
proxies, or CAPTCHA bypass.

The batch retained 200 Tetali rows and 86 Slepčev rows. Tetali's profile
still offered more publications at the 200-row cap, so its cache is marked
`capped_partial`. Slepčev's visible profile had no further rows and is marked
`complete_visible_profile`. Two rows had no displayed year and 18 had no
displayed venue across the two profiles; those fields remain blank. Two
Slepčev citation counts included an asterisk in the displayed source; the
cache retains that display alongside the numeric count. Each publication's
ID, title, authors, venue, year, citation display, and citation URL was
checked against saved HTML, and the source JSON and HTML checksums were
verified. The [dated second-batch receipt](https://github.com/cmccomb/map-of-research/blob/automation/map-snapshot/status/browser-capture-2026-10-24-batch-2.json)
records the profile IDs, coverage, counts, and checksums. These are captured
profile observations; the number of distinct or mapped works depends on the
later normalization and map-eligibility audit.

## Stopped direct-profile attempt

A later headed Chrome attempt opened the direct Scholar ID `ms9O2zoAAAAJ`
linked from [David R. Owen's CMU page](https://www.cmu.edu/math/people/faculty/owen.html).
The profile request returned HTTP 404 at about 19:16 UTC on 2026-10-24, so
collection stopped immediately and the browser was closed. No publication
rows were captured and no second profile was requested. The CMU-linked ID
remains in the registry; this access gap is not evidence that Owen has no
publications or that the linked identity is invalid. The
[dated failure receipt](https://github.com/cmccomb/map-of-research/blob/automation/map-snapshot/status/browser-attempt-2026-10-24-3.json)
records the HTTP status, stopped batch, and closed browser.

## Verified David R. Owen works

The stopped Scholar attempt left Owen without a profile cache. His
[CMU faculty page](https://www.cmu.edu/math/people/faculty/owen.html)
independently lists the three works below, and each publisher's DOI page
confirms the title, authors, venue, and year. Four curated authorship
observations attribute the three works to Owen and, for the 1999 paper,
also to Fonseca. They retain the CMU and publisher URLs and the verification
date; they do not claim to be captured Scholar observations.

| Year | Work | DOI evidence |
| --- | --- | --- |
| 1999 | Structured Deformations as Energy Minimizers in Models of Fracture and Hysteresis, with Irene Fonseca | [Mathematics and Mechanics of Solids](https://journals.sagepub.com/doi/abs/10.1177/108128659900400304) |
| 1998 | Antiplane Shear Flows in Viscoplastic Solids Exhibiting Isotropic and Kinematic Hardening, with J. M. Greenberg | [SIAM Journal on Applied Mathematics](https://epubs.siam.org/doi/10.1137/S0036139997314940) |
| 1993 | Structured deformations of continua, with Gianpietro Del Piero | [Archive for Rational Mechanics and Analysis](https://link.springer.com/article/10.1007/BF00375133) |

Fonseca's existing Scholar cache contains the 1999 title and year. A local
normalization check resolved her cached observation and the two curated
authorships to one DOI work with both CMU faculty IDs. The 1998 and 1993
DOI works are separate. Owen's full publication coverage remains unknown
until a later source becomes available.

## Next verified collaboration batch (staged for review)

Five DOI works add six curated faculty observations, without claiming a new
Scholar capture. Each authorship was checked against a CMU faculty identity
and the linked publisher record. The publisher's full ScienceDirect pages
were inaccessible during this review, so the two Elsevier DOI records also
use the [CMU Center for Nonlinear Analysis record](https://www.math.cmu.edu/cna/Publications/publication_auto.php?cnacode=25-CNA-006)
or [Ian Tice's CMU publications page](https://www.math.cmu.edu/~iantice/publications.html)
for the author list. The staged Scholar caches already contain observations
for the other registered coauthors listed below; those observations are
retained rather than duplicated as curated faculty rows.

| Year | DOI work | New curated faculty observations | Existing cached coauthors |
| --- | --- | --- | --- |
| 2025 | [Accretion and ablation in deformable solids using an Eulerian formulation: A finite deformation numerical method](https://www.sciencedirect.com/science/article/pii/S0022509625000523) | [Noel J. Walkington](https://www.cmu.edu/math/people/faculty/walkington.html) | Anthony Rollett, Kaushik Dayal |
| 2022 | [Accretion and ablation in deformable solids with an Eulerian description: examples using the method of characteristics](https://journals.sagepub.com/doi/10.1177/10812865211054573) | Noel J. Walkington | Kaushik Dayal |
| 2023 | [Traveling Wave Solutions to the Free Boundary Incompressible Navier-Stokes Equations](https://onlinelibrary.wiley.com/doi/10.1002/cpa.22084) | [Ian Tice](https://www.cmu.edu/math/people/faculty/tice.html) | Giovanni Leoni |
| 2019 | [Traces for homogeneous Sobolev spaces in infinite strip-like domains](https://www.sciencedirect.com/science/article/pii/S0022123619300254) | Ian Tice | Giovanni Leoni |
| 2022 | [Comparison of simulated and measured grain volume changes during grain growth](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.6.033402) | [David Kinderlehrer](https://www.cmu.edu/math/people/faculty/kinderlehrer.html), [Robert M. Suter](https://www.cmu.edu/physics/people/faculty/suter.html) | Kaushik Dayal, Gregory S. Rohrer |

Suter's [CMU Physics page](https://www.cmu.edu/physics/people/faculty/suter.html)
also names the 2022 grain-growth work and identifies him as an emeritus
Physics professor with a courtesy Materials Science and Engineering
appointment. The [APS DOI record](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.6.033402)
lists his and Kinderlehrer's names and CMU affiliations. The other authors
of the five works who are not in the current included registry are preserved
in the full author strings; no faculty identity is inferred for them.

A local normalization check using the staged 220-profile cache resolved each
DOI to exactly one map-eligible work. The six new curated observations joined
seven already cached Scholar observations across those five works. This
validation establishes the proposed links in the producer's normalized data;
the batch is not in a public snapshot until a later upload and site deploy.

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

The first CMU-wide normalized pre-layout snapshot has 38,592 observations and
218 cached profiles across Engineering, SCS, and MCS. It yields 280 distinct
Math-linked works, 274 of which pass the initial map-eligibility policy; the
later spatial audit can refine visibility. Those counts precede the second
direct profile batch above. Missing caches for other Math people remain a
collection gap, not evidence of no publications.

## Bounded Scholar batch 3 (staged for review)

A direct browser capture of the [CMU-linked Ian Tice profile](https://www.cmu.edu/math/people/faculty/tice.html)
returned all 63 visible Scholar rows. A second direct capture of the
[CMU-linked Robert Pego profile](https://www.cmu.edu/math/people/faculty/pego.html)
returned all 196 visible rows. The requests were 107 seconds apart. The
browser was closed after the second capture; no other profile was requested
in this batch. Each captured row has its Scholar observation ID, citation
link, and the raw HTML/JSON capture hashes in the snapshot receipt. The
validated local raw snapshot has 39,147 observations: 39,129 from 222 cached
Scholar profiles and 18 curated observations. It has not been uploaded.

The separate reviewed-exclusion manifest adds 37 decisions keyed to exact
Scholar observation IDs: three for Tice and 34 for Pego. Before these
decisions, all 63 Tice rows and 191 of Pego's 196 rows passed automatic
quality screening. After them, 60 Tice rows and 157 Pego rows remain
eligible. The 37 excluded observations remain in the raw table with their
reasons and evidence; they do not create map authorships. Nineteen Pego rows
are [NSF awards](https://www.nsf.gov/awardsearch/), and four are labels for
[CMU's Center for Nonlinear Analysis](https://www.cmu.edu/math/cna/).
Other entries include a Tice seminar and journal title, fragments, and
corrupted duplicate citations. Pego's [CMU publication list](https://www.math.cmu.edu/~bobpego/)
confirms the retained scholarly works behind the CNA report-number variants.
The original [2026 Hausdorff corrigendum](https://sites.math.duke.edu/~jliu/pdf/LiuPego_TAMS_2026.pdf)
is retained; only a garbled duplicate title is excluded.

The [Majda memorial](https://people.math.wisc.edu/~stechmann/publications/MajdaMemorial-EngquistSouganidisStechmannVicol2023preprint.pdf)
is kept provisionally. Its front byline has four authors, while Pego
contributed a signed section, so an automatic false-attribution decision
would overstate what the evidence establishes. No unique identified
research work was removed in this review.
