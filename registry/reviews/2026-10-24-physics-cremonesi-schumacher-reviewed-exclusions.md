# Physics Scholar observation review: Cremonesi and Schumacher

The [source capture draft](https://github.com/cmccomb/map-of-research/pull/156) preserves all 1,963 browser-visible Matteo Cremonesi rows and 543 Reinhard Schumacher rows, with HTML, ordered JSON, caches, and receipts. This separate producer draft records **27 exact observation exclusions**: 20 Cremonesi and seven Schumacher. It changes no source row, registry membership, snapshot, manifest, or upload. A producer rebuild must wait for source integration and the current trusted upload to finish.

The review matched each candidate to its saved row identity, title, year, authors, venue, and citation count. Primary publisher, collaboration, conference, institutional, or author-preprint records then established a false attribution, a nonpublication, or the one published work represented by duplicate rows. Truncated collaboration author strings alone were never used to reject a genuine paper. The manifest keys each decision by full `scholar_id:author_pub_id`; title/year similarity is only a way to find candidates.

## Cremonesi: three false attributions and one presentation

| Exclude capture row and ID suffix | Decision and primary evidence |
| --- | --- |
| 27 `:FRqDyidJaO0C` | The [2010 CDF PRL paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.105.012001) and its [full collaboration preprint](https://arxiv.org/pdf/1004.3224) omit Cremonesi from the author list. |
| 55 `:WVyJMqqT78IC` | The [2011 CMS PRD record](https://cms-results.web.cern.ch/cms-results/public-results/publications/BPH-10-003/index.html) links the [full preprint and member appendix](https://arxiv.org/pdf/1012.5545), which omit Cremonesi. |
| 356 `:FPJr55Dyh1AC` | The [2011 CDF PRL paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.152001) and its [full preprint](https://arxiv.org/pdf/1103.2782) omit Cremonesi. |
| 1376 `:6ftYtcnYaCAC` | The [CHEP 2019 program](https://indico.cern.ch/event/773049/contributions/3476048/) identifies Nick Smith's COFFEA [slides](https://indico.cern.ch/event/773049/contributions/3476048/attachments/1937453/3211202/ncsmith-chep2019-coffea.pdf). This is a presentation file; retain the distinct 2020 EPJ article `:zxzvZ9-XIW8C`. |

The three paper rows use `false_profile_attribution`; the slide deck uses `nonpublication`. These are exact paper and file decisions, not a judgment about Cremonesi's CMS membership or COFFEA contributions.

## Cremonesi: sixteen duplicate observations

Each child below is a damaged or fragment-author-list copy of the retained publication. The saved child/parent rows agree on the work's title and year after accounting for title math or punctuation; the retained row has the publication locator and the primary record identifies one paper. The child rows remain in the source capture.

| Primary work record | Exclude child capture rows and suffixes | Retain suffix |
| --- | --- | --- |
| [2018 JPCS 1085 042030](https://cds.cern.ch/record/2291606/files/Gutsche_2018_J._Phys.__Conf._Ser._1085_042030.pdf) | 1956 `:6AKj_8xLoccC` (uppercase, undated) | `:lgwcVrK6X84C` |
| [CMS BTV-16-002, 2018 JINST 13 P05011](https://cms-results.web.cern.ch/cms-results/public-results/publications/BTV-16-002/index.html) | 1599 `:qjuL_XCUnM8C`; 1631 `:W91e3rS6dHEC` | `:9qwnyNWB-oEC` |
| [CMS PRF-14-001, 2017 JINST 12 P10003](https://cms-results.web.cern.ch/cms-results/public-results/publications/PRF-14-001/index.html) | 1771 `:gTud6kY3p2wC`; 1855 `:S0RksyIsHSIC` | `:P1qO8dLd1z8C` |
| [CMS SUS-16-048, 2018 Phys. Lett. B 782](https://cms-results.web.cern.ch/cms-results/public-results/publications/SUS-16-048/index.html) | 1574 `:q22ZieUsQXoC`; 1654 `:5uu-MYFt2EcC` | `:E7VqQtBCVmcC` |
| [CMS BPH-15-004, 2017 Phys. Lett. B 771](https://cms-results.web.cern.ch/cms-results/public-results/publications/BPH-15-004/index.html) | 1805 `:vYYylRVofzEC`; 1809 `:csTP5AdqY_wC` | `:VN7nJs4JPk0C` |
| [CMS HIG-18-011, 2019 Phys. Lett. B 795](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-18-011/) | 1445 `:T-Bu49-jKQgC`; 1483 `:-XtG2q47PdUC` | `:2P1L_qKh6hAC` |
| [CMS EXO-17-015, 2019 Phys. Lett. B 795](https://cms-results.web.cern.ch/cms-results/public-results/publications/EXO-17-015/index.html) | 1422 `:Yp96WNAcdNsC`; 1523 `:ve7iT2ZEuL4C` | `:q3oQSFYPqjQC` |
| [CMS B2G-18-001, 2019 JHEP 03 127](https://cms-results.web.cern.ch/cms-results/public-results/publications/B2G-18-001/) | 1440 `:aOEnwurrEPcC`; 1516 `:TiLqlu47W2oC` | `:C-Rn0OCouf8C` |
| [CMS HIG-17-035, 2018 PRL 120 231801](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-17-035/index.html) | 1669 `:IF0FbId8XVwC` | `:kvJssbFybhEC` |

The B2G child at row 1440 loses the W-prime symbol; the HIG child at row 1669 loses the t-bar symbol. Those display losses do not create separate works. The CMS official pages identify one publication per collaboration code. The [full BTV-16-002 preprint](https://arxiv.org/pdf/1712.07158) includes M. Cremonesi in its collaboration author list.

## Schumacher: seven exact decisions

| Exclude capture row and ID suffix | Code | Evidence and retained work, if applicable |
| --- | --- | --- |
| 75 `:7H_jS4BsgvYC` | `false_profile_attribution` | The [IUPAC technical report](https://publications.iupac.org/pac/83/7/1485/index.html) lists Barber, Karol, Nakahara, Vardaci, and Vogt, with no Schumacher. |
| 289 `:N6_Y7JlWxwsC` | `false_profile_attribution` | This row maps to the [publisher's 2014 ebook of Polybios-Lexikon Band I, Lieferung 1](https://www.degruyterbrill.com/document/doi/10.1524/9783050078076/html), whose [2014 catalog record](https://www.schweitzer-online.de/ebook/Alpha-Gamma/9783050078076/A17146324/) credits Collatz, Helms, and Schäfer. It does not credit Schumacher on this exact volume. |
| 379 `:R22Rs3tN8aoC` | `nonpublication` | The exact [AIP DOI 10.1063/1.5118368](https://doi.org/10.1063/1.5118368) identifies a HYP2018 **group photo**, which is front matter rather than a research article. The [AIP proceedings volume](https://printorders.aip.org/proceedings/2130) identifies the edited collection; keep the separately indexed volume `:SWgZeABleR0C`. |
| 380 `:i91s68tWr-MC` | `corrupted_duplicate` | Its formula-stripped title and blank venue copy the 2019 GlueX contribution retained as `:2vr6o8x5NLkC`. The [author preprint](https://arxiv.org/abs/1810.04688) supplies the full formula, authors, and published DOI. |
| 353 `:tz746QTLzJkC` | `corrupted_duplicate` | The title replaces formulas with “[formula presented].” [APS PRL 91 222002](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.91.222002) matches retained `:hqOjcs7Dif8C`. |
| 409 `:1_W9tMSvGuwC` | `corrupted_duplicate` | The uppercase, single-author copy omits the article locator. The [Moriya–Schumacher author preprint](https://arxiv.org/abs/1309.4472) matches retained `:P7Ujq4OLJYoC`, IJMPCS 26 1460084. |
| 481 `:mNrWkgRL2YcC` | `nonpublication` | Its title is a person/institution list copied from page 20 of the [1996 Particle Data Group report](https://www.osti.gov/servlets/purl/469140), a Crystal Ball experiment roster, not a work title. |

The Polybios decision concerns **Band I, Lieferung 1 only**. A [different volume's publisher title page](https://api.pageplace.de/preview/DT0400.9783050073958_A23039951/preview-9783050073958_A23039951.pdf) credits a Reinhard Schumacher as a collaborator in 1975. The identity of that collaborator relative to the CMU physicist is unresolved; no series-wide attribution inference follows. Likewise, six older history and economics rows with an `R Schumacher` name remain untouched pending distinct-person evidence.

## Limits and gate

The source ledger reports 125 exact casefold/whitespace Cremonesi title/year groups (260 rows), while a broader punctuation-stripping screen reports 216 groups (446 rows). These are **review queues**, not 216 exclusions. No title/year-wide rule, blanket consortium exclusion, or profile-wide deduction was applied. The 27 decisions above were checked one row at a time; other candidates remain in the source.

The producer test fixes the 27 exact count and checks that all stated retained parent IDs remain unexcluded. This draft targets `main` and depends on [source PR #156](https://github.com/cmccomb/map-of-research/pull/156). Do not merge or rebuild from this draft during the current trusted upload.
