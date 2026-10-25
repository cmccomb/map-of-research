# Physics Scholar observation review: Cremonesi and Schumacher

The [source capture draft](https://github.com/cmccomb/map-of-research/pull/156) preserves all 1,963 browser-visible Matteo Cremonesi rows and 543 Reinhard Schumacher rows, with HTML, ordered JSON, caches, and receipts. This separate producer draft records **57 exact observation exclusions**: 35 Cremonesi and 22 Schumacher. It changes no source row, registry membership, snapshot, release manifest, or upload. Both public maps remain pinned to the published 0586c913 release; this review requires independent approval before any merge or producer rebuild.

The review matched each candidate to its saved row identity, title, year, authors, venue, and citation count. Primary publisher, collaboration, conference, institutional, or author-preprint records then established a false attribution, a nonpublication, or the one published work represented by duplicate rows. Truncated collaboration author strings alone were never used to reject a genuine paper. The manifest keys each decision by full `scholar_id:author_pub_id`; title/year similarity is only a way to find candidates.

## Cremonesi: four false attributions and one presentation

| Exclude capture row and ID suffix | Decision and primary evidence |
| --- | --- |
| 27 `:FRqDyidJaO0C` | The [2010 CDF PRL paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.105.012001) and its [full collaboration preprint](https://arxiv.org/pdf/1004.3224) omit Cremonesi from the author list. |
| 55 `:WVyJMqqT78IC` | The [2011 CMS PRD record](https://cms-results.web.cern.ch/cms-results/public-results/publications/BPH-10-003/index.html) links the [full preprint and member appendix](https://arxiv.org/pdf/1012.5545), which omit Cremonesi. |
| 356 `:FPJr55Dyh1AC` | The [2011 CDF PRL paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.152001) and its [full preprint](https://arxiv.org/pdf/1103.2782) omit Cremonesi. |
| 1949 `:Z7R3Ocg27JUC` | The exact title/year matches [CMS FWD-11-001](https://cms-results.web.cern.ch/cms-results/public-results/publications/FWD-11-001/index.html), published in 2013. Its [full author appendix](https://arxiv.org/pdf/1210.6718) omits Cremonesi; his [CMU CV](https://www.cmu.edu/physics/people/faculty/documents/cremonesi-cv.pdf) dates CMS membership from January 2015. |
| 1376 `:6ftYtcnYaCAC` | The [CHEP 2019 program](https://indico.cern.ch/event/773049/contributions/3476048/) identifies Nick Smith's COFFEA [slides](https://indico.cern.ch/event/773049/contributions/3476048/attachments/1937453/3211202/ncsmith-chep2019-coffea.pdf). This is a presentation file; retain the distinct 2020 EPJ article `:zxzvZ9-XIW8C`. |

The four paper rows use `false_profile_attribution`; the slide deck uses `nonpublication`. These are exact paper and file decisions, not a judgment about Cremonesi's other CMS work or COFFEA contributions.

## Cremonesi: thirty duplicate observations

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
| [CMS B2G-18-002, 2020 EPJC 80 237](https://cms-results.web.cern.ch/cms-results/public-results/publications/B2G-18-002/) | 969 `:6Dd5luMImnEC` (formula placeholders) | `:GwaQhVSQhKEC` |
| [CMS HIG-15-007, 2018 EPJC 78 708](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-15-007/index.html) | 1597 `:1DdUqAzPfXgC` (formula placeholders) | `:USX2HHcnDqcC` |
| [CMS TOP-16-005, 2017 EPJC 77 172](https://cms-results.web.cern.ch/cms-results/public-results/publications/TOP-16-005/index.html) | 1760 `:X0ywoi6J0UgC` (formula placeholders) | `:OcBU2YAGkTUC` (linked arXiv version) |
| [CMS TOP-15-015, 2017 EPJC 77 467](https://cms-results.web.cern.ch/cms-results/public-results/publications/TOP-15-015/) | 1744 `:1L_zwHEQtZoC` (formula placeholders) | `:VOx2b1Wkg3QC` |
| [CMS SMP-16-017, 2018 EPJC 78 165](https://cms-results.web.cern.ch/cms-results/public-results/publications/SMP-16-017/index.html) | 932 `:afsF9h1fxg0C` (formula placeholders) | `:GUYAmugLYisC` |
| [CMS HIN-19-011, 2022 PLB 829 137062](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-19-011/index.html) | 1266 `:besYCffB3V8C` (formula placeholder) | `:8p-ueveQw4wC` |
| [CMS SMP-17-012, 2019 EPJC 79 94](https://cms-results.web.cern.ch/cms-results/public-results/publications/SMP-17-012/index.html) | 1461 `:lDOOmgye57wC` (blank venue) | `:DvbUhBKpgQUC` |
| [CMS FSQ-15-005, 2018 JHEP 07 161](https://cms-results.web.cern.ch/cms-results/public-results/publications/FSQ-15-005/index.html) | 1550 `:cIAfX1pVE14C` (fragment authors and institution venue) | `:1sJd4Hv_s6UC` |
| [CMS EGM-17-001, 2021 JINST 16 P05014](https://cms-results.web.cern.ch/cms-results/public-results/publications/EGM-17-001/index.html) | 1323 `:XEKcsfOMjpkC` (fragment authors, blank venue) | `:u76CbRzuV8AC` |
| [CMS HIG-17-031, 2019 EPJC 79 421](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-17-031/) | 1525 `:uQdrZ3qaPmcC` (truncated title, blank venue) | `:IyxfXMz2bNAC` |
| [CMS HIN-16-010, 2017 PLB 765 193](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-16-010/index.html) | 1841 `:ey9_mCyZ-VYC` (fragment authors, blank venue) | `:9shLKfS_uJEC` |
| [CMS HIG-19-004, 2020 PLB 805 135425](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-19-004/) | 1360 `:-mA2wk0MEBkC` (fragment authors, blank venue) | `:5Klqo5HVOaoC` |
| [CMS BPH-19-003, 2020 PLB 803 135345](https://cms-results.web.cern.ch/cms-results/public-results/publications/BPH-19-003/index.html) | 1336 `:XeQYkT-08pIC` (incomplete locator); 1960 `:CXI6bF9CpJ4C` (undated OCR and formula loss) | `:It3QlvBKHbMC` |

The B2G child at row 1440 loses the W-prime symbol; the HIG child at row 1669 loses the t-bar symbol. Those display losses do not create separate works. The CMS official pages identify one publication per collaboration code. [NLM independently maps](https://pubmed.ncbi.nlm.nih.gov/28408859/) the formula-placeholder row 1760 to EPJC 77, 172 and its DOI; the CMS page links that work to the retained arXiv row. The [full BTV-16-002 preprint](https://arxiv.org/pdf/1712.07158) includes M. Cremonesi in its collaboration author list. The [SMP-16-017 erratum](https://cms-results.web.cern.ch/cms-results/public-results/publications/SMP-16-017/index.html), captured separately as `:y4Au4brZMfoC`, remains eligible.

## Schumacher: twenty-two exact decisions

| Exclude capture row and ID suffix | Code | Evidence and retained work, if applicable |
| --- | --- | --- |
| 75 `:7H_jS4BsgvYC` | `false_profile_attribution` | The [IUPAC technical report](https://publications.iupac.org/pac/83/7/1485/index.html) lists Barber, Karol, Nakahara, Vardaci, and Vogt, with no Schumacher. |
| 289 `:N6_Y7JlWxwsC` | `false_profile_attribution` | This row maps to the [publisher's 2014 ebook of Polybios-Lexikon Band I, Lieferung 1](https://www.degruyterbrill.com/document/doi/10.1524/9783050078076/html), whose [2014 catalog record](https://www.schweitzer-online.de/ebook/Alpha-Gamma/9783050078076/A17146324/) credits Collatz, Helms, and Schäfer. It does not credit Schumacher on this exact volume. |
| 379 `:R22Rs3tN8aoC` | `nonpublication` | The exact [AIP DOI 10.1063/1.5118368](https://doi.org/10.1063/1.5118368) identifies a HYP2018 **group photo**, which is front matter rather than a research article. The [AIP proceedings volume](https://printorders.aip.org/proceedings/2130) identifies the edited collection; keep the separately indexed volume `:SWgZeABleR0C`. |
| 380 `:i91s68tWr-MC` | `corrupted_duplicate` | Its formula-stripped title and blank venue copy the 2019 GlueX contribution retained as `:2vr6o8x5NLkC`. The [author preprint](https://arxiv.org/abs/1810.04688) supplies the full formula, authors, and published DOI. |
| 353 `:tz746QTLzJkC` | `corrupted_duplicate` | The title replaces formulas with “[formula presented].” [APS PRL 91 222002](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.91.222002) matches retained `:hqOjcs7Dif8C`. |
| 409 `:1_W9tMSvGuwC` | `corrupted_duplicate` | The uppercase, single-author copy omits the article locator. The [Moriya–Schumacher author preprint](https://arxiv.org/abs/1309.4472) matches retained `:P7Ujq4OLJYoC`, IJMPCS 26 1460084. |
| 481 `:mNrWkgRL2YcC` | `nonpublication` | Its title is a person/institution list copied from page 20 of the [1996 Particle Data Group report](https://www.osti.gov/servlets/purl/469140), a Crystal Ball experiment roster, not a work title. |
| 531 `:4QKQTXcH0q8C` | `nonpublication` | The [De Gruyter issue contents](https://www.degruyterbrill.com/document/doi/10.1515/jbwg-1991-320301/html) use “Abhandlungen, Studien, Miszellen” as a section heading followed by separately titled articles; the saved row is that heading. |
| 347 `:U0iAMwwPxtsC` | `nonpublication` | The [original Jefferson Lab document](https://www.jlab.org/exp_prog/proposals/16/LOI12-16-001.pdf) labels this exact item a Letter of Intent to PAC 44 for a proposed experiment. |
| 435 `:ODE9OILHJdcC` | `nonpublication` | The [original Jefferson Lab document](https://www.jlab.org/Hall-B/general/caa/caa_cmu_nov05.pdf) labels this exact item a 2005 CLAS data-analysis proposal. |
| 334 `:LK8CI43ZvvMC` | `nonpublication` | [Jefferson Lab PR12-17-007](https://misportal.jlab.org/mis/physics/experiments/viewProposal.cfm?paperId=923) labels the exact title an original, deferred proposal and separates related publications. |
| 290 `:86PQX7AUzd4C` | `nonpublication` | [Jefferson Lab PR12-12-002](https://misportal.jlab.org/mis/physics/experiments/viewProposal.cfm?paperId=721) identifies this arXiv-posted item as a PAC 39 experiment proposal, rather than a paper reporting results. |
| 539 `:LO7wyVUgiFcC` | `nonpublication` | The [PAC33 original](https://www.jlab.org/exp_prog/proposals/08/PR-08-013.pdf) is a proposal for future neutron form-factor measurements to Q² = 8.0 (GeV/c)². |
| 542 `:i2xiXl-TujoC` | `nonpublication` | The [PAC34 original](https://www.jlab.org/exp_prog/proposals/09/PR12-09-019.pdf) is a proposal for future neutron form-factor measurements to Q² = 18.0 (GeV/c)². |
| 541 `:edDO8Oi4QzsC` | `nonpublication` | The [PAC35 original](https://hallaweb.jlab.org/collab/PAC/PAC35/PR-10-005-GMN.pdf) is a proposal for future neutron form-factor measurements at Q² = 16.0 and 18.0 (GeV/c)². |

The following seven saved rows each copy a single APS article with formula placeholders or typography changes. The article number and primary record identify the retained publication:

| APS record | Exclude child capture rows and suffixes | Retain suffix |
| --- | --- | --- |
| [PRL 88 122001 (2002)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.88.122001) | 449 `:L24QuVWYgZ0C` | `:IjCSPb-OGe4C` |
| [PRC 65 034607 (2002)](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.65.034607) | 448 `:hHIA4WEVY-EC`; 447 `:gFrPXmx1TSsC` | `:hkOj_22Ku90C` |
| [PRC 68 035202 (2003)](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.68.035202) | 442 `:ymY9cBF3mdcC` | `:a0OBvERweLwC` |
| [PRC 68 032201 (2003)](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.68.032201) | 443 `:M0leSnx2MbUC`; 62 `:IWHjjKOFINEC` | `:ZqE1mSdD_DYC` |
| [PRL 91 022002 (2003)](https://doi.org/10.1103/PhysRevLett.91.022002) | 440 `:hNSvKAmkeYkC` | `:kNdYIx-mwKoC` |

The seven Jefferson Lab proposal decisions classify only the exact saved proposal documents; they do not exclude subsequent papers from those experiments. The [Hy Trac producer review](2026-10-24-physics-trac-reviewed-exclusions.md) already treats an author-hosted NSF experiment proposal as a nonpublication.

The Polybios decision concerns **Band I, Lieferung 1 only**. A [different volume's publisher title page](https://api.pageplace.de/preview/DT0400.9783050073958_A23039951/preview-9783050073958_A23039951.pdf) credits a Reinhard Schumacher as a collaborator in 1975. The identity of that collaborator relative to the CMU physicist is unresolved; no series-wide attribution inference follows. APS explicitly lists R. A. Schumacher at CMU on the genuine PRD 67 092001 paper `:iyewoVqAXLQC`; its `F₂` title repair belongs in the source draft, with no exclusion.

## Unresolved exact rows

| Saved row | Open evidence |
| --- | --- |
| Cremonesi `:zmHQPunddckC` (1926) | Generic “CMS Collaboration” title, 2014, with a fragmentary author list. His CMU CV dates CMS membership from 2015, but multiple 2014 CMS records share this generic label; there is no unique primary paper mapping. |
| Cremonesi `:QNiCEvFIB1MC` (1430) | “CORC,” 2019, has an ATLAS author fragment but no identifiable primary work. |
| Cremonesi `:wlzmIqt2EaEC` (1962) | “CERN Document Server” has no year, venue, or identifiable underlying record. |
| Cremonesi `:PDAgwb0ELvkC` (68) | The saved `A. Tumasyan et al` title, 2022 year, and tracker-upgrade venue conflict with the [official CMS TDR](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/TDR-17-001/index.html), which is a 2017 report. A second saved row has the tracker title. No unique exact-record mapping yet supports a cache repair or exclusion. |
| Cremonesi `:8NFFclwVBVUC`, `:f2QdLyYkFyUC`, `:Gm6cJkd1Yu0C` | The titles map to [CMS TRG-17-001](https://cms-results.web.cern.ch/cms-results/public-results/publications/TRG-17-001/index.html) and [CMS JME-18-001](https://cms-results.web.cern.ch/cms-results/public-results/publications/JME-18-001/index.html), but these rows carry journal locators while retained high-citation rows carry arXiv locators. Hold the version and retained-row choice. |
| Cremonesi `:UBSsHvkqiq4C`, `:OlGoE9RybnsC`, `:QjNCP7ux8QYC` | The [HIN-18-015](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-18-015/index.html) and [BPH-13-001](https://cms-results.web.cern.ch/cms-results/public-results/publications/BPH-13-001/index.html) records show related preprint and journal variants; hold until the retained row is chosen for each family. |
| Schumacher `:6bLC7aUMtPcC` (412) | “Project Narrative Experimental Medium Energy Physics Carnegie Mellon University” resembles a grant narrative, but the [DOE award list](https://science.osti.gov/-/media/_/pdf/universities/sc_statebystate_awards_FY2011.pdf) does not identify the exact captured document. |
| Schumacher `:F9fV5C73w3QC` (480) | Saved title calls this a 1996 Crystal Ball proposal; an [experiment summary](https://lss.fnal.gov/archive/other/bnl-34518.pdf) identifies the team and proposed study, but the exact proposal document remains unverified. |
| Schumacher `:_mQi-xiA4oYC`, `:pRWBApOjXDcC`, `:adHtZc2wMuEC`, `:oldoQiaHq2UC`, `:VnuxuLaQPLMC`, `:M0j1y4EgrScC` | Six 1977–88 history/economics rows have an `R Schumacher` author. A same-name Greek lexicon collaborator and the CMU physicist's education timeline suggest a namesake, but no direct author identity link to these six rows has been established. Hold all six. |

## Limits and gate

The source ledger reports 125 exact casefold/whitespace Cremonesi title/year groups (260 rows), while a broader punctuation-stripping screen reports 216 groups (446 rows). These are **review queues**, not 216 exclusions. No title/year-wide rule, blanket consortium exclusion, or profile-wide deduction was applied. The 57 decisions above were checked one row at a time; unresolved candidates remain in the source. Three other genuine Cremonesi CDF papers with literal `<?format ?>` markers retain their IDs and receive cache-only title repairs in PR #156. After its nine cache repairs, the remaining 12 `[Formula]` cache-title rows are all exact duplicate exclusions in this draft; no unexcluded cache title retains those literal markers. This marker scan does not certify every title's spelling or attribution.

The producer test fixes the 57 exact count and checks that all stated retained parent IDs remain unexcluded. This draft targets `main` and depends on [source PR #156](https://github.com/cmccomb/map-of-research/pull/156). Keep both drafts staged for independent review; do not merge, rebuild, or upload from this draft.
