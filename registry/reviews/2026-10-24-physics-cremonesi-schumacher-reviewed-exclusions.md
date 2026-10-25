# Physics Scholar observation review: Cremonesi and Schumacher

The [source capture draft](https://github.com/cmccomb/map-of-research/pull/156) preserves all 1,963 browser-visible Matteo Cremonesi rows and 543 Reinhard Schumacher rows, with HTML, ordered JSON, caches, and receipts. This separate producer draft records **100 exact observation exclusions**: 77 Cremonesi and 23 Schumacher. It also stages a publisher preflight for unresolved Physics rows; it changes no source row, registry membership, snapshot, release manifest, or upload. This review requires independent approval before any merge or producer rebuild; the draft has no effect on a published map.

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

## Cremonesi: seventy-two duplicate observations

Each child below is a damaged or incomplete observation of the retained publication. The available saved title, year, author, and venue fields were checked against the cited primary work record and retained row. Some children lack a year or retain only a title prefix; the primary record and retained publication locator resolve those cases. The child rows remain in the source capture.

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

The following fourteen further children have fragment author lists and no displayed citations. Thirteen have blank venues, and one has an institution venue. Each maps to the retained publication named by the official CMS record; the cited arXiv row for HIG-18-002 remains eligible.

| Primary work record | Exclude child capture row and suffix | Retain suffix |
| --- | --- | --- |
| [CMS HIN-14-005, 2017 EPJC 77 252](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-14-005/index.html) | 1819 `:jvrRlaHg2sAC` | `:9pM33mqn1YgC` |
| [CMS HIG-17-026, 2019 JHEP 03 026](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-17-026/index.html) | 1435 `:zBs93F57jOAC` | `:cK4Rrx0J3m0C` |
| [CMS FSQ-16-012, 2019 PLB 797 134826](https://cms-results.web.cern.ch/cms-results/public-results/publications/FSQ-16-012/index.html) | 1494 `:YEBtnu9hOHUC` | `:09LM3QYkMKUC` |
| [CMS HIG-17-023, 2019 PLB 793 520](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-17-023/index.html) | 1499 `:zTJoPluU4X4C` | `:urP0JZOBBUsC` |
| [CMS HIG-16-044, 2018 PLB 780 501](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-16-044/index.html) | 1676 `:IX653JsL2_EC` | `:kF1pexMAQbMC` |
| [CMS HIG-18-002, 2019 PRD 99 112003](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-18-002/index.html) | 1511 `:kxd3qP2_5uAC` | `:ziOE8S1-AIUC` (cited arXiv row) |
| [CMS HIN-16-001, 2018 PLB 782 474](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-16-001/index.html) | 1608 `:_IwLajd9sWMC` | `:Tyk-4Ss8FVUC` |
| [CMS HIG-17-030, 2019 PRL 122 121803](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-17-030/index.html) | 1488 `:o4Qvs5Y5TLQC` | `:MLfJN-KU85MC` |
| [CMS SMP-16-017, 2018 EPJC 78 165](https://cms-results.web.cern.ch/cms-results/public-results/publications/SMP-16-017/index.html) | 1629 `:lu5AzqVF4qYC` | `:GUYAmugLYisC` |
| [CMS EXO-16-048, 2018 PRD 97 092005](https://cms-results.web.cern.ch/cms-results/public-results/publications/EXO-16-048/index.html) | 1703 `:PPrs6GlykQcC` | `:uWQEDVKXjbEC` |
| [CMS BPH-15-008, 2018 PLB 781 517](https://cms-results.web.cern.ch/cms-results/public-results/publications/BPH-15-008/index.html) | 1641 `:t7izwRedFcYC` | `:cWzG1nlazyYC` |
| [CMS EXO-17-012, 2018 PRL 120 221801](https://cms-results.web.cern.ch/cms-results/public-results/publications/EXO-17-012/index.html) | 1668 `:ZyBjE3zI-i8C` | `:cFHS6HbyZ2cC` |
| [CMS SUS-18-002, 2019 EPJC 79 444](https://cms-results.web.cern.ch/cms-results/public-results/publications/SUS-18-002/index.html) | 1410 `:6hPiuOuOds4C` (institution venue); 1527 `:Gs1AF5H0x94C` (blank venue) | `:Ggqy4mf-avwC` (cache title and locator repaired in source draft) |

The next twenty zero-citation children duplicate cited records among the higher-ranked saved rows. The child at row 1198 has a JHEP venue; its cited parent receives the missing article locator in source PR #156. Rows 1860 and 1880 retain the cited arXiv observation because the child has no independent published-version record. The tranche stops at parent row 102; row 101 has the same displayed citation count and remains in the review queue.

| Primary work record | Exclude child capture row and suffix | Retain suffix |
| --- | --- | --- |
| [EXO-16-050, EPJC 79 280](https://cms-results.web.cern.ch/cms-results/public-results/publications/EXO-16-050/index.html) | 1412 `:FtLLHOsGVzkC` | 51 `:4X0JR2_MtJMC` |
| [SMP-16-002, PLB 766 268](https://cms-results.web.cern.ch/cms-results/public-results/publications/SMP-16-002/index.html) | 1824 `:AdGrMoNItN4C` | 57 `:PffFD9Wx8kQC` |
| [HIN-16-023, PLB 790 270](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-16-023/index.html) | 1479 `:X41XOdD1uaEC` | 60 `:nPT8s1NX_-sC` |
| [EXO-15-010, PRD 94 112004](https://cms-results.web.cern.ch/cms-results/public-results/publications/EXO-15-010/index.html) | 1901 `:eJjLl3UG7CkC` | 61 `:F2UWTTQJPOcC` |
| [SUS-16-033, PRD 96 032003](https://cms-results.web.cern.ch/cms-results/public-results/publications/SUS-16-033/index.html) | 1800 `:rCuJAniQoLoC` | 62 `:1DsIQWDZLl8C` |
| [B2G-16-028, EPJC 78 707](https://cms-results.web.cern.ch/cms-results/public-results/publications/B2G-16-028/index.html) | 1692 `:QZWLLlSfqgYC` | 65 `:yB1At4FlUx8C` |
| [SUS-19-006, JHEP 10 244](https://cms-results.web.cern.ch/cms-results/public-results/publications/SUS-19-006/index.html) | 1503 `:AbQWx2m_oG8C` | 67 `:bkKuixW_xMkC` |
| [HIN-16-025, EPJC 78 509](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-16-025/index.html) | 1674 `:PQ6BjWB6CiEC` | 70 `:zJ9_mGRecVUC` |
| [EXO-22-024, JHEP 08 215](https://cms-results.web.cern.ch/cms-results/public-results/publications/EXO-22-024/index.html) | 1198 `:cvOuzwNwsG0C` | 71 `:p39txVt3toMC` |
| [TOP-16-003, PLB 772 752](https://cms-results.web.cern.ch/cms-results/public-results/publications/TOP-16-003/index.html) | 1872 `:1l3MdapXzAoC` | 73 `:KNjnJ3z-R6IC` |
| [EXO-14-005, PRL 117 031802](https://cms-results.web.cern.ch/cms-results/public-results/publications/EXO-14-005/index.html) | 1894 `:OA9cGvo6324C` | 76 `:WqliGbK-hY8C` |
| [B2G-16-016, arXiv:1706.04260](https://cms-results.web.cern.ch/cms-results/public-results/publications/B2G-16-016/index.html) | 1860 `:HmWvvuJwtxYC` | 86 `:bVQMTfhMCi4C` |
| [HIN-16-006, PRL 120 142302](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-16-006/index.html) | 1686 `:ddB7do2jUx8C` | 87 `:HtS1dXgVpQUC` |
| [HIG-13-032, arXiv:1603.06896](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-13-032/index.html) | 1880 `:kmu2AJ0NrTYC` | 89 `:KxtntwgDAa4C` |
| [SMP-16-001, PLB 763 280](https://cms-results.web.cern.ch/cms-results/public-results/publications/SMP-16-001/index.html) | 1899 `:8EvVLpklxGMC` | 93 `:N2osnxbUuuUC` |
| [HIN-16-011, PRL 119 152301](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-16-011/index.html) | 1846 `:rwEhk56xNqMC` | 94 `:-_dYPAW6P2MC` |
| [CDF-D0-HIGGS-2013, PRD 88 052014](https://lss.fnal.gov/archive/2013/pub/fermilab-pub-13-081-e.pdf) | 1940 `:3I1Fh14WnZcC` | 96 `:EYYDruWGBe4C` |
| [HIN-16-009, PRL 118 122301](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-16-009/index.html) | 1856 `:PcT55Ow6fAIC` | 98 `:3NQIlFlcGxIC` |
| [EXO-16-018, PRL 117 051802](https://cms-results.web.cern.ch/cms-results/public-results/publications/EXO-16-018/index.html) | 1903 `:qt-6tCTBDsQC` | 99 `:ZzlSgRqYykMC` |
| [HIG-17-013, PLB 793 320](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-17-013/index.html) | 1480 `:ZpfHopiqs50C` | 102 `:pqnbT2bcN3wC` |

Six remaining literal `\documentclass` titles are damaged copies of published EPJC papers. The journal issue and article locator, together with the CMS publication record, resolve the shorter title prefixes.

| Primary work record | Exclude child capture row and suffix | Retain suffix |
| --- | --- | --- |
| [FSQ-16-007, EPJC 79 702](https://cms-results.web.cern.ch/cms-results/public-results/publications/FSQ-16-007/index.html) | 972 `:sdbE4nmAF2cC` | 286 `:0Kh4an1R61UC` |
| [SMP-16-007, EPJC 78 701](https://cms-results.web.cern.ch/cms-results/public-results/publications/SMP-16-007/index.html) | 979 `:4ENpXO9GHvQC` | 153 `:OP4eGU-M3BUC` |
| [TOP-17-015, EPJC 79 123](https://cms-results.web.cern.ch/cms-results/public-results/publications/TOP-17-015/index.html) | 1388 `:KJj4oxv5wnIC` | 792 `:6VlyvFCUEfcC` |
| [SMP-16-015, EPJC 78 965](https://cms-results.web.cern.ch/cms-results/public-results/publications/SMP-16-015/index.html) | 1529 `:ugwUUJJV9MUC` | 191 `:HP8h_xu6YbIC` |
| [BPH-16-003, EPJC 78 939](https://cms-results.web.cern.ch/cms-results/public-results/publications/BPH-16-003/index.html) | 1530 `:UrmhOAI2Q4QC` | 931 `:t4WtaE3RIIAC` |
| [SMP-16-011, EPJC 77 746](https://cms-results.web.cern.ch/cms-results/public-results/publications/SMP-16-011/index.html) | 1778 `:0-VSGmPhd2EC` | 251 `:TjyTfcaXIY8C` |

The [CMS HIN-24-002 publication](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-24-002/index.html) identifies the 2025 PRL 135 article 132301. Exclude its formula-placeholder row 1159 `:tATOuK1XnKgC` and publisher-only venue row 1145 `:XFIFPR1yFx4C`; retain cited row 687 `:_-FgD5s_Z1cC`. The retained title's missing square-root symbol is repaired in source PR #156.

The B2G child at row 1440 loses the W-prime symbol; the HIG child at row 1669 loses the t-bar symbol. Those display losses do not create separate works. The CMS official pages identify one publication per collaboration code. [NLM independently maps](https://pubmed.ncbi.nlm.nih.gov/28408859/) the formula-placeholder row 1760 to EPJC 77, 172 and its DOI; the CMS page links that work to the retained arXiv row. The [full BTV-16-002 preprint](https://arxiv.org/pdf/1712.07158) includes M. Cremonesi in its collaboration author list. The [SMP-16-017 erratum](https://cms-results.web.cern.ch/cms-results/public-results/publications/SMP-16-017/index.html), captured separately as `:y4Au4brZMfoC`, remains eligible.

## Schumacher: twenty-three exact decisions

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
| 480 `:F9fV5C73w3QC` | `nonpublication` | The [Ruđer Bošković Institute's original 1996 annual report](https://fulir.irb.hr/131/1/Godisnji_izvjestaj_1996_pdf.pdf), p. 51, lists the saved title and R. A. Schumacher's team as **BNL AGS Proposal 924 (1996)** under internal publications and preprints. The [BNL experiment catalog](https://lss.fnal.gov/archive/other/bnl-34518.pdf) confirms the same proposed Crystal Ball study; this decision does not cover later result papers. |
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

Seventeen exact IDs remain in this unresolved table: ten Cremonesi and seven Schumacher. Sixteen currently pass the automatic map-quality check; Cremonesi `:OlGoE9RybnsC` is filtered as garbled index text. Tracker companion row 547 and three Schumacher same-work pairs are additional holds. They stay in the raw source capture pending record identity, attribution, or retained-version evidence.

| Saved row | Open evidence |
| --- | --- |
| Cremonesi `:zmHQPunddckC` (1926) | Generic “CMS Collaboration” title, 2014, with a fragmentary author list. His CMU CV dates CMS membership from 2015, but multiple 2014 CMS records share this generic label; there is no unique primary paper mapping. |
| Cremonesi `:QNiCEvFIB1MC` (1430) | “CORC,” 2019, has an ATLAS author fragment but no identifiable primary work. |
| Cremonesi `:wlzmIqt2EaEC` (1962) | “CERN Document Server” has no year, venue, or identifiable underlying record. |
| Cremonesi `:PDAgwb0ELvkC` (68) | The saved `A. Tumasyan et al` title, 2022 year, and tracker-upgrade venue conflict with the [official CMS TDR](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/TDR-17-001/index.html), which is a 2017 report. A second saved row has the tracker title. No unique exact-record mapping yet supports a cache repair or exclusion. |
| Cremonesi `:8NFFclwVBVUC`, `:f2QdLyYkFyUC`, `:Gm6cJkd1Yu0C` | The titles map to [CMS TRG-17-001](https://cms-results.web.cern.ch/cms-results/public-results/publications/TRG-17-001/index.html) and [CMS JME-18-001](https://cms-results.web.cern.ch/cms-results/public-results/publications/JME-18-001/index.html), but these rows carry journal locators while retained high-citation rows carry arXiv locators. Hold the version and retained-row choice. |
| Cremonesi `:UBSsHvkqiq4C`, `:OlGoE9RybnsC`, `:QjNCP7ux8QYC` | The [HIN-18-015](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-18-015/index.html) and [BPH-13-001](https://cms-results.web.cern.ch/cms-results/public-results/publications/BPH-13-001/index.html) records show related preprint and journal variants; hold until the retained row is chosen for each family. |
| Schumacher `:6bLC7aUMtPcC` (412) | “Project Narrative Experimental Medium Energy Physics Carnegie Mellon University” resembles a grant narrative, but the [DOE award list](https://science.osti.gov/-/media/_/pdf/universities/sc_statebystate_awards_FY2011.pdf) does not identify the exact captured document. |
| Schumacher `:_mQi-xiA4oYC`, `:pRWBApOjXDcC`, `:adHtZc2wMuEC`, `:oldoQiaHq2UC`, `:VnuxuLaQPLMC`, `:M0j1y4EgrScC` | Six 1977–88 history/economics rows have an `R Schumacher` author. A same-name Greek lexicon collaborator and the CMU physicist's education timeline suggest a namesake, but no direct author identity link to these six rows has been established. Hold all six. |

## Limits and gate

The source ledger reports 125 exact casefold/whitespace Cremonesi title/year groups (260 rows), while a broader punctuation-stripping screen reports 216 groups (446 rows). These are **review queues**, not 216 exclusions. No title/year-wide rule, blanket consortium exclusion, or profile-wide deduction was applied. The 100 decisions above were checked one row at a time; the 17 unresolved candidates remain in the source. Three other genuine Cremonesi CDF papers with literal `<?format ?>` markers retain their IDs and receive cache-only title repairs in PR #156. Across both current profile caches, all 13 titles containing `formula` (case-insensitive) placeholders and all six Cremonesi titles with `\documentclass` commands are exact duplicate exclusions in this draft. The marker scan does not certify every title's spelling or attribution.

The producer test fixes the 100 exact count and checks that all stated retained parent IDs remain unexcluded. The publisher preflight reads the source status next to the raw snapshot. If either Physics profile is present while `map_quality_hold` is true, `producer_review_ready` is false, or unresolved exact IDs remain, it rejects the **entire upload before embeddings or dataset writes**. A later release also requires the producer exclusion counts to match the source status. Raw captures and snapshot rows remain intact. The gate is staged only in this draft; it is not live on `main`. If a Physics snapshot is later authorized, merge this trusted publisher change before building or uploading that snapshot. Keep both drafts staged for independent review; do not merge, rebuild, or upload from this draft.
