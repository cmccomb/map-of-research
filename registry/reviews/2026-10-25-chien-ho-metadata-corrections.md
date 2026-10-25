# Chien Ho Scholar metadata corrections — 2026-10-25

The raw 582-row Scholar capture for `8iLLOQgAAAAJ` is immutable. Seven exact
profile-publication IDs have enough source evidence to repair effective
`title`, `year`, or `venue` fields. The corrections live in
`data/curated/scholar-metadata-corrections.json`; each entry guards the original
three fields and records a source URL, specific basis, and review date. The
publisher applies them after validating the raw snapshot and before embedding,
quality assessment, and work identity generation. `source_record_json` and the
snapshot remain unchanged. Published `profile_publications` records expose the
effective fields alongside `metadata_correction_*` provenance; the upload
receipt records the number applied and the manifest checksum.

| Capture row | Scholar citation suffix | Effective repair | Evidence |
| --- | --- | --- | --- |
| 91 | `xyvS_IvSCKsC` | Replace the encyclopedia container title with the 2000 chapter title; retain the observed venue pending edition reconciliation. | [CMU bibliography 223](https://www.cmu.edu/bio/people/faculty/images/ho_publications.pdf) |
| 326 | `WJVC3Jt7v1AC` | Date the Chien Ho and Yue Yuan revision to 2010 and use its Wiley encyclopedia venue. | [Wiley version 2.0](https://onlinelibrary.wiley.com/doi/abs/10.1002/9780470015902.a0001345.pub2), [CMU bibliography 305](https://www.cmu.edu/bio/people/faculty/images/ho_publications.pdf) |
| 348 | `OTTXONDVkokC` | Repair the OCR title to “deoxy form” and simplify the supplement page citation. | [PubMed PMID 16703421](https://pubmed.ncbi.nlm.nih.gov/16703421/) |
| 360 | `lvd772isFD0C` | Use the 2003 Springer chapter year and series citation. | [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-642-56036-1_6) |
| 393 | `gKiMpY-AVTkC` | Correct the 2004 ISMRM meeting number from 11 to 12. | [CMU coauthor CV](https://users.ece.cmu.edu/~moura/vitae/moura-resume.pdf), [ISMRM meeting notice](https://cds.ismrm.org/protected/MRPulse/MRPulsev10_3.pdf) |
| 445 | `nRpfm8aw39MC` | Repair the damaged “Inflammatory” title, use the 2012 book year and pages 335–351. | [CMU bibliography 318](https://www.cmu.edu/bio/people/faculty/images/ho_publications.pdf), [original book preview](https://api.pageplace.de/preview/DT0400.9781439807521_A38252935/preview-9781439807521_A38252935.pdf) |
| 543 | `TesyEGJKHF4C` | Replace `1712`, the abstract number, with publication year 2011. | [ISMRM abstract 1712](https://archive.ismrm.org/2011/1712.html) |

Two confirmed-author rows **remain held** and have no correction entry:

| Capture row | Scholar citation suffix | Missing evidence |
| --- | --- | --- |
| 486 | `CdxZDUztZiMC` | The [coauthor CV](https://ifpoolscouldtalk.com/Exhibits/KochanckCV.pdf) cites Journal of Neurotrauma 24:1230, while Scholar cites 24:1231. An original journal abstract or issue page is needed to settle pagination. |
| 490 | `sA9dB-pw3HoC` | [CMU bibliography 193](https://www.cmu.edu/bio/people/faculty/images/ho_publications.pdf) dates the original encyclopedia chapter to 1995; Scholar describes a 2007 eMagRes edition. A publisher record for this exact edition is needed to decide whether to correct its year and venue or retain it as a separate edition. |

This is a metadata repair only. Attribution review and the decision to release
Chien Ho's profile remain separate; these seven entries do not themselves
make the 582-row profile publication-ready.
