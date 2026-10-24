# Physics Scholar observation review, 2026-10-24

This review checks 19 exact Google Scholar observations against the saved complete-visible-profile captures and linked source records. The 15 captures cover 3,319 visible rows and are staged separately in draft [PR #80](https://github.com/cmccomb/map-of-research/pull/80). The 19 decisions concern seven of those profiles; the later Meyer and Shi captures have preliminary flags only. Each row below gives a profile and `author_pub_id` suffix; the manifest records the full exact ID. No new Scholar profile was opened for this review.

The decisions are recorded in [`data/curated/scholar-exclusions.json`](../../data/curated/scholar-exclusions.json). They affect research-map eligibility only. The raw observations, including their original titles, authors, dates, and citation counts, remain intact. The 48 preliminary candidates in PR #80's status ledger are not excluded.

| Profile and raw row | `author_pub_id` suffix | Exact decision and source |
| --- | --- | --- |
| Kozii 37 | `HDshCWvjkbEC` | [APS Publisher's Note](https://link.aps.org/doi/10.1103/PhysRevB.97.099901) about the original research article retained in row 19. It is a citable correction notice, but not a separate research work for this map. |
| Kozii 38 | `mB3voiENLucC` | [APS Publisher's Note](https://link.aps.org/doi/10.1103/PhysRevB.95.179902) about the original research article retained in row 5; the same research-work distinction applies. |
| Tarnopolsky 49 | `Wp0gIr-vW9MC` | The [World Scientific memorial volume](https://doi.org/10.1142/10811) is credited to its editors. Tarnopolsky authored a [chapter](https://dare.uva.nl/id/cc56c740-02d1-4dca-aded-2ea8e1454e63), not the whole edited book. |
| Tarnopolsky 62 | `M05iB0D1s5AC` | An author-line fragment was imported as the title of the [Vacua on the Brink of Decay chapter](https://dare.uva.nl/id/cc56c740-02d1-4dca-aded-2ea8e1454e63); the correctly titled chapter remains in row 38. |
| Tarnopolsky 63 | `M3ejUd6NZC8C` | The captured title names the [Oxford journal issue](https://academic.oup.com/ptep/issue/2016/12), not a separate authored work. |
| Breivik 25 | `K3LRdlH-MEoC` | [APS](https://doi.org/10.1103/PhysRevD.106.063512) credits *Slow-roll inflation at N3LO* only to Auclair and Ringeval. |
| Breivik 96 | `uWQEDVKXjbEC` | The [published battery article](https://sites.utexas.edu/maple/files/2025/10/Telmasre_2025_ECS_Adv._4_030503.pdf) credits Telmasre and coauthors; Breivik's name occurs in an unrelated recommended-article item. |
| Dodelson 27 | `3NQIlFlcGxIC` | The captured 1971 *Modern Cosmology* is [D. W. Sciama's book](https://astro.ing.iac.es/library/showentry.php?entryflag=book&refnr=2513), not Scott Dodelson's later book of the same name. |
| Dodelson 442 | `XoXfffV-tXoC` | The captured string is the opening names and page numbers of a [SLAC proceedings author index](https://www.slac.stanford.edu/econf/C990809/docs/AuthorIndex.pdf), not a Dodelson work. |
| Trac 167 | `hImZayhCRfQC` | The [published Ara OB1a article](https://research.unipd.it/handle/11577/2525616) credits Baume, Carraro, Comerón, and de Elía; the captured ACT author text is conflated with this paper. |
| Trac 182 | `fhMX52Pd6iQC` | The captured person-name title is the cover of [Zheng Zheng's presentation](https://member.ipmu.jp/igm/proc/zheng_zheng.pdf), which lists Hy Trac among collaborators. |
| Croft 241 | `v1_lew4L6wgC` | [NSF award 1109730](https://api.nsf.gov/services/v1/awards/1109730.json), not a publication. |
| Croft 243 | `MAUkC_7iAq8C` | [NSF award 1009781](https://api.nsf.gov/services/v1/awards/1009781.json), not a publication. |
| Croft 249 | `-7ulzOJl1JYC` | [NSF award 0749212](https://api.nsf.gov/services/v1/awards/0749212.json), not a publication. |
| Croft 254 | `kVjdVfd2voEC` | [NSF award 0205978](https://api.nsf.gov/services/v1/awards/0205978.json), not a publication. |
| Croft 263 | `LjlpjdlvIbIC` | A [CMU-hosted Croft curriculum vitae](https://www.cmu.edu/physics/people/faculty/documents/croft_cv.pdf), not a research work; the captured year 1909 is erroneous. |
| Palmese 531 | `4e5Qn2KL_jwC` | [NSF award 2308193](https://api.nsf.gov/services/v1/awards/2308193.json), not a publication. |
| Palmese 338 | `nRpfm8aw39MC` | The full [*Chinese Physics C* article](https://inspirehep.net/files/5d327409805a0d882d626fc7354d6435), DOI `10.1088/1674-1137/ac9895`, has the BESIII author list and does not name Palmese. |
| Palmese 544 | `hHIA4WEVY-EC` | The same [BESIII article](https://inspirehep.net/files/5d327409805a0d882d626fc7354d6435) is repeated with a conflated author list naming Palmese; its complete published author list does not. |

These are exact observation decisions. They do not infer that other publications by these researchers are invalid. The source capture PR remains separate from this producer-manifest change, and a Physics-inclusive snapshot must reconcile the registry and rebuild the Parquet and manifest before upload.
