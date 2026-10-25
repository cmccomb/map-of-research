# Subha R. Das Scholar attribution: next high-citation review

**Status:** four exact false attributions are integrated into the draft producer ledger, exclusion manifest, and reviewed-subset proposal. The 670-row Scholar capture, trusted held release, snapshot, and public map are unchanged. Both PRs remain draft under the release freeze.

The [CMU Chemistry biography](https://www.cmu.edu/chemistry/people/faculty/das.html) identifies the intended researcher as **Subha R. Das**. These four saved rows match publications by other named Das researchers. The match uses each Scholar row's title, year, coauthors, and, where present, journal citation. A visible `S Das` initial or subject area alone was not enough for an exclusion.

| Capture row | Citation ID suffix | Captured citations | Exact primary evidence | Attribution |
| ---: | --- | ---: | --- | --- |
| 114 | `PuOEWVtPfzwC` | 9 | [2016 enterovirus article, PubMed PMID 27651091](https://pubmed.ncbi.nlm.nih.gov/27651091/), DOI `10.1038/emi.2016.101` | The full author list names **Suman R Das**, affiliated with the J. Craig Venter Institute. Scholar truncates its visible byline before Das. |
| 142 | `ji7lAbPyDbYC` | 7 | [2018 infectious-diarrhea review, PubMed PMID 29928670](https://pubmed.ncbi.nlm.nih.gov/29928670/), DOI `10.1016/j.jcmgh.2018.02.009` | The full author list names **Soumita Das**, affiliated with UC San Diego Pathology. The [publisher's issue page](https://www.cmghjournal.org/issue/S2352-345X%2818%29X0004-6) independently displays her full name with this exact title. |
| 149 | `vnF2_uLGgtgC` | 7 | [2014 original IDOSI publisher PDF](https://idosi.org/gjp/8(3)14/7.pdf), DOI `10.5829/idosi.gjp.2014.8.3.83111` | The article byline and correspondence name **Srijit Das** of UKM Anatomy. |
| 211 | `QD3KBmkZPeQC` | 4 | [2019 original *Scientia Iranica* article](https://scientiairanica.sharif.edu/article_20339.html), DOI `10.24200/sci.2018.20339`, and [BIT Mesra's Dr. Sudip Das bibliography](https://bitmesra.ac.in/Show_profile/79325A4639596B676543575576366379524C6C4650673D3D) | The publisher's `S. Das`, P. Kumar, J. K. Prasad byline, BIT affiliation, volume 26(1), and pages 358–366 match the exact work claimed on **Sudip Das**'s institutional profile. Scholar's saved title drops the `fl` in “flow”; the other keys resolve that typo. |

All four IDs occur exactly once in the saved 670-row capture and were `unresolved_held` before this pass. The [machine-readable evidence CSV](2026-10-25-chem-das-next-high-citation-evidence.csv) preserves the full IDs and captured metadata. The 27 citations in this table are the counts displayed at capture time, not a completeness measure.

| Partition | Before this pass | Draft after this pass |
| --- | ---: | ---: |
| Primary-source positives | 49 | 49 |
| Exact exclusions | 199 | 203 |
| Held rows, including two specifically held edge cases | 422 | 418 |
| Captured Scholar rows | 670 | 670 |

Other high-citation candidates remain held without an exact identity match. For example, row 84 has an author-uploaded manuscript naming Sanjib Das, but no exact original publisher or institutional record was established in this pass; row 102 has only a matching aerospace topic and initials; row 95 has only `S. R. Das` at a different institution. None was changed by topic, initials, or a secondary citation alone.

The 49 positives still form an incomplete public profile, so the trusted release remains **held**. This review does not authorize a subset release, snapshot, upload, merge, or public map change.

Provenance: source capture `das-visible-670-2026-10-24.json` SHA-256 `f14571052ab088778779b919b2874b328745cdacc706be0680732b6458e9b997`; frozen cache SHA-256 `85700e7ee25079ed30633df91137e84cc0ef6c0e6628166f80acf4c133f4c1df`.
