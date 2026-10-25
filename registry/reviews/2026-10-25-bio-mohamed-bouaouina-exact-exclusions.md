# Mohamed Bouaouina visible Scholar profile: exact-row review — 2026-10-25 UTC

This offline review uses the saved 23-row Google Scholar capture for `JDo1onIAAAAJ`, its receipt, [CMU's identity record](https://scholars.cmu.edu/3528-mohamed-bouaouina), and independent publisher or institutional records. I did not reopen Scholar. The receipt reports 23 displayed rows, 23 unique citation IDs, no remaining “Show more,” and no challenge. The raw JSON SHA-256 is `565a2a69c45e7afc17b0d4f8aa6b76eb6fb2fe82348b52a4f207bd1298a726ab`, verified here. This is complete coverage of the **visible profile at capture time**, not proof that Scholar holds his entire academic bibliography.

## Exact decisions

Preserve **all 23 raw observations**. The manifest excludes three exact IDs from map authorships. A separate local disposition CSV records all 23 IDs. Seventeen rows are provisionally eligible and three remain explicitly uncertain pending a publisher copy. The count of eligible observations does not equal a count of distinct papers.

| Row · exact ID suffix | Code | Source-backed reason |
| --- | --- | --- |
| 18 `Zph67rFs4hoC` | `corrupted_duplicate` | Same title, year, journal, volume, and pages as complete row 11. The [*Journal of Cell Science* issue](https://journals.biologists.com/jcs/issue/125/23) lists one article, *Zasp regulates integrin activation*, with Bouaouina first and the full author order. Row 18 rearranges those names. |
| 22 `3fE2CSJIrl8C` | `nonpublication` | Its supposed title is an alphabetized author-index fragment. The page references point to separate JBC papers: [David J. Filman on page 6154](https://www.sciencedirect.com/science/article/pii/S0021925820571937), [Mingui Fu on page 6337](https://pubmed.ncbi.nlm.nih.gov/18178554/), and [Yukio Fujiki on page 6136](https://pubmed.ncbi.nlm.nih.gov/18174172/), all in volume 283, issue 10. These are unrelated authors of distinct papers, not a Bouaouina publication. |
| 23 `MXK_kJrjxJIC` | `nonpublication` | The displayed title is only a page-viii *Journal of Biological Chemistry* running header, and the author string is an alphabetical index beginning Abrams, Abramson, Agrawal. [JBC issue contents](https://www.sciencedirect.com/journal/journal-of-biological-chemistry/vol/295/issue/50) distinguish an “Author Index” on pages vii–viii from research articles; [a volume-290 article PDF](https://www.jbc.org/article/S0021925820402091/pdf) exhibits the same running-header form. The exact issue behind the captured fragment was not identified, but the fragment itself is an index page, not a paper title. |

## Retained and unresolved

- Rows 15 and 16 have the same authors, journal, year, volume, issue, and pages, but French and English titles. The English title may be a translation of the French article; I did not locate the original publisher record to prove whether they are one source item. Both are retained and explicitly marked uncertain, with **no automatic deduplication**.
- Row 17 is a 2014 *Angiogenesis* abstract, while row 8 is a 2014 *Development* article on related dynamin and angiogenesis work. Distinct venue and title support retaining both source versions.
- Row 19 is a 2006 *Matrix Biology* abstract related to the 2008 *Journal of Biological Chemistry* article in row 3. It remains a distinct source observation.
- Row 20 is a conference-workshop proceedings entry with many coauthors. Its exact individual authorship is not resolved by this pass; it stays eligible pending an original proceedings check.

No raw title, author string, citation count, or year was changed. Missing dates on rows 22–23 were not used as exclusion evidence; their contents and independent journal records were.

The raw source capture is separate. This draft review must stay unmerged until trusted upload and consumer-pin verification.
