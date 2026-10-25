# Two COSINE rows in the held Hyung J. Kim Scholar capture

The saved `bC4aq-4AAAAJ` Google Scholar capture contains 918 observations
(SHA-256 `ea04610e120039df7aaf83c402d193021f328b319cd41cfabf541f3942069867`).
Neither the raw capture nor the whole-profile release hold changes here. This
review excludes only two exact observations whose original publisher records
name a different researcher.

| Capture row and Scholar observation | Original source and decision |
| --- | --- |
| 325 `bC4aq-4AAAAJ:mgoTDWlsYNUC` | [Nature correction, DOI `10.1038/s41586-019-0890-3`](https://www.nature.com/articles/s41586-019-0890-3) gives the exact title, 2019 date, and COSINE-100 collaborator list naming **Hongjoo Kim** at Kyungpook National University. [PubMed's complete collaborator list](https://pubmed.ncbi.nlm.nih.gov/30664749/) independently spells the name. The publisher classifies this as an Author Correction to the 2018 paper, Nature 566:E2; the Scholar venue's `566, 1` page is inaccurate. Exclude this observation from CMU Hyung J. Kim. |
| 366 `bC4aq-4AAAAJ:8JTMrWI6FdcC` | [Communications Physics original article, DOI `10.1038/s42005-025-02067-4`](https://www.nature.com/articles/s42005-025-02067-4) exactly matches the 2025 title and volume 8 article 135. Its full byline names **Hongjoo Kim**, and its author affiliations place him in Kyungpook National University's Department of Physics. Exclude this observation from CMU Hyung J. Kim. |

The rebuild checks both Scholar IDs, captured titles, years, author strings,
and venue strings before writing the two `false_profile_attribution` decisions.
The full capture remains partitioned into **122 source-confirmed CMU works,
476 exact exclusions, and 320 held observations**. Initials-only COSINE rows
172 and 331 remain held; the two full-name records above do not automatically
resolve other collaboration papers. No trusted Kim cache, snapshot, or public
map changes in this draft.
