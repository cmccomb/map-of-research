# Chemistry Scholar release review: Murshid, Faulkner, and Silva

This producer draft pairs with [source PR #250](https://github.com/cmccomb/map-of-research/pull/250). That source draft records complete visible Google Scholar captures of 49, 5, and 41 rows respectively and retains every raw observation. The trusted uploader reads `registry/people.csv` from producer main, so this draft repeats the three source registry rows byte for byte: each has a verified Scholar ID and an official CMU source for primary Chemistry department. The typed department home is derived from `primary_department_id=chem`.

| Person | Scholar ID | Cache SHA-256 | Exact exclusions | Provisional holds |
| --- | --- | --- | ---: | ---: |
| [Nimer Murshid](https://scholars.cmu.edu/13260-nimer-murshid) | `MPxFaHYAAAAJ` | `37f993f07e75d5dc9aaeb60cc50429bcd7cfdbad255e7939ec4c9ba80f05d0de` | 3 | 2 |
| [Simon Faulkner](https://scholars.cmu.edu/9471-simon-faulkner) | `tPp2aaoAAAAJ` | `9d64f37c39179c78af9635e1cf5afd53a26ca00bfeee093a8866ff8d7d3a09f8` | 1 | 0 |
| [Gloria Silva](https://www.cmu.edu/chemistry/people/faculty/silva.html) | `vF51LxIAAAAJ` | `01908b11b53c9d9659b96720dec6f69816cabdd68ac1c9e9b590d8d42968b522` | 0 | 2 |

The release decisions pin all 95 original rows and exact cache hashes. The hold manifest also pins the reviewed exclusion set for Nimer (three full citation IDs) and an empty exclusion set for Gloria. The publisher checks source row indices, citation IDs, and original Scholar citation URLs before suppressing held contributions to mapped works and coauthor edges. Simon has no provisional holds, so his release decision has no hold-set fields.

## Exact duplicate exclusions

| Scholar row | Decision and primary evidence |
| --- | --- |
| Murshid 31 `MPxFaHYAAAAJ:hFOr9nPyWt4C` | Malformed second copy of retained row 3, the [Nanomaterials article](https://doi.org/10.3390/nano11113084). |
| Murshid 35 `MPxFaHYAAAAJ:-f6ydRqryjwC` | Malformed second copy of retained row 24, the [Sustainability article](https://doi.org/10.3390/su15054342). |
| Murshid 40 `MPxFaHYAAAAJ:hC7cP41nSMkC` | Malformed second copy of retained row 34, the [Gels article](https://doi.org/10.3390/gels9060465). |
| Faulkner 5 `tPp2aaoAAAAJ:2osOgNQ5qMEC` | Reduced-byline copy of retained row 1. The [publisher DOI record](https://doi.org/10.1016/j.ica.2005.08.034) identifies one 2006 article by Cardell, Hogarth, and Faulkner. |

All four use the producer's `corrupted_duplicate` code. The exact exclusions preserve the original cache and raw profile rows while preventing a second mapped work or authorship.

## Provisional holds

| Scholar row | Unresolved point |
| --- | --- |
| Murshid 48 `MPxFaHYAAAAJ:WF5omc3nYNoC` | Fragmentary journal-name title without year or venue; the underlying item is unknown. |
| Murshid 49 `MPxFaHYAAAAJ:zYLM7Y9cAGgC` | Footnote-suffixed title without year or venue; item and output type are unresolved. |
| Silva 33 `vF51LxIAAAAJ:dhFuZR0502QC` | 2020 strawberry-delivery preprint appears related to her 2022 PNAS article, but the exact version relation needs confirmation. |
| Silva 41 `vF51LxIAAAAJ:M3ejUd6NZC8C` | Undated Spanish archive entry has a malformed institutional venue; type, date, and authorship context need review. |

These rows are held, not labeled false attributions. Each hold's exact original Scholar citation URL is stored in `scholar-provisional-holds.json`. Silva's 2022 strawberry article remains retained: the [primary PubMed record](https://pubmed.ncbi.nlm.nih.gov/35943988/) names Gloria L. Silva with CMU affiliation. Her three identifiable ACS meeting abstracts also remain available to the ordinary output-type policy.

## Release dependencies

Producer primary-home gate #242, Collins exclusions #247, and provisional-hold gate #243 have merged. This review branch can merge on producer main before the source drafts because its SHA-bound release checks prevent publication until the matching complete caches are present. The three producer registry rows are byte-identical to source PR #250. A combined no-upload preview with source PRs #118 and #250 preserved all 95 raw rows, applied four exact exclusions and four holds, and confirmed held IDs contribute no mapped work or authorship. Next merge the producer review, then the two source-only drafts; merge the reviewed snapshot only after confirming all hashes, run the trusted upload, and verify both live maps.
