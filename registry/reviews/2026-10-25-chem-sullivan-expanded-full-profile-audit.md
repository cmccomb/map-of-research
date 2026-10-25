# Ryan Sullivan: expanded 250-row Scholar review (2026-10-25)

The saved Google Scholar page has 250 visible rows, with the final Show more control disabled. The sanitized JSON and HTML match row by row. The source-bound candidate cache at `data/captures/sullivan-visible-250-2026-10-25.candidate-cache.json` preserves every row and passes `validate_complete_receipt`; its SHA-256 is `960b296af5cf0c9e3aab6be32844ec2fe2ad7f2a2b3e331cfc4c857f4df57fd7`. It is evidence for review, **not** the canonical `data/authors/wXSgFr4AAAAJ.json` cache. That canonical file remains the frozen 196-row version.

The prior audit of the 58 new-only Scholar IDs proposed 35 new exact exclusions and identified 15 unresolved rows. Screening all 250 titles, bylines, venues, and years exposed 25 additional uncertain rows that share IDs with the frozen cache. The [machine-readable inventory](2026-10-25-chem-sullivan-full-profile-hold-inventory.json) pins the saved JSON/HTML and candidate-cache hashes, each current row ID and index, and the reason for holding it. It contains **40 current rows**: 15 new-only and 25 shared legacy. The 51 exact exclusions in producer draft #217 do not overlap them. This is a risk screen, not independent item-level confirmation of every remaining row.

Several shared rows have direct evidence of false attribution or a nonpublication container:

- Scholar row 99, *Can Subdaily LST Be Constructed in High-Latitude Regions Using Polar Orbiting Satellites?*, has [publisher-deposited metadata](https://api.crossref.org/works/10.1109/LGRS.2025.3546797) assigning Ryan C. Sullivan to Argonne National Laboratory, ORCID `0000-0002-3500-1842`, rather than CMU Sullivan, ORCID `0000-0003-0701-7158`.
- Scholar row 181, *Weekly Cycle in NEXRAD Rain Rates*, is listed in the [official 98th AMS program](https://ams.confex.com/ams/98Annual/webprogram/meeting.html) under Ryan Sullivan of NSF in Norwood, Minnesota.
- Scholar row 229, *First in situ investigation of a dark wind streak on Mars*, assigns R. Sullivan to [Cornell Astronomy](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008JE003102). The other Mars rows 230, 231, and 234 need exact item review; [another paper by the same planetary scientist](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008JE003101) provides the identity lead.
- Scholar row 194, *Atmospheric Ice Formation Processes and Impacts I*, is an [AGU session](https://studio.m-anage.com/agu/fm15/webprogram/Session10991.html) convened by CMU Sullivan, rather than one of the session's authored abstracts. The event page separately lists his A11Q-06 abstract.
- Rows 243 and 247 are PCCP themed-issue headings, 245 is labeled supplemental material, 246 is a clipped title, and 250 is an author-list fragment. Their exact parent items and duplicate status need review before they appear as works.

The four IDs in the 196-row cache but absent from the 250-row capture also have [source-backed draft dispositions](2026-10-25-chem-sullivan-prior-id-audit.json). One [AGU 2017 A13J-05 abstract](https://agu.confex.com/agu/fm17/meetingapp.cgi/Paper/297700) has a native curated recovery; two old ORNL dataset variants point to the current excluded row 101; and an old EGU25 abstract points to current row 134. The candidate-cache hash is now recorded, but the gate does not yet support the false-attribution rekey classification. These draft dispositions are not a gate exception.

Some initially unusual subject matter is genuinely attributable to CMU Sullivan: the [PLOS One aerosolized-infection paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0273194) gives `rsullivan@cmu.edu`, the [environmental conflict-of-interest paper](https://pubmed.ncbi.nlm.nih.gov/37943968/) gives CMU Chemistry and Mechanical Engineering, and [NASA's AEROICESTUDY report](https://ntrs.nasa.gov/citations/20205009903) names R. Sullivan at CMU. These rows are not held merely because their titles differ from Sullivan's usual aerosol-chemistry topics.

## Release dependencies

1. Resolve the 40 current-row holds or pin them in a source-bound provisional-hold manifest. Keep all 250 observations in raw source data.
2. Extend the historical-ID retention rule narrowly for two old false-attribution IDs mapping to an exact excluded current ID; keep the AGU native recovery and EGU rekey separately verified.
3. Review the 51 draft exclusions against the current producer mainline and release only a single exact, nonoverlapping decision set. The current source branch does not promote this cache or rebuild a snapshot.
