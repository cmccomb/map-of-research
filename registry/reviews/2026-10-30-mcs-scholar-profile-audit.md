# MCS Scholar-profile and collaboration-coverage audit — 2026-10-30

The [published collaboration graph](https://cmccomb.com/map-of-collaboration/)
serves the immutable [research dataset revision `5073c244`](https://huggingface.co/datasets/ccm/cmu-engineering-publications/tree/5073c24450e2f239ca49f674cbc81287b915045b).
Its MCS view contains 210 distinct included faculty, teaching faculty, and
emeriti. The four departmental memberships sum to 211 because one person is
cross-appointed. Before this review, 116 of the 210 people had a verified
Scholar ID; 115 of those had mapped publications. Two people without Scholar
IDs had mapped works from separately verified sources. Thus 117 MCS people
had mapped publications and 93 did not. Among the 117, 83 had at least one
observed collaborator and 34 had no edge anywhere in the CMU graph. Four
additional people had an edge only to a faculty member outside MCS.

The MCS-induced graph had 114 internal edges and 140 components, including
131 single-person components. Every mapped work belonging to the 34 globally
isolated MCS people had only one cataloged faculty ID in the published
artifact. This audit found no dropped graph-builder edge between two IDs on
the same work. It also found no duplicate DOI across MCS profiles; seven
undated exact-title groups involved pairs that already have edges. A
high-similarity title check found no candidate near-duplicate linking an
isolated MCS person to another MCS profile; that heuristic is not an exhaustive
bibliographic reconciliation. These observations describe **indexed**
coauthorship, not the completeness of CMU collaboration.

All 115 MCS Scholar caches that contribute mapped works are marked
`complete_visible_profile` in the published source snapshot. None is a capped
or legacy partial cache. Matteo Cremonesi, Colin Morningstar, Diana Parno,
and Tina Kahniashvili have complete visible captures despite being globally
isolated in this graph. The sole registered MCS Scholar ID without a canonical
cache belongs to [Hyung J. Kim](https://www.cmu.edu/chemistry/people/faculty/kim.html).
His 918-row visible profile was captured but deliberately withheld pending
attribution review because it mixes namesake works; the [Kim review
branch](https://github.com/cmccomb/map-of-research/tree/codex/chem-kim-chapters-namesakes-20261025)
is nonoperational and does not authorize publication. Re-fetching that account
would not resolve its identity holds.

A saved-source identity sanity check found matching Scholar IDs, direct
profile URLs, captured user IDs, and row counts for all 115 canonical MCS
caches; no capture JSON or HTML was missing. No new identity mismatch appeared.
The non-CMU or absent affiliation text on the saved Scholar headers for Clive
Newstead, Noel Walkington, Sergey Nadtochiy, and Tina Kahniashvili is already
documented with separate identity evidence in the registry and earlier
reviews. The `Hiller NL` profile name is likewise a previously reviewed
variant of Luisa Hiller. This check used the saved source and made no new
Scholar requests.

## Two confirmed one-sided coauthorships

The published artifact contains these eligible works, but assigns each to
only one of two named, currently rostered MCS faculty. The missing people
have no verified Scholar ID in the registry.

| Work and primary evidence | Present faculty ID | Missing faculty ID |
| --- | --- | --- |
| [1999 Golgi transport paper](https://pubmed.ncbi.nlm.nih.gov/10233155/), DOI `10.1091/mbc.10.5.1445`; the [CMU Lee Lab bibliography](https://labs.bio.cmu.edu/lee/publications/) names Tina Lee and Adam Linstedt | Tina Lee | Adam Linstedt |
| [2024 galaxy-clustering methods paper](https://academic.oup.com/mnras/article/531/4/4954/7691261), DOI `10.1093/mnras/stae1444`; the publisher names Rachel Mandelbaum and Scott Dodelson with CMU Physics affiliations | Scott Dodelson | Rachel Mandelbaum |

These are source-attribution gaps, not layout or graph-builder omissions.
The owning surface is `map-of-research`'s curated source and publisher. A
single verified paper does not establish either missing person's whole-career
publication history: the existing
[`person-publication-holds.json`](../../data/curated/person-publication-holds.json)
policy would keep a newly added one-paper observation out of the public graph.
Defer those two edges until a complete Scholar capture or reviewed whole-career
bibliography is available and the exact works pass the source and publication
gates. Do not insert edges directly into the visualization.

## Three candidate profile identities held

The following candidate Scholar profiles were checked for identity only. The
named profile headers were compared with the official CMU faculty pages.
None of their publication rows was captured or added to the map in this
review. A matching profile header, CMU email, or link from a Scholar profile
to a CMU page does not meet the registry's direct CMU-owner link requirement
for an exact Scholar ID. Read-only checks of the three official faculty pages,
their linked CMU-hosted CVs, and the Alison and Heinrich CMU-hosted lab sites
found no direct links to these exact IDs. The registry therefore leaves the
three IDs and their ID-source fields blank. Their explicit Physics homes remain
supported by the named CMU faculty pages.

| Person | Candidate ID (held) | Identity evidence and official corroboration |
| --- | --- | --- |
| John Alison | [`U0xIDHkAAAAJ`](https://scholar.google.com/citations?user=U0xIDHkAAAAJ) | The profile identifies a CMU Physics associate professor, shows a verified `cmu.edu` email, and lists experimental particle physics; [CMU Physics](https://www.cmu.edu/physics/people/faculty/alison.html) matches the role and field. |
| Frank Heinrich | [`JpCZd9kAAAAJ`](https://scholar.google.com/citations?user=JpCZd9kAAAAJ) | The profile identifies Frank Heinrich at CMU and NIST with a verified `cmu.edu` email and biophysics research; [CMU Physics](https://www.cmu.edu/physics/people/faculty/heinrich.html) matches. |
| Valentina Dutta | [`wI8rr3AAAAAJ`](https://scholar.google.com/citations?user=wI8rr3AAAAAJ) | The profile identifies a CMU Physics assistant professor and links her [CMU Physics page](https://www.cmu.edu/physics/people/faculty/dutta.html); its name, role, and field match. |

Recorded MCS Scholar-ID coverage remains 116 of 210 people; the 94 blank IDs
remain unresolved. Among active research faculty, 27 lack a verified ID: five
in Biological Sciences, three in Chemistry, ten in Mathematical Sciences, and
nine in Physics. Current
individual CMU faculty pages, 18 linked lab or personal sites, and seven
CMU-hosted CVs supplied no additional owner-linked Scholar IDs in this pass.
Focused Scholar profile searches did not establish any other exact identity
among the remaining active Bio, Chemistry, or Physics faculty. The earlier
[Mathematical Sciences account-coverage audit](2026-10-25-math-scholar-account-coverage.md) covers
the unresolved Math people. Search absence does not establish that a person
lacks a Scholar profile.

[Dutta's CMU-hosted CV](https://www.cmu.edu/physics/people/faculty/documents/dutta_cv.pdf)
says her name appears on every CMS and LDMX paper. Alison's official page
also lists many CMS and earlier ATLAS collaboration papers. Before a full
profile capture or public release, these large-consortium records need an
explicit review of attribution and what a coauthorship edge should convey.
Holding these candidate IDs changes no mapped works or graph edges.
