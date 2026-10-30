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

## Eight confirmed one-sided coauthorships

The published artifact contains these eligible works, but assigns each to
only one of two named, currently rostered MCS faculty. The missing people
have no verified Scholar ID in the registry.

| Work and primary evidence | Present faculty ID | Missing faculty ID |
| --- | --- | --- |
| [1999 Golgi transport paper](https://pubmed.ncbi.nlm.nih.gov/10233155/), DOI `10.1091/mbc.10.5.1445`; the [CMU Lee Lab bibliography](https://labs.bio.cmu.edu/lee/publications/) names Tina Lee and Adam Linstedt | Tina Lee | Adam Linstedt |
| [2024 galaxy-clustering methods paper](https://academic.oup.com/mnras/article/531/4/4954/7691261), DOI `10.1093/mnras/stae1444`; the publisher names Rachel Mandelbaum and Scott Dodelson with CMU Physics affiliations | Scott Dodelson | Rachel Mandelbaum |
| [2014 sea-urchin gene-network paper](https://pubmed.ncbi.nlm.nih.gov/24496631/), DOI `10.1242/dev.105585` | C. Joel McManus | Charles Ettensohn |
| [2014 rainbow Hamilton-cycles paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/rsa.20475), DOI `10.1002/rsa.20475`; also on [Loh's CMU site](https://www.math.cmu.edu/~ploh/docs/research/20120722-rainbow-ham.pdf) | Alan Frieze | Po-Shen Loh |
| [2016 Kompaneets-equation paper](https://epubs.siam.org/doi/10.1137/15M1054730), DOI `10.1137/15M1054730` | Robert Pego | Gautam Iyer |
| [2022 red-light ATRP paper](https://par.nsf.gov/servlets/purl/10340015), DOI `10.1021/acsmacrolett.2c00080` | Krzysztof Matyjaszewski | Linda A. Peteanu |
| [2007 strand-invasion paper](https://pubs.acs.org/doi/abs/10.1021/ja074886j), DOI `10.1021/ja074886j`; corroborated by the [CMU Armitage bibliography](https://www.chem.cmu.edu/groups/army/pub/) | Bruce Armitage | Danith Ly |
| [2020 MassiveBlackII paper](https://academic.oup.com/mnras/article/491/3/4116/5634270), DOI `10.1093/mnras/stz3240` | Tiziana Di Matteo | Rachel Mandelbaum |

These are source-attribution gaps, not layout or graph-builder omissions.
All eight faculty pairs are absent from the published collaboration edge list,
so each is a concrete contributor to the apparent fragmentation. Their eventual
release could add up to eight distinct edges, subject to source and publication
review; the table is not an approval to publish those person observations.
The owning surface is `map-of-research`'s curated source and publisher. A
single verified paper does not establish either missing person's whole-career
publication history: the existing
[`person-publication-holds.json`](../../data/curated/person-publication-holds.json)
policy would keep a newly added one-paper observation out of the public graph.
Defer these eight attributions until a complete Scholar capture or reviewed whole-career
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

At this checkpoint, recorded MCS Scholar-ID coverage remained 116 of 210 people;
94 IDs were blank. Among active research faculty, 27 lacked a verified ID: five
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

## Follow-on Physics profile checks

The [CMU Engineering biography for Stephanie Tristram-Nagle](https://engineering.cmu.edu/mcf/directory/bios/tristram-nagle-stephanie.html)
directly links the exact Scholar ID [`PFPNTa4AAAAJ`](https://scholar.google.co.il/citations?hl=en&user=PFPNTa4AAAAJ).
Her [CMU Physics page](https://www.cmu.edu/physics/people/faculty/stn.html)
supports the typed Physics home. This review records that ID and home in the
registry. It captures no publication rows and changes no graph edge. Recorded
MCS Scholar-ID coverage is now 117 of 210, with 93 blank IDs; the active-faculty
count above is unchanged because this is an emerita appointment.

Three other Scholar profiles have matching names, CMU-verified email, fields,
and official Physics biographies, but the reviewed CMU and self-managed pages
do not link their exact IDs. Their candidate IDs remain unrecorded under the
same owner-link rule applied above. Their Physics homes are recorded from the
official pages, independently of Scholar verification.

| Person and official home | Candidate Scholar ID (held) |
| --- | --- |
| [Mathias Lösche](https://www.cmu.edu/physics/people/faculty/loesche.html) | [`Wpq8ChcAAAAJ`](https://scholar.google.com/citations?user=Wpq8ChcAAAAJ) |
| [John Nagle](https://www.cmu.edu/physics/people/faculty/nagle.html) | [`a9Boo7wAAAAJ`](https://scholar.google.com/citations?user=a9Boo7wAAAAJ) |
| [Robert M. Suter](https://www.cmu.edu/physics/people/faculty/suter.html) | [`jF3_kJEAAAAJ`](https://scholar.google.com/citations?user=jF3_kJEAAAAJ) |

Suter already has one public curated DOI observation with David Kinderlehrer,
[`10.1103/PhysRevMaterials.6.033402`](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.6.033402),
despite lacking a complete publication-history source and a person-level hold.
His self-managed site labels its lists
[`Recent Publications (2014–2020)`](https://www.andrew.cmu.edu/user/suter/RecentPubs.html)
and [selected publications](https://www.andrew.cmu.edu/user/suter/Archive_RMS.html);
neither claims whole-career coverage. Review the pre-existing Suter visibility
decision before a future source publication. This audit does not alter that
already published work or its edge.

## Whole-bibliography leads and additional missing Physics ties

The [CMU Physics page for John Nagle](https://www.cmu.edu/physics/people/faculty/nagle.html)
links a [“List of All Publications”](https://www.cmu.edu/physics/people/faculty/publist2025nagle.docx).
Read-only inspection found 218 listed entries spanning 1966–2026. It is a
credible whole-career source candidate, but its internal 2023 header is stale,
only 30 entries state a DOI, and two 2026 entries are submitted manuscripts.
Identity, work matching, duplicates, and publication eligibility need review
before source ingestion. Nagle currently has zero mapped works and zero edges.

Two papers on that list are already mapped, but only for Markus Deserno:

| Work and primary evidence | Published work ID | Missing faculty observations |
| --- | --- | --- |
| [2019 Poisson-ratio paper](https://www.cmu.edu/physics/people/faculty/nagle.html), DOI [`10.1039/C9SM01290G`](https://doi.org/10.1039/C9SM01290G) | `work-92ea0dceaa429d283e461ac3794b2f615acb3d4ec6fe850e0d6003a1204b1a0f` | John Nagle |
| [2026 bending-moduli paper](https://www.cmu.edu/physics/people/faculty/nagle.html), DOI [`10.1016/j.bpj.2025.12.037`](https://doi.org/10.1016/j.bpj.2025.12.037) | `work-944bdbee2fe13439787a309e6c6f0d3dbaa7c42ba4e06dca215b31af1b02943a` | John Nagle and Stephanie Tristram-Nagle |

The three faculty pairs among Nagle, Tristram-Nagle, and Deserno all lack a
published edge. Other exact papers in Nagle's CMU list connect him to
[Frank Heinrich (2025)](https://doi.org/10.1039/D4SM01312C),
[Heinrich and Tristram-Nagle (2023)](https://pubmed.ncbi.nlm.nih.gov/36566351/),
[Heinrich, Tristram-Nagle, and Lösche (2016)](https://doi.org/10.1016/j.bbamem.2016.09.010),
and [Suter (1996)](https://pubmed.ncbi.nlm.nih.gov/8785298/).
Those example works are absent from the current publication artifact. Name
matches across the 218 entries suggest more joint work, but the exact records
and person-level publication coverage must be reconciled before any edge is
released. Author names alone do not create faculty observations in the
publisher.

[Rachel Mandelbaum's CMU-hosted publication page](https://www.andrew.cmu.edu/user/rmandelb/publist.html)
labels its ADS link “All publications” and says its links are full lists. This
is a promising whole-career source for her two missing edges above, but the
legacy ADS search has a 200-result parameter and did not resolve in this
review. Check identity, coverage, and any result cap before using it.
[Linda Peteanu's group list](https://groups.chem.cmu.edu/peteanu/publications/)
contains the red-light ATRP paper but omits older papers listed by her
[archived CMU group page](https://www.chem.cmu.edu/groups/peteanu/peteanu_group/rep_pubs.html).
[Danith Ly's group list](https://groups.chem.cmu.edu/ly/publications/)
contains the strand-invasion paper but omits at least a
[2021 paper](https://pubmed.ncbi.nlm.nih.gov/34214178/). Neither group list
establishes whole-career completeness. Existing unmerged author-site reviews
for Gautam Iyer and Po-Shen Loh should be reconciled before restarting those
source paths.
