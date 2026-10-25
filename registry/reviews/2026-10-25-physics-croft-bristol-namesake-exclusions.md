# Rupert Croft: seven exact Bristol namesake exclusions

The saved Rupert Croft source capture in draft PR #80 contains 267 rows. Its receipt hashes match the unchanged raw JSON and HTML. Each of the seven IDs below occurs once in both the ordered raw capture and converted author cache, at the same row number and title. Their saved citation counts total **933**. This producer review changes neither that source nor a snapshot.

[CMU identifies Rupert A. C. Croft](https://www.cmu.edu/physics/people/faculty/croft.html) as a cosmology professor. In a contemporary [CMS/TOTEM proceedings paper](https://pos.sissa.it/035/022/pdf), **Richard Croft** names the University of Bristol and gives his Bristol address while describing his CMS/TOTEM trigger research. The primary records below place their `R. Croft` with that Bristol team. These are exact observation decisions, not a rejection of Rupert Croft's other work or every CERN/CMS row.

| Capture row | Exact `author_pub_id` suffix | Primary work evidence | Decision |
| ---: | --- | --- | --- |
| 13 | `6bLC7aUMtPcC` | [CMS Technical Design Report Volume I](https://inspirehep.net/files/5baab21b09e5774ce5a706eaa4aad00a), author roster under University of Bristol | `false_profile_attribution` |
| 26 | `rHJHxKgnXwkC` | [CMS heavy-ion addendum](https://physics.iyte.edu.tr/wp-content/uploads/sites/51/2019/05/demir62.pdf), Bristol author group | `false_profile_attribution` |
| 49 | `Ade32sEp0pkC` | [HERA-LHC Proceedings Part A](https://arxiv.org/pdf/hep-ph/0601012), `R. Croft` affiliation 31, Bristol | `false_profile_attribution` |
| 77 | `SjuI4pbJlxcC` | [CMS/TOTEM forward-physics report](https://inspirehep.net/files/cac9ca36140cd4eb07f981859bf6e8ec), `R. Croft` affiliation 10, Bristol; reference 104 cites his Bristol thesis | `false_profile_attribution` |
| 141 | `WC9gN4BGCRcC` | [Exact CMS/TOTEM Level-1 study in HERA-LHC Proceedings Part B](https://arxiv.org/pdf/hep-ph/0601013), `R. Croft` affiliation c, Bristol | `false_profile_attribution` |
| 154 | `3NQIlFlcGxIC` | [CMS Note 2006/054](https://inspirehep.net/files/940439019050d9d4f57d747ca9d653d4), `R. Croft` under University of Bristol | `false_profile_attribution` |
| 157 | `q-HalDI95KYC` | Second saved rendering of the [same heavy-ion addendum](https://physics.iyte.edu.tr/wp-content/uploads/sites/51/2019/05/demir62.pdf), with the Bristol author | `false_profile_attribution` |

The first five Croft exclusions already on producer main are NSF award or CV `nonpublication` decisions. This review adds seven separate `false_profile_attribution` decisions. Rows 250 `:LPtt_HFRSbwC` and 264 `:DBa1UEJaJKAC` remain **unresolved and unexcluded**: row 250 needs final exact-row metadata comparison against the heavy-ion report, and row 264 needs a work-versus-section decision. Continue to hold the Croft source profile until its remaining quality review is complete. No Scholar page was opened for this review.

## Source readiness limit

The current producer `validate_complete_receipt` gate fails on the held Croft source at row 29. A read-only comparison of all 267 saved HTML, raw JSON, and cache rows found seven HTML-to-raw title extraction omissions at rows 29, 56, 60, 127, 129, 215, and 216. The cache titles equal the raw JSON, so the mismatch predates these producer exclusions. The omitted text includes `Lyα`, `z > 2`, and TeX alpha glyphs. This branch makes no source repair or publisher exception and does not make the Croft profile ingestion-ready.
