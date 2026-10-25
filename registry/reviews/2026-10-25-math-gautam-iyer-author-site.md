# Gautam Iyer author-site publication review (2026-10-25)

This is a **review extraction**, not a Google Scholar capture or a released source-graph update. The [CMU Mathematical Sciences faculty page](https://www.cmu.edu/math/people/faculty/iyer.html) identifies Gautam Iyer and links to his [CMU-hosted personal website](https://www.math.cmu.edu/~gautam/sj/index.html). Its **All Publications** tab currently displays 47 entries. This means 47 displayed entries were captured, not that every work in his academic history has been independently established.

## Capture and integrity

- Source: `https://www.math.cmu.edu/~gautam/sj/index.html`
- Captured at: `2026-10-25T09:48:40+00:00` from an HTTP 200 response; 163,654 HTML bytes.
- Source HTML SHA-256: `264cca147620a53e60d51b83b97339b3556646cb906cc46fdae53df6e500107b`. The exact HTML is saved at [`data/author_sites/gautam-iyer-2026-10-25.html`](../../data/author_sites/gautam-iyer-2026-10-25.html).
- Extraction boundary: the 47 direct `<li>` children of `div#pubs-all > ul`. The Recent Publications and topic tabs repeat many of these works and were excluded.
- [Author-site CSV](2026-10-25-math-gautam-iyer-author-site.csv): 47 consecutive row indices, 47 unique normalized title/year pairs, 47 titles, 47 author strings containing Iyer, 47 years, and 47 linked work URLs. Its SHA-256 is `e3f481cb22ede0500ccb5bc1f950a012e0c442058865876644b5bf69a01dde79`.
- The CSV keeps the author's title, initials, year, and venue **as listed**. `work_url_as_linked` is the venue/publisher link when shown, otherwise an author-hosted manuscript PDF. These links were recorded, not all independently opened. Row 44 is a Ph.D. thesis.

## Metadata gaps

- 20 of 47 entries have a DOI link on the author page. The other 27 have **no DOI link there**; this does not establish that they lack a DOI.
- 40 of 47 entries show a venue. The seven newest entries have no venue shown, though each has an author-hosted manuscript link.
- The [DOI audit CSV](2026-10-25-math-gautam-iyer-doi-audit.csv) checks all 20 linked DOIs against the publisher-supplied Crossref [Works API](https://api.crossref.org/works/). Each DOI's registry author list includes an Iyer. Its SHA-256 is `188388e7b8bcbdcbeeea40946cd4baa27b91456d5075685a1215f6311e4017ee`.
- A separate [unlinked DOI audit](2026-10-25-math-gautam-iyer-unlinked-doi-audit.json) reviews rows 32–34 from Crossref, arXiv, and publisher evidence. Its SHA-256 is `ffa1f0913353f1b7770a6cdb324fad5aea995d86744bd415c43c517461c99da9`. These DOI values were **not** linked from the author page; its 47 raw rows and 20 linked-DOI audit rows are unchanged.
- No exact Google Scholar account is registered for Iyer in `registry/people.csv` at this review. If one is verified later, Scholar should remain the project's primary source for his displayed profile, with this author-site list used to inspect omissions and conflicts.

## Exact DOI and year reconciliation

Ten author-listed years differ from the linked DOI's Crossref publication year. The author-site year remains unchanged in the raw row; the normalized observation uses the DOI publication year. The source records do not establish why the two dates differ, so no manuscript-date claim is made.

| Row | DOI | Author year | DOI publication year |
| --- | --- | ---: | ---: |
| 8 | `10.1088/1361-6544/adc652` | 2024 | 2025 |
| 11 | `10.1090/qam/1668` | 2024 | 2023 |
| 14 | `10.1098/rsta.2021.0024` | 2021 | 2022 |
| 16 | `10.1137/21M1394497` | 2021 | 2022 |
| 17 | `10.1007/s00285-021-01615-0` | 2020 | 2021 |
| 18 | `10.1007/s00332-020-09637-6` | 2019 | 2020 |
| 20 | `10.1090/tran/8195` | 2019 | 2021 |
| 22 | `10.4310/CMS.2020.v18.n7.a2` | 2018 | 2020 |
| 23 | `10.30757/ALEA.v18-48` | 2018 | 2021 |
| 26 | `10.1214/17-AOP1196` | 2016 | 2018 |

Row 22's author-page title, “Anomalous diffusion in one and two dimensional combs,” differs substantially from the DOI record. The [author-hosted manuscript](https://www.math.cmu.edu/~gautam/sj/research/papers/201809-fcomb.pdf) and [publisher PDF](https://www.intlpress.com/site/pub/files/_fulltext/journals/cms/2020/0018/0007/CMS-2020-0018-0007-a002.pdf) both title the work “Anomalous Diffusion in Comb-Shaped Domains and Graphs” and name Samuel Cohn, Gautam Iyer, James Nolen, and Robert L. Pego. The DOI belongs to that listed work; the author-page title and 2018 year remain in the raw capture as observed.

## Three independently reviewed DOI identities

All three author-listed years already equal the Crossref publication years, so these reviews change the DOI-backed work identity without changing the displayed year. The audit retains the author's original title, byline, and blank DOI field alongside the publisher title/byline, Crossref response hash, and evidence URLs.

| Author-site row | Reviewed DOI | Independent evidence |
| ---: | --- | --- |
| 32 | `10.4310/cms.2015.v13.n1.a12` | [Communications in Mathematical Sciences publisher PDF](https://archive.intlpress.com/site/pub/files/_fulltext/journals/cms/2015/0013/0001/CMS-2015-0013-0001-a012.pdf) and [arXiv:1308.4735](https://arxiv.org/abs/1308.4735) confirm the same five-author work; [Crossref](https://api.crossref.org/works/10.4310%2FCMS.2015.v13.n1.a12) records the 2015 journal DOI. |
| 33 | `10.1214/14-aap1008` | [arXiv:1212.6451](https://arxiv.org/abs/1212.6451) explicitly links the journal DOI; the [IMS issue PDF](https://imstat.org/publications/aap/aap_25_2/AAP_25_2.pdf) and [Crossref](https://api.crossref.org/works/10.1214%2F14-AAP1008) confirm the Iyer–Leger–Pego article. |
| 34 | `10.1063/1.4738637` | [arXiv:1202.5671](https://arxiv.org/abs/1202.5671) explicitly links the journal DOI; Iyer's [journal-form manuscript](https://www.math.cmu.edu/~gautam/sj/research/papers/201202-extdyn.pdf) and [Crossref](https://api.crossref.org/works/10.1063%2F1.4738637) confirm the Iyer–Pego–Zarnescu article. |

These reviews do not alias Pego's distinct Scholar observations. Row 22's preprint/version family and row 4's arXiv native identifier also remain separate pending exact alias rules. No new Scholar account is claimed for Iyer.

## Staged ingestion decision

The [`author_site` capture manifest](../../data/author_sites/gautam-iyer-2026-10-25.json) binds the exact HTML, extraction CSV, 20 linked DOI reviews, and three independently reviewed unlinked DOI identities. The loader requires all 47 rows, exact review counts and row identities, unique manuscript-backed native IDs, the official CMU identity link, and the row-44 thesis classification. It permits blank DOI and venue only for this distinct source type. A DOI is used for work identity only when it appears in a pinned exact review; DOI-like strings inside an unreviewed publisher link or raw JSON do not count. Free-text coauthor names do not create faculty edges.

**No source snapshot or public graph has been updated.** The capture is marked `staged_for_explicit_promotion`; an explicit CLI argument is needed to include it in a raw snapshot preview, and the publisher rejects any snapshot containing `author_site` until a separate exact release gate and consumer provenance display are ready. A verified Scholar account should later take precedence for populated profile fields while retaining these 47 source observations for reconciliation. The author site proves what its All Publications section displayed on the capture date; it does not certify lifetime completeness or provide citation counts.
