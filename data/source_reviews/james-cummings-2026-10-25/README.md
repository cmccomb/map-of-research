# James Cummings bibliography evidence (review only)

**Release hold:** No exact Google Scholar profile has been verified for James
Cummings. These records do not enter a publication snapshot, Hugging Face, or a
collaboration graph. Google Scholar remains the preferred publication ground
truth if a verified account becomes available. This review preserves what the
other sources actually show; it does not certify a complete lifetime count.

[CMU identifies Cummings as a Mathematical Sciences professor](https://www.cmu.edu/math/people/faculty/cummings.html).
His [CMU Scholars profile](https://scholars.cmu.edu/2104-james-cummings/publications)
links ORCID `0000-0002-7913-0427` and the researcher-maintained
[papers page](https://www.math.cmu.edu/users/jcumming/papers/). The pinned
sources and every source checksum are in [the manifest](manifest.json).

## Scope and reconciliation

| Source | Captured extent | What it proves |
| --- | ---: | --- |
| CMU Scholars, five pages | 108 distinct object IDs, 25+25+25+25+8 | Full visible CMU Elements list on 2026-10-25, not a deduplicated work count |
| Researcher-maintained papers page | 51 numbered items | 50 correspond to Elements IDs; item 14 is site-only unpublished talk slides. The page says it was last updated 2017-10-23. |
| ORCID works linked by CMU Scholars | 51 groups | 49 have DOIs exactly matching Elements records. Two published chapters have no Elements *publication* records; Elements instead calls their related talks presentations. ORCID is independently incomplete. |
| Publisher DOI records | 4 omissions | Two published chapters and two edited volumes absent as publication manifestations in Elements |

The [108-row Elements crosswalk](elements-crosswalk-108.csv), [51-item author
crosswalk](researcher-crosswalk-51.csv), and [51-row ORCID
crosswalk](orcid-crosswalk-51.csv) preserve each source observation and its
relationship to the others. The [publisher omission
ledger](publisher-omissions-4.json) gives DOI, contributor role, and primary
publisher URL for four missing publication manifestations. The 13 candidate
same-work Elements pairs share a reviewed `work_group` without discarding the
underlying records: `499952/608884`, `153674/608887`, `80684/608889`,
`80690/608888`, `80687/428468`, `313301/428481`, `80706/313300`,
`80712/608890`, `106520/608892`, `80707/608894`, `608891/608881`,
`608893/608882`, and `80686/612226`. These are version or repository
correspondences, not 13 additional publications.

The two chapter manifestations are [*Large Cardinal Properties of Small
Cardinals*](https://link.springer.com/chapter/10.1007/978-94-015-8988-8_2)
(Springer, 1998; Elements presentation `341749`) and [*Compactness and
incompactness phenomena in set
theory*](https://www.cambridge.org/core/books/logic-colloquium-01/compactness-and-incompactness-phenomena-in-set-theory/509478D98A6929EF1A6408564701480A)
(Cambridge print volume, 2005; Elements presentation `341761`). Cambridge
posted the latter online in 2017; that is an online manifestation date, not a
second work. The missing edited volumes are [*Appalachian Set Theory
2006–2012*](https://www.cambridge.org/core/books/appalachian-set-theory/6C9D56F8A40832C8A58E47B330521F3F)
(Cambridge, 2012; Cummings and Ernest Schimmerling, editors) and
[*Foundations of Mathematics: Logic at
Harvard*](https://www.ams.org/books/conm/690/) (AMS, 2017; Andrés Eduardo
Caicedo, Cummings, Peter Koellner, and Paul B. Larson, editors). The
`Appalachian` Crossref record says 2009, but Cambridge's publication dates are
in 2012; the ledger retains this conflict instead of silently choosing the
registration year. Editorship does not imply authorship of every chapter.

## Genre and attribution holds

- All 35 Elements presentations remain presentations. The two related to
  published chapters have separate publisher-backed publication records;
  neither talk is converted into a chapter.
- Elements book `619986`, *Higher Recursion Theory and Set Theory*, credits four
  people in an author-style field. [The publishing institute identifies them
  as editors](https://ims.nus.edu.sg/publications/lecture-notes-series/), and
  the pinned Crossref response types the work as an edited book. No article
  coauthor edges follow from that editorial credit.
- Elements `327935` has the literal title `$22` and says journal article.
  [The publisher record for DOI `10.2307/421078`](https://doi.org/10.2307/421078)
  and Crossref identify it as a review by James W. Cummings. It is held from
  research-article display until its title and genre are repaired.
- Elements `313300`, `313301`, and `313302` have figshare repository DOIs and
  journal-style labels. The first two correspond to separately registered
  articles. The third, *Limits, Regularity and Removal for Finite Structures*,
  remains an author-owned [arXiv preprint](https://arxiv.org/abs/1412.8084)
  in this review. The author page's 2017 “to appear” claim is not a verified
  final journal publication.
- Elements `80711` says `Conference`, whereas its registered DOI and venue
  identify a *Notre Dame Journal of Formal Logic* article. Elements chapter
  `787912` is a genuine 2018 Springer education commentary by James Cummings,
  Elise Lockwood, and Keith Weber; the CMU-linked person attribution and
  registered DOI are preserved without treating it as a set-theory paper.

The same-name [2019 digital humanities article](https://link.springer.com/article/10.1007/s42803-019-00016-6)
was found during search and was **not** attributed to this CMU mathematician.
Source records require a CMU person link, exact ORCID link, researcher-owned
listing, or primary publisher match; a name-only search result is insufficient.

## Remaining release boundary

The author page stops in 2017, ORCID's latest work is from 2024, and Elements
shows 2025 entries. The two edited volumes demonstrate a concrete gap in
Elements, and the genre errors mean its 108 rows cannot be used as a
whole-career count. No exact Scholar account has been located or collected.
Every row here is marked `release_held`; none is graph eligible. Promotion
requires source-model support for editor and reviewer roles, chapter/talk
manifestations, repository copies, and explicit work identity decisions, plus
the approved release handoff.

Run the offline provenance gate with:

```sh
python data/source_reviews/james-cummings-2026-10-25/validate_capture.py
```

`load_staged.py` stages **all 214 raw source observations** from those pinned
sources: 108 CMU Elements, 51 researcher-site citations, 51 ORCID groups, and
four publisher registrations. Overlapping observations remain separate, with
the reviewed crosswalk and raw source records attached. This is neither 214
distinct works nor a complete career publication count. Contributor roles
and genres remain explicit, including talks, reviews, and edited volumes.
Every row has release, map, and collaboration eligibility set to false; the
loader does not feed the production snapshot builder.
