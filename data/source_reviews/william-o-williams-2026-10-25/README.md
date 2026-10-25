# William O. Williams: held whole-researcher source review

This review concerns `person-williamowilliams-d43b120b`, the [CMU Mathematical
Sciences professor emeritus](https://www.cmu.edu/math/people/faculty/williams.html)
whose research includes mechanics, thermodynamics, and tensegrity. CMU's faculty
page calls its eight-item list **Select Publications**. It does not establish a
complete bibliography. No exact Google Scholar account is verified. **Lifetime
publication completeness is unestablished.** The 65 crosswalk rows are source
candidates and manifestations, not 65 distinct works or releasable Scholar rows.

The official CMU page and [older researcher page](https://www.math.cmu.edu/~wow/williams.html)
establish identity and selected titles. The [researcher PDF directory](https://www.math.cmu.edu/~wow/papers/)
exposes eight author PDFs; `author-pdf-index.json` records their URLs, content
hashes, sizes, and inspected first-page facts without committing the PDFs.
[Rice's repository thesis](https://repository.rice.edu/handle/1911/88982)
identifies **William Orville Williams** and a 1964 master's thesis with the same
title as his 1965 journal article. The thesis and journal article remain separate
manifestations while their content relation is unreviewed. The thesis PDF is
also represented by a hash and first-page facts, without its content.

## Source accounting

| Source | Frozen evidence | Interpretation |
| --- | --- | --- |
| [CMU faculty profile](https://www.cmu.edu/math/people/faculty/williams.html) | `cmu-profile.html.gz`, `cmu-selected-8.json` | Eight selected citations; never a completeness claim. |
| [Researcher PDF directory](https://www.math.cmu.edu/~wow/papers/) | `author-papers-index.html.gz`, `author-pdf-index.json` | Eight PDFs, including two Primer versions and an unlisted Snelson manuscript. |
| [OpenAlex author](https://openalex.org/A5111526463) | `openalex-author.json`, `openalex-works.json.gz` | 60 discovery rows returned by the author filter. Its author summary says 68 works; that mismatch is unresolved. |
| [Crossref API](https://api.crossref.org/) | `crossref-57.json.gz`, `crossref-marching.json` | Deposited publisher metadata for 57 indexed DOIs plus the Micheletti–Williams paper omitted by the author filter. |
| [CMU KiltHub](https://kilthub.cmu.edu/) and [DataCite](https://api.datacite.org/) | `kilthub-search.json`, `kilthub-records.json`, `kilthub-crosswalk-6.csv`, `datacite-cauchy.json` | Six CMU Mathematics Technical Reports and their likely journal-version relationships; repository DOI for one appears in OpenAlex. |
| [Rice repository](https://repository.rice.edu/handle/1911/88982) | `rice-ms-thesis-item.json`, `rice-thesis-pdf-index.json` | One exact-name 1964 master's thesis. |
| Review crosswalk | `candidate-crosswalk-65.csv` | 60 OpenAlex rows and five other source candidates; all held. |

`manifest.json` gives capture time, source endpoints, exact file hashes, counts,
person ID, and the review hold. Raw metadata and the site HTML are preserved;
the three HTML snapshots are gzipped byte-for-byte.
The KiltHub API search used `POST /v2/articles/search` with
`{"search_for":"\"William O. Williams\"","group":9960,"page_size":100}`;
each of its six returned IDs was fetched individually. The OpenAlex request
used `filter=author.id:A5111526463&per-page=200`, and each Crossref DOI was
fetched individually. These queries are discovery evidence, not completeness
proof.
The two DOI-free OpenAlex rows are preserved too. The 1979 Sampaio–Williams
paper lacks a located journal primary record; its primary-evidence field is
blank. The 1993 Gurtin–Williams report has an [NTIS record](https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/ADA266641.xhtml)
and is classified as a report, not a research article.

## Reconciliation that prevents false edges and counts

- CMU calls the 2007 Micheletti–Williams paper *An Evolutionary Procedure*;
  its [linked author PDF](https://www.math.cmu.edu/~wow/papers/marching.pdf)
  and [journal DOI](https://doi.org/10.2140/jomms.2007.2.857) say *A marching
  procedure*. Both source strings remain visible in the frozen evidence.
- The two *Axioms for work and energy in general continua* DOIs are distinct
  [Part I](https://doi.org/10.1007/bf00251432) and
  [Part II](https://doi.org/10.1007/bf00255667). Conversely,
  [this DOI](https://doi.org/10.1007/bf00247688) is a one-page corrigendum to
  the 1969 Owen–Williams article.
- Three rows are book reviews. The [*Physics Today* review](https://physicstoday.aip.org/reviews/the-physics-of-deformation-and-flow)
  is by Williams; Crossref also lists the reviewed book's authors. The
  [SIAM review](https://doi.org/10.1137/1033045) is by Williams, not by the
  book's author. The ASME *Recent Developments in Thermomechanics of Solids*
  row appears to be a review, but its full publisher page was inaccessible;
  its role and byline remain held. No reviewed-book creator becomes a coauthor.
- Three book-chapter DOIs appear to repeat journal articles, and six KiltHub
  reports appear to precede or accompany journal articles. These are version
  candidates, not independently counted research works. The two *Primer* PDFs
  are versions of one manuscript; no final journal publication is confirmed.
- KiltHub's 1967 *A note on Cauchy's theorem* is a title-variant candidate of
  the 1968 [journal article](https://doi.org/10.1016/0022-247x(68)90181-9).
  OpenAlex uses the later repository DOI year; the review retains the
  repository's 1967 year.

`python scripts/validate_williams_source_review.py` checks the frozen hashes,
source row counts, version crosswalk, and release hold. Its source-only loader
returns 65 observations with no canonical work IDs, no canonical coauthor IDs,
and all release, map, and collaboration flags false. This directory has no
import path into collection, snapshot building, or upload.
