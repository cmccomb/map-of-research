# Robin Neumayer bibliography capture (review only)

CMU [identifies Robin Neumayer](https://www.cmu.edu/math/people/faculty/neumayer.html)
as Mathematical Sciences faculty and links her
[researcher-owned page](https://www.math.cmu.edu/~rneumaye/). Its four bibliography
sections display 30 entries. The linked [June 2026 CV](https://www.math.cmu.edu/~rneumaye/CVNeumayer2026.pdf)
has 30 numbered entries. The 31-row [crosswalk](cv-crosswalk.csv) preserves all
of them: 29 paired entries, CV-only item 24, and the site-only expository note
*The Yamabe problem*. The CV-only item is also present in a
[published Oberwolfach report](https://ems.press/content/serial-article-files/47471).

The [source audit](source-audit.json) keeps every website observation's wording,
section, position, links, row hash, CV item number, and arXiv correspondence.
The original author HTML, CV PDF, and arXiv XML are retained byte for byte.
The audit also pins the public CMU identity page, CMU Scholars search result,
and unresolved ORCID name candidate; their captured source files are retained
here. [The manifest](manifest.json) pins every source checksum and the crosswalk.
All 23 website research entries match all 23 results in the captured
[arXiv author query](https://export.arxiv.org/api/query?search_query=au%3A%22Neumayer%2C_Robin%22&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending).
These are source correspondences, not 23 additional works.

The [31-row genre and DOI ledger](doi-genre-hold-ledger.csv) records every
observation's research/preprint, article, survey, report, thesis, or note
classification. It reconciles **18 exact registered DOIs** (16 research papers
and two book chapters) against the original Crossref or DataCite responses in
[`doi-registry/`](doi-registry/) and the compact
[DOI evidence index](exact-doi-evidence.json). Six rows retain specific
publication identity or date holds. The
[review decisions](doi-genre-review.md) explain them and separate actual
publication bylines from “joint work with” notes in Oberwolfach summaries.

**Release hold:** No exact Google Scholar account is verified. This capture is
not Scholar data and does not establish complete lifetime output beyond the
visible website and June CV. The ORCID name candidate has no public work or
affiliation evidence and is not assigned to Neumayer. Proceedings, surveys,
the thesis, and the expository note have now been typed in the review ledger,
but six row-level holds prevent whole-union ingestion. The author page does not
link a DOI; every DOI in the ledger was separately verified from a registration
agency. No row is inserted into a source snapshot,
person registry, Hugging Face dataset, or public map.

## Staged 31-row loader

[`load_staged.py`](load_staged.py) reads the pinned captures, crosswalk, and
genre/DOI review into **31 source-only records**. It verifies source checksums,
identities, CV coverage, the 18 registered DOI responses, and the six row-level
holds before returning records. Each record retains the raw source observation,
CV transcription when present, and review ledger row. The loader marks every
record release-held and graph-ineligible; it never builds a snapshot or writes a
dataset. Report summaries, surveys, thesis, and lecture notes have explicit
noncollaboration genres. The three Oberwolfach summaries use their actual solo
publication bylines; their "joint work with" notes remain only in raw source
evidence.

The CV-only Oberwolfach item is `source_kind=researcher_cv`, with the CV and
published EMS report as evidence. It does not cite the author webpage. The
site-only Yamabe lecture notes have `year=null`; January 2018 is a lecture
date, not a claimed publication date. The 2018 MATRIX event label and 2020
chapter DOI are both retained while that row's display year stays unresolved.

**Schema hold:** The main snapshot accepts only Scholar and curated records;
the draft author-site path models an author webpage and requires a year on each
Iyer row. Neither path currently represents this CV-only record as its own
source kind. The staged loader therefore has no snapshot integration. Before
promotion, add a source kind for researcher CV evidence, support nullable
publication years in the author-owned source adapter, and complete row-level
release review. A verified Google Scholar account would supersede this fallback
as publication ground truth.
