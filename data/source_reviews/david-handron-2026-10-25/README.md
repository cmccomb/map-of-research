# David Handron: held whole-researcher source review

Captured 2026-10-25 for `person-davidhandron-66faa357`. This is a bounded
source review, not a publication snapshot or release input. No Google Scholar
ID or self-asserted ORCID was verified. The registry identity and blank Scholar
ID are unchanged.

## Captured source scope

| Source | Attributed observations | Finding |
| --- | ---: | --- |
| [CMU Math profile](https://www.cmu.edu/math/people/faculty/handron.html) | 1 | Its **Select Publications** section lists the 2002 journal article only. |
| [Handron's CMU-hosted personal page](https://www.math.cmu.edu/~handron/) | 2 | The **Papers and Preprints** section lists that article and an arXiv-linked Morse-complex preprint. The page still shows a Fall 2010 teaching schedule, so it cannot establish a current career-complete list. |
| [Rice University thesis record](https://repository.rice.edu/items/3f2f6423-2661-454a-8d2e-e17e4814b734) | 1 | A 1999 doctoral thesis by **David Gerard Christian Handron**. It shares the later article's title but remains a separate thesis manifestation and genre. The pinned Rice API record supplies the raw metadata. |
| [arXiv exact-author query](https://export.arxiv.org/api/query?search_query=au%3A%22David+G.+C.+Handron%22&start=0&max_results=100) | 1 | `math/0406486v2`, “The Morse Complex for a Morse Function on a Manifold with Corners.” The next page is empty; the broader “David Handron” query returns the same entry. |
| [Publisher-registered DOI metadata](https://api.crossref.org/works/10.1016%2FS0166-8641%2802%2900036-6) | 1 | Elsevier's `Topology and its Applications` article, DOI `10.1016/S0166-8641(02)00036-6`, volume 126, issues 1–2, pages 83–118. |
| KiltHub name search | 0 | Both captured result pages are empty. |

The six attributed source observations are retained in
`source-observations-6.csv`, linked provisionally into **three distinct
manifestation groups**: thesis, journal article, and preprint. These are not
lifetime publication totals. The one selected CMU citation and both
author-maintained list entries remain verbatim in separate CSVs. Raw source
responses and SHA-256 hashes are pinned in `manifest.json`.

## Attribution and genre holds

- The Rice thesis and 2002 journal article have the same title. The repository
  identifies the former as a 1999 doctoral thesis; the DOI identifies the
  latter as a 2002 journal article. They must not be flattened into one
  publication record before Scholar reconciliation.
- Three official AMS meeting program or abstract leads are recorded in
  `uncaptured-conference-leads-3.csv`. Direct PDF retrieval was denied, so
  these are **not** ingested source observations and are not asserted to be
  peer-reviewed papers. The 2006 abstract's title differs from the two listed
  papers, making it a useful follow-up lead.
- The author page's Fall 2010 schedule is direct evidence of staleness. Its
  two-item list could have been complete then, but cannot certify his entire
  career through the present.
- A single direct Scholar author search redirected to Google sign-in. It did
  not provide an author result, and no Scholar ID was inferred from a name
  match. The Rice frontend yielded an empty HTTP 202 response, but the
  institution's API record was captured instead.

Every attributed observation and conference lead is `graph_eligible=false`.
Do not create publication or coauthor edges, upload these rows to the Hub, or
present the three groups as a complete researcher until an exact Scholar
account is verified and fully paginated against these sources. If no Scholar
account exists, a current researcher-controlled complete bibliography and an
explicit exception decision would be required.

Run `PYTHONPATH=. uv run --extra dev python scripts/validate_handron_source_review.py`
to check source preservation, attribution, version and genre separation, and
the release holds.
