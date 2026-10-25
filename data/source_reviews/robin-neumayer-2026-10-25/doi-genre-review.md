# Robin Neumayer: exact genre, version, and DOI review

**Decision:** The full visible 31-item union remains source-only and release-held.
The [row ledger](doi-genre-hold-ledger.csv) accounts for every item: 23 research
papers/preprints, five website proceedings/surveys, one CV-only report, one
doctoral thesis, and one website-only set of expository notes. The 23 research
items match the 23 records in the saved arXiv author feed. Direct registration
agency lookups verify **18 exact DOIs** (16 research articles and two book
chapters); the [evidence index](exact-doi-evidence.json) identifies each
Crossref/DataCite record and its preserved raw response. A missing DOI is not
replaced with a similar-title candidate.

The [researcher bibliography](https://www.math.cmu.edu/~rneumaye/) and linked
[June 2026 CV](https://www.math.cmu.edu/~rneumaye/CVNeumayer2026.pdf) give the
capture scope. They do not establish a current Scholar account or independently
prove lifetime completeness. Six items have unresolved release metadata:

| Row | Hold |
| ---: | --- |
| 8 | The site and CV cite a 2024 Harvard CMSA book-series placement, and [arXiv 2210.16852](https://arxiv.org/abs/2210.16852) identifies the manuscript. No exact publisher chapter record or DOI was verified. |
| 24 | The [2022 EMS Oberwolfach report](https://ems.press/content/serial-article-files/46965) starts Neumayer's contribution at p. 1563. Her [linked excerpt](https://www.math.cmu.edu/~rneumaye/owr-2022geometry.pdf) says the underlying research was joint with Maggi and Tomasetti; the report contribution is headed with Neumayer's name alone. The site/CV cite the whole workshop report's pp. 1551–1601, so contribution extent and graph eligibility remain held. |
| 26 | The [2019 EMS Oberwolfach report](https://ems.press/content/serial-article-files/46813) starts Neumayer's contribution at p. 2046. Her [linked excerpt](https://www.math.cmu.edu/~rneumaye/OWR19.pdf) mentions joint work with Leonardi and Saracco, but the contribution byline is Neumayer alone. Keep report authorship separate from the joint research it discusses. |
| 27 | The site/CV use the 2018 MATRIX Annals label; the [Springer chapter DOI](https://api.crossref.org/works/10.1007/978-3-030-38230-8_20) records 2020 publication. The chapter is identified, but a source-graph year rule must preserve the 2018 event versus 2020 publication distinction. |
| 30 | The [author-hosted PDF](https://www.math.cmu.edu/~rneumaye/YamabeProblem.pdf) identifies notes from January 2018 lectures. It gives no publication date or publisher. This is an expository note, and 2018 must not silently become a publication year. Similar-title DOI search returns works by other authors. |
| 31 | The June CV still calls this contribution “to appear,” while the [2023 EMS report](https://ems.press/content/serial-article-files/47471) publishes it under Neumayer's name. It summarizes the research in row 8 but is a distinct report contribution. Confirm contribution extent and map eligibility before ingestion. |

Other exact identity decisions are preserved in the ledger. Row 11's Ars
Inveniendi Analytica DOI is registered at [DataCite](https://api.datacite.org/dois/10.15781/e7f3-a487),
not Crossref. Row 13's website title reverses the order of two phrases; its
[author manuscript](https://www.math.cmu.edu/~rneumaye/dpFinal.pdf),
[arXiv record](https://arxiv.org/abs/2010.15663), and
[journal DOI](https://api.crossref.org/works/10.2140/gt.2023.27.227)
identify one research article. Row 25 is a separately published
[book-chapter survey](https://api.crossref.org/works/10.1142/9789811273223_0003)
by the same three authors, not a duplicate of row 13. Row 29's
[dissertation title page](https://www.math.cmu.edu/~rneumaye/NeumayerThesis.pdf)
says *Minimality and stability*, reversing the website/CV wording; it is one
2017 thesis and is not a coauthored publication. Preserve the site/CV and
registered years separately for rows 7, 16, and 21.

No `author_site` observation enters a snapshot in this PR. The existing
publisher still blocks that source type, and the six holds prevent a faithful
31-item ingestion. A verified full Google Scholar profile would take priority
for Scholar bibliographic fields; these source observations should remain
separate evidence for reconciliation.
