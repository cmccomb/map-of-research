# Jane Panangaden: complete visible-source review

**Status:** Review only. This bundle is outside the snapshot builder and adds
no publication or coauthor edge to the public maps. It captures every visible
entry in two bounded author sources on 25 October 2026; it does not establish
an exhaustive career bibliography or a verified Google Scholar account.

[CMU's Math profile](https://www.cmu.edu/math/people/faculty/panangaden.html)
identifies Jane Panangaden as an Assistant Teaching Professor in Mathematical
Sciences. Her self-asserted
[ORCID 0000-0002-6214-2130](https://orcid.org/0000-0002-6214-2130)
names CMU as her 2026 employer and contains **five journal-article DOI
groups**. The [2026 publisher page](https://link.springer.com/article/10.1007/s43034-026-00512-1)
independently puts the same ORCID on her article with Peter Burton. The exact
[arXiv author query](https://export.arxiv.org/api/query?search_query=au:%22Panangaden%2C_Jane%22&start=0&max_results=100)
reports **seven entries**, all naming Jane. Five are earlier versions of the
five DOI articles, and two have no matched journal DOI in these sources.

The [2016 McGill master's thesis](https://www.erudit.org/en/theses/mcgill/P/?page=12&sort_by=date_desc)
is a version of arXiv:2110.01678: the posted PDF's cover identifies the same
title, author, institution, and April 2016 degree, despite its 2021 arXiv upload.
The [2022 Caltech dissertation](https://thesis.caltech.edu/14622/) is a separate
doctoral work, DOI `10.7907/xrba-8471`. Its subject overlaps the 2020/2024
modular-curves paper and the 2023 higher-weight preprint, but genre and scope
differ; those three remain separate provisional work families.

| Provisional family | Source observations | Genre pending final review |
| --- | --- | --- |
| Reflectionless CMV matrices | 2014 arXiv; 2015 journal DOI | Article |
| Energy conservation, counting statistics | 2014 arXiv; 2015 journal DOI | Article |
| Reflection probabilities of one-dimensional Schrödinger operators | 2015 arXiv; 2017 journal DOI | Article |
| Quantum statistical mechanics and modular curves | 2020 arXiv; 2024 journal DOI | Article |
| Furstenberg conjecture formulations | 2024 arXiv; 2026 journal DOI | Article |
| Energy full counting statistics in return-to-equilibrium | 2016 McGill thesis; 2021 arXiv posting | Master's thesis |
| Higher-weight limiting modular symbols | 2023 arXiv | Preprint |
| Quantum statistical mechanics, noncommutative geometry, and boundary of modular curves | 2022 Caltech thesis | Doctoral thesis |

The four-author 2015 *Energy Conservation, Counting Statistics, and Return to
Equilibrium* [journal article](https://link.springer.com/article/10.1007/s11005-015-0769-7)
and Panangaden's solo 2016 master's thesis have similar titles but distinct
authors and content. The 2026 ORCID month (July) also differs from the
[publisher's online date](https://link.springer.com/article/10.1007/s43034-026-00512-1)
(21 April); source dates remain separate in the observation table.

The [Math account review](../../../registry/reviews/2026-10-25-math-scholar-account-coverage.json)
records a bounded Scholar author search with no verified Jane Panangaden
account. Search results for Prakash Panangaden are a separate person and are
excluded. Neither CMU's page nor the public ORCID record links an exact
Scholar profile. ORCID omits both theses and the 2023 preprint, while arXiv
only covers deposited versions. No researcher-owned full CV or bibliography
was found. These 14 source observations across eight provisional work families
remain held; citation counts and whole-career completeness are unknown.

`manifest.json` gives SHA-256 hashes of the five raw source captures and the
master's PDF cover transcript. The full master's PDF SHA-256 is recorded there
without adding the PDF to Git. Run
`python3 scripts/validate_panangaden_source_review.py` from the repo root to
verify raw captures, source counts, identity, and the 14-row crosswalk.
