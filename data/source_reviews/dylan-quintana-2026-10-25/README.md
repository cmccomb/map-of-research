# Dylan Quintana: held whole-researcher source review

This review covers included CMU Mathematical Sciences teaching faculty member
`person-dylanquintana-ca051f35`. [CMU's profile](https://www.cmu.edu/math/people/faculty/quintana.html)
and [2023 interview](https://www.cmu.edu/mcs/news-events/2023/1107-dylan-quintana-math-education)
connect his 2015 CMU mathematics and materials science degrees, 2021 University
of Chicago mathematics doctorate, and CMU appointment. Publisher affiliations
tie the materials and coding papers to those two stages of his career. **No exact
Google Scholar account or complete career bibliography was verified. Lifetime
completeness is unestablished.**

## Confirmed core, by publication genre

| Year | Genre | Work | Version evidence |
| --- | --- | --- | --- |
| 2015 | Journal article | [Multiferroicity in doped hexagonal LuFeO3](https://doi.org/10.1103/PhysRevB.92.054435) | [2014 arXiv preprint](https://arxiv.org/abs/1406.7793) explicitly links the journal DOI; title and byline changed. |
| 2016 | Journal article | [Bayesian method for the analysis of diffraction patterns using BLAND](https://doi.org/10.1107/S1600576716016423) | [IUCr issue](https://journals.iucr.org/j/issues/2016/06/00/) and Crossref give D. J./Dylan J. Quintana. |
| 2020 | Conference paper | [List Decoding of Direct Sum Codes](https://doi.org/10.1137/1.9781611975994.85) | [arXiv full version](https://arxiv.org/abs/2011.05467) says it belongs to SODA 2020. |
| 2020 | Conference paper | [Unique Decoding of Explicit ε-balanced Codes Near the Gilbert–Varshamov Bound](https://doi.org/10.1109/FOCS46700.2020.00048) | [arXiv version](https://arxiv.org/abs/2011.05500) has the same title and byline. |
| 2021 | Doctoral dissertation | [Adventures in Decoding Direct Sum Codes](https://knowledge.uchicago.edu/records/3c4qz-van37) | UChicago version DOI `10.6082/uchicago.3032`; DataCite links OpenAlex's concept DOI `10.6082/he9v3-2g538` to this record. |

[CMU's selected publications list](https://www.cmu.edu/math/research/docs/sample-faculty-publications-2018-2023.pdf)
shows only the two 2020 coding papers because it covers publications since 2018.
The two earlier journal articles are independently verified by the
[Physical Review B publisher](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.054435),
[IUCr journal issue](https://journals.iucr.org/j/issues/2016/06/00/), and
[NIST](https://www.nist.gov/publications/bayesian-method-analysis-diffraction-patterns-using-bland).
The 2015 publisher lists his CMU affiliation. NIST's 2016 page misspells his
surname `Quintanta`; the journal and Crossref spell `Quintana`.

The 16-row crosswalk also retains a [2014 APS abstract](https://meetings-archive.aps.org/mar/2014/m6/8),
2014 [NIST student report](https://www.nist.gov/system/files/documents/2017/05/09/Promo-book-for-SURF-2014.pdf),
[2015 ACA poster](https://acas.memberclicks.net/assets/Meeting/ProgramBooks/2015%20PB.pdf),
[2016 ACA talk](https://acas.memberclicks.net/assets/Meeting/ProgramBooks/2016%20PB.pdf),
[2016 ACNS talk](https://www.osti.gov/servlets/purl/1343116), a 2016 APS
abstract indexed by OpenAlex, three preprint rows, and two duplicate metadata
rows. These genres remain distinct from the four peer-reviewed papers. The
direct 2016 APS abstract PDF returned 403, so its exact publisher byline remains
unverified and its row is discovery-only. The 2015 NIST row duplicates the PRB
paper; OpenAlex's second FOCS arXiv row duplicates the same preprint.

Crossref dates SODA to January 2020; OpenAlex labels that record 2019. OpenAlex
calls the dissertation an article, while UChicago classifies it as a dissertation.
Both original source values are retained. [A separate Dylan Quintana MD](https://www.fullcircleidaho.org/health_care_team_mem/dylan-quintana-md/)
has College of Idaho chemistry and University of Washington medical training;
clinical works under that name are excluded from this CMU review.

## Release boundary and completeness

The exact OpenAlex author record reports 12 indexed works, while its filtered
works query returned 11 raw rows. All 11 are in `openalex-works.json` and mapped
in `candidate-crosswalk-16.csv`; five additional primary-source observations
bring the crosswalk to 16 manifestations. Five inspected PDFs are indexed by
hash and evidence note without committing the PDFs. The CMU interview said in
November 2023 that Quintana had not researched since his thesis; it cannot
establish his complete history through 2026. His old UChicago personal URL now
returns 404, and a direct Scholar author search redirected to sign-in/CAPTCHA.
No Scholar ID or Scholar publication IDs are assigned.

`manifest.json` pins raw source responses and the crosswalk by SHA-256. Run
`python scripts/validate_dylan_source_review.py` to check source integrity,
all 11 discovery rows, DOI/version/genre decisions, namesake exclusion, and the
release hold. The loader emits no canonical work or coauthor IDs. This review
does not alter a source snapshot, Hub dataset, or map.
