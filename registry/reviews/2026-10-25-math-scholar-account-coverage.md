# Mathematical Sciences Scholar account coverage — 2026-10-25

This is a dated identity and cache audit. The [machine-readable ledger](2026-10-25-math-scholar-account-coverage.json) records the coverage totals and a separate search disposition for all 29 people without a verified ID. Existing IDs and their cache rows remain in the source registry and immutable snapshot. This review does not assign an ID or change a publication source.

| Included Math people | Verified Scholar IDs | Captured rows visible at collection | Missing verified ID | Incomplete known-ID caches |
| ---: | ---: | ---: | ---: | ---: |
| 65 | 36 | 4,080 | 29 | 0 |

The roster is from producer `main` at `68ab26ed4e857224ea4789238d95ec3e838ef454`. The cache evidence is from immutable source commit `9bb7474a431a941cd9a63b1af2c17b1f8871d4c3` on `automation/map-snapshot`, checked against that branch's current remote tip on October 25. Each of the 36 cache files has `coverage=complete_visible_profile`, `more_remain=false`, and a displayed count equal to its publication-row count. Recomputing SHA-256 over the saved capture bytes matched all **36 JSON and 36 HTML** hashes recorded in the caches (72/72); no raw capture file was missing. Collection timestamps span October 24–25. These facts establish the saved visible profile at collection time; they do not prove Google Scholar contains the researcher's entire academic history or that a profile has not changed since collection.

## Unverified accounts

The CMU Mathematical Sciences roster lists the following included people without a verified Scholar ID. The `search_result` in the JSON distinguishes a search with no matching account returned from searches that returned only unrelated or ambiguous people. **Neither result proves that a Scholar account does not exist.**

| Role | People |
| --- | --- |
| Faculty (10) | Agoston Pisztora; Clinton Conley; Ernest Schimmerling; Gautam Iyer; James Cummings; Martin Larsson; Po-Shen Loh; Richard Statman; Robin Neumayer; Theresa Anderson |
| Teaching (10) | Anthony Weston; David Handron; David Offner; Dylan Quintana; Gregory Johnson; Irina Gheorghiciuc; Jane Panangaden; Jonathan Simone; Noha Abdelghany; Tim Flaherty |
| Emeriti (9) | Jack Schaeffer; Luc C. Tartar; Manuel Blum; Marion Oliver; Robert F. Sekerka; Roy Nicolaides; Russell C. Walker; Shlomo Ta'asan; William O. Williams |

On October 25, we checked direct Scholar links on the 27 available [CMU Math profile pages](https://www.cmu.edu/math/people/faculty/index.html) for these people; none linked a Scholar profile. A small, logged-in Chrome batch searched Scholar's **Profiles** author search for all 29 names, with subject or affiliation qualifiers where needed. No result had a verified CMU address, researcher-controlled link, or matching publication history sufficient to assign an exact ID. Search results included unrelated namesakes: Manuel Blum profiles in medicine and at Freiburg, Noha Abdelghany in Hebrew literature, several other Martin Larssons outside mathematical finance, and non-Math results for James Cummings. The research tabs were closed after inspection. Searches are point-in-time and incomplete as an existence test.

Researcher-maintained publication lists can corroborate future candidates before a whole-profile Scholar capture:

| Person | Researcher publication source |
| --- | --- |
| [Po-Shen Loh](https://www.math.cmu.edu/~ploh/docs.shtml) | CMU-hosted papers and documents |
| [Martin Larsson](https://sites.google.com/view/martin-larsson) | Researcher site |
| [Clinton Conley](https://www.math.cmu.edu/~clintonc/) | CMU-hosted researcher site |
| [Ernest Schimmerling](https://www.math.cmu.edu/~eschimme/publications.html) | CMU-hosted publications |
| [James Cummings](https://www.math.cmu.edu/users/jcumming/papers/) | CMU-hosted papers |
| [Theresa Anderson](https://faculty.math.cmu.edu/anderson/research/) | CMU-hosted research list |
| [Gautam Iyer](https://www.math.cmu.edu/~gautam/) | CMU-hosted publications |
| [Robin Neumayer](https://www.math.cmu.edu/~rneumaye/) | CMU-hosted publications |
| [David Offner](https://davidoffner.wordpress.com/research/) | Researcher-maintained research list |
| [Robert F. Sekerka](https://sekerkaweb.phys.cmu.edu/SekerkaPublications.pdf) | CMU-hosted bibliography |

The remaining 19 have no researcher publication URL recorded in this pass; this is an audit gap, not a claim that they have no publications. The next exact profile candidate should be cross-checked against the relevant official CMU identity and its publication history. When a profile is verified, collect **every row visible after Show More is exhausted** in one capture, save the HTML and structured JSON with hashes, and reconcile the complete cache through the source and publication gates. Keep the Scholar ID blank until identity evidence is strong enough. Primary display homes and all joint memberships stay as recorded in the registry.

Correction on October 25: [Sekerka's individual CMU Math profile](https://www.cmu.edu/math/people/faculty/sekerka.html) was located after the 27-page pass above. It links his [CMU-hosted personal site](https://sekerkaweb.phys.cmu.edu/), which in turn links the bibliography in the table. The individual profile and personal site's HTML were checked for a direct Scholar account link; neither supplies one. The JSON ledger now points to the individual profile, bringing the individually checked Math profile pages to 28 without changing Sekerka's Scholar identity decision.
