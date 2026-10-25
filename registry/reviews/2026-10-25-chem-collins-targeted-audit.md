# Collins truncated-bylines and output-type audit — 2026-10-25

This review uses the complete 361-row, disabled-Show-More Scholar capture in
draft source PR #118, its exact `author_pub_id` values, and the 31 reviewed
exclusions in draft producer PR #247. No raw Scholar observation was changed.
The [row inventory](2026-10-25-chem-collins-truncated-bylines.csv) identifies
all 61 nonexcluded rows whose short Scholar byline omits `Collins` or `TJ` and
ends in an ellipsis. The nontruncated row 42 is separate: [CMU lists its title
under Collins's articles](https://www.cmu.edu/news/experts/terrence.collins).

Forty-nine of the 61 truncated rows have exact normalized-title matches in
[Collins's CMU-linked ORCID works](https://orcid.org/0000-0003-2611-9184).
This is positive corroboration of the Scholar profile row, not proof of every
individual author's role. Of the twelve unmatched rows, eleven have the
following source-level positives. Row 274 remains held for attribution:

| Rows | Source check and decision |
| --- | --- |
| 23, 29, 61 | [ACS](https://pubs.acs.org/doi/10.1021/ja972922g), [EPA HERO](https://hero.epa.gov/reference/3243967), and [PubMed](https://pubmed.ncbi.nlm.nih.gov/38592230/) name Collins on their respective works. Retain. |
| 79, 102, 131 | [Ecologic](https://www.ecologic.eu/19670) names Collins on *Primary plastic polymers*; [CMU IGS](https://www.cmu.edu/igs/pub/index.html) lists the exact TAML catalysis title; Collins and Ryabov's [CV](https://greensciencepolicy.org/docs/cpsc-petition-supporting-statements-15-03-31.pdf) lists the desulfurization conference work. Retain. |
| 173, 175, 192 | [CMU](https://www.cmu.edu/news/experts/terrence.collins), [PubMed](https://pubmed.ncbi.nlm.nih.gov/31249429/), and the [original *Inorganic Syntheses* chapter](https://sites.lsa.umich.edu/jbuss/wp-content/uploads/sites/811/2020/08/inorganic-synthesis33.pdf) name Collins. Row 192 has inaccurate Scholar volume-wide pages; retain its genuine chapter, with metadata correction pending. |
| 218, 228 | The [Wiley issue](https://chemistry-europe.onlinelibrary.wiley.com/toc/15213765/2015/21/16) credits Collins on a separate cover picture; [ACS DOI metadata](https://api.crossref.org/works/10.1021/jacs.6c11419) names Collins and his ORCID on the JACS paper. Retain. |
| 274 | The UK histopathology abstract's expanded author identity remains unresolved. Preserve the raw row under its existing provisional hold. |

## Output-type decisions

| Rows | Source check and decision |
| --- | --- |
| 141 | Scholar indexes a 1999 *ChemInform* abstract for *The Design of Green Oxidants*. [Oxford's book](https://books.google.com/books?id=fbhTAAAAMAAJ) and a [CMU-hosted scholarly bibliography](https://cepac.cheme.cmu.edu/pasi2008/slides/ofelia/library/reading/Life-Cycle_Approaches_for_Assessing_Green_Chemistry_Technologies.pdf) establish the original 1998 chapter, pages 46–71; the ChemInform record is an abstract index. Hold the Scholar row pending a source-bound chapter rekey. Row 211, another ChemInform index of an already retained original paper, is among the 31 exact exclusions. |
| 350 | Scholar shows an undated *Advanced Oxidation Process Mid-year Report* by Horwitz and Collins without venue. The [DOE final report](https://www.osti.gov/servlets/purl/821583) and [EPA record](https://hero.epa.gov/reference/1580300/) establish a separate 2003 final report and the authors, but the exact mid-year copy was not located. Hold pending report identity and type. |
| 356 | Scholar shows an undated and unvenued estradiol-degradation item by Mills, Arias-Salazar, Ryabov, and Collins. The [2015 *Scientific Reports* article](https://www.nature.com/articles/srep10511) confirms those authors' related research but does not identify this exact output. Hold pending an exact copy or venue. |
| 358, 360, 361 | The former is a distinct [2014 credited research poster](https://www.researchgate.net/publication/305619657_Degradation_and_Removal_of_Biological_Effects_of_Ethinylestradiol_by_Green_Chemistry_%27TAML_Hydrogen_Peroxide_Activators%27_-_a_Possible_Green_Solution_to_EDC_Pollution); Gordon College hosts the [2010 phenolic-polymerization poster](https://www.gordon.edu/download.cfm?file=%5Cdownload%5Cgalleries%5C2010_Physical+Sciences_Poster_Winner.pdf) and [2011 structural-determination poster](https://www.gordon.edu/download/galleries/Kristen%20Poster%20URS%202011.pdf), each naming Collins. Retain as credited posters despite Scholar's missing year and venue. |
| 218, 234 | Wiley's [2015 issue](https://chemistry-europe.onlinelibrary.wiley.com/toc/15213765/2015/21/16) and [2023 issue](https://chemistry-europe.onlinelibrary.wiley.com/toc/15213765/29/69) credit Collins on cover items separate from research articles. Retain both distinct outputs. |
| 64, 73, 167, 236 | Scholar has distinct patent publication or application numbers. The [2001 patent record](https://patents.google.com/patent/US6241779B1/en) and [2021 patent record](https://patents.google.com/patent/US10926248B2/en) name Collins as inventor. Similar patent titles alone do not establish duplicate publications; retain the patent class pending any exact family-level review. |
| 265, 266, 279 | Scholar explicitly identifies ACS meeting abstracts and a symposium lecture, with `TJ Collins` in the short byline. These are distinct scholarly outputs, not article surrogates; retain. |

The three added holds bring the provisional set to seven: capture rows
141, 214, 274, 299, 350, 355, and 356. With 31 exact exclusions, 323 of the
361 raw Scholar rows remain map-eligible candidates subject to ordinary quality
checks. This is a source-bound release decision, not certification of all 323
remaining records. Source PR #118 and producer PRs #247 and #243 remain draft;
none of this audit authorizes a snapshot rebuild or upload during the central
release freeze.
