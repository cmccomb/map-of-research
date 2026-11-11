# David Offner: source-bounded whole-researcher review

**Status:** Review only. This directory is not loaded by the snapshot builder,
does not assert a Google Scholar account, and cannot publish graph edges.
The source union is bounded by the pages captured on 25 October 2026, not a
claim that every work in Offner's career has been found.

## Identity and source scope

[CMU's faculty page](https://www.cmu.edu/math/people/faculty/offner.html)
places David Offner in Mathematical Sciences. [CMU Scholars](https://scholars.cmu.edu/5568-david-offner/publications)
identifies the same person by name, CMU email, and ORCID
`0000-0002-3413-7991`, and links his [owned bibliography](https://davidoffner.wordpress.com/research/).
His registry person ID is `person-davidoffner-91600258`; the included Math
membership already exists. No exact Scholar author-account URL was verified.

The two retained CMU Elements pages contain **29 different object IDs** with
Offner as a resolved author, including 15 journal and 14 preprint records.
Pagination reports 1–25 and 26–29 of 29 with `totalIsLowerBound=false`.
`elements-29.csv` retains each source title, type, date, DOI, ordered author
list, and object ID; the two HTML files retain the full source payloads. The
site's 29 rows are observations, **not 29 distinct papers**.

`elements-crosswalk-29.csv` records ten preprint-to-journal version pairs and
therefore **19 distinct research works**: 15 journal papers and four
standalone preprints. The journal record is the proposed canonical display
version for each pair. The preprint record remains independently inspectable.
The [Gibson arXiv record](https://arxiv.org/abs/2107.07450) directly names its
2024 journal DOI despite “two” versus “2” in the titles. The
[Han arXiv record](https://arxiv.org/abs/1712.02496) and
[Springer article](https://link.springer.com/article/10.1007/s00373-018-1914-8)
agree on Han and Offner, title, and abstract despite HTML/TeX math markup and
a stray hyphen in the Elements journal title. The
[2019 *Integers* article](https://math.colgate.edu/~integers/t18/t18.pdf)
lists the same seven authors as its preprint, although Elements abbreviates
the journal names. The 2021 Elements date for arXiv:1905.10114 is a later
record date; it does not replace the preprint's 2019 first submission.

For standalone `791732`, Elements displays the March 2017 revision year.
[arXiv's version history](https://arxiv.org/abs/1607.03471) gives the first
submission as 12 July 2016. The crosswalk keeps the original Elements date
and proposes **2016** as the canonical first-publication year for review.
The 2025 Li–Offner–Ye preprint, `791733`, is in CMU Elements but absent from
the older author page; [arXiv](https://arxiv.org/abs/2501.01571) confirms its
byline and date.

## Author page, genres, and holds

`author-20.csv` preserves all 20 entries and links from the retained owned
bibliography: 14 journal, four preprint, and two “Other Publications.” Its 18
research entries map to CMU canonical works, including an earlier preprint
version of the 2025 cycle article. The two additional outputs are:

* Offner's 20 August 2012 *MAA Reviews* book review. The
  [MAA archive](https://old.maa.org/node/112869) and
  [Westminster College notice](https://www.westminster.edu/about/news/release.cfm?id=3955)
  verify the author and genre.
* John Bonomo and Offner's “Solution to Problem 1868,” *Mathematics Magazine*
  85 (2012), 154–155. [Offner's hosted scan](https://davidoffner.wordpress.com/wp-content/uploads/2016/05/problem1868.pdf)
  names those two as the solution's authors. Donald Knuth proposed the
  problem; he is not a coauthor of their solution.

These give **21 distinct known listed outputs before any thesis**. Their
`book_review` and `problem_solution` genres are retained separately from
research articles; neither has been authorized as a research collaboration
edge. The 2022 CV is pinned for historical comparison. It identifies the
2009 thesis *Extremal problems on the hypercube* and Oleg Pikhurko as advisor,
but does not list the thesis under publications. Dissertation inclusion needs
an archival record and an explicit map policy; advisor status alone is not a
coauthor relation.

There may be later works absent from these sources. A verified Scholar
account, if found, must be captured in full and reconciled against all 21
observed outputs. Citation counts remain unknown in this review.

Run `python3 scripts/validate_offner_source_review.py` from the repository
root to validate source checksums, both embedded CMU payloads, author-page
entries, author identity, all ten version mappings, genres, and counts.
