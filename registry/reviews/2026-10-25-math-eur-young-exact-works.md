# Christopher Eur and Michael Young: exact Scholar work review

Reviewed 2026-10-25 against the complete, source-only Scholar captures staged in
PR #226 at commit `c95a1a9`, including its repaired SVG title text. This
producer review changes effective bibliographic fields and work
identity only. All 96 raw Scholar rows, including their observed titles, years,
venues, and exact citation IDs, remain in the source captures and caches and
will be retained when the new snapshot is built.

| Researcher | Earlier Scholar observation | Version of record | DOI |
| --- | --- | --- | --- |
| Christopher Eur | `8Ufu7vkAAAAJ:W7OEmFMy1HYC` (2020 arXiv:2011.14182) | `8Ufu7vkAAAAJ:WF5omc3nYNoC` (2021 *Le Matematiche*) | `10.4418/2021.76.2.10` |
| Christopher Eur | `8Ufu7vkAAAAJ:_FxGoFyzp5QC` (2022 JMM manuscript/booklet) | `8Ufu7vkAAAAJ:0EnyYjriUFMC` (2024 *Bulletin AMS*) | `10.1090/bull/1803` |
| Michael Young | `l0rzNYsAAAAJ:d1gkVwhDpl0C` (Scholar says 2012) | `l0rzNYsAAAAJ:aqlVkmm33-oC` and `l0rzNYsAAAAJ:W7OEmFMy1HYC` (two observations of the 2017 journal article) | `10.1007/s10878-016-0103-z` |

For Eur's first pair, [arXiv:2011.14182](https://arxiv.org/abs/2011.14182)
and the [journal article](https://lematematiche.dmi.unict.it/index.php/lematematiche/article/view/2256)
give the same title and four authors; the journal PDF assigns the DOI and 2021
volume and pages. For his survey, [Eur's CMU publication page](https://www.math.cmu.edu/~ceur/notes.html)
identifies the work as prepared for the JMM Current Events Bulletin and then
published in *Bulletin AMS*. The [AMS meeting booklet](https://www.ams.org/meetings/lectures/2023-Booklet-Master-EBOOK.pdf)
and [2024 journal article](https://www.ams.org/journals/bull/2024-61-01/S0273-0979-2023-01803-X/S0273-0979-2023-01803-X.pdf)
carry the same essay and abstract under revised titles. The earlier JMM row
uses the alias schema's `preprint_of` relationship for that earlier text, even
though it also circulated in the meeting booklet; the 2024 journal row supplies
the canonical year and venue.

For Young, [arXiv:1512.06413](https://arxiv.org/abs/1512.06413) carries the
earlier row's title and four authors. The [Springer article](https://link.springer.com/article/10.1007/s10878-016-0103-z)
has those authors and the same abstract under its revised journal title, volume
34, pages 736–741 (2017). The two 2017 Scholar observations already share the
same normalized title and year; both receive the DOI alias so their shared work
also joins the earlier-title row. Scholar's `2012` year and `Appl. Math` venue
for that row have no independent source confirmation and remain literal raw
observations; the canonical work uses the journal's 2017 year and venue.

Three exact metadata repairs use source guards. Eur's
`8Ufu7vkAAAAJ:hqOjcs7Dif8C` has an appended `: C. Eur` author fragment in
the title; [Springer's version of record](https://link.springer.com/article/10.1007/s00029-024-00979-7)
has the title without it. Young's `l0rzNYsAAAAJ:ufrVoPGSRksC` and
`l0rzNYsAAAAJ:mVmsd5A6BfQC` both cite arXiv:1803.01933, but one has
`1803` as its year and the other preserves the mathematical expression as a
literal `\times` from the saved SVG aria-label. [The arXiv record](https://arxiv.org/abs/1803.01933)
dates the work to 2018 and gives its full title and authors. Effective
corrections give these
two rows the same complete plain-text title and 2018 year, so the existing
normalized title/year rule joins them without a false journal DOI.

The exact same-title/year journal pairs for Young's 2014 principal-rank paper
and 2015 signed-Petersen paper already collapse under the producer's ordinary
identity rule. No extra aliases are added for those pairs. Young's
`l0rzNYsAAAAJ:UeHWp8X0CEIC` [Scholar observation](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=l0rzNYsAAAAJ&pagesize=100&citation_for_view=l0rzNYsAAAAJ:UeHWp8X0CEIC)
literally identifies an IMA research-group **proposal** and has no year or
publication venue. It receives one exact `nonpublication` exclusion from the
map while remaining in the raw Scholar capture. No exact public proposal
document was found. The distinct [2017 AIM workshop report](https://aimath.org/pastworkshops/zeroforcingrep.pdf)
has different organizers and is not evidence for the IMA proposal. Other
near-title pairs remain untouched.

These aliases are release-gated: the currently published snapshot lacks the
Eur and Young citation IDs. Trusted upload must validate them against a new
snapshot built from the complete Scholar source captures after PR #226's source
release. Do not publish this producer manifest against the earlier snapshot.
