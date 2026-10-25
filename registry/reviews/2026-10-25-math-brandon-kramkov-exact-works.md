# Deborah Brandon and Dmitry Kramkov: exact Scholar work review

Reviewed 2026-10-25 against the complete, source-only Scholar captures in
draft PR #227 at commit `190abc9`. Brandon has 23 visible rows and Kramkov has
33; every citation ID is distinct. Both captured pages have disabled “Show
more,” no visible challenge or error, and two redacted XSRF values. The 56
structured rows match the saved HTML, including Kramkov's restored row-21
mathematical title. The source branch preserves every observation. This
producer review changes only exact, effective eligibility and DOI identity.

## Published versions of the same work

| Researcher | Earlier or duplicate Scholar ID | Journal or proceedings Scholar ID | DOI |
| --- | --- | --- | --- |
| Brandon | `9eegHpAAAAAJ:2osOgNQ5qMEC`, `9eegHpAAAAAJ:YsMSGLbcyi4C` (1989 manuscript and CMS Technical Summary Report #89-37) | `9eegHpAAAAAJ:u-x6o8ySG0sC` (1991 *SIAM Journal on Mathematical Analysis*) | `10.1137/0522005` |
| Brandon | `9eegHpAAAAAJ:roLk4NBRz8UC` (SPIE [2192-08] duplicate) | `9eegHpAAAAAJ:LkGwnXOMwfcC` (1994 SPIE proceedings) | `10.1117/12.174243` |
| Brandon | `9eegHpAAAAAJ:WF5omc3nYNoC` (SPIE [1919-13] duplicate) | `9eegHpAAAAAJ:9yKSN-GCB0IC` (1993 SPIE proceedings) | `10.1117/12.148401` |
| Kramkov | `InIipCMAAAAJ:zYLM7Y9cAGgC` (2006 manuscript with truncated title) | `InIipCMAAAAJ:d1gkVwhDpl0C` (2007 *Stochastic Processes and their Applications*) | `10.1016/j.spa.2007.04.014` |
| Kramkov | `InIipCMAAAAJ:5nxA0vEk-isC` (2011 arXiv:1110.3248, earlier title) | `InIipCMAAAAJ:u5HHmVD_uO8C` (2014 *Stochastic Processes and their Applications*) | `10.1016/j.spa.2013.06.017` |

The [original CMS report](https://citeseerx.ist.psu.edu/document?doi=fce486d353653e485b17ebb74c2adba399d7080d&repid=rep1&type=pdf)
has Brandon's exact title, author, report number, and June 1989 date. The
[SIAM version of record](https://epubs.siam.org/doi/abs/10.1137/0522005)
has the same author, title, and research abstract, with 1991 volume 22,
pages 72–106. The two SPIE DOI records, [1994](https://api.crossref.org/works/10.1117/12.174243)
and [1993](https://api.crossref.org/works/10.1117/12.148401), were deposited
by the publisher and give the exact author teams, titles, proceedings volumes,
and first pages. The bracketed Scholar titles repeat those SPIE paper numbers;
they do not establish new research works.

Kramkov's [CMU-hosted 2006 manuscript](https://www.math.cmu.edu/~bachelier/seminar_6f/Kramkov.pdf)
and the [2007 Elsevier paper](https://www.sciencedirect.com/science/article/pii/S0304414907000932)
share the complete title and author team; the earlier Scholar row clips the
last word. [Kramkov's CMU publication list](https://www.math.cmu.edu/~kramkov/publications.html)
connects the earlier title to arXiv:1110.3248; the
[arXiv record](https://arxiv.org/abs/1110.3248) explicitly links the later
[2014 Elsevier paper](https://www.sciencedirect.com/science/article/pii/S0304414913001877)
by DOI and journal reference, with Kramkov and Predoiu as authors. All 11
observations get exact DOI aliases, while the later year and venue represent
each published work. The version-of-record title-selection code is staged
separately in producer PR #231; release this review after that change.

## Exact records kept out of the publication graph

The [NSF award API record 9103199](https://api.nsf.gov/services/v1/awards/9103199.json)
identifies Brandon's “Mathematical Sciences: Nonlocal Models in Continuum
Mechanics” row `9eegHpAAAAAJ:MXK_kJrjxJIC` as an award. The corresponding
[award 0505414](https://api.nsf.gov/services/v1/awards/0505414.json) identifies
Kramkov's “Topics in Mathematical Finance” row
`InIipCMAAAAJ:4TOpqqG69KYC`; [award 0353556](https://api.nsf.gov/services/v1/awards/0353556.json)
identifies the GOALI fellowship row `InIipCMAAAAJ:qxL8FJ1GzNcC`. Each is
excluded as an exact `nonpublication`, with its raw Scholar row retained.

Three Kramkov rows are presentations rather than research publications. The
[2014 slide deck](https://www.hse.ru/data/2014/07/15/1312438456/BMR_Radner.pdf)
itself says “Backward martingale representation and the existence of
endogenously complete equilibria” is a presentation based on two separately
identified papers; this is `InIipCMAAAAJ:qjMakFHDy7sC`.
[MathNet's 2011 seminar page](https://www.mathnet.ru/eng/present3343) identifies
“A model for price impact” as a Kramkov talk, row
`InIipCMAAAAJ:Zph67rFs4hoC`. Its
[Kramkov events list](https://www.mathnet.ru/eng/person23594) identifies
“Utility-based methods in Mathematical Finance. Lecture 4,” row
`InIipCMAAAAJ:KlAtU1dfN6UC`, as a 2015 school video lecture. These three
exact rows are also excluded from the publication graph, while the captured
observations remain available for review.

The Brandon viscoelasticity rows `9eegHpAAAAAJ:5nxA0vEk-isC` and
`9eegHpAAAAAJ:IjCSPb-OGe4C` refer to a final technical report whose author
list includes Brandon. Their bibliographic relationship and publication
eligibility need more direct document-level review; both stay untouched. The
Kramkov row `InIipCMAAAAJ:0EnyYjriUFMC` is a separate 2015 research paper;
it is not collapsed with the 2014 presentation. Other similar titles without
exact version evidence remain unchanged.

The currently published snapshot lacks the Brandon and Kramkov citation IDs.
Trusted upload must validate these exact aliases and exclusions against a new
snapshot built from PR #227's complete source captures. Do not publish this
producer manifest against the earlier snapshot.
