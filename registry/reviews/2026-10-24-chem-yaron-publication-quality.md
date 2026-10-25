# David Yaron Scholar capture: offline publication quality review

Reviewed 2026-10-24 from the saved, redacted 242-row capture for Scholar ID
`HMSZExQAAAAJ`. The capture reports `more_remain: false`, no challenge, and no
page error. The paired JSON and HTML hashes, every exact `author_pub_id`, the
reason for each decision, a primary evidence URL, and retained parent IDs are
in [the structured ledger](2026-10-24-chem-yaron-publication-quality.json).
The HTML has 242 row elements and its 242 citation IDs exactly match the JSON.
The evidence review made no Google Scholar request. Publisher and CMU pages
were checked for the final duplicate decision. This PR stages only
exact graph decisions; it does not edit the cache or a release snapshot.

## Staged exact graph exclusions

| Reason | Count | Exact captured row indices | Evidence |
| --- | ---: | --- | --- |
| NSF awards, not publications | 20 | 121, 122, 125, 136, 145, 146, 149, 153, 158, 167, 173, 174, 180, 187, 189, 194, 201, 203, 204, 211 | Each captured title was checked against its matching [NSF awards API](https://api.nsf.gov/services/v1/awards/2337209.json); the JSON ledger contains the exact award URL for every row. Row 136 is Linda Peteanu's award. |
| Other nonpublications | 4 | 190, 233, 234, 235 | CMU lists the parent of `Section III. B` as [Arindam Chowdhury's dissertation](https://www.chem.cmu.edu/alumni/phd/doc_degr/doc_degr-recipient.pdf). Two rows are [LearnLab project/wiki pages](https://learnlab.org/mediawiki-1.44.2/index.php?title=Stoichiometry_Study), which themselves cite the formal paper. One row is [supporting information](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00491) for a retained ACS article. |
| False Yaron authorship | 1 | 20 | The exact [ACM journal article](https://www.research.ed.ac.uk/en/publications/plan-recognition-and-visualization-in-exploratory-learning-enviro/) has only Amir and Gal; a separate same-titled book chapter includes Yaron. |
| Corrupted duplicate records | 11 | 112, 113, 114, 128, 129, 199, 207, 232, 236, 239, 242 | Each maps to a fuller retained record in the same 242-row capture. Examples include an [RSC article](https://pubs.rsc.org/en/content/articlehtml/2018/me/c7me00131b) repeated with no year and just `D Yaron`, the [Wiley article](https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/chem.202300477) repeated as the title `Chemistry—A European`, and an [ACS paper](https://pubs.acs.org/doi/10.1021/ed5002009) repeated with website terms-of-use text prefixed. For row 242, [Yaron's CMU bibliography](https://www.chem.cmu.edu/groups/yaron/pub.html) identifies its short title and five authors as the [2003 ACS article](https://pubs.acs.org/doi/10.1021/jp0272769) already captured at row 30. |

Total: **36 staged exclusions**: 24 `nonpublication`, 1
`false_profile_attribution`, and 11 `corrupted_duplicate`. The raw capture
must retain all 242 observations; these are graph eligibility decisions.

## Deferred and retained

Eleven rows remain deliberately deferred in the JSON ledger. In particular,
row 89 is a generic 1985 symposium title; rows 109 and 110 may be related
preprints; rows 200 and 202 have the same title in different meeting venues;
and undated rows 238 and 241 need source reconciliation. Row 237, a 2015
conference talk, has an ambiguous captured author string; an old CV names its
presenter but does not establish its full byline. Keep it until the original
abstract or program can resolve Yaron's authorship. Row 116 has a
malformed year (`6`) but appears to be a genuine book chapter; correct its
metadata only after checking a primary book record. The 2024 ChemRxiv row 120
and 2025 journal row 85 have similar titles but are kept separate pending the
preprint policy.

Do not use a truncated Scholar author cell to discard education or computing
work. CMU's [original handheld-device report](https://www.cs.cmu.edu/~pebbles/papers/CMU-CS-00-152.pdf)
names Yaron despite his absence in the captured short author cell. The
[ChemCollective publication list](https://chemcollective.org/about_us/research)
and ACS issue pages similarly confirm Yaron on additional learning-science
papers, including [ORCCA Tutor](https://pubs.acs.org/toc/jceda8/99/2) and
[Making Lemonade out of Lemons](https://pubs.acs.org/toc/jceda8/97/9).

The 36 exact graph exclusions are staged in the producer manifest. The normalized
242-row cache preserves every ID and retains 11 distinct parent records cited
by 14 exclusion decisions. The raw
capture remains staged outside this PR; release it only with its source receipt,
then verify that excluded observations contribute no graph authorship.
