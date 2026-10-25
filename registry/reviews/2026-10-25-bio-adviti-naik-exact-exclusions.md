# Adviti Naik visible Scholar profile: exact-row review — 2026-10-25 UTC

This offline review uses the saved 44-row Google Scholar capture for
`lKAvvmoAAAAJ`, its capture receipt, [CMU's identity record](https://scholars.cmu.edu/13090-adviti-naik),
and original publisher records. I did not reopen Scholar. The receipt reports
44 displayed rows, 44 unique citation IDs, one “Show more” click, no rows
remaining, and no challenge. The saved JSON SHA-256 is
`16fa11dd8b2d519670a7c94582b2c6ea9c751afb11c1269f4cf90de122e6a723`.
This is complete coverage of the **visible profile at capture time**, not proof
that Scholar contains every publication in Naik's academic history.

## Exact decisions

Preserve **all 44 raw observations**. Exclude only these two exact Scholar IDs
from map authorships. Both are corrupt duplicate observations of complete
articles retained on the same profile.

| Row · exact ID suffix | Code | Original-source evidence |
| --- | --- | --- |
| 25 `ZeXyd9-uunAC` | `corrupted_duplicate` | The row repeats the beginning and author list of retained row 15 but ends the title after “Neuropilin-1” and places “PlGF, Neuropilin-1: A Promising Therapeutic Target…” in its venue field. [Frontiers](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2019.00323/full) identifies one complete 2019 article by Al-Zeheimi, Naik, and colleagues, DOI `10.3389/fonc.2019.00323`. The latter venue text comes from a distinct [IntechOpen chapter](https://www.intechopen.com/chapters/1197987) published in 2024 by Al-Zeheimi and Adham, without Naik. Retain row 15 `ufrVoPGSRksC`. |
| 39 `YsMSGLbcyi4C` | `corrupted_duplicate` | It gives only the first part of row 14's title, the same 2014 year, and the same three authors, with no venue. [PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003993) gives one complete article, DOI `10.1371/journal.pcbi.1003993`. Retain row 14 `d1gkVwhDpl0C`. |

## Retained observations and limits

The other 42 exact IDs have no demonstrated false attribution or nonpublication
in this pass. They remain Scholar-grounded observations, not 42 independently
verified distinct journal papers. In particular:

- Row 26 is Naik's separate 2013 [doctoral dissertation](https://d-nb.info/1053779119/04), not a duplicate of the later PLOS paper.
- Row 27 appears in the [MATHMOD 2012 proceedings](https://www.asim-gi.org/fileadmin/user_upload_argesim/ARGESIM_Publications_OA/MATHMOD_Publications_OA/MATHMOD_2012_AR38/AR38_AM142_MATHMOD_2012_Abstract_Volume.pdf), although Scholar's compressed author string needs a metadata correction before being called canonical.
- Row 37 is a 2021 bioRxiv version related to row 9's 2022 journal article. A separate version is retained under the existing source policy.
- Rows 28 and 29 are separately listed by the [IJMS publisher](https://www.mdpi.com/1422-0067/27/18/8127) and [AACR](https://aacrjournals.org/cancerres/article/86/14_Supplement/A015/786444/Abstract-A015-Targeting-polyamine-metabolism-to), respectively. The AACR item is an abstract, not the journal review.
- Rows 42–44 have no Scholar year; rows 43–44 resemble symposium abstracts. No original proceedings evidence established that any is a duplicate or false attribution, so none is excluded. Their missing years should remain visible as source metadata gaps.

No raw title, author string, venue, year, citation count, or row was changed.
These two exclusion decisions append to the 872-entry main manifest after the
Ihab, Mohamed, and Nesrine reviews, yielding 874 exact entries. Later Bio review
branches must preserve all earlier entries during sequential integration. This
review is separate from the raw-source capture and should remain draft until
trusted upload and both consumer pins are verified.
