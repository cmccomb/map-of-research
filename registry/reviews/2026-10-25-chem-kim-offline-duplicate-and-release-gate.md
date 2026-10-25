# Hyung J. Kim: one exact duplicate and remaining release gate

This review uses the unchanged, complete visible 918-row Google Scholar capture
for `bC4aq-4AAAAJ` in draft source PR #118. Its JSON SHA-256 is
`ea04610e120039df7aaf83c402d193021f328b319cd41cfabf541f3942069867`;
the paired redacted HTML SHA-256 is
`a5f1a45612a885ca42174926f3c43bc437fd7146da4b92f241930e6adfd9bc94`.
The receipt records exhausted Show More, no visible page error, and no canonical
Kim cache. This follow-up made no Scholar request and preserves all raw rows.

## Exact duplicate decision

Capture row 795, `bC4aq-4AAAAJ:pUxgyZctzPYC`, repeats the title
"Muon detector for the COSINE-100 experiment," the 2018 year, *Journal of
Instrumentation* volume 13 issue 2, and its leading Prihtiadi/Adhikari
authors from row 156, `bC4aq-4AAAAJ:hNSvKAmkeYkC`. Row 795 omits the article
number and prepends the collaboration name. The [KNU institutional article
record](https://research.knu.ac.kr/en/publications/muon-detector-for-the-cosine-100-experiment/)
identifies the one article as T02007 and links the H. J. Kim author to
Kyungpook Physics professor Hong Joo Kim. Row 156 was already excluded as a
false attribution. Mark row 795 as an exact `corrupted_duplicate`; retain
both raw observations and the earlier row 156 decision. The rebuild pins row
795's Scholar ID, title, year, issue, leading authors, and row 156 ID.

## Complete capture ledger after this decision

| Disposition | Rows | Release meaning |
| --- | ---: | --- |
| Exact exclusions | 388 | 8 baseline plus 380 proposed exact IDs; only row 795 is new here. |
| CMU-linked ORCID title/year signals | 99 | Positive person signal, subject to output review. |
| Other CMU/publisher positive signals | 10 | Source-specific positive person signal. |
| Provisional attribution/output holds | 421 | No person or publication decision implied. |
| **Raw Scholar observations** | **918** | **No Kim cache, snapshot, or map release.** |

The 421 holds consist of 416 unmatched rows, two KNU ORCID matches lacking a
DOI, two KNU ORCID year discrepancies, and one initials-only supergravity
paper imported into the CMU-linked ORCID. In the held set, 29 rows display at
least 50 citations; 12 lack a year, 19 lack a venue, and 11 have a repeated
normalized title in the capture. These counts overlap. Their subject areas
and truncated bylines are screening clues only.

Other repeated-title pairs stay held. The 2020 arXiv/2021 journal COSINE
paper, the 2024 arXiv/2025 journal background-model paper, and the patent
application/grant pair may represent distinct outputs or versions. Rows 336
and 793 share journal metadata, but no independent article-number or DOI
record was available in this offline pass. None gets a duplicate decision by
title alone.

## Gate before any Kim release

Keep the canonical cache absent and the profile off public maps until the
producer hold gate can pin the 918-row source hash, the then-current exact
exclusion IDs, and all unresolved citation IDs. A future release should
retain 918 raw observations, prevent every held and excluded ID from adding
mapped works/authorships/coauthor edges, and label the remaining positive
rows as reviewed only to the strength of their source evidence. Reconcile
these draft decisions with producer main after the Math upload freeze; do not
infer that the 109 positive signals cover Kim's complete academic history.

Priority for the next offline or bounded primary-source pass: resolve the
four KNU ORCID edge cases by original author-linked article records, the
initials-only supergravity attribution by a full author identity record,
then the 29 high-citation held rows and repeated-title pairs. No topic-based
or initials-only bulk exclusions are justified.

## Title-variant leads from the saved ORCID responses

An offline similarity screen of the held Scholar titles against the saved
CMU-linked and KNU-linked ORCID work summaries found the following *leads*.
None changes the ledger, because the source titles differ, a Scholar title is
truncated, or a distinct correction/version may exist. An original article's
full author identity and output type remain to be checked before a decision.

| Scholar row and exact ID | ORCID work lead | Remaining check |
| --- | --- | --- |
| 480 `bC4aq-4AAAAJ:R3hNpaxXUhUC` | CMU ORCID put-code `28834308`, DOI `10.1063/1.473606`, 1997; title spells part one as `.1.` rather than Scholar's `. I.`. | Publisher full Hyung J. Kim byline and article match. |
| 587 `bC4aq-4AAAAJ:u0Mu_IsstPMC` | CMU ORCID Crossref-supplied put-code `170388136`, DOI `10.1021/jasms.1c00346`, 2022; Scholar truncates the title. | Publisher full author and affiliation record. |
| 388 `bC4aq-4AAAAJ:sszUF3NjhM4C` | KNU ORCID Scopus-supplied put-code `153825460`, DOI `10.1109/TNS.2008.920252`, 2008; Scholar drops subscript and decay notation. | Original IEEE/KNU author-linked work record. |
| 88 `bC4aq-4AAAAJ:oi2SiIJ9l4AC` | KNU ORCID Scopus-supplied put-code `153825511`, DOI `10.1016/j.jallcom.2018.03.266`, 2018; Scholar truncates the title. | Original article's full author identity. |
| 669 `bC4aq-4AAAAJ:SAZ1SQo2q1kC` | KNU ORCID Scopus-supplied put-code `153825155`, DOI `10.1016/j.jlumin.2019.116815`, 2020; Scholar truncates the title. | Original article's full author identity. |

The correction rows 643, 709, and 796 resemble underlying CMU ORCID papers,
but a correction is a distinct record. Their own DOI/byline must be reviewed;
they are not duplicates of the parent papers by title resemblance alone.
