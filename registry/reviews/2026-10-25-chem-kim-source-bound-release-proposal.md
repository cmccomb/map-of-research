# Hyung J. Kim: complete Scholar capture and held release proposal

This is a **nonoperational review proposal**. Hyung J. Kim remains off the
public map. The complete browser capture from source PR #118 has 918 distinct
Scholar citation IDs and no remaining profile rows. Its unchanged JSON SHA-256
is `ea04610e120039df7aaf83c402d193021f328b319cd41cfabf541f3942069867`;
the paired redacted HTML SHA-256 is
`a5f1a45612a885ca42174926f3c43bc437fd7146da4b92f241930e6adfd9bc94`.
The capture is a complete **visible Scholar profile**, not proof that every
publication in Kim's academic history appears there.

## Release blocker measured against the current publisher

At producer head `fa8b087`, before the latest row decisions, I converted the saved 918 rows to the ordinary
snapshot shape in memory, added inert embedding vectors, and called
`build_dataset_tables` with the existing `scholar-exclusions.json`. This made
**409 of 410 held observations map eligible**. The one exception,
`bC4aq-4AAAAJ:KaMxkj08jr0C` (“I-SEEC 2012”), was rejected by the generic
`event_or_session_label` quality rule. All 394 exact exclusions were
suppressed, and all 114 positive person signals were eligible. The publisher
currently has no hold set, so installing the complete cache before a verified
hold gate would put unresolved attributions into graph works and edges.

## Conservative exact-ID proposal

The [prepared release fragment](proposals/scholar-profile-releases.kim-proposal.json)
and [122 source confirmations](proposals/scholar-attribution-reviews.kim-proposal.json)
partition every captured ID as follows:

| Disposition | IDs | Sorted-ID SHA-256, joined by newline without a final newline |
| --- | ---: | --- |
| Confirmed source-backed works for a possible reviewed subset | 122 | `7ed2f1779372ab3cf3060b8014e96ec62780eb058e9499ff8b4b7fc15bd2b101` |
| Exact false-attribution, duplicate, or nonpublication exclusions | 459 | `ab551e678f7b05ad85ff459717e35b9aabebc3e0d6748857041e3b822a47dc8a` |
| Held from any map release | 337 | `7023441ccbde8f48e1ad1be39019025805beef152003226fefe43db8ced8d46c` |

The 337 holds comprise observations that still lack enough output-level review
for this conservative subset. The 122 confirmed entries have exact Scholar
IDs and work-specific publisher or CMU evidence in the linked fragment and in
the [historical](2026-10-25-chem-kim-historical-primary-retentions.md),
[group bibliography](2026-10-25-chem-kim-lab-bibliography-follow-up.md), and
[five latest](2026-10-25-chem-kim-five-primary-retentions.md) review notes,
plus the [first eight CMU group matches](2026-10-25-chem-kim-group-eight-primary-retentions.md)
and [ten further group/publisher matches](2026-10-25-chem-kim-group-ten-primary-retentions.md),
followed by [eleven AIP/PLOS/BKCS matches](2026-10-25-chem-kim-group-eleven-primary-retentions.md)
and [ten recent CMU faculty/publisher matches](2026-10-25-chem-kim-faculty-ten-primary-retentions.md),
followed by [twelve systematic official-bibliography/publisher matches](2026-10-25-chem-kim-official-bibliography-twelve-primary-retentions.md)
and [28 full-name publisher matches with CMU or matching ORCID identity](2026-10-25-chem-kim-publisher-fullname-twentyeight-primary-retentions.md),
then [the final 20 exact CMU ORCID signals with original publisher and institutional evidence](2026-10-25-chem-kim-remaining-twenty-primary-retentions.md)
and [seven chapters/proceedings outputs plus two explicit namesakes](2026-10-25-chem-kim-chapters-namesakes-primary-review.md).
Five additional [Hyojoon Kim/Kapral exact outputs](2026-10-25-chem-kim-hyojoon-kapral-five-primary-review.md)
have full-name publisher or institutional identity evidence and are excluded
from CMU Kim by exact Scholar ID.
Seven further [Hyung Joon Kim neuroscience outputs](2026-10-25-chem-kim-hyung-joon-neuro-seven-primary-review.md)
have original full-name bylines or exact coauthor-lab bibliography records;
these are also excluded only by their exact Scholar IDs.
For the [1999 lung adenocarcinoma row](2026-10-25-chem-kim-lung-absent-author-primary-review.md),
NLM's complete original-journal byline contains no H. J. Kim at all, so that
exact citation ID is excluded as an absent-author attribution.
Three more [JBC and KSTAR output checks](2026-10-25-chem-kim-kstar-jbc-three-primary-review.md)
exclude one full-name namesake, one article with no H. J. Kim in its complete
147-author list, and one Haejin Kim fusion-institute article by exact ID.
Three [KIST fuel-cell team comparisons](2026-10-25-chem-kim-kist-fuel-cell-three-primary-review.md)
exclude exact initials-only articles by a documented full-name coauthor/team
bridge to Hyoung-Juhn Kim; this inference does not extend to the remaining
fuel-cell holds.
Eight [KNU person-linked scintillator records and two renderings of one
Springer erratum](2026-10-25-chem-kim-scintillator-erratum-ten-primary-review.md)
identify Hong Joo Kim, not CMU's Hyung J. Kim, by exact work metadata. The
erratum is excluded for wrong-person attribution; corrections are not
categorically treated as nonpublications.
[Five original US patent/application front pages](2026-10-25-chem-kim-five-patent-inventors-primary-review.md)
expand five saved initials-only inventor bylines to other full names and exclude
only those exact Scholar IDs.
[Six original publisher or publisher-deposited full-author records](2026-10-25-chem-kim-six-full-author-sensor-community-review.md)
identify Hyeonsook Kim on one IEEE output and Hyung Joon Kim on five KIST
sensor outputs. The six exact Scholar IDs are excluded; neighboring
initials-only records stay held.
[Eleven further exact publisher, society, and institutional author records](2026-10-25-chem-kim-eleven-full-author-primary-review.md)
identify Ho Jin, Hee Joung, Hae Jin, HyeJung, Hee-Jin, and Hyun-Jung Kim
on other works; only those eleven exact Scholar IDs are excluded.
[Six further exact-work author checks](2026-10-25-chem-kim-six-exact-author-primary-review.md)
identify Hong Joo, Ho Jin, Han-Joon, and Hwi Kim on four outputs, and rule out
the CMU chemist from two complete author lists. Only the six exact citation
IDs are newly excluded; neighboring initials-only works remain held.
[Seven further exact author and duplicate checks](2026-10-25-chem-kim-seven-exact-author-and-duplicate-review.md)
confirm one published Biophysical Journal abstract for CMU Kim, exclude three
Ho Jin/H. Jin Kim works, and remove three redundant Scholar renderings of
already confirmed CMU outputs by exact DOI and citation ID.
Published ACS/RSC correction notices at rows 643, 709, and 796 have their
own DOI/page evidence and remain distinct works. The 122 confirmed rows do
**not** represent Kim's whole publication history.
The [additional high-citation check](2026-10-25-chem-kim-three-high-impact-holds.md)
keeps three more initials-only records held despite suggestive related sources.
A [separate high-citation QC pass](2026-10-25-chem-kim-high-citation-holds-qc.md)
documents 14 exact works that still lack a direct full-name attribution.

The offline [preparer](prepare_chem_kim_reviewed_subset.py) recreates a
candidate `data/authors/bC4aq-4AAAAJ.json`-shaped cache and both proposal
fragments from the saved capture and review ledger. The candidate cache is
kept outside canonical `data/authors` and has byte SHA-256
`f4b7b37efcd3310a05ff82057488ecb286450e3db0dc390b4e26b125fc378b2e`.
A no-upload snapshot build/validation from that candidate retained 918 Kim
Scholar rows plus 18 existing curated observations, with 918 unique Kim IDs.
Its `browser_capture.coverage` marker protects the complete capture from a
later bounded collector refresh.

To reproduce locally with the verified source file:

```sh
python3 -m registry.reviews.prepare_chem_kim_reviewed_subset \
  --capture /path/to/kim-visible-918-2026-10-24.json \
  --output-dir /tmp/kim-review-proposal
```

Before any public use, a trusted producer gate must compare the cache SHA-256,
all 918 exact IDs, the 122 confirmed IDs, and the 459 exact exclusion IDs to
these lists; it must prevent all 337 held rows from creating works,
authorships, or coauthor edges. Even after that guard exists, Kim should stay
off public maps until the subset is clearly represented as incomplete or the
remaining outputs receive work-level review. This proposal does not edit the
canonical cache, publisher manifests, snapshot, or public map.
