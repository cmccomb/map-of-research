# Avigad and Boffi whole-profile source review

The guarded 2026-10-24 direct-profile captures contain all **260 Avigad** and **48 Boffi** Scholar rows visible when `Show more` disabled. The paired redacted HTML and structured JSON, canonical author caches, and SHA-256 receipts are staged in a separate draft source PR for the next snapshot cycle. No row is erased from raw source evidence.

## Exact Avigad exclusions proposed

The accompanying manifest adds **16 exact citation IDs**, rather than a broad title or venue rule:

- **11 NSF award records.** The [NSF award API](https://resources.research.gov/common/webapi/awardapisearch-v1.htm) returns matching project titles and award IDs for the eleven Scholar rows labeled `NSF Award`: 2434614, 2229099, 1615444, 1068829, 0937208, 0700174, 0713945, 0612754, 0401042, 0070600, and 9614851. The manifest links each individual primary record. These are funding records, even where Avigad is the principal investigator.
- **Four false profile attributions.** [Stanford's *Concepts* entry](https://plato.stanford.edu/entries/concepts/) credits Eric Margolis and Stephen Laurence, [Dagstuhl's SAT 2023 volume](https://drops.dagstuhl.de/entities/volume/LIPIcs-volume-271) credits editors Meena Mahajan and Friedrich Slivovsky, [Dagstuhl's ITP 2019 volume](https://drops.dagstuhl.de/entities/volume/LIPIcs-volume-141) credits John Harrison, John O'Leary, and Andrew Tolmach, and [Cambridge's winter-meeting report](https://www.cambridge.org/core/journals/bulletin-of-symbolic-logic/article/abs/20102011-winter-meeting-of-the-association-for-symbolic-logic/B67DD3C47BC2D614836168A0650D2709) credits Michael Mislove. None credits Avigad as the author/editor of the captured work.
- **One issue-front-matter record.** Cambridge places `RSL volume 4 issue 2 Cover and Front matter` in the [Front Cover and matter section](https://www.cambridge.org/core/journals/review-of-symbolic-logic/issue/7F1227F3183293BA74805866F5E137C8), pages f1–f4. Scholar lists editorial names, including Avigad, as its authors; it is not his research publication.

These decisions remove the identified rows from *mapped works after a later snapshot rebuild*. The 260 raw profile rows remain available for audit, including their source titles and citation IDs.

## Entries retained or unresolved

Avigad rows 54, 136, 146, 154, 166, 168, and 245 have displayed author strings without `Avigad`. Some are conference-volume containers; others may have truncated contributor lists or a real editorial role. The initial review does not apply an exclusion to any of these seven until the work's primary record is checked. Scholar also omits years on 19 Avigad rows and venue text on 60; those literal gaps remain in the source cache.

Boffi rows 11, 13, 18, 25, 45, and 46 omit `Boffi` only because Scholar truncates the displayed author list. [Boffi's own publication page](https://nmboffi.github.io/) names him on all six. They remain in the cache and are not proposed for exclusion. No Boffi row has a missing displayed year or venue.

This review covers visible Scholar profiles on the capture date, not a proof that either profile indexes every work ever produced. The new 308 observations are outside the frozen 43,701-row combined release.
