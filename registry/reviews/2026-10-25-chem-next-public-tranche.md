# Chemistry next public tranche: producer readiness

Reviewed against producer `main` `71d45720`, the last successful shared source snapshot `8477a325`, and its completed dataset upload receipt (`f194bdab`). This note records a producer candidate and does not authorize an upload.

## Complete-profile coverage

The [Chemistry roster recheck](2026-10-25-chemistry-roster-recheck.md) contains 38 included faculty, teaching faculty, and emeriti: 23 verified Scholar IDs and 15 unresolved. The shared source snapshot contains a `complete_visible_profile` cache for **21 of those 23 verified IDs**. This includes Kietrys (36 captured rows), Gil (195), Bernhard (228), Collins (361), and the other 17 verified researchers with source files. The [last successful upload receipt](https://github.com/cmccomb/map-of-research/blob/8477a325/status/last-successful-upload.json) pins that source snapshot; this review found no newer author cache on the shared source branch. The 21 are therefore already in the publication source, subject to their ordinary exact exclusions, provisional holds, and profile-release decisions. An additional producer release entry for those ordinary profiles would not supply new observations.

| Missing verified-ID profile | Current evidence | Gate |
| --- | --- | --- |
| Hyung J. Kim, `bC4aq-4AAAAJ` | A separately staged [918-row raw capture and initial review](2026-10-24-chem-kim-collins-exclusions.md) exist. The canonical author cache is withheld while attribution review continues. The open [Kim exact-source draft #394](https://github.com/cmccomb/map-of-research/pull/394) reports 123 confirmed, 548 proposed or prior exclusions, and 247 held, all in an inactive proposal. | Keep all 918 rows outside the shared snapshot and public works until the remaining holds and whole-profile source gate are resolved. |
| Subha R. Das, `FrF1YXQAAAAJ` | A complete 670-row canonical cache is staged on the [held source branch](https://github.com/cmccomb/map-of-research/tree/codex/chem-das-complete-held-source-20261025), separate from the shared snapshot. The local `codex/chem-held-consolidation-das-20261025` review ledger partitions 68 confirmed, 488 excluded, and 114 held. | Keep the whole profile held; 114 unresolved rows and a source/producer gate remain. The local review branch is not part of this PR. |

## Smallest safe producer change

The accompanying [official-source identity review](2026-10-25-chem-unresolved-identity-source-batch.md) adds typed Chemistry display homes for the 15 unresolved-ID people and five CMU-linked ORCIDs. It assigns **zero** unverified Scholar IDs and changes **zero** publication observations, works, authorship links, or graph edges. Expected new visible works and connections from this draft: **0**. It improves affiliation evidence for existing roster people; a public dataset rebuild would still require the normal release checks and a separately authorized upload.

The next publication gain depends on a complete, source-bound profile outside the 21 already present. Kim and Das are the only verified-ID Chemistry profiles missing from the shared snapshot, and both have unresolved attribution gates. A bounded exact-source review may reduce those holds, but neither current draft supports a partial public release. The 15 without verified Scholar IDs cannot enter a Scholar-grounded publication tranche until a direct source-linked identity and full visible-profile capture are available.
