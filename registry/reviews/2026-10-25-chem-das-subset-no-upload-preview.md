# Subha R. Das: local reviewed-subset preview, no upload

This is a safety preview of the separate [reviewed-subset proposal](2026-10-25-chem-das-reviewed-subset-proposal.json). The trusted upload manifest still has Das in `state: held`. The preview used draft source branch `codex/chem-das-complete-held-source-20261025` at `9f6b0b6`, which carried the **earlier 51,139-row baseline**. Building its staged 670-row Das cache produced 51,809 local raw rows. This source predates the settled Bernhard and Sullivan Chemistry releases, so these totals are provisional and must be rerun on the later source head. The Das cache SHA-256 was `85700e7ee25079ed30633df91137e84cc0ef6c0e6628166f80acf4c133f4c1df`; the new local raw snapshot SHA-256 was `1c2edd3277809c753abe91805bd9526b37caddbd688557a4d68078f97ab9e2b0`. The raw snapshot validated with its source registry and curated works. Neither source nor producer draft was merged or uploaded.

The producer loaded the exact positive attribution, exclusion, and proposed three-way release manifests; validated the subset partition against the 670 cache IDs; applied metadata corrections; then filtered observations **before** embeddings and public table construction. The table preview used fixed dummy vectors solely to run the deterministic table builder; it did not calculate a final layout, publish a map artifact, or contact Hugging Face.

| Check | Result |
| --- | ---: |
| Raw observations / Das source observations | 51,809 / 670 |
| Proposed Das positive / excluded / held IDs | 8 / 185 / 477 |
| Das observations filtered before public tables | 662 |
| Das public `profile_publications` / `authorships` / mapped works | 8 / 8 / 8 |
| Das person Scholar observation count / mapped work count | 8 / 8 |
| Held or excluded Das observation IDs in public works, authorships, or publication view | 0 |
| Active trusted `held` manifest: Das rows retained in public frame | 0 |

The local machine-readable receipt is `/Users/work/Documents/Codex/2026-10-24/wh/work/chem-das-no-upload-preview/das-subset-validation.json` (proposal SHA-256 `da3698791d36263f0428d723c5a84178b6d304fc11b9e21415fc11fa66d61897`). This preview proves that the gate keeps unreviewed rows out of public tables **for these exact draft heads**. It does not establish a complete Das publication history. A fresh paired preview is required after the source and producer branches are restacked onto settled main, and public release remains gated on the user's completeness requirement.
