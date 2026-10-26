# Next ten SCS Scholar sources, one combined release

This paired draft integrates the two five-profile, complete saved Scholar batches without merging their stale raw snapshots: source/producer #449/#450 and #451/#452. Both draft raw Parquet files independently began from the older 59,434-row baseline and would omit the first fifteen SCS captures if merged directly. This source branch instead rebuilds from the post-upload source/status tip `aeb9caf22403b0b895135a1e550438be38952e3d`, whose successful receipt pins immutable Hub `9e604bfd7864221930531279afd0aa2a119814b7`; the producer is based on main `620e85adc5614159f96a5f5a45c9ecf8307116c7`.

| Batch | Complete profiles | Captured raw rows | Exact provisional holds | Ordinary gate candidates |
| --- | ---: | ---: | ---: | ---: |
| Peng, Shankar, Goel, Sadeh, Silwal | 5 | 827 | 333 | 494 |
| Bajcsy, Liu, Held, Pathak, Ramanan | 5 | 993 | 217 | 776 |
| **Combined** | **10** | **1,820** | **550** | **1,270** |

The rebuilt source snapshot has 64,981 rows: all 63,161 prior observations remain field for field equal, including the first fifteen SCS profiles and 43 curated rows, plus exactly 1,820 rows from ten distinct Scholar IDs and stable person IDs. The source and producer registries use identical bytes. Ten complete redacted captures, canonical caches, and browser receipts reconcile to their release entries. Every uncertain citation ID is bound by the exact hold manifest; these 550 rows remain held until separate evidence supports release.

All ten included scholars have an evidenced primary academic home. Peng and Shankar use CSD, Goel and Sadeh use S3D, and Silwal plus the five RI faculty use the typed Robotics Institute home. RI is retained as an institute rather than labeled a department. Joint memberships remain in the registry.

The source snapshot SHA-256 is `294284cfa254963ef6969c0c9e51c7bad8e6b319e2c6341929d6d94c75a4b260`. Independent saved-capture and raw row audits are at `/Users/work/Documents/Codex/2026-10-26/cmu-scs-next-ten-combined/work/next-ten-{capture-receipt,raw-independent}-audit.json`.

This pair remains draft until both hosted CI checks and an exact-parent seven-file no-upload preview against `9e604bfd7864221930531279afd0aa2a119814b7` pass, including prior point/authorship/membership/pair preservation and held-row leakage checks. The source raw branch triggers upload when merged, so merge producer first and source last only after central review.
