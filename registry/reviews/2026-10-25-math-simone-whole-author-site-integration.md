# Jonathan Simone: held whole author-site integration

**Decision:** Stage the 11 public work families from the [source review](../../data/source_reviews/jonathan-simone-2026-10-25/README.md) together as `curated_verified`. This is a draft release candidate; it has not been merged or uploaded. No Google Scholar account has been verified, and no Scholar ID is assigned. The CMU Math registry person remains `person-jonathansimone-9415e63e`, with primary department Math in MCS.

The immutable reviewed source commit is `3efca3dfe67eecc89884bc82b2e90521a1657f86` (source [PR #371](https://github.com/cmccomb/map-of-research/pull/371)). Each curated row pins that commit and the candidate JSON SHA-256 `18f1caff031dc31bc28659a8e1597cd8ab3b04620c188d7951c76c82f84dbd0b`. The researcher-maintained [Publications and Preprints](https://www.math.cmu.edu/~jsimone/research.html) page supplies ten entries, the [CV](https://www.math.cmu.edu/~jsimone/JonathanSimoneCV.pdf) supplies the same current nine publication families plus an in-preparation hold, the [exact arXiv author search](https://export.arxiv.org/api/query?search_query=au:%22Simone%2C_Jonathan%22&start=0&max_results=100) confirms ten paper/preprint IDs, and [Virginia's thesis record](https://api.datacite.org/dois/10.18130/V39Z90B6J) confirms the separately listed dissertation. This establishes completeness of the visible author-declared public list at the review date, not proof that no unlisted work exists. Scholar will take precedence if an exact account is later verified and completely captured.

| Source family | Rows | Work treatment |
| --- | ---: | --- |
| DOI journal article plus its arXiv version | 8 | One DOI work per family; arXiv ID retained in raw provenance |
| Accepted preprint without verified journal DOI | 1 | Stable `arxiv:2405.00500` native ID |
| Older standalone preprint | 1 | Stable `arxiv:1710.06702` native ID |
| Doctoral dissertation | 1 | DOI work, `doctoral_dissertation` type, no collaboration edge |
| CV-only manuscript explicitly marked in preparation | 0 | Held outside curated works |

For *On χ-slice pretzel links*, the [publisher DOI record](https://api.crossref.org/works/10.1016/j.topol.2025.109578) supplies volume 375 and 2025, while the current author site and CV say volume 377 and 2026. The source review preserves the discrepancy. The integration uses the publisher metadata, and it does not merge this paper with the distinct 2026 toroidal-plumbing article.

## No-upload preview against the hosted dataset

The comparison pins the current hosted [dataset revision `38f119d7`](https://huggingface.co/datasets/ccm/cmu-engineering-publications/tree/38f119d71692cdb9112210217b532a7dd7c1633a). SHA-256 checks confirm that all four local baseline table files exactly match that revision's hosted LFS metadata. The local snapshot uses the same frozen Scholar cache source commit `77fbb9c514bb58e33ef76b94965fdcfa67ef4a15` plus this 11-row curated batch. It validates at 51,890 raw observations and 39 curated rows (the live source has 51,879 and 28). The table delta is deterministic and makes no network writes:

| Table | Hosted `38f119d7` | Candidate | Difference |
| --- | ---: | ---: | ---: |
| People | 2,001 | 2,001 | 0 |
| Works | 47,743 | 47,754 | +11 |
| Authorships | 49,629 | 49,640 | +11 |
| Profile publications | 51,879 | 51,890 | +11 |

None of the 11 work IDs or observation IDs collides with the hosted tables. Simone's current person row has zero publication observations; the candidate person row has **11 curated, zero Scholar, 11 mapped**. All 11 mapped works retain `source_types=["curated_verified"]`; all 11 observations have a blank `scholar_id` and preserve their author-site source record. Ten papers retain ordinary publication eligibility if an exact mapped coauthor is later found; the dissertation may appear as a research work but cannot make a coauthor edge. The draft does not assign a DOI to the accepted preprint, expose the in-preparation manuscript, or convert the older standalone preprint into the distinct journal article.

The normalized candidate has exactly 11 faculty authorships, all for Simone's Math/MCS person record, and no external coauthor IDs. Every work's `person_ids`, `department_ids`, and `school_ids` are exactly `[Simone]`, `[math]`, and `[mcs]`; free-text paper bylines do not create faculty or external nodes. With no second mapped faculty on any work, this batch adds **zero collaboration edges**. The [collaboration graph filter](https://github.com/cmccomb/map-of-collaboration/blob/main/assets/collaboration-core.js) shows connected faculty by default, so Simone will appear only with **Show unconnected faculty** or an explicit search selection. His 11 works remain visible in the research publication map. A future exact coauthor attribution can change the collaboration result.

This preview checks Simone's table identity, provenance, publication type, map eligibility, and count delta. It does **not** establish exact retention of all prior 51,879 hosted observations or every prior work and authorship. Before any release, the central publisher must stage the complete output and compare the prior observation, work, and authorship rows against the hosted `38f119d7` baseline, allowing only separately reviewed changes. The central lane must also run production embeddings, map layout, and hosted release gates. The local preview receipt is saved outside the repository at `/Users/work/Documents/Codex/2026-10-24/wh/work/simone-integration-preview/preview.json`.
