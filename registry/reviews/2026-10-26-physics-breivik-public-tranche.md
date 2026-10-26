# Katelyn Breivik Physics source-bound release review

This draft releases the complete 131-row saved Scholar capture for Katelyn Breivik (`AcW2EXwAAAAJ`), whose exact Scholar ID is linked from [CMU Physics](https://www.cmu.edu/physics/people/faculty/breivik.html). The source capture, author cache, and snapshot remain unchanged. The producer binds the profile to the cache SHA-256 `f22c819fb75a356f42dcf10031222303d054c1f6f65dc8e6d50b18583b3a5aff`.

The nine already-reviewed exact exclusions remain in the producer manifest, including false attributions, nonpublications, and corrupted duplicates. The release adds 52 exact provisional holds and leaves 70 rows for the ordinary work and map quality gates. The hold manifest records every saved capture index, full citation ID, original Scholar citation URL, and reason. Its exclusion and hold digests are `8e952ea7aa66a16b6aeac16fc12462ae0cec20bf0883f9cefb7fc4f7a60779e1` and `43d1cff74a53a8bd7cd8fa06211f4d31ba7ba88e01a4341402e0aa7d28825544`. These holds make no adverse authorship claim.

| Hold group | Capture rows | Review boundary |
| --- | --- | --- |
| Meeting abstracts and presentations | 40, 52, 61, 65, 77, 85, 86, 87, 89, 90, 91, 97, 100, 101, 102, 105, 106, 107, 109, 110, 111, 113, 115, 116, 118, 120, 122, 123, 124, 125, 126, 127 | Await a consistent map-point and collaboration-link rule. |
| Software and repository records | 51, 74, 78, 98, 108, 131 | Await output-type and citation-version policy. |
| Blank venue | 24, 62, 103, 121, 129 | Verify primary source and output type. |
| Metadata fragment or career guide | 46, 104 | Row 46 may index row 8; row 104 needs output-type classification. |
| Related versions | 6, 40, 43, 52, 60, 61, 73, 75, 76, 114 | Reconcile paired journal, arXiv, and white-paper records before mapping. |

The source review ledger at `status/physics-breivik-release-review-2026-10-26.json` records category overlaps, exact row sets, and counts. The source status changes only after every saved row has a disposition. Row 115 and row 116 author-order cache repairs and their source ledgers are present in the source tip. The historical draft PRs #207 and #257 remain open; their source and repair artifacts are already present at the current source tip, so this pair uses the current source/status and producer parents.

The Chemistry continuity guard now recognizes the exact immutable Hunt/Deserno/Feenstra Hub parent `42b30e512fb941a8cbe78d67399bd3771372cb68`. A direct comparison of its 19 reviewed Chemistry works with the prior Physics Hub parent found one changed diagnostic: work `work-a0cfe68d0794a4afe20e8a55b401afed795f14a8445cb9dc54275a84acb6b3d4` now has `map_region_prior_eligible_override=false`. The guard pins this exact parent-specific value, retains the other reviewed override checks, and has a parameterized regression test.

Release remains draft. Exact-parent preview, focused tests, and hosted CI are gates before any merge or upload.
