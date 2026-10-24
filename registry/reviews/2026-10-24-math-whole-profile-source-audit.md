# Mathematical Sciences Scholar source review

Reviewed 2026-10-24 against the saved complete visible-profile captures. Google Scholar remains the primary source for each observed row. This review identifies only exact citation IDs that independent sources show to be grants, front matter, false attributions, or corrupted duplicates. Every captured Scholar row stays in the raw source table; the exclusion manifest prevents a reviewed row from becoming a mapped work or authorship.

The manifest adds **219** exact-ID decisions to the 268 present on this branch's producer `main` baseline. Each decision states its reason and source URL. All 219 IDs are unique and occur in the staged complete Math captures; the existing 268 decisions are preserved.

| Review scope | New exact exclusions | Decision record |
| --- | ---: | --- |
| Frieze and Bukh initial review | 45 | `2026-10-24-math-frieze-bukh-initial-decisions.json` |
| Frieze and Bukh second pass | 45 | `2026-10-24-math-frieze-bukh-second-pass-decisions.json` |
| Owen | 25 | `2026-10-24-math-owen-source-decisions.json` |
| Howell, Shkolnikov, Grossberg | 13 | `2026-10-24-math-howell-shkolnikov-grossberg-decisions.json` |
| Lehoczky, Shreve, Cornuéjols | 70 | `2026-10-24-math-lehoczky-shreve-cornuejols-decisions.json` |
| Published Leoni, Tkocz, Mackey captures | 21 | `2026-10-24-math-leoni-tkocz-mackey-triage.md` |

The Lehoczky/Shreve/Cornuéjols ledger records all 108 candidates: 70 exclusions, seven genuine items needing title repair, and 31 unresolved entries retained. The genuine items include Lehoczky's authored MathSciNet review, Shreve's separate Volume I and Volume II errata, Shreve's chapter in *Proceedings of Symposia in Applied Mathematics* 57, and Cornuéjols's report. Frieze's 1995 technical report and 1982 paper are likewise retained in the second-pass record. These decisions avoid discarding real academic work because Scholar clipped a title or merged a page header into it.

The other ledgers also preserve unresolved rows instead of turning uncertainty into exclusion. A title repair would require a separate, provenance-preserving display metadata change; this manifest changes only map eligibility. “Complete visible profile” means all rows shown by that Scholar profile at capture time, not independent proof of the researcher's lifetime publication total.

This source review can merge before the staged captures without losing records. The next validated snapshot build will apply these decisions to the saved raw rows; the live maps change only after that snapshot is uploaded and both sites are deployed.
