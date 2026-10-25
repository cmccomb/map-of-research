# Chemistry decisions in the frozen Bernhard region release

The [exact 19-row manifest](../../data/curated/scholar-chem-region-review.json)
binds each decision to its saved Scholar citation, complete profile cache,
frozen source row, `bb88ddf0` work fingerprint, and independent evidence.
The frozen `f8d7ffec` raw snapshot and every full Scholar cache remain intact.
The producer applies these decisions in memory after validating the original
curated manifests, preserving the byte-pinned Sullivan release gate.

| Historical change | Include | Hold | Exclude |
| --- | ---: | ---: | ---: |
| Restore | 3 | 3 | 1 |
| Reexclude | 7 | 2 | 3 |

The two Donahue AGU meeting presentations are explicitly typed as
`conference_presentation` and included as scholarly meeting outputs. They are
not journal papers; one is solo and adds no coauthor edge.
Matyjaszewski's 1979 *Polimery* article is the third restored output. Seven
previously visible works retain their points: two Kietrys posters, Collins's
ACS response and viewpoint, Armitage's News & Views article, Matyjaszewski's
book section, and his *Macromolecules* article.

The two already-visible Kietrys poster rows retain their generic Scholar
`publication` type in this visibility review. Organizer entries confirm the
poster format, but their full abstract/byline has not been recovered; a
separate context and collaboration review should resolve that type and any
coauthor links.

Five rows stay raw under holds. They are the two Collins translated features,
Kurnikova's HARLEM software row with an unsupported 1999 journal venue,
Gil's multi-group signed support feature, and a Collins ACS meeting abstract
whose exact byline still needs the original abstract. Four exact rows are
excluded from mapping and coauthor edges: Washburn's spliced citation,
Matyjaszewski's falsely attributed Cesa column, the unattributed publisher
notice in *Progress in Polymer Science*, and the truncated duplicate
"Living" Scholar row. The publisher-deposited Crossref record for
[the notice](https://api.crossref.org/works/10.1016/S0079-6700(02)00114-4)
has no authors; the adjacent [Berry–Matyjaszewski editorial](https://api.crossref.org/works/10.1016/S0079-6700(02)00115-6)
has a separate DOI and page.

The exact composite effect changes from +99/−156 to +95/−149 Bernhard/Math/
Chemistry map points, with the Sullivan +11/−29 transition unchanged. The
expected frozen-parent map has 42,853 points. Before any release, run a new
seven-file first-write-intercepted preview from the exact PR head and pinned
source commit `8be6a240`; verify the current Hub parent is still `bb88ddf0`,
the staged `profile_publications`, `authorships`, `people`, `works`, and map
parity checks pass, and zero Hub writes occurred. The prior #328 preview
receipted its former 42,850-point candidate and cannot approve this payload.
