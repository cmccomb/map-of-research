# One-sided MCS coauthor evidence pilot — 2026-10-30

One Scholar profile can report a joint paper before the other faculty member's
profile does. The four entries in
`data/curated/reported-faculty-coauthors.json` below use that
one-sided report as collaboration evidence for an already mapped work. They
remain separate from the target person's own publication-history count and
from the `authorships` table. Each reporter has an eligible released Scholar
row; a primary publisher or CMU source resolves the abbreviated coauthor
byline to the named, included MCS faculty member.

| Reporter → underreported coauthor | Exact Scholar row | Primary byline evidence |
| --- | --- | --- |
| Alan Frieze → Po-Shen Loh, DOI `10.1002/rsa.20475` | `EqKRQ4YAAAAJ:j7_hQOaDUrUC` | [CMU-hosted paper](https://www.math.cmu.edu/~ploh/docs/research/20120722-rainbow-ham.pdf) |
| Robert Pego → Gautam Iyer, DOI `10.1137/15M1054730` | `neJ3NIUAAAAJ:LPZeul_q3PIC` | [CMU-hosted paper](https://www.math.cmu.edu/~gautam/sj/research/papers/201512-kompaneets.pdf); [SIAM record](https://epubs.siam.org/doi/10.1137/15M1054730) |
| Markus Deserno → John Nagle, DOI `10.1039/C9SM01290G` | `1J28ExgAAAAJ:tKAzc9rXhukC` | [Nagle's CMU Physics bibliography](https://www.cmu.edu/physics/people/faculty/nagle.html) |
| C. Joel McManus → Charles Ettensohn, DOI `10.1242/dev.105585` | `28yZqbcAAAAJ:eQOLeE2rZwMC` | [PubMed record](https://pubmed.ncbi.nlm.nih.gov/24496631/) |

All four target people have included MCS roster identities. Their exact
published works currently have only the reporter's faculty ID and no edge to
the target. The source-frozen assertions pin the raw Scholar title and byline,
observation ID, work ID, and primary source. A changed publication identity,
eligibility decision, source checksum, or duplicate assertion fails closed.
The publisher carries only point-level provenance; the collaboration consumer
may add a visibly labeled reported-coauthor edge without presenting these
isolated reports as a complete publication history.

A broader strict byline screen found 209 *review leads* on 200 of 14,726
eligible works with exactly one MCS faculty ID. Those leads span 91
reporter–target pairs and 56 potential targets. Even strict initial and surname
matches include namesakes. No other lead is approved by this pilot. Consortium
bylines, ellipses, and ambiguous names remain outside automatic inference.
This producer change does not collect Scholar profiles, upload a snapshot, or
publish the map. A release must satisfy the current parent-bound continuity,
private-subset, and artifact review gates.
