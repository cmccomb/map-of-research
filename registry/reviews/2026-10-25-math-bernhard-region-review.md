# Math decisions for the frozen Bernhard region change

This draft reviews 31 exact Scholar observations associated with Mathematical
Sciences memberships. The [source-bound decision manifest](../../data/curated/scholar-region-review.json)
records each saved citation ID, capture index, cache and source-record checksums,
work ID, current primary display home, evidence URL, and decision. It contains
16 includes, three exclusions, and 12 provisional holds. The 12 held rows
remain in their complete raw Scholar profiles, but contribute no mapped work or
authorship. Candidate clean citation IDs from the working review are **not**
aliases. No source snapshot or person registry row changes in this draft.
The work fingerprints use the publisher's own field normalization against
immutable `bb88ddf0` works; a frozen [31-work baseline fixture](../../tests/fixtures/bernhard-math-bb88-works.json)
checks this calculation in CI. The earlier standalone repair audit serialized
numeric fields differently, so its work hashes are not used for this gate.

Within these 31 Math decisions, the visibility differences from the
`bb88ddf0` dataset are four gains and two losses:

| Action | Exact Scholar citation ID | Evidence |
| --- | --- | --- |
| Include | `A0ISJPUAAAAJ:q3CdL3IzO_QC` | [Shreve's textbook contents](https://link.springer.com/book/9780387401010) identifies the separately indexed 2004 *Change of Numéraire* chapter. |
| Include | `7G7zH2AAAAAJ:IWHjjKOFINEC` | [Publisher record](https://www.tandfonline.com/doi/abs/10.1080/00029890.2022.2128607) identifies Frick and Peng's *What Can You Draw?* article. |
| Include | `0Nj-v7sAAAAJ:VL0QpB8kHFEC` | [Author preprint](https://arxiv.org/abs/2404.18342) identifies the Leoni–Spector work. Scholar's captured title is truncated; this draft does not silently rewrite it. |
| Include | `P_AmD6MAAAAJ:dQ2og3OwTAUC` | [Wiley/Crossref record](https://api.crossref.org/works/10.1002/9781118445112.stat05071) identifies Lehoczky's *Markov Chains* reference chapter. |
| Exclude | `VQfvIfQAAAAJ:IUKN3-7HHlwC` | Saved Scholar row 249 is an editor's referee thank-you list, not Avigad authorship. An [Oxford acknowledgments page](https://academic.oup.com/jigpal/article-abstract/14/6/902/747606) confirms the record type; **that page is an example, not an exact match to the undated saved row**. |
| Exclude | `VQfvIfQAAAAJ:yFnVuubrUp4C` | [University event notice](https://www.illc.uva.nl/NewsandEvents/Events/Conferences?displayMode=month&month=04&year=2020) identifies a cancelled 2020 conference event, not a coauthored paper. |

The third exclusion, Shreve's duplicate Cyrillic record
`A0ISJPUAAAAJ:BUYA1_V_uYcC`, was already hidden in `bb88ddf0` and stays raw.
The [Russian State Library catalog](https://search.rsl.ru/ru/record/01001252173)
identifies the 1985 Nauka translation also represented by a clean Scholar row.
Nine unreadable Frieze citations, two damaged Fonseca–Leoni–Müller citations,
and one unidentifiable Shreve citation remain held pending exact document or
version evidence. The full rationale for each is in the decision manifest.

Twenty-one routes have Math as their primary display home. Ten have Math as
an included secondary membership; the existing primary homes remain their
visual homes. This draft changes no primary department, school, membership, or
coauthor identity.

## Composite release boundary

The older 280-work Bernhard manifest identifies the exact rows for immutable
source and parent checks. Only the 31 Math rows have reviewed decisions. The
other 249 retain their published `bb88ddf0` region decisions, topics, map
placement, and work payloads. The same release includes the complete Ryan
Sullivan Scholar capture in source commit
`8be6a2400cf72dd959c874bfcada8624540820ef`. The exact source snapshot,
curated decision files, baseline Hub revision `bb88ddf0`, and staged dataset
payloads are checked before an atomic Hub write. Across the combined release,
the reviewed Math effect is four restored and two removed map points; Sullivan
adds 11 and removes 29. The resulting point set is 42,909 works.

The release remains held. Its publisher preview must intercept the first Hub
write and compare all seven staged files against the immutable parent before
any dispatch. A changed Hub parent or source/producer commit requires a new
preview and review. This note records the proposed release gate, not a claim
that the dataset has been published or an authorization to retry the upload.
