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

The only proposed visibility differences from the live `bb88ddf0` dataset are
four gains and two losses:

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

## Release boundary

The optional `math_region_review` workflow dispatch requires the frozen
`f7a0eee8` source snapshot and live Hub revision `bb88ddf0`. It applies the
ordinary bb88 continuity path and admits only the four reviewed gains above.
The two reviewed losses come from exact exclusions. The older
`restore_prior_regions` path replays all 280 region changes and is blocked
from the bb88 baseline pending a separate review. Any bb88 snapshot upload
without the Math flag fails before a Hub write, so merging this draft alone cannot silently
publish the three exclusions and 12 holds without the four positive decisions.
After another release advances Hub beyond bb88, this frozen dispatch needs a
new review and baseline; sequence it with Sullivan and Bio accordingly.

Run a no-upload preview against immutable bb88 and verify exactly four gains,
two losses, no unrelated visibility changes, preserved raw rows and primary
homes, and zero held contributions before dispatch. This note describes a
draft policy, not a released dataset.
