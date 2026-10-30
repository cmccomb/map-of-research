# SCS code-model topic map hold

The initial decision applied to the four-profile SCS source snapshot
`9cfe0e2d6157d0d70371172e0164683baabacaa1bc3daef3931db03a4bb9c017`
on public Hub parent
`6964644510ac2debbc1b94e2fa4a1e07a8a93c25`.

Daniel Fried's captured Scholar rows for *StarCoder: may the source be with
you!* (`sJDqACEAAAAJ:r0BpntZqJG4C`) and *SantaCoder: don't reach for the
stars!* (`sJDqACEAAAAJ:_Qo2XoVZTnwC`) are genuine publications. His
[owner CV](https://dpfried.github.io/cv/Fried-CV-web.pdf) lists both titles;
the local row ledger records the exact CV entries. The primary
[StarCoder paper](https://arxiv.org/abs/2305.06161) and
[SantaCoder paper](https://arxiv.org/abs/2301.03988) describe language
models for code.

The same source and producer tree yielded a local stage in which both works
were excluded as `low_information_region` and a hosted Linux stage in which
both were mapped under `abstract elementary classes`. That label does not
describe these papers. The global topic fit changes with small numerical
differences between build machines. This exact release therefore keeps both
raw rows and Daniel's authorships, while withholding the two work map points
under `reviewed_topic_hold` until a source-backed topic decision is reviewed.
The hold is about the map's topic placement, not attribution.

All other work, author, and map decisions follow the standing release gates.
This hold does not change the existing public Hub revision or other snapshots.

## Carry forward from the public four-profile release

The four-profile release became immutable Hub revision
`af0e147a935f5f2a557ea6b3f837890c46ede0c2`. Its public works table has
both exact work IDs above with `map_eligible=false` and the sole exclusion
reason `reviewed_topic_hold`; its profile-publication table retains Daniel's
two raw Scholar observations as eligible source rows, and each work has only
Daniel's one observation. The raw `source_record_json` SHA-256 values are
`d2971922fba75add5e2323351e778d044b2c6c22895edc50a7284de926a98b9b`
for StarCoder and
`aeb96f99bb3ab7206d919187aa6efcc9cad1320cae13c766607a2346c6c39acb`
for SantaCoder.

The subsequent twelve-profile source commit
`79b77d1da5e7dece2dcfb0a7b77348ed2d9782ba` has snapshot SHA-256
`3059c70d73c6a820e8e1f89158424f291ff56991be5b73eaa7538e861f37eb52`.
Daniel's author cache is byte-identical to the preceding SCS source cache
(SHA-256 `161a00332212f6d4c0b20893ab85d990b500373db157234459531cd0e65eda5d`).
The two exact titles occur only in Daniel's author cache in this source
snapshot. The local no-write preview against `af0e` failed before upload with
exactly these two previously held observations as unexplained map gains, zero
unexplained losses, and zero changed mapped bindings. No new source evidence
reviewed their topic placement.

The publisher now carries this hold forward only when both prior public works
retain the exact hold, both prior and current Daniel source fingerprints
match, and the work fingerprints and observation memberships remain unchanged.
A partial or changed state fails closed. This keeps the genuine raw rows and
authorships while preventing numerical topic refits from making either map
point visible. A future reviewed topic decision requires an explicit change
to this gate.
