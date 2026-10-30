# Ann B. Lee selected-only local preview decision

This ledger supports an isolated no-write preview only. It is not a public release authorization.

- Stable person: `person-annlee-3bb27a6a`; exact owner-linked Scholar ID: `pDQsaXgAAAAJ`.
- CMU owner and typed primary home source: https://www.stat.cmu.edu/~annlee/ (`department:stats`).
- Complete saved Chrome observation: 103 rows. The private original-document gate verified 37 preliminary rows.
- Selected-only source stage: 36 rows. One additional document-verified row remains private because its captured citation count includes a footnote marker rejected by the current decimal-only parser.
- Withheld: 60 holds and 7 exclusions. No withheld IDs, raw captures, or HTML enter the source stage.
- Two selected Ann rows correspond to existing public Dejan Slepcev works; the preview must preserve both prior work IDs and Dejan bindings while adding Ann.
- Historical Hub4203 local preview source commit: `843f753599313a57db4e70574c2e307e3a9fefaf`. Current local selected-only source candidate: `cb6e5b298f6805b1eb8af2ccffe953025da2846c`; current exact public parent: Hub `df71f79e48d3af7a1cc3eb0996e474564a4a7784`. The df71 all-hold placement review is a separate source-relative manifest; it is not release approval.
- Historical Ann-only source selection manifest SHA-256: `708c614b3ef3ac47c53d68f83331381b9930a88578663f0948d7e5e182986400`.
- Historical Ann-only source private-subset preflight receipt SHA-256: `47abbd10e492efbba54017faf48979bd06ea30942a26f8b28a724be2e4a6ac74`.
- Ann selected cache SHA-256: `7f23312165a6e5938e03ca1772381ecb12e2a40c274acfc91b703bda03399c3a`.
- Historical Ann-only staged private review SHA-256: `e5687e65c6161c0d080564b0cfe415973f1f73c2b10f4b1930e75f0c769f4fc8`.

- Current df71 all-hold placement manifest SHA-256: `e29aecb4ccb62d0330a212dff4a1af80d187ee87951e9639dc981449928d31de`; 69 works, all coordinates null and topics pending.
- Historical df71 no-write preview on producer `41c353b8480e54324e37d2253d7ae64d74500796` and source `cb6e5b298f6805b1eb8af2ccffe953025da2846c` intercepted the first write, but a second independent works-table audit found 156 prior region override-flag changes. That preview and its first audit are superseded and do not approve publication. The final producer PR head requires a new exact-parent no-write preview and independent full-field works-table check before release.

## Private review evidence

- `outputs/ann-lee-103-private-attribution-review-2026-10-30.json` SHA-256 `6f53080ce049fa288f81a65607cdfd5cf16fe282062303a5e3f2a980f044e1f9`
- `outputs/ann-lee-all-53-original-document-gates-private-integration-2026-10-30.json` SHA-256 `4a94bec663a15a7bcc7625b719057344f599c2f92f60a816d5f73a97763b64e4`
- `outputs/ann-lee-37-private-release-readiness-audit-2026-10-30.json` SHA-256 `b2c3c37f1e265c698fd7a76b692cd9e1a28a72c5990d2cf517840ea7d2ea359b`
- `outputs/ann-lee-36-selected-only-private-stage-boundary-2026-10-30.json` SHA-256 `1263050d3b5de138112f034ce0cc55210aae52f6525c7bc9a5a7238c74e0f754`

## Release gate

The producer selected-only decision must pass local tests and an exact-parent first-write-intercepted preview. Before any public change, independently verify 67 Ann withheld IDs absent, all old work IDs and faculty bindings retained, the two shared Dejan work IDs unchanged, and no faculty pair loss. Confirm hosted CI and both served maps after an authorized combined release.
