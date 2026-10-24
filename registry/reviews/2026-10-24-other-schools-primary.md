# Four-school display-primary review — 2026-10-24

The [dated primary ledger](2026-10-24-other-schools-primary.csv) covers every
person with a registered Google Scholar ID and at least one membership added
through the CFA, Dietrich, Heinz, or Tepper review. It has **105 distinct
people**: **103 resolved** display-primary units and **two explicitly
unresolved** cases. The three `primary_department_*` columns match the schema
proposed in [draft PR #35](https://github.com/cmccomb/map-of-research/pull/35);
`basis` is an audit note. The current registry does not yet have those three
columns, so this ledger preserves the decisions for integration after that
schema change. It neither removes joint memberships nor adds publication data.

For a person with one included academic unit, the official membership source
supports that unit. For multiple included units, the reviewers used current
explicit professor, courtesy, joint, academic-position, or major/minor wording
when available. Where current appointments were coequal and public degree
records allowed a bounded count, the choice is marked as an **inference** in
the ledger and the relevant school review. Twelve resolved rows use this
advising-count method:

| Person | Display unit | Count/evidence location |
| --- | --- | --- |
| Baruch Fischhoff | EPP | EPP PhD advisee plus department/institute CV wording; [Dietrich review](2026-10-24-dietrich.md) |
| Byron Yu | Neuroscience Institute | 12 Neural Computation, 3 ECE, 1 BME; [Dietrich review](2026-10-24-dietrich.md) |
| Corey Harper | Civil and Environmental Engineering | 5 CEE, 0 Heinz on current lab roster; [Heinz review](2026-10-24-heinz.md) |
| David Creswell | Psychology | 1 Psychology, 0 NI in located student directories; [Dietrich review](2026-10-24-dietrich.md) |
| Ken Koedinger | HCII | 7 HCII, 2 Psychology, 1 Machine Learning in degree-labeled PIER records; [Dietrich review](2026-10-24-dietrich.md) |
| Lujo Bauer | S3D | 4 Societal Computing, 3 ECE in current cohort; tentative, [Heinz review](2026-10-24-heinz.md) |
| Lorrie Cranor | S3D | 7 Societal Computing, 2 EPP in current cohort; [Heinz review](2026-10-24-heinz.md) |
| Maggie Henderson | Psychology | 3 Psychology, 0 NI in located student directories; [Dietrich review](2026-10-24-dietrich.md) |
| Matthew A. Smith | Neuroscience Institute | 10 Neural Computation/Systems Neuroscience, 4 BME; [Dietrich review](2026-10-24-dietrich.md) |
| Steven M. Chase | Neuroscience Institute | 4 Neural Computation, 1 ECE, 1 BME; [Dietrich review](2026-10-24-dietrich.md) |
| Timothy Verstynen | Psychology | 1 Psychology, 0 NI in located student directories; [Dietrich review](2026-10-24-dietrich.md) |
| Tzahi Cohen-Karni | Materials Science and Engineering | 8 MSE, 7 BME current students; [Dietrich review](2026-10-24-dietrich.md) |

The machine-readable ledger records the exact source URL and basis. Counts are documented public subsets,
not complete lifetime advising histories. A joint-degree or co-advised student
is counted once per documented degree unit. An explicit current academic
position can settle a display choice even when a bounded advising subset
leans elsewhere, as [Tepper's Nicholas Muller review](2026-10-24-tepper.md)
documents. The original joint memberships remain included.

The two unresolved rows are:

- **Berit C. R. Royer:** LCAL lists a part-time adjunct lecturer. No included
  CMU faculty unit is documented by this review.
- **Chris Hendrickson:** current CEE, EPP, and Heinz emeritus appointments are
  coequal in official sources. Two located CEE doctoral advisees do not
  establish a majority across those units; see the [Heinz review](2026-10-24-heinz.md).

The ledger also retains the Civil and Environmental Engineering choice for
Destenie Nock from draft PR #35's EPP catalog evidence. Her new CFA
Architecture membership is affiliate and excluded. A campus-wide review of
earlier Engineering coequal appointments can revisit that prior choice without
changing this four-school roster.
