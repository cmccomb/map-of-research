# Hyung J. Kim: eight exact CMU group and publisher matches

This bounded review starts with the unchanged, complete visible 918-row
Google Scholar capture for `bC4aq-4AAAAJ` (JSON SHA-256
`ea04610e120039df7aaf83c402d193021f328b319cd41cfabf541f3942069867`).
The [CMU Kim Group publication list](https://www.chem.cmu.edu/groups/kim/pub.html)
identifies the group, spells out each article's authors, title, journal,
volume, pages, and year, and links the publisher record. Its locally reviewed
HTML bytes have SHA-256
`f1ed23585510d2e3a7414e9152fdd1503cc2e89a5e50bbead003c089558828e8`.
For each row below, publisher-deposited DOI metadata also names **Hyung J.
Kim** in full. The Scholar title, year, author sequence, journal, volume, and
pages agree with the CMU list and deposited record; row 145's deposited
metadata supplies its first page and the CMU list supplies the full range.

| Capture row and exact Scholar ID | Publisher-deposited DOI record | Exact output and full byline |
| --- | --- | --- |
| 9 `bC4aq-4AAAAJ:d1gkVwhDpl0C` | [ACS `10.1021/nn901916m`](https://api.crossref.org/works/10.1021%2Fnn901916m) | *Nanoporous Carbon Supercapacitors in an Ionic Liquid: A Computer Simulation Study*, ACS Nano 4:2345–2355 (2010); Youngseon Shim, Hyung J. Kim. |
| 23 `bC4aq-4AAAAJ:Tyk-4Ss8FVUC` | [ACS `10.1021/nn900195b`](https://api.crossref.org/works/10.1021%2Fnn900195b) | *Solvation of Carbon Nanotubes in a Room-Temperature Ionic Liquid*, ACS Nano 3:1693–1702 (2009); Youngseon Shim, Hyung J. Kim. |
| 41 `bC4aq-4AAAAJ:kNdYIx-mwKoC` | [ACS `10.1021/jp203458b`](https://api.crossref.org/works/10.1021%2Fjp203458b) | *Graphene-Based Supercapacitors: A Computer Simulation Study*, J. Phys. Chem. C 115:23574–23583 (2011); Youngseon Shim, YounJoon Jung, Hyung J. Kim. |
| 49 `bC4aq-4AAAAJ:LkGwnXOMwfcC` | [ACS `10.1021/jp907394v`](https://api.crossref.org/works/10.1021%2Fjp907394v) | *Molecular Interactions in 1-Ethyl-3-methylimidazolium Acetate Ion Pair: A Density Functional Study*, J. Phys. Chem. A 113:10397–10404 (2009); Nilesh R. Dhumal, Hyung J. Kim, Johannes Kiefer. |
| 89 `bC4aq-4AAAAJ:W7OEmFMy1HYC` | [ACS `10.1021/ar700061r`](https://api.crossref.org/works/10.1021%2Far700061r) | *Solvation, Solute Rotation and Vibration Relaxation, and Electron-Transfer Reactions in Room-Temperature Ionic Liquids*, Acc. Chem. Res. 40:1130–1137 (2007); Youngseon Shim, Daun Jeong, Swati Manjari, Moo Young Choi, Hyung J. Kim. |
| 111 `bC4aq-4AAAAJ:_FxGoFyzp5QC` | [ACS `10.1021/jp0703859`](https://api.crossref.org/works/10.1021%2Fjp0703859) | *Free Energy and Dynamics of Electron-Transfer Reactions in a Room Temperature Ionic Liquid*, J. Phys. Chem. B 111:4510–4519 (2007); Youngseon Shim, Hyung J. Kim. |
| 145 `bC4aq-4AAAAJ:9ZlFYXVOiuMC` | [RSC `10.1039/B921725H`](https://api.crossref.org/works/10.1039%2FB921725H) | *Fragility, Stokes–Einstein Violation, and Correlated Local Excitations in a Coarse-grained Model of an Ionic Liquid*, PCCP 12:2001–2010 (2010); Daun Jeong, M. Y. Choi, Hyung J. Kim, YounJoon Jung. |
| 166 `bC4aq-4AAAAJ:3fE2CSJIrl8C` | [ACS `10.1021/jp067316z`](https://api.crossref.org/works/10.1021%2Fjp067316z) | *Effects of Solute Electronic Polarizability on Solvation in a Room-Temperature Ionic Liquid*, J. Phys. Chem. B 111:4920–4925 (2007); Daun Jeong, Youngseon Shim, M. Y. Choi, Hyung J. Kim. |

The group list has 27 normalized exact title/year matches to the held ORCID
person signals, but this decision promotes only the eight above. A similar
title alone is not an exact match: row 41 is the 2011 simulation article, not
a later graphene oxide paper. The other candidate rows remain held until
their author attribution and output type receive the same review.

`rebuild_chem_kim_918_review.py` guards these eight decisions by immutable
Scholar citation ID, original title, and original year. The review ledger now
has **23 source-confirmed works**, **91 exact CMU-linked ORCID signals awaiting
output review**, **410 other held rows**, and **394 exact exclusions**. All 918
raw observations remain preserved. The trusted Kim release state is still
`held`; the 23 confirmations are evidence for a nonoperational proposal and
do not represent his complete publication history.
