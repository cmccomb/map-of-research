# Three exact KNU, COSINE, and AMoRE author identities in Kim's held capture

The saved CMU Chemistry Hyung J. Kim Google Scholar capture has 918 observations
(SHA-256 `ea04610e120039df7aaf83c402d193021f328b319cd41cfabf541f3942069867`).
This review keeps that full raw capture and the whole-profile public hold intact. It
excludes exactly three held observations with source-linked non-CMU authors.

| Capture row and Scholar ID | Source-backed decision |
| --- | --- |
| 353 `bC4aq-4AAAAJ:RgznTc0nqo4C` | [Kyungpook National University's exact 2024 article record](https://research.knu.ac.kr/en/publications/measurements-of-low-energy-nuclear-recoil-quenching-factors-for-n/) gives Physical Review C 110:014614, DOI `10.1103/PhysRevC.110.014614`, and links its H. J. Kim author directly to [Hong Joo Kim's KNU Physics faculty page](https://research.knu.ac.kr/en/persons/hong-joo-kim/). This is Hong Joo Kim, not CMU Chemistry's Hyung J. Kim. |
| 408 `bC4aq-4AAAAJ:e0LTWoPxLYMC` | [Korea University's institutional record](https://scholar.korea.ac.kr/item/c10ee6c6-a308-44a3-a039-13d2f39d10c1) gives the exact title, DOI `10.1103/77p2-rnbw`, Physical Review Letters 135(23), and a complete byline naming Hongjoo Kim. [APS's original article](https://journals.aps.org/prl/abstract/10.1103/77p2-rnbw) assigns H. J. Kim to Kyungpook National University Physics. This is not CMU Hyung J. Kim. |
| 519 `bC4aq-4AAAAJ:oXKBmVzQOggC` | [Frontiers' exact 2024 article](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2024.1362209/full), DOI `10.3389/fphy.2024.1362209`, lists **two** H. J. Kim authors, with publisher-linked ORCIDs [`0000-0002-8265-5283`](https://pub.orcid.org/v3.0/0000-0002-8265-5283/person) and [`0000-0001-9787-4684`](https://pub.orcid.org/v3.0/0000-0001-9787-4684/person). The ORCID public person records name them Ho-Jong Kim and HongJoo Kim respectively; neither is CMU Hyung J. Kim. |

The rebuild checks all three Scholar IDs, captured titles, years, venue strings,
and captured author strings before writing `false_profile_attribution` decisions.
Only those three held rows change. The 918 observations now partition into **122
source-confirmed CMU works, 479 exact exclusions, and 317 held observations**.
Other initials-only COSINE and AMoRE records, including rows 162, 172, and 331,
remain held. The review does not change any trusted Kim cache, public map,
snapshot, or Hub dataset.
