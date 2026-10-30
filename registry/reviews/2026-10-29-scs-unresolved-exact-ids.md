# SCS unresolved Scholar ID discovery — 2026-10-29

## Scope and method

At the `2876a97269e83572cdd5172078dfc94a3fcd2b16` producer baseline, 317 included SCS people had 176 recorded Scholar IDs and 141 unresolved IDs. This discovery found 22 exact links from direct citation links on exact named CMU Experts or owner pages, or on personal and lab pages directly linked by named official CMU pages. [PR #479](https://github.com/cmccomb/map-of-research/pull/479) independently integrated the same Ann Lee and Larry A Wasserman IDs while this branch was being prepared. This review proposes the other 20 IDs. Each ID was matched to an existing stable person ID. Google Scholar profiles were not opened, and no name-only or search-result inference was used.

Earlier bounded scans of MLD, LTI, RI, S3D, CSD, and HCII/CBD were retained. New paths included HCII affiliated faculty, CMU Experts, the official MLD Website index, and directly linked CMU and personal sites. Three legacy `http://www.cs.cmu.edu/~...` owner links were separately confirmed to serve the same direct citation anchor over HTTPS before recording the HTTPS source URL. The dated local evidence is in the coordinating chat's `outputs/scs-*-2026-10-29.json` files.

## Directly evidenced IDs

| Person and stable ID | Scholar ID | Direct link source | Primary academic home |
| --- | --- | --- | --- |
| Aaron Steinfeld (`person-aaronsteinfeld-5443ab0b`) | `CQ39OowAAAAJ` | [named owner page](https://hcii.cmu.edu/people/aaron-steinfeld) | `institute/ri` — [source](https://hcii.cmu.edu/people/aaron-steinfeld) |
| Alan Montgomery (`person-alanmontgomery-6a20c040`) | `uvsSrnoAAAAJ` | [named owner page](https://www.andrew.cmu.edu/user/alm3/) | `academic_area/tepper:marketing` — [source](https://www.cmu.edu/tepper/faculty-and-research/academic-areas/marketing) |
| Andrej Risteski (`person-andrejristeski-52580d0a`) | `NWPDSEsAAAAJ` | [named owner page](https://www.andrejristeski.org/) | `department/mld` — [source](https://ml.cmu.edu/peopleindexes/core-faculty-index.v1.json) |
| Andrew Begel (`person-andrewbegel-f147eec8`) | `S2CAx3cAAAAJ` | [named owner page](https://hcii.cmu.edu/people/andrew-begel) | `department/s3d` — [source](https://hcii.cmu.edu/people/andrew-begel) |
| Barnabas Poczos (`person-barnabaspoczos-1e82e4e8`) | `sUriZlUAAAAJ` | [named owner page](https://www.cs.cmu.edu/~bapoczos/) | `department/mld` — [source](https://ml.cmu.edu/peopleindexes/core-faculty-index.v1.json) |
| Bryan Wilder (`person-bryanwilder-6e2b6ae3`) | `dQ0dT2sAAAAJ` | [named owner page](https://bryanwilder.github.io) | `department/mld` — [source](https://ml.cmu.edu/peopleindexes/core-faculty-index.v1.json) |
| Christian Cuba Samaniego (`person-christiancubasamaniego-791497de`) | `F5JTlEcAAAAJ` | [named owner page](https://christiansami.wixsite.com/cecubasamaniego) | `department/cbd` — [source](https://www.cmu.edu/cbd/people/cuba-samaneigo.html) |
| Fernando De la Torre Frade (`person-fernandodelatorrefrade-857d94c1`) | `YB8_6gkAAAAJ` | [named owner page](https://experts.cmu.edu/expert/fernando.delatorre/Fernando-De-la-Torre) | `institute/ri` — [source](https://www.ri.cmu.edu/ri-faculty/fernando-de-la-torre-frade/) |
| Graham Neubig (`person-grahamneubig-767add33`) | `wlosgkoAAAAJ` | [named owner page](https://www.phontron.com/publications) | `institute/lti` — [source](https://www.lti.cs.cmu.edu/people/faculty/neubig-graham.html) |
| Hammad Ahmad (`person-hammadahmad-dcffc18f`) | `7Oy1s9YAAAAJ` | [named owner page](https://hammadahmad.io/) | `department/s3d` — [source](https://s3d.cmu.edu/people/core-faculty/ahmad-hammad.html) |
| Justine Cassell (`person-justinecassell-9da8e0be`) | `wkC9xHMAAAAJ` | [named owner page](https://hcii.cmu.edu/people/justine-cassell) | `institute/lti` — [source](https://www.lti.cs.cmu.edu/people/faculty/cassell-justine.html) |
| Katerina Fragkiadaki (`person-katerinafragkiadaki-42756ecb`) | `FWp7728AAAAJ` | [named owner page](https://www.cs.cmu.edu/~katef/) | `department/mld` — [source](https://ml.cmu.edu/peopleindexes/core-faculty-index.v1.json) |
| Max Simchowitz (`person-maxsimchowitz-e43dffcb`) | `HjmSOFEAAAAJ` | [named owner page](https://msimchowitz.github.io/) | `department/mld` — [source](https://ml.cmu.edu/peopleindexes/core-faculty-index.v1.json) |
| Minchen Li (`person-minchenli-272be5ea`) | `Gk7K7VkAAAAJ` | [named owner page](https://www.cs.cmu.edu/~minchenl/) | `department/csd` — [source](https://www.cs.cmu.edu/~minchenl/) |
| Rashmi Korlakai Vinayak (`person-rashmikorlakaivinayak-75014cd2`) | `gUO59T8AAAAJ` | [named owner page](https://www.cs.cmu.edu/~rvinayak/) | `department/csd` — [source](https://www.csd.cs.cmu.edu/people/faculty/rashmi-korlakai-vinayak) |
| Rayid Ghani (`person-rayidghani-0861bb94`) | `_ufWH4IAAAAJ` | [named owner page](https://experts.cmu.edu/expert/rayid.ghani/Rayid-Ghani) | `department/mld` (existing) — [source](https://www.heinz.cmu.edu/faculty-research/profiles/ghani-rayid) |
| Ruslan Salakhutdinov (`person-ruslansalakhutdinov-f2200ba2`) | `ITZ1e7MAAAAJ` | [named owner page](https://www.cs.cmu.edu/~rsalakhu/) | `department/mld` — [source](https://ml.cmu.edu/peopleindexes/core-faculty-index.v1.json) |
| Todd Mowry (`person-toddmowry-6c3bbf9d`) | `MYsIXF4AAAAJ` | [named owner page](https://www.toddcmowry.org/publications-1) | `department/csd` — [source](https://www.toddcmowry.org/) |
| William W Cohen (`person-williamwcohen-c58db582`) | `8ys-38kAAAAJ` | [named owner page](https://www.cs.cmu.edu/~wcohen/) | `department/mld` — [source](https://ml.cmu.edu/peopleindexes/core-faculty-index.v1.json) |
| Yang Liu (`person-yangliu-b57c54d4`) | `LrHm5cgAAAAJ` | [named owner page](https://yangpliu.github.io/pdf/cv.pdf) | `department/csd` — [source](https://yangpliu.github.io/) |

The primary home shown above is the registry's one selected typed display home. Rayid Ghani's existing explicit home was retained. Alan Montgomery's prior `school/tepper` home was refined to the official Tepper Marketing academic area because the [current roster](https://www.cmu.edu/tepper/faculty-and-research/academic-areas/marketing) names him as Professor of Marketing, while his [CMU Scholars profile](https://scholars.cmu.edu/186-alan-montgomery) identifies his current academic position at Tepper. The school-level choice is invalid under the publication gate when an included department membership also exists; his MLD membership remains included. All newly set homes use the owning unit's registry type and an official or owner source; joint memberships remain unchanged.

## Held exact links and excluded candidates

- **Christopher Timperley:** The named [CyLab page](https://www.cylab.cmu.edu/directory/bios/timperley-christopher.html) and directly linked personal page expose `btEn24cAAAAJ`, but current CMU pages conflict between S3D and RI/NREC as his academic home. Hold the ID until the current home and membership are reconciled.
- **Sivaraman Balakrishnan:** His [owner page](https://www.stat.cmu.edu/~siva/) exposes `o7yFQXUAAAAJ`. Current pages describe a joint Statistics and MLD appointment without distinguishing the primary home. Hold the ID pending contemporary primary-home evidence.
- **David Mortensen:** The [Changeling Lab](https://changelinglab.github.io/) footer exposes `2iS5aeoAAAAJ` near his contact information, but the citation link itself does not name him. Hold pending an exact owner-attributed link.
- **Amy Melniczuk:** The prior HCII scan found an exact citation href, but the documented identity and home issue remains unresolved. Do not integrate it.
- **Jessica Hammer:** Her ID is already recorded. Keep her documented identity/home hold; this PR does not change her row or perform capture.
- A CSD “Google Scholar” link for William Kuszmaul points only to the generic Scholar homepage. Lab-wide links and links for named coauthors were excluded.

The two IDs already integrated by PR #479 and the 20 rows proposed here would increase recorded SCS IDs from 176 to 198, leaving 119 unresolved. This is registry source work only. It does not collect Scholar data, alter the immutable 696 Hub, upload a snapshot, or deploy a map.
