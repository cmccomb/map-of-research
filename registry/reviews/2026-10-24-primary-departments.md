# Primary academic home review, 2026-10-24

The [row-level review ledger](2026-10-24-primary-departments.csv) records all
187 people needing an explicit review: 165 included people with multiple
candidate units, two included people with only a campus facet, and 20
excluded-only people with multiple recorded units. It reconciles the
[four-school primary review](2026-10-24-other-schools-primary.md) and the
[Physics primary review](2026-10-24-physics-primary-departments.csv) with the
earlier Engineering, SCS, and MCS choices. All memberships remain available
for filters. A display primary is not an assertion about a university HR
primary appointment.

| Population | People | One included academic unit | School fallback | Reviewed source | Inferred profile title | Inferred PhD advisees | First-listed display tie-break | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Included roster | 1,480 | 1,096 | 154 | 189 | 10 | 20 | 11 | 0 |
| Excluded-only roster | 521 | 0 | 0 | 0 | 0 | 0 | 0 | 521 |
| Entire registry | 2,001 | 1,096 | 154 | 189 | 10 | 20 | 11 | 521 |

The included count incorporates the later correction of Itamar Vigdorovich's
future Fall 2027 Math appointment, which remains excluded pending its start.

The explicit choices include their source URL, verification date, and basis in
`people.csv`. `reviewed` means a source-backed map display choice;
`school_fallback` means only a broad school-level home is supported by an
included shared roster; `inferred_profile_title` means the selected academic
unit is named in a CMU faculty-page title despite a joint appointment; and
`inferred_advisee_count` means that unit has more distinct PhD advisees with
publicly stated degree departments in the available lab roster. The latter
two are editorial inferences with limits recorded in the ledger. The typed
home fields distinguish actual departments from institutes, constituent
schools, Tepper academic areas, broad school fallbacks, and reviewed campus
display homes. The legacy
`primary_department_id` is populated only for actual departments.

`display_tie_break_first_listed` is an explicitly reviewed visual choice for
joint appointments whose official CMU titles list multiple academic units. The
[Philosophy profile for Jeremy Avigad](https://www.cmu.edu/dietrich/philosophy/people/faculty/jeremy-avigad.html)
titles him “Professor of Philosophy and Mathematical Sciences,” so the map
displays Philosophy. The
[Provost's University Professors list](https://www.cmu.edu/leadership/the-provost/university-professors)
titles Robert F. Sekerka “University Professor of Physics and Mathematics,
Emeritus,” so the map displays Physics. First-listed order is a display
tie-break, not an institutional rank or an exclusion of the Math appointment.
All four department memberships remain included for filters and collection.
Avigad's [faculty-authored student list](https://www.andrew.cmu.edu/user/avigad/students.html),
cross-checked against the [CMU Pure and Applied Logic alumni list](https://www.logic.cmu.edu/alumni.html)
and [Philosophy graduate programs](https://www.cmu.edu/dietrich/philosophy/graduate/phd/),
places 8 of 17 listed PhD advisees in Philosophy programs, 5 in Mathematical
Sciences, and 4 in Computer Science. This corroborates the displayed Philosophy
home but is a count of listed advisees, not an institutional primary appointment;
the chosen display rule remains first-listed department. Sekerka's
[CMU-hosted CV](https://sekerkaweb.phys.cmu.edu/SekerkaCV2015.html)
documents the Physics and Mathematics appointments but has no degree-coded
advisee list, so the same title-order display rule supplies his visual home.
[Jenelle Feather's Psychology profile](https://www.cmu.edu/dietrich/psychology/directory/core-training-faculty/feather-jenelle.html)
titles her Assistant Professor of Psychology and the Neuroscience Institute;
Psychology is the labeled display tie-break, with both memberships retained.
[Mark Kamlet's SDS profile](https://www.cmu.edu/dietrich/sds/people/faculty/mark-kamlet.html)
and [Heinz profile](https://www.heinz.cmu.edu/faculty-research/profiles/kamlet-mark)
both list Social and Decision Sciences before Heinz in their joint-appointment
description. The map uses SDS as a labeled visual choice while retaining his
CMIST, Heinz, and SDS memberships; his current CMIST graduate committee role
does not establish an institutional primary academic appointment.
[Paul Fischbeck's Engineering profile](https://engineering.cmu.edu/directory/bios/fischbeck-paul.html)
lists Engineering and Public Policy before Social and Decision Sciences, so
the map uses EPP as a reviewed display choice. The SDS course catalog lists
SDS first, making a consistent first-listed tie-break unavailable. His two
emeritus appointments remain included.

Every included person now has a display home. Edward Rubin and Elias Towe
have labeled first-listed choices; their joint memberships remain included,
and the reviews do not claim a complete advisee comparison.

[Karimulla Shaikh](https://sv.cmu.edu/directory/faculty-and-researchers/bios/shaikh-karimulla.html)
and [Ravi Thomas](https://sv.cmu.edu/directory/faculty-and-researchers/bios/thomas-ravi.html)
have reviewed `campus` display homes from their CMU Silicon Valley teaching
memberships. Their III adjunct memberships remain excluded. The campus
choice does not assert an academic department or change publication
collection.

[Newell R. Washburn's faculty-authored 2020 CV](https://purebase.com/wp-content/uploads/2020/10/NRW_cv_2020.pdf)
lists 14 current and former PhD advisees: eight Chemistry, three BME, and
three MSE. It describes MSE as a courtesy appointment. His
[current group roster](https://groups.chem.cmu.edu/washburn/people/) also
identifies two Chemistry PhD students. Chemistry is a labeled historical
advising-count inference. The
[Siyang Zheng lab roster](https://www.cmu.edu/bme/zhenglab/team/index.html)
identifies six BME PhD students and one ECE PhD student; BME is an inferred
display primary from a page that appears to be dated around 2024–25.
[Andrew Ilyas lists three current students](https://andrewilyas.com/):
[Tori Qiu](https://cos.s3d.cmu.edu/people/students/qiu-torri.html) is in
Societal Computing/S3D, [Matan Shtepel](https://csd.cmu.edu/people/doctoral-student/matan-shtepel)
is in CSD, and [Jeffrey Wu](https://ml.cmu.edu/people/phd-students) is listed
under MLD PhD students. Among Ilyas's *two included faculty appointments*,
the observed degree-department count is S3D 1 versus ECE 0. S3D is a bounded
display inference from the current listed students; the CSD and MLD students
are retained in the evidence, but Ilyas has no included CSD or MLD faculty
appointment. This does not assert a university HR primary or a complete
advising history.
[Chris Hendrickson's 2026–27 CMU CEE catalog entry](https://coursecatalog.web.cmu.edu/schools-colleges/collegeofengineering/departmentofcivilandenvironmentalengineering/)
titles him Hamerschlag University Professor Emeritus *of Civil and Environmental
Engineering*. The [College of Engineering profile](https://engineering.cmu.edu/directory/bios/hendrickson-chris.html)
also records EPP and Heinz College. Located doctoral records span CEE, EPP,
and Heinz, without a complete lifetime denominator. CEE is a title-based
display inference from the current CMU catalog; all three memberships remain
included.
[Byron Yu's CMU-hosted group roster](https://users.ece.cmu.edu/~byronyu/people.html)
names 12 Neural Computation PhD advisees, three ECE advisees, and one BME
advisee among its degree-coded current and former students. His inferred
display primary is Neuroscience Institute; the roster may not represent every
past advisee. [Lujo Bauer's CMU-hosted student roster](https://users.ece.cmu.edu/~lbauer/)
lists 12 degree-labeled CMU ECE PhD students (three current, nine former)
versus eight Societal Computing PhD students (four current, four former).
Entries explicitly credited to another advisor are excluded from this count.
The former-student section includes people Bauer says he advised *or worked
with closely*, so ECE is a labeled inference from his listed PhD students,
not a claim about a complete advising history. This full-roster comparison
supersedes the current-cohort-only S3D choice in the earlier
[Heinz](2026-10-24-heinz.md) and
[four-school](2026-10-24-other-schools-primary.md) reviews.
[Tzahi Cohen-Karni's CMU lab roster](https://www.cmu.edu/cohenkarnilab/people/index.html)
supports the separate MSE inference from eight current MSE and seven current
BME PhD students, with both appointments retained.

[Pedro Ferreira's CMU-hosted faculty profile](https://www.contrib.andrew.cmu.edu/~pedrof/)
names both the Heinz School of Information Systems and Management (ISM) and
Engineering and Public Policy (EPP). His
[January 2025 CV](https://www.heinz.cmu.edu/faculty-research/profiles/profiles/pedro-ferreira/_files/pmfcv012025.pdf)
names ten CMU Heinz PhD students and nine EPP PhD students across graduated
primary, graduated co-advised, and current groups; one Temple co-advisee is
outside the CMU comparison. The CV identifies the Heinz students at the
*college* level, so the choice of ISM is an inference combining that narrow
10-to-9 lead with the self-profile's explicit constituent-school position.
His EPP, Heinz ISM, and shared Heinz memberships remain included. This
supersedes the school-level `heinz-faculty` placeholder in the earlier
[Heinz review](2026-10-24-heinz.md). The
[Heinz and Tepper hierarchy review](2026-10-24-heinz-tepper-primary-units.csv)
also records 23 Tepper people listed in one official academic area and five
listed in two; academic areas are not departments, and the shared roster
containers do not by themselves identify a person's constituent home.
[Niepa's current lab roster](https://www.cheme.engineering.cmu.edu/research-groups/niepa-lab.html)
lists nine doctorate students. Five have published ChemE degree evidence:
[Camila Cue](https://engineering.cmu.edu/news-events/news/2026/06/08-fellowships.html),
[Hannah Gedde](https://www.cmu.edu/news/stories/archives/2025/september/physical-pursuits-help-cmu-faculty-attain-a-work-life-balance-that-matters),
[Shakira Martínez Vásquez](https://www.cheme.engineering.cmu.edu/directory/bios/niepa-tagbo.html),
[Abraham Polanco](https://www.cheme.engineering.cmu.edu/directory/bios/niepa-tagbo.html),
and [David Soares](https://sites.google.com/site/ewgomez/members/lab-alumni).
The last source is Soares's former Penn State lab rather than CMU.
Three have published BME degree evidence:
[Amy Apgar](https://engineering.cmu.edu/news-events/news/2025/08/15-digital-humanism.html),
[Ann Badia](https://engineering.cmu.edu/news-events/news/2025/04/10-cmlh-fellows.html),
and [Anghea Dolisca](https://engineering.cmu.edu/about-us/community-engagement/student-council.html).
Rawia Marafi's CMU degree department is not explicitly stated. Even if she is
in BME, the observed ChemE count leads 5 to 4. ChemE is an inferred display
primary based on this current roster, not a claim about all past advisees or
Niepa's university HR appointment.
[Mitchell Small's CMU profile](https://epp.engineering.cmu.edu/directory/bios/small-mitchell.html)
titles him EPP emeritus while describing a joint CEE appointment, and
[Theresa Mayer's CMU profile](https://www.ece.cmu.edu/directory/bios/Theresa%20Mayer.html)
titles her ECE professor while describing a joint MSE appointment. These are
title-based display choices, with both memberships retained.

The CMU-Africa roster unit is treated as an academic department under
Engineering for this dataset. Its 12 included people with no other included
department inherit it as their display primary. For the three cross-appointed
people, [Tim Brown's current profile](https://www.africa.engineering.cmu.edu/about/contact/directory/bios/brown-tim.html)
titles him Professor of EPP and Director of CMU-Africa, and
[Barry Rawn's Engineering profile](https://engineering.cmu.edu/directory/bios/rawn-barry.html)
titles him ECE teaching professor working at CMU-Africa. Their display choices
use those professorship titles while retaining the CMU-Africa department.
[Ismaila Dabo's current profile](https://www.africa.engineering.cmu.edu/about/contact/directory/bios/dabo-ismaila.html)
titles both CMU-Africa and MSE; a
[CMU MSE appointment account](https://www.mse.engineering.cmu.edu/news/2024/09/12-dabo-global-partnership.html)
describes his joining the MSE faculty to establish its first substantial
presence at CMU-Africa. That context supports MSE as a reviewed display
choice while retaining CMU-Africa. None of these choices asserts a university
HR primary appointment. CMU Silicon Valley is a campus
facet, so it cannot be a primary. Only substantive included academic
memberships can supply a primary; excluded courtesy, affiliate, and adjunct
memberships remain filterable but do not supply one. All 521 excluded-only
people are therefore unresolved, including Elizabeth Wayne, whose CMU role is
adjunct. [Rafal Wlodarski's current ECE teaching profile](https://www.sv.cmu.edu/directory/faculty-and-researchers/bios/wlodarski-rafal.html)
supports an included ECE membership alongside the Silicon Valley campus.

The 487 Scholar-ID holders include the later Bio/Chemistry IDs, directly linked
profiles in four more schools, and David R. Owen's user-supplied current ID.
All 457 included Scholar-ID holders have a typed display home. The other 30
Scholar-ID holders are excluded-only and have no display primary.
The earlier Bio/Chemistry ID additions retain their identity provenance. The
later four-school review changed some display choices when it added substantive
cross-appointments; Alison Barth, for example, now has included Biological
Sciences and Neuroscience Institute memberships and a reviewed Bio choice.
The three [directly verified Bio Scholar profiles](2026-10-24-bio-scholar-verified-profiles.md)
also have eligible primaries: C. Joel McManus has a reviewed Biological Sciences
choice across Bio and CBD; Elizabeth Ransey and En Cai derive Biological Sciences
from their sole included unit.
[Murphy's current CBD directory](https://www.cmu.edu/cbd/people/voting-faculty.html)
titles him Computational Biology professor emeritus; his
[CMU-hosted CV](https://murphylab.cbd.cmu.edu/MurphyShortCV.htm) dates the
Biological Sciences professorship to 2021. CBD is the reviewed display primary
for the merged identity, while both department records are retained.
