# Kim Scholar capture: seven exact-author and duplicate decisions

This review uses the complete, unchanged 918-row saved Scholar capture for
profile `bC4aq-4AAAAJ`. It changes seven **held** citation IDs only. The
source-attribution decision and the separate question of public map eligibility
must not be conflated. Kim's entire profile remains held from public release.

| Capture row | Scholar citation ID | Decision | Exact output evidence |
| ---: | --- | --- | --- |
| 129 | `bC4aq-4AAAAJ:GUYAmugLYisC` | Exclude other person | The [original *Neurology* 75:1432–1438 article](https://www.neurology.org/doi/10.1212/WNL.0b013e3181f88191), DOI `10.1212/WNL.0b013e3181f88191`, identifies its H. J. Kim author as correspondent **Ho Jin Kim** at the National Cancer Center. The saved title, 2010 year, venue, and citation ID match. |
| 442 | `bC4aq-4AAAAJ:urP0JZOBBUsC` | Exclude other person | The [original *Neurology* 76:672–674 article](https://www.neurology.org/doi/10.1212/WNL.0b013e31820c30af), DOI `10.1212/WNL.0b013e31820c30af`, identifies the last H. J. Kim author as correspondent **Ho Jin Kim** at the National Cancer Center. The saved title, 2011 year, venue, and citation ID match. |
| 447 | `bC4aq-4AAAAJ:jtI9f0ekYq0C` | Exclude other person | The [original Berkeley-hosted IEEE CDC paper](https://people.eecs.berkeley.edu/~sastry/pubs/PDFs%20of%20Pubs2000-2005/Publications%20of%20Postdocs/ShimDavid/ShimDecentralized2003.pdf), DOI `10.1109/CDC.2003.1271710`, prints second author **H. Jin Kim** at Berkeley EECS, not CMU Chemistry. [KAIST's exact-output record](https://pure.kaist.ac.kr/en/publications/decentralized-nonlinear-model-predictive-control-of-multiple-flyi/) independently gives H. Jin Kim, and [coauthor Shankar Sastry's CV](https://people.eecs.berkeley.edu/~sastry/Sastry-cv26.pdf) gives the saved longer title, same author team, year, and pages 3621–3626. |
| 802 | `bC4aq-4AAAAJ:sSrBHYA8nusC` | Confirm CMU author, published abstract | [Elsevier's publisher-deposited DOI record](https://api.crossref.org/works/10.1016%2Fj.bpj.2014.11.1162) gives **Hyunjin Kim, Hyung J. Kim, Miguel Llinás** for the exact title in *Biophysical Journal* 108(2):210a (2015), DOI `10.1016/j.bpj.2014.11.1162`. The [original Biophysical Society 2015 program](https://www.biophysics.org/Portals/0/EasyDNNnews/Uploads/2169/2015_Program_Guide.pdf) prints the same full byline under poster 1052-Pos Board B3. [CMU's Kim faculty page](https://www.cmu.edu/chemistry/people/faculty/kim.html) documents his plasminogen work with Llinás, and the [CMU Kim group roster](https://www.chem.cmu.edu/groups/kim/people/) identifies Hyunjin Kim as a member. This is a published meeting abstract, not a full research article. |
| 892 | `bC4aq-4AAAAJ:AXPGKjj_ei8C` | Exclude duplicate of confirmed row 359, `bC4aq-4AAAAJ:QIV2ME_5wuYC` | The [AIP deposited DOI record](https://api.crossref.org/works/10.1063%2F1.478952) names **Hyung J. Kim** at CMU Chemistry for one 1999 *JCP* 110(19):9656–9665 article, DOI `10.1063/1.478952`. Row 892 repeats the row 359 authors, year, volume, issue, and pages under a category-prefixed/truncated title, with no separate citation count. |
| 893 | `bC4aq-4AAAAJ:BqipwSGYUEgC` | Exclude duplicate of confirmed row 897, `bC4aq-4AAAAJ:bnK-pcrLprsC` | The [SPIE deposited DOI record](https://api.crossref.org/works/10.1117%2F12.306113) names **Hyung J. Kim**, Bursulaya, Jeon, and Zichi for the one 1998 proceedings paper at DOI `10.1117/12.306113`, volume 3273, pages 90–99. Row 893 is the same title and team as row 897, rendered with metadata prefixes and no separate citation count. |
| 896 | `bC4aq-4AAAAJ:XiVPGOgt02cC` | Exclude duplicate of confirmed row 94, `bC4aq-4AAAAJ:eQOLeE2rZwMC` | The [AIP deposited DOI record](https://api.crossref.org/works/10.1063%2F1.477102) names **Hyung J. Kim** at CMU Chemistry for one 1998 *JCP* 109(12):4911–4919 article, DOI `10.1063/1.477102`. Row 896 repeats row 94's authors, year, volume, issue, and pages under a category-prefixed/truncated title, with no separate citation count. |

The three duplicate exclusions do not deny CMU authorship. They prevent the
same DOI from creating two distinct graph works. Rows 456 and 501 still have
only initials in the original APS records and stay held; title or research area
alone does not establish either author's full identity.

After these exact decisions: **122 confirmed, 459 excluded, 337 held = 918**.
The confirmed count includes one newly identified published abstract, not
three extra copies of already confirmed papers. The original Scholar capture
and offline candidate cache remain byte-identical, and no Kim cache or subset
is installed on the public map.
