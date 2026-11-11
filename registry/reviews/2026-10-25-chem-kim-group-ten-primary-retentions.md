# Hyung J. Kim: ten further CMU group and publisher matches

This bounded review uses the unchanged complete 918-row Google Scholar capture
for `bC4aq-4AAAAJ` (JSON SHA-256
`ea04610e120039df7aaf83c402d193021f328b319cd41cfabf541f3942069867`).
Each Scholar row below matches an entry in the [CMU Kim Group's publication
list](https://www.chem.cmu.edu/groups/kim/pub.html) by title, year, journal,
volume, and pages. Each linked DOI registration deposited by ACS or RSC names
**Hyung J. Kim** in full. The ACS records also give his CMU Chemistry
affiliation. The 2011 RSC registration gives only the first page; the CMU list
confirms the full range. Scholar's displayed authors agree with the full DOI
bylines, rather than merely matching initials.

| Capture row and exact Scholar ID | Publisher DOI | Full DOI byline; publication details |
| --- | --- | --- |
| 63 `bC4aq-4AAAAJ:Y0pCki6q_DkC` | [ACS `10.1021/jp802595r`](https://api.crossref.org/works/10.1021%2Fjp802595r) | Youngseon Shim, Hyung J. Kim; *J. Phys. Chem. B* 112:11028–11038 (2008). |
| 86 `bC4aq-4AAAAJ:ULOm3_A8WrAC` | [ACS `10.1021/jp1122322`](https://api.crossref.org/works/10.1021%2Fjp1122322) | Nilesh R. Dhumal, Hyung J. Kim, Johannes Kiefer; *J. Phys. Chem. A* 115:3551–3558 (2011). |
| 170 `bC4aq-4AAAAJ:0EnyYjriUFMC` | [ACS `10.1021/jp992582g`](https://api.crossref.org/works/10.1021%2Fjp992582g) | Badry D. Bursulaya, Jonggu Jeon, Chia-Ning Yang, Hyung J. Kim; *J. Phys. Chem. A* 104:45–52 (2000 print). |
| 232 `bC4aq-4AAAAJ:mVmsd5A6BfQC` | [ACS `10.1021/jp9065407`](https://api.crossref.org/works/10.1021%2Fjp9065407) | Youngseon Shim, Hyung J. Kim; *J. Phys. Chem. B* 113:12964–12972 (2009). |
| 254 `bC4aq-4AAAAJ:Zph67rFs4hoC` | [ACS `10.1021/jp002146q`](https://api.crossref.org/works/10.1021%2Fjp002146q) | Jonggu Jeon, Hyung J. Kim; *J. Phys. Chem. A* 104:9812–9815 (2000). |
| 273 `bC4aq-4AAAAJ:IWHjjKOFINEC` | [ACS `10.1021/jp105021b`](https://api.crossref.org/works/10.1021%2Fjp105021b) | Youngseon Shim, Hyung J. Kim; *J. Phys. Chem. B* 114:10160–10170 (2010). |
| 281 `bC4aq-4AAAAJ:HDshCWvjkbEC` | [RSC `10.1039/C0CP01845G`](https://api.crossref.org/works/10.1039%2FC0CP01845G) | Youngseon Shim, YounJoon Jung, Hyung J. Kim; *PCCP* 13:3969–3978 (2011; CMU supplies the last page). |
| 468 `bC4aq-4AAAAJ:TQgYirikUcIC` | [ACS `10.1021/jp710128p`](https://api.crossref.org/works/10.1021%2Fjp710128p) | Youngseon Shim, Hyung J. Kim; *J. Phys. Chem. B* 112:2637–2643 (2008). |
| 687 `bC4aq-4AAAAJ:isC4tDSrTZIC` | [ACS `10.1021/jp075270j`](https://api.crossref.org/works/10.1021%2Fjp075270j) | Youngseon Shim, Hyung J. Kim; *J. Phys. Chem. B* 112:585–594 (2008 print). |
| 723 `bC4aq-4AAAAJ:blknAaTinKkC` | [ACS `10.1021/jp053578+`](https://api.crossref.org/works/10.1021%2Fjp053578%2B) | Sudha Dorairaj, Jonggu Jeon, Hyung J. Kim; *J. Phys. Chem. A* 110:429–435 (2006 print). |

The DOI metadata has an earlier online year for rows 170, 687, and 723. Its
`published-print` year, the CMU group list, and the Scholar journal issue agree.
The raw Scholar title and year remain unchanged.

`rebuild_chem_kim_918_review.py` guards all ten decisions by Scholar ID, exact
saved title/year, and DOI in the linked CMU ORCID work snapshot. The draft
ledger now partitions all 918 rows into **33 source-confirmed works, 394 exact
exclusions, and 491 held rows** (81 CMU ORCID person signals needing output
review and 410 other holds). The trusted release stays `held`; this is a
nonoperational proposal and does not add a canonical Kim cache or a public
partial profile.
